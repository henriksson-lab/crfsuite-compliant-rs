//! Cross-validation: Rust CQDB reader vs C model data.
//! Load the label and attribute CQDBs from a real model file and verify
//! that all lookups produce identical results.

use crfsuite_compliant_rs::cqdb::lookup3;
use crfsuite_compliant_rs::cqdb::{CqdbReader, CqdbWriter};
use std::path::PathBuf;

fn project_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).to_path_buf()
}

fn model_bytes() -> Vec<u8> {
    std::fs::read(project_root().join("test_data/model_c.bin")).expect("model_c.bin not found")
}

fn read_u32(buf: &[u8], off: usize) -> u32 {
    u32::from_le_bytes([buf[off], buf[off + 1], buf[off + 2], buf[off + 3]])
}

/// Parse the model header to find the label and attribute CQDB offsets and sizes.
/// Returns ((offset, size), (offset, size)) for label CQDB and attr CQDB.
/// Size is taken from the CQDB chunk's own size field (not the gap between offsets,
/// which may include alignment padding added by the model writer).
fn find_cqdb_sections(model: &[u8]) -> ((usize, usize), (usize, usize)) {
    assert!(&model[0..4] == b"lCRF", "not a CRF model");

    let off_labels = read_u32(model, 32) as usize;
    let off_attrs = read_u32(model, 36) as usize;

    // Read the size from each CQDB chunk's own header (offset +4)
    let label_size = read_u32(model, off_labels + 4) as usize;
    let attr_size = read_u32(model, off_attrs + 4) as usize;

    ((off_labels, label_size), (off_attrs, attr_size))
}

#[test]
fn test_label_cqdb_reader() {
    let model = model_bytes();
    let ((off, size), _) = find_cqdb_sections(&model);
    let cqdb_buf = &model[off..off + size];

    let reader = CqdbReader::open(cqdb_buf).expect("failed to open label CQDB");
    let num = reader.num();
    assert!(num > 0, "label CQDB should have entries");

    // Read all labels via to_string and verify round-trip with to_id
    for id in 0..num as i32 {
        let s = reader
            .to_string(id)
            .unwrap_or_else(|| panic!("to_string({}) failed", id));
        let got_id = reader
            .to_id(s)
            .unwrap_or_else(|| panic!("to_id({:?}) failed", s));
        assert_eq!(
            got_id, id,
            "round-trip failed for {:?}: got {} expected {}",
            s, got_id, id
        );
    }

    // Known labels from the training data
    let expected_labels = ["B-NP", "I-NP", "B-VP", "B-PP"];
    for label in &expected_labels {
        assert!(reader.to_id(label).is_some(), "label {:?} not found", label);
    }
    assert!(reader.to_id("NONEXISTENT").is_none());
}

#[test]
fn test_attr_cqdb_reader() {
    let model = model_bytes();
    let (_, (off, size)) = find_cqdb_sections(&model);
    let cqdb_buf = &model[off..off + size];

    let reader = CqdbReader::open(cqdb_buf).expect("failed to open attr CQDB");
    let num = reader.num();
    assert!(num > 0, "attr CQDB should have entries");

    // Round-trip all attributes
    for id in 0..num as i32 {
        let s = reader
            .to_string(id)
            .unwrap_or_else(|| panic!("attr to_string({}) failed", id));
        let got_id = reader
            .to_id(s)
            .unwrap_or_else(|| panic!("attr to_id({:?}) failed", s));
        assert_eq!(got_id, id, "attr round-trip failed for {:?}", s);
    }

    // Known attributes from the training data
    assert!(reader.to_id("w[0]=the").is_some());
    assert!(reader.to_id("pos[0]=DT").is_some());
    assert!(reader.to_id("NONEXISTENT_ATTR").is_none());
}

#[test]
fn test_lookup3_against_c() {
    // Cross-validate lookup3 hash against known values computed by C.
    // We'll verify by checking that CQDB lookups work (which internally
    // depend on hash correctness).
    let model = model_bytes();
    let ((off, size), _) = find_cqdb_sections(&model);
    let reader = CqdbReader::open(&model[off..off + size]).unwrap();

    // If to_id works for every label, the hash is correct
    let num = reader.num();
    for id in 0..num as i32 {
        let s = reader.to_string(id).unwrap();
        let found = reader.to_id(s);
        assert_eq!(found, Some(id), "hash-based lookup failed for {:?}", s);
    }
}

#[test]
fn test_lookup3_known_values() {
    // Test hash of empty string (just null terminator)
    let h = lookup3::hashlittle(&[0], 0);
    // Value from C: hashlittle("\0", 1, 0) = should be deterministic
    assert_ne!(h, 0);

    // Test that same string always gives same hash
    let h1 = lookup3::hash_string("hello");
    let h2 = lookup3::hash_string("hello");
    assert_eq!(h1, h2);

    // Different strings give different hashes (with high probability)
    let h3 = lookup3::hash_string("world");
    assert_ne!(h1, h3);
}

#[test]
fn test_cqdb_writer_matches_c_label_cqdb() {
    // Read the label CQDB from the C model, extract all entries,
    // rebuild it with the Rust writer, and compare bytes.
    let model = model_bytes();
    let ((off, size), _) = find_cqdb_sections(&model);
    let c_cqdb = &model[off..off + size];

    let reader = CqdbReader::open(c_cqdb).unwrap();
    let num = reader.num();

    // Rebuild using Rust writer with entries in ID order
    let mut writer = CqdbWriter::new(0);
    for id in 0..num as i32 {
        let s = reader.to_string(id).unwrap();
        writer.put(s, id);
    }
    let rust_cqdb = writer.close();

    assert_eq!(
        c_cqdb.len(),
        rust_cqdb.len(),
        "Label CQDB size mismatch: C={} Rust={}",
        c_cqdb.len(),
        rust_cqdb.len()
    );
    assert_eq!(c_cqdb, &rust_cqdb[..], "Label CQDB bytes differ");
}

#[test]
fn test_cqdb_writer_matches_c_attr_cqdb() {
    let model = model_bytes();
    let (_, (off, size)) = find_cqdb_sections(&model);
    let c_cqdb = &model[off..off + size];

    let reader = CqdbReader::open(c_cqdb).unwrap();
    let num = reader.num();

    let mut writer = CqdbWriter::new(0);
    for id in 0..num as i32 {
        let s = reader.to_string(id).unwrap();
        writer.put(s, id);
    }
    let rust_cqdb = writer.close();

    assert_eq!(
        c_cqdb.len(),
        rust_cqdb.len(),
        "Attr CQDB size mismatch: C={} Rust={}",
        c_cqdb.len(),
        rust_cqdb.len()
    );
    assert_eq!(c_cqdb, &rust_cqdb[..], "Attr CQDB bytes differ");
}
