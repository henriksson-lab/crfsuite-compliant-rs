//! Test: write a model with Rust, read it back with Rust ModelReader, verify all data.

use crfsuite_compliant_rs::crf1d::feature::{self, Feature, FT_STATE, FT_TRANS};
use crfsuite_compliant_rs::model::ModelReader;
use crfsuite_compliant_rs::model_writer;
use std::path::PathBuf;
use std::process::Command;

fn project_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
}

fn rust_bin() -> PathBuf {
    if let Some(bin) = option_env!("CARGO_BIN_EXE_crfsuite-rs") {
        return PathBuf::from(bin);
    }
    project_root().join("target/debug/crfsuite-rs")
}

fn c_bin() -> PathBuf {
    project_root().join("crfsuite/frontend/.libs/crfsuite")
}

fn c_lib_path() -> PathBuf {
    project_root().join("crfsuite/lib/crf/.libs")
}

fn read_u32(buf: &[u8], offset: usize) -> u32 {
    u32::from_le_bytes(buf[offset..offset + 4].try_into().unwrap())
}

fn assert_zero_padding(buf: &[u8], start: usize, end: usize) {
    assert!(
        buf[start..end].iter().all(|&b| b == 0),
        "expected zero padding in byte range {start}..{end}"
    );
}

#[test]
fn test_model_roundtrip() {
    // Create some features
    let features = vec![
        Feature {
            ftype: FT_STATE,
            src: 0,
            dst: 0,
            freq: 2.0,
        },
        Feature {
            ftype: FT_STATE,
            src: 0,
            dst: 1,
            freq: 1.0,
        },
        Feature {
            ftype: FT_STATE,
            src: 1,
            dst: 0,
            freq: 1.0,
        },
        Feature {
            ftype: FT_STATE,
            src: 1,
            dst: 1,
            freq: 3.0,
        },
        Feature {
            ftype: FT_TRANS,
            src: 0,
            dst: 0,
            freq: 1.0,
        },
        Feature {
            ftype: FT_TRANS,
            src: 0,
            dst: 1,
            freq: 2.0,
        },
        Feature {
            ftype: FT_TRANS,
            src: 1,
            dst: 0,
            freq: 1.0,
        },
        Feature {
            ftype: FT_TRANS,
            src: 1,
            dst: 1,
            freq: 1.0,
        },
    ];
    let weights = vec![0.5, -0.3, 0.2, 0.7, 0.1, -0.1, 0.4, -0.2];
    let labels = vec!["B-NP".to_string(), "I-NP".to_string()];
    let attrs = vec!["w[0]=the".to_string(), "pos[0]=DT".to_string()];

    let (attr_refs, label_refs) = feature::init_references(&features, attrs.len(), labels.len());

    let model_bytes = model_writer::write_model(
        &features,
        &weights,
        &labels,
        &attrs,
        &label_refs,
        &attr_refs,
    );

    // Read it back
    let model = ModelReader::open(&model_bytes).expect("failed to read written model");
    assert_eq!(model.num_labels(), 2);
    assert_eq!(model.num_attrs(), 2);

    // Check labels
    assert_eq!(model.to_label(0), Some("B-NP"));
    assert_eq!(model.to_label(1), Some("I-NP"));

    // Check attrs
    assert_eq!(model.to_attr(0), Some("w[0]=the"));
    assert_eq!(model.to_attr(1), Some("pos[0]=DT"));

    // Check features are readable
    let num_feats = model.num_features();
    assert_eq!(num_feats, 8); // all have non-zero weights

    for fid in 0..num_feats {
        let f = model
            .get_feature(fid)
            .unwrap_or_else(|| panic!("feature {} missing", fid));
        assert!(f.weight.is_finite());
    }

    // Check label refs point to transition features
    for lid in 0..2 {
        let refs = model.get_labelref(lid);
        for &fid in refs {
            let f = model.get_feature(fid as u32).unwrap();
            assert_eq!(f.ftype, FT_TRANS as u32);
        }
    }

    // Check attr refs point to state features
    for aid in 0..2 {
        let refs = model.get_attrref(aid);
        for &fid in refs {
            let f = model.get_feature(fid as u32).unwrap();
            assert_eq!(f.ftype, FT_STATE as u32);
        }
    }
}

#[test]
fn test_model_with_pruning() {
    // Some features have zero weight → should be pruned
    let features = vec![
        Feature {
            ftype: FT_STATE,
            src: 0,
            dst: 0,
            freq: 1.0,
        },
        Feature {
            ftype: FT_STATE,
            src: 0,
            dst: 1,
            freq: 1.0,
        },
        Feature {
            ftype: FT_STATE,
            src: 1,
            dst: 0,
            freq: 1.0,
        }, // zero weight
        Feature {
            ftype: FT_TRANS,
            src: 0,
            dst: 0,
            freq: 1.0,
        },
        Feature {
            ftype: FT_TRANS,
            src: 0,
            dst: 1,
            freq: 1.0,
        }, // zero weight
    ];
    let weights = vec![0.5, -0.3, 0.0, 0.1, 0.0]; // features 2,4 pruned
    let labels = vec!["A".to_string(), "B".to_string()];
    let attrs = vec!["x".to_string(), "y".to_string()];

    let (attr_refs, label_refs) = feature::init_references(&features, attrs.len(), labels.len());

    let model_bytes = model_writer::write_model(
        &features,
        &weights,
        &labels,
        &attrs,
        &label_refs,
        &attr_refs,
    );

    let model = ModelReader::open(&model_bytes).expect("failed to read model");
    assert_eq!(model.num_features(), 3); // only 3 active features
    assert_eq!(model.num_attrs(), 1); // only attr 0 is active (attr 1 had zero weight)
}

#[test]
fn test_model_section_layout_offsets_and_padding() {
    let features = vec![
        Feature {
            ftype: FT_STATE,
            src: 0,
            dst: 0,
            freq: 1.0,
        },
        Feature {
            ftype: FT_STATE,
            src: 1,
            dst: 1,
            freq: 1.0,
        },
        Feature {
            ftype: FT_STATE,
            src: 2,
            dst: 0,
            freq: 1.0,
        },
        Feature {
            ftype: FT_TRANS,
            src: 0,
            dst: 1,
            freq: 1.0,
        },
        Feature {
            ftype: FT_TRANS,
            src: 1,
            dst: 0,
            freq: 1.0,
        },
    ];
    let weights = vec![0.5, 0.0, -1.25, 0.75, 0.0];
    let labels = vec!["A".to_string(), "B".to_string()];
    let attrs = vec!["x".to_string(), "unused".to_string(), "z".to_string()];
    let (attr_refs, label_refs) = feature::init_references(&features, attrs.len(), labels.len());

    let bytes = model_writer::write_model(
        &features,
        &weights,
        &labels,
        &attrs,
        &label_refs,
        &attr_refs,
    );

    let total_size = read_u32(&bytes, 4) as usize;
    let off_features = read_u32(&bytes, 28) as usize;
    let off_labels = read_u32(&bytes, 32) as usize;
    let off_attrs = read_u32(&bytes, 36) as usize;
    let off_labelrefs = read_u32(&bytes, 40) as usize;
    let off_attrrefs = read_u32(&bytes, 44) as usize;

    assert_eq!(&bytes[0..4], b"lCRF");
    assert_eq!(total_size, bytes.len());
    assert_eq!(&bytes[8..12], b"FOMC");
    assert_eq!(read_u32(&bytes, 12), 100);
    assert_eq!(read_u32(&bytes, 16), 0);
    assert_eq!(read_u32(&bytes, 20), 2);
    assert_eq!(read_u32(&bytes, 24), 2);

    assert_eq!(off_features, 48);
    assert!(off_features < off_labels);
    assert!(off_labels < off_attrs);
    assert!(off_attrs < off_labelrefs);
    assert!(off_labelrefs < off_attrrefs);

    assert_eq!(&bytes[off_features..off_features + 4], b"FEAT");
    assert_eq!(
        read_u32(&bytes, off_features + 4) as usize,
        off_labels - off_features
    );
    assert_eq!(read_u32(&bytes, off_features + 8), 3);

    assert_eq!(&bytes[off_labels..off_labels + 4], b"CQDB");
    let labels_end = off_labels + read_u32(&bytes, off_labels + 4) as usize;
    assert_eq!(labels_end, off_attrs);

    assert_eq!(&bytes[off_attrs..off_attrs + 4], b"CQDB");
    let attrs_end = off_attrs + read_u32(&bytes, off_attrs + 4) as usize;
    assert!(attrs_end <= off_labelrefs);
    assert_eq!(off_labelrefs % 4, 0);
    assert_zero_padding(&bytes, attrs_end, off_labelrefs);

    assert_eq!(&bytes[off_labelrefs..off_labelrefs + 4], b"LFRF");
    assert_eq!(read_u32(&bytes, off_labelrefs + 8), 4);
    let labelrefs_end = off_labelrefs + read_u32(&bytes, off_labelrefs + 4) as usize;
    assert!(labelrefs_end <= off_attrrefs);
    assert_eq!(off_attrrefs % 4, 0);
    assert_zero_padding(&bytes, labelrefs_end, off_attrrefs);

    assert_eq!(&bytes[off_attrrefs..off_attrrefs + 4], b"AFRF");
    assert_eq!(read_u32(&bytes, off_attrrefs + 8), 2);
    assert_eq!(
        off_attrrefs + read_u32(&bytes, off_attrrefs + 4) as usize,
        total_size
    );
}

#[test]
fn test_unusual_weight_bits_roundtrip() {
    let nan = f64::from_bits(0x7ff8_0000_0000_1234);
    let features = vec![
        Feature {
            ftype: FT_STATE,
            src: 0,
            dst: 0,
            freq: 1.0,
        },
        Feature {
            ftype: FT_STATE,
            src: 0,
            dst: 1,
            freq: 1.0,
        },
        Feature {
            ftype: FT_TRANS,
            src: 0,
            dst: 0,
            freq: 1.0,
        },
        Feature {
            ftype: FT_TRANS,
            src: 0,
            dst: 1,
            freq: 1.0,
        },
        Feature {
            ftype: FT_STATE,
            src: 1,
            dst: 0,
            freq: 1.0,
        },
    ];
    let weights = vec![0.0, -0.0, f64::MIN_POSITIVE, -1.0e200, nan];
    let labels = vec!["A".to_string(), "B".to_string()];
    let attrs = vec!["x".to_string(), "y".to_string()];
    let (attr_refs, label_refs) = feature::init_references(&features, attrs.len(), labels.len());

    let model_bytes = model_writer::write_model(
        &features,
        &weights,
        &labels,
        &attrs,
        &label_refs,
        &attr_refs,
    );
    let model = ModelReader::open(&model_bytes).expect("failed to read unusual-weight model");

    assert_eq!(model.num_features(), 3);
    assert_eq!(model.num_attrs(), 1);
    assert_eq!(
        model.get_feature(0).unwrap().weight.to_bits(),
        f64::MIN_POSITIVE.to_bits()
    );
    assert_eq!(
        model.get_feature(1).unwrap().weight.to_bits(),
        (-1.0e200f64).to_bits()
    );
    assert_eq!(
        model.get_feature(2).unwrap().weight.to_bits(),
        nan.to_bits()
    );
}

#[test]
fn test_minimal_trained_model_bytes_match_c() {
    if !c_bin().exists() {
        eprintln!("SKIP: C crfsuite binary not found");
        return;
    }

    let tmp = tempfile::tempdir().unwrap();
    let train = tmp.path().join("train.iwa");
    std::fs::write(&train, "A\tx\n\n").unwrap();
    let c_model = tmp.path().join("c.bin");
    let rs_model = tmp.path().join("rs.bin");

    let c = Command::new(c_bin())
        .args([
            "learn",
            "-a",
            "lbfgs",
            "-p",
            "max_iterations=0",
            "-m",
            c_model.to_str().unwrap(),
            train.to_str().unwrap(),
        ])
        .env("LD_LIBRARY_PATH", c_lib_path())
        .output()
        .expect("spawn C minimal model train");
    assert!(c.status.success(), "C minimal model train failed");

    let rs = Command::new(rust_bin())
        .args([
            "learn",
            "-a",
            "lbfgs",
            "-p",
            "max_iterations=0",
            "-m",
            rs_model.to_str().unwrap(),
            train.to_str().unwrap(),
        ])
        .output()
        .expect("spawn Rust minimal model train");
    assert!(rs.status.success(), "Rust minimal model train failed");

    let c_bytes = std::fs::read(c_model).unwrap();
    let rs_bytes = std::fs::read(rs_model).unwrap();
    assert_eq!(c_bytes, rs_bytes, "minimal trained model bytes differ");
}
