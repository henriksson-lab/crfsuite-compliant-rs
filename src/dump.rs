use std::io::Write;

use crate::model::ModelReader;

/// Dump a model in human-readable format, matching the C `crf1dm_dump` output exactly.
pub fn dump_model(model: &ModelReader, out: &mut impl Write) -> std::io::Result<()> {
    // File header — note: num_features from file header is 0 (C reads it directly from the struct)
    let num_features_header = 0u32; // C stores 0 in the header
    writeln!(out, "FILEHEADER = {{")?;
    writeln!(out, "  magic: lCRF")?;
    writeln!(out, "  size: {}", model.header_size())?;
    writeln!(out, "  type: FOMC")?;
    writeln!(out, "  version: {}", model.header_version())?;
    writeln!(out, "  num_features: {}", num_features_header)?;
    writeln!(out, "  num_labels: {}", model.num_labels())?;
    writeln!(out, "  num_attrs: {}", model.num_attrs())?;
    writeln!(out, "  off_features: 0x{:X}", model.off_features())?;
    writeln!(out, "  off_labels: 0x{:X}", model.off_labels())?;
    writeln!(out, "  off_attrs: 0x{:X}", model.off_attrs())?;
    writeln!(out, "  off_labelrefs: 0x{:X}", model.off_labelrefs())?;
    writeln!(out, "  off_attrrefs: 0x{:X}", model.off_attrrefs())?;
    writeln!(out, "}}")?;
    writeln!(out)?;

    // Labels
    writeln!(out, "LABELS = {{")?;
    for i in 0..model.num_labels() {
        let s = model.to_label(i as i32).unwrap_or("?");
        writeln!(out, "  {:5}: {}", i, s)?;
    }
    writeln!(out, "}}")?;
    writeln!(out)?;

    // Attributes
    writeln!(out, "ATTRIBUTES = {{")?;
    for i in 0..model.num_attrs() {
        let s = model.to_attr(i as i32).unwrap_or("?");
        writeln!(out, "  {:5}: {}", i, s)?;
    }
    writeln!(out, "}}")?;
    writeln!(out)?;

    // Transitions
    writeln!(out, "TRANSITIONS = {{")?;
    for i in 0..model.num_labels() {
        let refs = model.get_labelref(i as i32);
        for &fid in refs {
            if let Some(f) = model.get_feature(fid as u32) {
                let from = model.to_label(f.src as i32).unwrap_or("?");
                let to = model.to_label(f.dst as i32).unwrap_or("?");
                writeln!(out, "  ({}) {} --> {}: {:.6}", f.ftype, from, to, f.weight)?;
            }
        }
    }
    writeln!(out, "}}")?;
    writeln!(out)?;

    // State features
    writeln!(out, "STATE_FEATURES = {{")?;
    for i in 0..model.num_attrs() {
        let refs = model.get_attrref(i as i32);
        for &fid in refs {
            if let Some(f) = model.get_feature(fid as u32) {
                let attr = model.to_attr(f.src as i32).unwrap_or("?");
                let to = model.to_label(f.dst as i32).unwrap_or("?");
                writeln!(out, "  ({}) {} --> {}: {:.6}", f.ftype, attr, to, f.weight)?;
            }
        }
    }
    writeln!(out, "}}")?;
    writeln!(out)?;

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn model_bytes() -> Vec<u8> {
        let root = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        std::fs::read(root.join("test_data/model_c.bin")).expect("model_c.bin")
    }

    #[test]
    fn test_dump_produces_output() {
        let data = model_bytes();
        let model = ModelReader::open(&data).unwrap();
        let mut out = Vec::new();
        dump_model(&model, &mut out).unwrap();
        let text = String::from_utf8(out).unwrap();
        assert!(text.contains("FILEHEADER"));
        assert!(text.contains("LABELS"));
        assert!(text.contains("TRANSITIONS"));
        assert!(text.contains("STATE_FEATURES"));
        assert!(text.contains("B-NP"));
    }
}
