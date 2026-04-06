//! Test: write a model with Rust, read it back with Rust ModelReader, verify all data.

use crfsuite_core::crf1d::feature::{self, Feature, FeatureRefs, FT_STATE, FT_TRANS};
use crfsuite_core::model::ModelReader;
use crfsuite_core::model_writer;

#[test]
fn test_model_roundtrip() {
    // Create some features
    let features = vec![
        Feature { ftype: FT_STATE, src: 0, dst: 0, freq: 2.0 },
        Feature { ftype: FT_STATE, src: 0, dst: 1, freq: 1.0 },
        Feature { ftype: FT_STATE, src: 1, dst: 0, freq: 1.0 },
        Feature { ftype: FT_STATE, src: 1, dst: 1, freq: 3.0 },
        Feature { ftype: FT_TRANS, src: 0, dst: 0, freq: 1.0 },
        Feature { ftype: FT_TRANS, src: 0, dst: 1, freq: 2.0 },
        Feature { ftype: FT_TRANS, src: 1, dst: 0, freq: 1.0 },
        Feature { ftype: FT_TRANS, src: 1, dst: 1, freq: 1.0 },
    ];
    let weights = vec![0.5, -0.3, 0.2, 0.7, 0.1, -0.1, 0.4, -0.2];
    let labels = vec!["B-NP".to_string(), "I-NP".to_string()];
    let attrs = vec!["w[0]=the".to_string(), "pos[0]=DT".to_string()];

    let (attr_refs, label_refs) = feature::init_references(&features, attrs.len(), labels.len());

    let model_bytes = model_writer::write_model(
        &features, &weights, &labels, &attrs, &label_refs, &attr_refs,
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
        let f = model.get_feature(fid).expect(&format!("feature {} missing", fid));
        assert!(f.weight.is_finite());
    }

    // Check label refs point to transition features
    for lid in 0..2 {
        let refs = model.get_labelref(lid);
        for &fid in &refs {
            let f = model.get_feature(fid as u32).unwrap();
            assert_eq!(f.ftype, FT_TRANS as u32);
        }
    }

    // Check attr refs point to state features
    for aid in 0..2 {
        let refs = model.get_attrref(aid);
        for &fid in &refs {
            let f = model.get_feature(fid as u32).unwrap();
            assert_eq!(f.ftype, FT_STATE as u32);
        }
    }
}

#[test]
fn test_model_with_pruning() {
    // Some features have zero weight → should be pruned
    let features = vec![
        Feature { ftype: FT_STATE, src: 0, dst: 0, freq: 1.0 },
        Feature { ftype: FT_STATE, src: 0, dst: 1, freq: 1.0 },
        Feature { ftype: FT_STATE, src: 1, dst: 0, freq: 1.0 }, // zero weight
        Feature { ftype: FT_TRANS, src: 0, dst: 0, freq: 1.0 },
        Feature { ftype: FT_TRANS, src: 0, dst: 1, freq: 1.0 }, // zero weight
    ];
    let weights = vec![0.5, -0.3, 0.0, 0.1, 0.0]; // features 2,4 pruned
    let labels = vec!["A".to_string(), "B".to_string()];
    let attrs = vec!["x".to_string(), "y".to_string()];

    let (attr_refs, label_refs) = feature::init_references(&features, attrs.len(), labels.len());

    let model_bytes = model_writer::write_model(
        &features, &weights, &labels, &attrs, &label_refs, &attr_refs,
    );

    let model = ModelReader::open(&model_bytes).expect("failed to read model");
    assert_eq!(model.num_features(), 3); // only 3 active features
    assert_eq!(model.num_attrs(), 1); // only attr 0 is active (attr 1 had zero weight)
}
