//! Feature extraction for CRF1d training (replaces crf1d_feature.c).
//!
//! Features are (type, src, dst) tuples with accumulated frequencies.
//! Uses BTreeMap for deterministic ordering (same lexicographic order as C's rumavl).

use std::collections::{BTreeMap, BTreeSet};

use crate::types::Instance;

pub const FT_STATE: i32 = 0;
pub const FT_TRANS: i32 = 1;

#[derive(Debug, Clone)]
pub struct Feature {
    pub ftype: i32,
    pub src: i32,
    pub dst: i32,
    pub freq: f64,
}

/// Feature reference: list of feature IDs associated with an attribute or label.
#[derive(Debug, Clone, Default)]
pub struct FeatureRefs {
    pub fids: Vec<i32>,
}

/// Generate features from training data.
///
/// Returns a vector of features with deterministic ordering by (type, src, dst).
pub fn generate_features(
    instances: &[Instance],
    num_labels: usize,
    _num_attrs: usize,
    min_freq: f64,
    connect_all_attrs: bool,
    connect_all_edges: bool,
) -> Vec<Feature> {
    // Use BTreeMap keyed by (type, src, dst) for deterministic ordering
    let mut feature_map: BTreeMap<(i32, i32, i32), f64> = BTreeMap::new();
    let mut observed_attrs = BTreeSet::new();

    for inst in instances {
        let t_max = inst.num_items();
        let weight = inst.weight;

        for t in 0..t_max {
            let label = inst.labels[t];
            let item = &inst.items[t];

            // State features: (FT_STATE, attr_id, label_id)
            for attr in &item.contents {
                observed_attrs.insert(attr.aid);
                let key = (FT_STATE, attr.aid, label);
                *feature_map.entry(key).or_insert(0.0) += weight * attr.value;
            }

            // Transition features: (FT_TRANS, prev_label, cur_label)
            if t > 0 {
                let prev_label = inst.labels[t - 1];
                let key = (FT_TRANS, prev_label, label);
                *feature_map.entry(key).or_insert(0.0) += weight;
            }
        }
    }

    // Optional: connect observed attributes to all labels (zero-freq state features).
    // CRFsuite adds these while visiting item contents, so attributes that only
    // exist in held-out data or dictionaries are not forced into the feature set.
    if connect_all_attrs {
        for aid in observed_attrs {
            for lid in 0..num_labels as i32 {
                feature_map.entry((FT_STATE, aid, lid)).or_insert(0.0);
            }
        }
    }

    // Optional: connect all label pairs (zero-freq transition features)
    if connect_all_edges {
        for i in 0..num_labels as i32 {
            for j in 0..num_labels as i32 {
                feature_map.entry((FT_TRANS, i, j)).or_insert(0.0);
            }
        }
    }

    // Convert to vector, filtering by min_freq
    feature_map
        .into_iter()
        .filter(|(_, freq)| *freq >= min_freq)
        .map(|((ftype, src, dst), freq)| Feature {
            ftype,
            src,
            dst,
            freq,
        })
        .collect()
}

/// Build reference arrays mapping attributes→features and labels→features.
///
/// Returns (attr_refs[num_attrs], label_refs[num_labels]).
pub fn init_references(
    features: &[Feature],
    num_attrs: usize,
    num_labels: usize,
) -> (Vec<FeatureRefs>, Vec<FeatureRefs>) {
    let mut attr_refs: Vec<FeatureRefs> = (0..num_attrs).map(|_| FeatureRefs::default()).collect();
    let mut label_refs: Vec<FeatureRefs> =
        (0..num_labels).map(|_| FeatureRefs::default()).collect();

    for (fid, f) in features.iter().enumerate() {
        match f.ftype {
            FT_STATE => {
                if (f.src as usize) < num_attrs {
                    attr_refs[f.src as usize].fids.push(fid as i32);
                }
            }
            FT_TRANS => {
                if (f.src as usize) < num_labels {
                    label_refs[f.src as usize].fids.push(fid as i32);
                }
            }
            _ => {}
        }
    }

    (attr_refs, label_refs)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{Attribute, Instance, Item};

    fn make_instance(items: Vec<Vec<(i32, f64)>>, labels: Vec<i32>) -> Instance {
        Instance {
            items: items
                .into_iter()
                .map(|attrs| Item {
                    contents: attrs
                        .into_iter()
                        .map(|(aid, val)| Attribute { aid, value: val })
                        .collect(),
                })
                .collect(),
            labels,
            weight: 1.0,
            group: 0,
        }
    }

    #[test]
    fn test_generate_basic() {
        let instances = vec![make_instance(
            vec![vec![(0, 1.0), (1, 1.0)], vec![(2, 1.0)]],
            vec![0, 1],
        )];
        let features = generate_features(&instances, 2, 3, 0.0, false, false);

        // Should have state features: (0,0,0), (0,1,0), (0,2,1)
        // And transition feature: (1,0,1)
        assert_eq!(features.len(), 4);

        // Features are ordered by (type, src, dst)
        assert_eq!(
            (features[0].ftype, features[0].src, features[0].dst),
            (0, 0, 0)
        );
        assert_eq!(
            (features[1].ftype, features[1].src, features[1].dst),
            (0, 1, 0)
        );
        assert_eq!(
            (features[2].ftype, features[2].src, features[2].dst),
            (0, 2, 1)
        );
        assert_eq!(
            (features[3].ftype, features[3].src, features[3].dst),
            (1, 0, 1)
        );
    }

    #[test]
    fn test_possible_states_only_use_observed_attributes() {
        let instances = vec![make_instance(vec![vec![(0, 1.0)]], vec![0])];
        let features = generate_features(&instances, 2, 2, 0.0, true, false);
        let keys: Vec<_> = features
            .iter()
            .map(|f| (f.ftype, f.src, f.dst, f.freq))
            .collect();

        assert_eq!(keys, vec![(FT_STATE, 0, 0, 1.0), (FT_STATE, 0, 1, 0.0)]);
    }

    #[test]
    fn test_init_references() {
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
            },
            Feature {
                ftype: FT_TRANS,
                src: 0,
                dst: 1,
                freq: 1.0,
            },
        ];
        let (attr_refs, label_refs) = init_references(&features, 2, 2);

        assert_eq!(attr_refs[0].fids, vec![0, 1]); // features 0,1 have src=0
        assert_eq!(attr_refs[1].fids, vec![2]); // feature 2 has src=1
        assert_eq!(label_refs[0].fids, vec![3]); // transition from label 0
        assert!(label_refs[1].fids.is_empty());
    }
}
