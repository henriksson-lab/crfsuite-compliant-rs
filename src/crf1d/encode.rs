//! CRF1d encoder for training (replaces crf1d_encode.c).
//!
//! Connects features, context, and model writer for training algorithms.

use crate::crf1d::context::{Crf1dContext, CTXF_MARGINALS, CTXF_VITERBI, RF_STATE, RF_TRANS};
use crate::crf1d::feature::{self, Feature, FeatureRefs};
use crate::model_writer;
use crate::types::Instance;

pub struct Crf1dEncoder {
    pub num_labels: usize,
    pub num_attributes: usize,
    pub num_features: usize,
    pub features: Vec<Feature>,
    pub attr_refs: Vec<FeatureRefs>,   // attribute → state feature IDs
    pub label_refs: Vec<FeatureRefs>,  // label → transition feature IDs
    pub ctx: Crf1dContext,
    weights: Vec<f64>,
    scale: f64,
    // Precomputed feature dst for fast indexing (avoids struct field access in hot loops)
    feature_dst: Vec<u32>,
    transition_fid: Vec<Option<u32>>,
}

#[derive(Debug, Clone)]
pub struct EncodedItemFeature {
    pub fid: u32,
    pub dst: u32,
    pub value: f64,
}

#[derive(Debug, Clone)]
pub struct EncodedInstance {
    pub items: Vec<Vec<EncodedItemFeature>>,
    pub labels: Vec<i32>,
    pub weight: f64,
}

impl Crf1dEncoder {
    /// Initialize the encoder from training data.
    pub fn new(
        instances: &[Instance],
        num_labels: usize,
        num_attrs: usize,
        min_freq: f64,
        possible_states: bool,
        possible_transitions: bool,
    ) -> Self {
        // Find maximum sequence length
        let max_t = instances.iter().map(|inst| inst.num_items()).max().unwrap_or(0);

        // Generate features
        let features = feature::generate_features(
            instances, num_labels, num_attrs, min_freq,
            possible_states, possible_transitions,
        );
        let num_features = features.len();

        // Build reference arrays
        let (attr_refs, label_refs) = feature::init_references(&features, num_attrs, num_labels);

        // Create CRF context
        let ctx = Crf1dContext::new(CTXF_MARGINALS | CTXF_VITERBI, num_labels, max_t);

        // Precompute feature dst array
        let feature_dst: Vec<u32> = features.iter().map(|f| f.dst as u32).collect();
        let mut transition_fid = vec![None; num_labels * num_labels];
        for (fid, feature) in features.iter().enumerate() {
            if feature.ftype == feature::FT_TRANS
                && 0 <= feature.src
                && 0 <= feature.dst
                && (feature.src as usize) < num_labels
                && (feature.dst as usize) < num_labels
            {
                transition_fid[feature.src as usize * num_labels + feature.dst as usize] =
                    Some(fid as u32);
            }
        }

        Crf1dEncoder {
            num_labels,
            num_attributes: num_attrs,
            num_features,
            features,
            attr_refs,
            label_refs,
            ctx,
            weights: Vec::new(),
            scale: 1.0,
            feature_dst,
            transition_fid,
        }
    }

    /// Encode instances for trainer hot loops after the feature set is fixed.
    pub fn encode_instances(&self, instances: &[Instance]) -> Vec<EncodedInstance> {
        instances.iter().map(|inst| self.encode_instance(inst)).collect()
    }

    fn encode_instance(&self, inst: &Instance) -> EncodedInstance {
        let mut items = Vec::with_capacity(inst.num_items());
        for item in &inst.items {
            let mut encoded = Vec::new();
            for attr in &item.contents {
                let aid = attr.aid as usize;
                if aid >= self.attr_refs.len() {
                    continue;
                }
                for &fid in &self.attr_refs[aid].fids {
                    encoded.push(EncodedItemFeature {
                        fid: fid as u32,
                        dst: self.feature_dst[fid as usize],
                        value: attr.value,
                    });
                }
            }
            items.push(encoded);
        }
        EncodedInstance {
            items,
            labels: inst.labels.clone(),
            weight: inst.weight,
        }
    }

    /// Set weights and compute transition scores.
    pub fn set_weights(&mut self, w: &[f64], scale: f64) {
        // Copy weights only if different length (reuse buffer)
        self.weights.resize(w.len(), 0.0);
        self.weights.copy_from_slice(w);
        self.scale = scale;
        self.ctx.reset(RF_TRANS);
        self.transition_score_from_stored();
    }

    /// Set transition and state scores for one instance without copying weights.
    pub fn set_weights_and_instance(&mut self, inst: &Instance, w: &[f64], scale: f64) {
        self.ctx.reset(RF_TRANS);
        self.transition_score(w, scale);
        self.ctx.set_num_items(inst.num_items());
        self.ctx.reset(RF_STATE);
        self.state_score(inst, w, scale);
    }

    /// Set transition scores from borrowed weights without copying them.
    pub fn set_transitions_from_weights(&mut self, w: &[f64], scale: f64) {
        self.ctx.reset(RF_TRANS);
        self.transition_score(w, scale);
    }

    /// Set one instance's state scores from borrowed weights without copying them.
    pub fn set_instance_from_weights(&mut self, inst: &Instance, w: &[f64], scale: f64) {
        self.ctx.set_num_items(inst.num_items());
        self.ctx.reset(RF_STATE);
        self.state_score(inst, w, scale);
    }

    /// Set one pre-encoded instance's state scores from borrowed weights.
    pub fn set_encoded_instance_from_weights(
        &mut self,
        inst: &EncodedInstance,
        w: &[f64],
        scale: f64,
    ) {
        self.ctx.set_num_items(inst.items.len());
        self.ctx.reset(RF_STATE);
        self.state_score_encoded(inst, w, scale);
    }

    /// Set an instance and compute state scores.
    pub fn set_instance(&mut self, inst: &Instance) {
        let t = inst.num_items();
        self.ctx.set_num_items(t);
        self.ctx.reset(RF_STATE);
        self.state_score_from_stored(inst);
    }

    fn transition_score_from_stored(&mut self) {
        let l = self.num_labels;
        let scale = self.scale;
        for i in 0..l {
            let trans_start = i * l;
            for &fid in &self.label_refs[i].fids {
                let dst = self.feature_dst[fid as usize] as usize;
                self.ctx.trans[trans_start + dst] = self.weights[fid as usize] * scale;
            }
        }
    }

    fn state_score_from_stored(&mut self, inst: &Instance) {
        let l = self.num_labels;
        let scale = self.scale;
        for t in 0..inst.num_items() {
            let item = &inst.items[t];
            let state_start = t * l;
            for attr in &item.contents {
                let aid = attr.aid as usize;
                if aid >= self.attr_refs.len() { continue; }
                let value = attr.value;
                for &fid in &self.attr_refs[aid].fids {
                    let dst = self.feature_dst[fid as usize] as usize;
                    self.ctx.state[state_start + dst] += self.weights[fid as usize] * value * scale;
                }
            }
        }
    }

    /// Compute state scores from feature weights for an instance.
    fn state_score(&mut self, inst: &Instance, w: &[f64], scale: f64) {
        let l = self.num_labels;
        for t in 0..inst.num_items() {
            let item = &inst.items[t];
            let state_start = t * l;
            for attr in &item.contents {
                let aid = attr.aid as usize;
                if aid >= self.attr_refs.len() { continue; }
                let vs = attr.value * scale;
                for &fid in &self.attr_refs[aid].fids {
                    let dst = self.feature_dst[fid as usize] as usize;
                    self.ctx.state[state_start + dst] += w[fid as usize] * vs;
                }
            }
        }
    }

    fn state_score_encoded(&mut self, inst: &EncodedInstance, w: &[f64], scale: f64) {
        let l = self.num_labels;
        for (t, item) in inst.items.iter().enumerate() {
            let state_start = t * l;
            for feature in item {
                self.ctx.state[state_start + feature.dst as usize] +=
                    w[feature.fid as usize] * feature.value * scale;
            }
        }
    }

    /// Compute transition scores from feature weights (instance-independent).
    fn transition_score(&mut self, w: &[f64], scale: f64) {
        let l = self.num_labels;
        for i in 0..l {
            let trans_start = i * l;
            for &fid in &self.label_refs[i].fids {
                let f = &self.features[fid as usize];
                self.ctx.trans[trans_start + f.dst as usize] = w[fid as usize] * scale;
            }
        }
    }

    /// Accumulate model expectations into gradient array.
    fn model_expectation(&self, inst: &Instance, g: &mut [f64], weight: f64) {
        let l = self.num_labels;
        let t_max = inst.num_items();

        // State feature expectations
        for t in 0..t_max {
            let item = &inst.items[t];
            let mexp_start = t * l;
            for attr in &item.contents {
                let aid = attr.aid as usize;
                if aid >= self.attr_refs.len() { continue; }
                let value = attr.value;
                let vw = value * weight;
                for &fid in &self.attr_refs[aid].fids {
                    let dst = self.feature_dst[fid as usize] as usize;
                    g[fid as usize] += self.ctx.mexp_state[mexp_start + dst] * vw;
                }
            }
        }

        // Transition feature expectations
        for i in 0..l {
            let mexp_start = i * l;
            for &fid in &self.label_refs[i].fids {
                let dst = self.feature_dst[fid as usize] as usize;
                g[fid as usize] += self.ctx.mexp_trans[mexp_start + dst] * weight;
            }
        }
    }

    fn model_expectation_encoded(&self, inst: &EncodedInstance, g: &mut [f64], weight: f64) {
        let l = self.num_labels;

        for (t, item) in inst.items.iter().enumerate() {
            let mexp_start = t * l;
            for feature in item {
                g[feature.fid as usize] +=
                    self.ctx.mexp_state[mexp_start + feature.dst as usize] * feature.value * weight;
            }
        }

        for i in 0..l {
            let mexp_start = i * l;
            for &fid in &self.label_refs[i].fids {
                let dst = self.feature_dst[fid as usize] as usize;
                g[fid as usize] += self.ctx.mexp_trans[mexp_start + dst] * weight;
            }
        }
    }

    /// Compute negative log-likelihood and gradients for the entire dataset.
    ///
    /// Returns the objective (negative log-likelihood).
    /// `w`: current weights, `g`: output gradient array (same length as w).
    pub fn objective_and_gradients_batch(
        &mut self,
        instances: &[Instance],
        w: &[f64],
        g: &mut [f64],
    ) -> f64 {
        let encoded = self.encode_instances(instances);
        self.objective_and_gradients_batch_encoded(&encoded, w, g)
    }

    pub fn objective_and_gradients_batch_encoded(
        &mut self,
        instances: &[EncodedInstance],
        w: &[f64],
        g: &mut [f64],
    ) -> f64 {
        let k = self.num_features;

        // Initialize gradients with negative observation expectations
        #[allow(clippy::needless_range_loop)]
        for i in 0..k {
            g[i] = -self.features[i].freq;
        }

        // Set transition scores (instance-independent)
        self.ctx.reset(RF_TRANS);
        self.transition_score(w, 1.0);
        self.ctx.exp_transition();

        let mut logl = 0.0f64;

        for inst in instances {
            let t_max = inst.items.len();
            if t_max == 0 { continue; }

            // Set up instance
            self.ctx.set_num_items(t_max);
            self.ctx.reset(RF_STATE);
            self.state_score_encoded(inst, w, 1.0);

            // Forward-backward
            self.ctx.exp_state();
            self.ctx.alpha_score();
            self.ctx.beta_score();
            self.ctx.marginals();

            // Compute log P(y|x) = score(y) - log_norm
            let score = self.ctx.score(&inst.labels);
            let lognorm = self.ctx.lognorm();
            let logp = score - lognorm;

            logl += logp * inst.weight;

            // Accumulate model expectations
            self.model_expectation_encoded(inst, g, inst.weight);
        }

        -logl // negative log-likelihood (we minimize this)
    }

    /// Compute one instance's negative log-likelihood and apply the online
    /// CRFsuite update: observed features are added and model expectations are
    /// subtracted from `w`, both scaled by `gain * inst.weight`.
    pub fn objective_and_gradients_online(
        &mut self,
        inst: &Instance,
        w: &mut [f64],
        scale: f64,
        gain: f64,
    ) -> f64 {
        self.set_weights_and_instance(inst, w, scale);
        self.ctx.exp_state();
        self.ctx.exp_transition();
        self.ctx.alpha_score();
        self.ctx.beta_score();
        self.ctx.marginals();

        let scaled_gain = gain * inst.weight;
        self.observation_expectation(inst, w, scaled_gain);
        self.model_expectation(inst, w, -scaled_gain);
        (-self.ctx.score(&inst.labels) + self.ctx.lognorm()) * inst.weight
    }

    pub fn objective_and_gradients_online_encoded(
        &mut self,
        inst: &EncodedInstance,
        w: &mut [f64],
        scale: f64,
        gain: f64,
    ) -> f64 {
        self.ctx.reset(RF_TRANS);
        self.transition_score(w, scale);
        self.set_encoded_instance_from_weights(inst, w, scale);
        self.ctx.exp_state();
        self.ctx.exp_transition();
        self.ctx.alpha_score();
        self.ctx.beta_score();
        self.ctx.marginals();

        let scaled_gain = gain * inst.weight;
        self.features_on_path_encoded(inst, &inst.labels, |fid, val| {
            w[fid as usize] += val * scaled_gain;
        });
        self.model_expectation_encoded(inst, w, -scaled_gain);
        (-self.ctx.score(&inst.labels) + self.ctx.lognorm()) * inst.weight
    }

    /// Run Viterbi on the current instance. Returns (labels, score).
    pub fn viterbi(&mut self, labels: &mut [i32]) -> f64 {
        self.ctx.viterbi(labels)
    }

    /// Score a path on the current instance.
    pub fn score(&self, path: &[i32]) -> f64 {
        self.ctx.score(path)
    }

    /// Partition factor for the current instance.
    pub fn partition_factor(&self) -> f64 {
        self.ctx.lognorm()
    }

    /// Accumulate observation expectations (features on gold path).
    pub fn observation_expectation(&self, inst: &Instance, w: &mut [f64], scale: f64) {
        let t_max = inst.num_items();

        for t in 0..t_max {
            let item = &inst.items[t];
            let gold_label = inst.labels[t] as usize;

            for attr in &item.contents {
                let aid = attr.aid as usize;
                if aid >= self.attr_refs.len() { continue; }
                for &fid in &self.attr_refs[aid].fids {
                    let f = &self.features[fid as usize];
                    if f.dst as usize == gold_label {
                        w[fid as usize] += attr.value * scale;
                    }
                }
            }

            if t > 0 {
                let prev_label = inst.labels[t - 1] as usize;
                if prev_label < self.label_refs.len() {
                    for &fid in &self.label_refs[prev_label].fids {
                        let f = &self.features[fid as usize];
                        if f.dst as usize == gold_label {
                            w[fid as usize] += scale;
                        }
                    }
                }
            }
        }
    }

    /// Accumulate model expectations into a gradient array (alternative signature for L2-SGD).
    pub fn model_expectation_into(&self, inst: &Instance, g: &mut [f64], weight: f64) {
        self.model_expectation(inst, g, weight);
    }

    /// Enumerate features on a given path (for online algorithms).
    pub fn features_on_path<F>(&self, inst: &Instance, path: &[i32], mut callback: F)
    where
        F: FnMut(i32, f64),
    {
        let t_max = inst.num_items();

        for t in 0..t_max {
            let item = &inst.items[t];
            let label = path[t] as usize;

            for attr in &item.contents {
                let aid = attr.aid as usize;
                if aid >= self.attr_refs.len() { continue; }
                for &fid in &self.attr_refs[aid].fids {
                    let f = &self.features[fid as usize];
                    if f.dst as usize == label {
                        callback(fid, attr.value);
                    }
                }
            }

            if t > 0 {
                let prev_label = path[t - 1] as usize;
                if prev_label < self.label_refs.len() {
                    for &fid in &self.label_refs[prev_label].fids {
                        let f = &self.features[fid as usize];
                        if f.dst as usize == label {
                            callback(fid, 1.0);
                        }
                    }
                }
            }
        }
    }

    /// Enumerate features on a path using a pre-encoded instance.
    pub fn features_on_path_encoded<F>(
        &self,
        inst: &EncodedInstance,
        path: &[i32],
        mut callback: F,
    )
    where
        F: FnMut(i32, f64),
    {
        let l = self.num_labels;
        for (t, item) in inst.items.iter().enumerate() {
            let label = path[t] as usize;

            for feature in item {
                if feature.dst as usize == label {
                    callback(feature.fid as i32, feature.value);
                }
            }

            if t > 0 {
                let prev_label = path[t - 1] as usize;
                if prev_label < l {
                    let index = prev_label * l + label;
                    if let Some(fid) = self.transition_fid[index] {
                        callback(fid as i32, 1.0);
                    }
                }
            }
        }
    }

    /// Save the model to bytes.
    pub fn save_model(
        &self,
        w: &[f64],
        label_strings: &[String],
        attr_strings: &[String],
    ) -> Vec<u8> {
        model_writer::write_model(
            &self.features,
            w,
            label_strings,
            attr_strings,
            &self.label_refs,
            &self.attr_refs,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{Attribute, Instance, Item};

    fn tiny_instance() -> Instance {
        Instance {
            items: vec![
                Item {
                    contents: vec![Attribute { aid: 0, value: 1.0 }],
                },
                Item {
                    contents: vec![Attribute { aid: 1, value: 1.0 }],
                },
            ],
            labels: vec![0, 1],
            weight: 1.0,
            group: 0,
        }
    }

    #[test]
    fn objective_and_gradients_match_hand_computed_fixture() {
        let instances = vec![tiny_instance()];
        let mut encoder = Crf1dEncoder::new(&instances, 2, 2, 0.0, false, false);
        let weights = vec![0.0; encoder.num_features];
        let mut gradient = vec![0.0; encoder.num_features];

        let objective = encoder.objective_and_gradients_batch(&instances, &weights, &mut gradient);

        assert_eq!(encoder.num_features, 3);
        assert_eq!(
            encoder
                .features
                .iter()
                .map(|f| (f.ftype, f.src, f.dst, f.freq))
                .collect::<Vec<_>>(),
            vec![(0, 0, 0, 1.0), (0, 1, 1, 1.0), (1, 0, 1, 1.0)]
        );
        assert!((objective - 4.0f64.ln()).abs() <= f64::EPSILON);
        assert_eq!(gradient, vec![-0.5, -0.5, -0.75]);
    }

    #[test]
    fn objective_and_gradients_apply_instance_weight_like_c() {
        let mut inst = tiny_instance();
        inst.weight = 2.0;
        let instances = vec![inst];
        let mut encoder = Crf1dEncoder::new(&instances, 2, 2, 0.0, false, false);
        let weights = vec![0.0; encoder.num_features];
        let mut gradient = vec![0.0; encoder.num_features];

        let objective = encoder.objective_and_gradients_batch(&instances, &weights, &mut gradient);

        assert!((objective - 2.0 * 4.0f64.ln()).abs() <= f64::EPSILON);
        assert_eq!(gradient, vec![-1.0, -1.0, -1.5]);
    }
}
