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
            let t_max = inst.num_items();
            if t_max == 0 { continue; }

            // Set up instance
            self.ctx.set_num_items(t_max);
            self.ctx.reset(RF_STATE);
            self.state_score(inst, w, 1.0);

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
            self.model_expectation(inst, g, inst.weight);
        }

        -logl // negative log-likelihood (we minimize this)
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
