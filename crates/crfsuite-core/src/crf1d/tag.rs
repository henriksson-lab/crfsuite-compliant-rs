/// Pure Rust CRF1d tagger — replaces crf1d_tag.c
///
/// Given a ModelReader, creates a tagger that can tag sequences.

use crate::crf1d::context::{Crf1dContext, CTXF_MARGINALS, CTXF_VITERBI, RF_STATE};
use crate::model::ModelReader;
use crate::types::Instance;

const LEVEL_NONE: i32 = 0;
const LEVEL_SET: i32 = 1;
const LEVEL_ALPHABETA: i32 = 2;

pub struct Crf1dTagger<'a> {
    model: &'a ModelReader<'a>,
    ctx: Crf1dContext,
    num_labels: usize,
    level: i32,
}

impl<'a> Crf1dTagger<'a> {
    pub fn new(model: &'a ModelReader<'a>) -> Self {
        let num_labels = model.num_labels() as usize;
        let mut ctx = Crf1dContext::new(CTXF_VITERBI | CTXF_MARGINALS, num_labels, 0);

        // Set transition scores from model
        ctx.set_num_items(0);
        Self::set_transition_scores(&mut ctx, model, num_labels);
        ctx.exp_transition();

        Crf1dTagger {
            model,
            ctx,
            num_labels,
            level: LEVEL_NONE,
        }
    }

    fn set_transition_scores(ctx: &mut Crf1dContext, model: &ModelReader, num_labels: usize) {
        for i in 0..num_labels {
            let refs = model.get_labelref(i as i32);
            let trans_start = i * num_labels;
            for &fid in &refs {
                if let Some(f) = model.get_feature(fid as u32) {
                    ctx.trans[trans_start + f.dst as usize] = f.weight;
                }
            }
        }
    }

    fn set_state_scores(&mut self, inst: &Instance) {
        let l = self.num_labels;
        let t_max = inst.num_items();

        for t in 0..t_max {
            let item = &inst.items[t];
            let state_start = t * l;
            for attr in &item.contents {
                let aid = attr.aid;
                if aid < 0 { continue; }
                let refs = self.model.get_attrref(aid);
                let value = attr.value;
                for &fid in &refs {
                    if let Some(f) = self.model.get_feature(fid as u32) {
                        self.ctx.state[state_start + f.dst as usize] += f.weight * value;
                    }
                }
            }
        }
    }

    fn ensure_level(&mut self, level: i32) {
        if level <= LEVEL_ALPHABETA && self.level < LEVEL_ALPHABETA {
            self.ctx.exp_state();
            self.ctx.alpha_score();
            self.ctx.beta_score();
        }
        self.level = level;
    }

    /// Set an instance on the tagger. Must be called before viterbi/score/etc.
    pub fn set(&mut self, inst: &Instance) {
        let t = inst.num_items();
        self.ctx.set_num_items(t);
        self.ctx.reset(RF_STATE);
        self.set_state_scores(inst);
        self.level = LEVEL_SET;
    }

    /// Run Viterbi decoding. Returns (labels, score).
    pub fn viterbi(&mut self) -> (Vec<i32>, f64) {
        let n = self.ctx.num_items;
        let mut labels = vec![0i32; n];
        let score = self.ctx.viterbi(&mut labels);
        (labels, score)
    }

    /// Compute score of a label path.
    pub fn score(&self, path: &[i32]) -> f64 {
        self.ctx.score(path)
    }

    /// Get log of the partition function. Requires alpha computation.
    pub fn lognorm(&mut self) -> f64 {
        self.ensure_level(LEVEL_ALPHABETA);
        self.ctx.lognorm()
    }

    /// Get marginal probability P(y_t = l | x). Requires alpha+beta.
    pub fn marginal_point(&mut self, label: i32, t: i32) -> f64 {
        self.ensure_level(LEVEL_ALPHABETA);
        self.ctx.marginal_point(label as usize, t as usize)
    }

    /// Get marginal probability of a partial path. Requires alpha+beta.
    pub fn marginal_path(&mut self, path: &[i32], begin: i32, end: i32) -> f64 {
        self.ensure_level(LEVEL_ALPHABETA);
        self.ctx.marginal_path(path, begin as usize, end as usize)
    }

    pub fn num_items(&self) -> usize {
        self.ctx.num_items
    }
}
