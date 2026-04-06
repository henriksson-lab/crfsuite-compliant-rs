/// CRF1d context: forward-backward algorithm, Viterbi decoding, marginal computation.
///
/// Matrices use row-major `[t * L + l]` indexing where:
///   t = time position (0..T)
///   l = label index (0..L)

use crate::vecmath::*;

pub const CTXF_VITERBI: i32 = 0x01;
pub const CTXF_MARGINALS: i32 = 0x02;

pub const RF_STATE: i32 = 0x01;
pub const RF_TRANS: i32 = 0x02;

pub struct Crf1dContext {
    pub flag: i32,
    pub num_labels: usize,
    pub num_items: usize,
    cap_items: usize,

    // Score matrices [T * L] or [L * L]
    pub state: Vec<f64>,       // [T][L] state scores (log domain)
    pub trans: Vec<f64>,       // [L][L] transition scores (log domain)

    // Forward-backward
    pub alpha_score: Vec<f64>, // [T][L]
    pub beta_score: Vec<f64>,  // [T][L]
    pub scale_factor: Vec<f64>,// [T]
    row: Vec<f64>,             // [L] scratch

    // Viterbi
    pub backward_edge: Vec<i32>, // [T][L]

    // Marginals (exponentiated)
    pub exp_state: Vec<f64>,   // [T][L]
    pub exp_trans: Vec<f64>,   // [L][L]
    pub mexp_state: Vec<f64>,  // [T][L] model expectations
    pub mexp_trans: Vec<f64>,  // [L][L] model expectations

    pub log_norm: f64,
}

impl Crf1dContext {
    pub fn new(flag: i32, num_labels: usize, cap_items: usize) -> Self {
        let l = num_labels;
        let mut ctx = Crf1dContext {
            flag,
            num_labels: l,
            num_items: 0,
            cap_items: 0,
            state: Vec::new(),
            trans: vec![0.0; l * l],
            alpha_score: Vec::new(),
            beta_score: Vec::new(),
            scale_factor: Vec::new(),
            row: Vec::new(),
            backward_edge: Vec::new(),
            exp_state: Vec::new(),
            exp_trans: if flag & CTXF_MARGINALS != 0 { vec![0.0; l * l] } else { Vec::new() },
            mexp_state: Vec::new(),
            mexp_trans: if flag & CTXF_MARGINALS != 0 { vec![0.0; l * l] } else { Vec::new() },
            log_norm: 0.0,
        };
        if cap_items > 0 {
            ctx.set_num_items(cap_items);
        }
        ctx.num_items = 0;
        ctx
    }

    pub fn set_num_items(&mut self, t: usize) {
        self.num_items = t;
        if self.cap_items >= t {
            return;
        }
        let l = self.num_labels;
        self.state = vec![0.0; t * l];
        self.alpha_score = vec![0.0; t * l];
        self.beta_score = vec![0.0; t * l];
        self.scale_factor = vec![0.0; t];
        self.row = vec![0.0; l];

        if self.flag & CTXF_VITERBI != 0 {
            self.backward_edge = vec![0i32; t * l];
        }
        if self.flag & CTXF_MARGINALS != 0 {
            self.exp_state = vec![0.0; t * l];
            self.mexp_state = vec![0.0; t * l];
        }
        self.cap_items = t;
    }

    pub fn reset(&mut self, flag: i32) {
        let t = self.num_items;
        let l = self.num_labels;
        if flag & RF_STATE != 0 {
            veczero(&mut self.state[..t * l]);
        }
        if flag & RF_TRANS != 0 {
            veczero(&mut self.trans[..l * l]);
        }
        if self.flag & CTXF_MARGINALS != 0 {
            veczero(&mut self.mexp_state[..t * l]);
            veczero(&mut self.mexp_trans[..l * l]);
            self.log_norm = 0.0;
        }
    }

    pub fn exp_state(&mut self) {
        let n = self.num_items * self.num_labels;
        veccopy(&mut self.exp_state[..n], &self.state[..n]);
        vecexp(&mut self.exp_state[..n]);
    }

    pub fn exp_transition(&mut self) {
        let n = self.num_labels * self.num_labels;
        veccopy(&mut self.exp_trans[..n], &self.trans[..n]);
        vecexp(&mut self.exp_trans[..n]);
    }

    // ── Row accessors ───────────────────────────────────────────────────

    #[inline]
    fn beta_mut(&mut self, t: usize) -> &mut [f64] {
        let l = self.num_labels;
        &mut self.beta_score[t * l..(t + 1) * l]
    }

    #[inline]
    pub fn state_score(&self, t: usize) -> &[f64] {
        let l = self.num_labels;
        &self.state[t * l..(t + 1) * l]
    }

    #[inline]
    pub fn state_score_mut(&mut self, t: usize) -> &mut [f64] {
        let l = self.num_labels;
        &mut self.state[t * l..(t + 1) * l]
    }

    #[inline]
    pub fn trans_score(&self, i: usize) -> &[f64] {
        let l = self.num_labels;
        &self.trans[i * l..(i + 1) * l]
    }

    #[inline]
    fn exp_trans_row(&self, i: usize) -> &[f64] {
        let l = self.num_labels;
        &self.exp_trans[i * l..(i + 1) * l]
    }

    // ── Forward algorithm (alpha scores) ────────────────────────────────

    pub fn alpha_score(&mut self) {
        let t_max = self.num_items;
        let l = self.num_labels;

        // t=0: alpha[0][j] = exp_state[0][j], then scale
        {
            let (alpha_part, _) = self.alpha_score.split_at_mut(l);
            alpha_part.copy_from_slice(&self.exp_state[..l]);
            let sum: f64 = alpha_part.iter().sum();
            self.scale_factor[0] = if sum != 0.0 { 1.0 / sum } else { 1.0 };
            let s = self.scale_factor[0];
            for v in alpha_part.iter_mut() { *v *= s; }
        }

        // t=1..T-1: alpha[t][j] = exp_state[t][j] * sum_i(alpha[t-1][i] * exp_trans[i][j])
        for t in 1..t_max {
            let prev_start = (t - 1) * l;
            let cur_start = t * l;

            // cur = 0
            self.alpha_score[cur_start..cur_start + l].fill(0.0);

            // cur += alpha[t-1][i] * exp_trans[i][*]  (matrix-vector product)
            for i in 0..l {
                let prev_i = self.alpha_score[prev_start + i];
                let trans = &self.exp_trans[i * l..(i + 1) * l];
                let cur = &mut self.alpha_score[cur_start..cur_start + l];
                vecaadd(cur, prev_i, trans);
            }

            // cur *= exp_state[t]
            {
                let (left, right) = self.alpha_score.split_at_mut(cur_start);
                let _ = left; // unused
                let cur = &mut right[..l];
                vecmul(cur, &self.exp_state[cur_start..cur_start + l]);
            }

            // Scale
            let sum: f64 = self.alpha_score[cur_start..cur_start + l].iter().sum();
            self.scale_factor[t] = if sum != 0.0 { 1.0 / sum } else { 1.0 };
            let s = self.scale_factor[t];
            vecscale(&mut self.alpha_score[cur_start..cur_start + l], s);
        }

        // log_norm = -sum(log(scale_factor[t]))
        self.log_norm = -self.scale_factor[..t_max].iter().map(|s| s.ln()).sum::<f64>();
    }

    // ── Backward algorithm (beta scores) ────────────────────────────────

    pub fn beta_score(&mut self) {
        let t_max = self.num_items;
        let l = self.num_labels;

        // t=T-1: beta[T-1][i] = scale[T-1]
        {
            let s = self.scale_factor[t_max - 1];
            let beta = self.beta_mut(t_max - 1);
            vecset(beta, s);
        }

        // t=T-2 down to 0
        for t in (0..t_max - 1).rev() {
            // row[j] = exp_state[t+1][j] * beta[t+1][j]
            let next_start = (t + 1) * l;
            for j in 0..l {
                self.row[j] = self.exp_state[next_start + j] * self.beta_score[next_start + j];
            }

            // beta[t][i] = scale[t] * sum_j(exp_trans[i][j] * row[j])
            let s = self.scale_factor[t];
            let cur_start = t * l;
            for i in 0..l {
                let trans_row = self.exp_trans_row(i);
                let mut dot = 0.0f64;
                for j in 0..l {
                    dot += trans_row[j] * self.row[j];
                }
                self.beta_score[cur_start + i] = dot * s;
            }
        }
    }

    // ── Marginal probabilities ──────────────────────────────────────────

    pub fn marginals(&mut self) {
        let t_max = self.num_items;
        let l = self.num_labels;

        // State marginals: p(t,i) = alpha[t][i] * beta[t][i] / scale[t]
        for t in 0..t_max {
            let start = t * l;
            let inv_scale = 1.0 / self.scale_factor[t];
            for j in 0..l {
                self.mexp_state[start + j] =
                    self.alpha_score[start + j] * self.beta_score[start + j] * inv_scale;
            }
        }

        // Transition marginals: sum over t of alpha[t][i] * exp_trans[i][j] * exp_state[t+1][j] * beta[t+1][j]
        for t in 0..t_max - 1 {
            let next_start = (t + 1) * l;

            // row[j] = exp_state[t+1][j] * beta[t+1][j]
            for j in 0..l {
                self.row[j] = self.exp_state[next_start + j] * self.beta_score[next_start + j];
            }

            let alpha_start = t * l;
            for i in 0..l {
                let fwd_i = self.alpha_score[alpha_start + i];
                let trans_start = i * l;
                let mexp_start = i * l;
                for j in 0..l {
                    self.mexp_trans[mexp_start + j] +=
                        fwd_i * self.exp_trans[trans_start + j] * self.row[j];
                }
            }
        }
    }

    pub fn marginal_point(&self, label: usize, t: usize) -> f64 {
        let l = self.num_labels;
        let start = t * l;
        self.alpha_score[start + label] * self.beta_score[start + label] / self.scale_factor[t]
    }

    pub fn marginal_path(&self, path: &[i32], begin: usize, end: usize) -> f64 {
        let l = self.num_labels;
        let fwd_start = begin * l;
        let bwd_start = (end - 1) * l;

        let mut prob = self.alpha_score[fwd_start + path[begin] as usize]
            * self.beta_score[bwd_start + path[end - 1] as usize]
            / self.scale_factor[begin];

        for t in begin..end - 1 {
            let state_start = (t + 1) * l;
            let next_label = path[t + 1] as usize;
            let cur_label = path[t] as usize;
            let trans_row = self.exp_trans_row(cur_label);
            prob *= trans_row[next_label]
                * self.exp_state[state_start + next_label]
                * self.scale_factor[t];
        }

        prob
    }

    // ── Score a path (log domain) ───────────────────────────────────────

    pub fn score(&self, labels: &[i32]) -> f64 {
        let t_max = self.num_items;

        let mut i = labels[0] as usize;
        let state0 = self.state_score(0);
        let mut ret = state0[i];

        for t in 1..t_max {
            let j = labels[t] as usize;
            let trans_row = self.trans_score(i);
            let state_t = self.state_score(t);
            ret += trans_row[j];
            ret += state_t[j];
            i = j;
        }
        ret
    }

    pub fn lognorm(&self) -> f64 {
        self.log_norm
    }

    // ── Viterbi decoding (log domain) ───────────────────────────────────

    pub fn viterbi(&mut self, labels: &mut [i32]) -> f64 {
        let t_max = self.num_items;
        let l = self.num_labels;

        // t=0: alpha[0][j] = state[0][j]
        {
            let state0 = &self.state[..l];
            self.alpha_score[..l].copy_from_slice(state0);
        }

        // t=1..T-1
        for t in 1..t_max {
            let prev_start = (t - 1) * l;
            let cur_start = t * l;
            let back_start = t * l;

            for j in 0..l {
                let mut max_score = f64::MIN;
                let mut argmax = 0i32;
                for i in 0..l {
                    let trans_row = &self.trans[i * l..(i + 1) * l];
                    let score = self.alpha_score[prev_start + i] + trans_row[j];
                    if max_score < score {
                        max_score = score;
                        argmax = i as i32;
                    }
                }
                self.backward_edge[back_start + j] = argmax;
                self.alpha_score[cur_start + j] = max_score + self.state[cur_start + j];
            }
        }

        // Find best final label
        let last_start = (t_max - 1) * l;
        let mut max_score = f64::MIN;
        labels[t_max - 1] = 0;
        for i in 0..l {
            if max_score < self.alpha_score[last_start + i] {
                max_score = self.alpha_score[last_start + i];
                labels[t_max - 1] = i as i32;
            }
        }

        // Backtrack
        for t in (0..t_max - 1).rev() {
            let back_start = (t + 1) * l;
            labels[t] = self.backward_edge[back_start + labels[t + 1] as usize];
        }

        max_score
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_viterbi_simple() {
        // 2 labels, 2 items
        let mut ctx = Crf1dContext::new(CTXF_VITERBI, 2, 2);
        ctx.set_num_items(2);

        // State scores: item 0 prefers label 0, item 1 prefers label 1
        ctx.state[0] = 2.0; ctx.state[1] = 0.0; // t=0: [2, 0]
        ctx.state[2] = 0.0; ctx.state[3] = 2.0; // t=1: [0, 2]

        // Transition: 0→1 is good (1.0), others neutral (0.0)
        ctx.trans[0] = 0.0; ctx.trans[1] = 1.0; // from 0: [0→0, 0→1]
        ctx.trans[2] = 0.0; ctx.trans[3] = 0.0; // from 1: [1→0, 1→1]

        let mut labels = vec![0i32; 2];
        let score = ctx.viterbi(&mut labels);

        // Best path: 0 → 1 (score = 2.0 + 1.0 + 2.0 = 5.0)
        assert_eq!(labels, vec![0, 1]);
        assert_eq!(score, 5.0);
    }

    #[test]
    fn test_forward_backward_normalization() {
        // After forward-backward, marginals at each position should sum to ~1.0
        let l = 3;
        let t = 4;
        let mut ctx = Crf1dContext::new(CTXF_VITERBI | CTXF_MARGINALS, l, t);
        ctx.set_num_items(t);

        // Set some state/trans scores
        for i in 0..t * l { ctx.state[i] = (i as f64) * 0.3 - 1.0; }
        for i in 0..l * l { ctx.trans[i] = (i as f64) * 0.2 - 0.5; }

        ctx.exp_state();
        ctx.exp_transition();
        ctx.alpha_score();
        ctx.beta_score();
        ctx.marginals();

        // Check marginals sum to ~1 at each position
        for pos in 0..t {
            let sum: f64 = (0..l).map(|j| ctx.marginal_point(j, pos)).sum();
            assert!(
                (sum - 1.0).abs() < 1e-10,
                "marginals at t={} sum to {} instead of 1.0", pos, sum
            );
        }
    }

    #[test]
    fn test_score_matches_viterbi() {
        let l = 3;
        let t = 3;
        let mut ctx = Crf1dContext::new(CTXF_VITERBI | CTXF_MARGINALS, l, t);
        ctx.set_num_items(t);

        for i in 0..t * l { ctx.state[i] = (i as f64) * 0.5 - 2.0; }
        for i in 0..l * l { ctx.trans[i] = (i as f64) * 0.3 - 1.0; }

        let mut labels = vec![0i32; t];
        let viterbi_score = ctx.viterbi(&mut labels);

        // Score the viterbi path
        let path_score = ctx.score(&labels);
        assert!(
            (viterbi_score - path_score).abs() < 1e-10,
            "viterbi score {} != path score {}", viterbi_score, path_score
        );
    }
}
