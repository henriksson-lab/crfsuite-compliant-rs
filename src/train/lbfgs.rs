//! L-BFGS training with L1/L2 regularization.
//! Uses the pure Rust `liblbfgs-compliant-rs` crate.

use crate::crf1d::encode::Crf1dEncoder;
use crate::train::LogFn;

use liblbfgs_compliant_rs::{self as lbfgs, LbfgsParam, LineSearch, OrthantWise};

#[allow(clippy::too_many_arguments)]
pub fn train_lbfgs(
    encoder: &mut Crf1dEncoder,
    instances: &[crate::types::Instance],
    c1: f64,
    c2: f64,
    max_iterations: i32,
    num_memories: i32,
    epsilon: f64,
    period: i32,
    delta: f64,
    linesearch: &str,
    max_linesearch: i32,
    log: &mut LogFn,
) -> Vec<f64> {
    let k = encoder.num_features;
    let mut w = vec![0.0f64; k];
    let num_features = encoder.num_features;

    let ls = match linesearch {
        "Backtracking" => LineSearch::BacktrackingWolfe,
        "StrongBacktracking" => LineSearch::BacktrackingStrongWolfe,
        _ => LineSearch::MoreThuente,
    };

    let orthantwise = if c1 > 0.0 {
        Some(OrthantWise { c: c1, start: 0, end: k as i32 })
    } else {
        None
    };

    let param = LbfgsParam {
        m: num_memories,
        epsilon,
        past: period,
        delta,
        max_iterations,
        linesearch: if orthantwise.is_some() { LineSearch::BacktrackingWolfe } else { ls },
        max_linesearch,
        orthantwise,
        ..LbfgsParam::default()
    };

    // Evaluate callback: objective + gradients
    let mut evaluate = |w: &[f64], g: &mut [f64], _step: f64| -> f64 {
        let f = encoder.objective_and_gradients_batch(instances, w, g);

        if c2 > 0.0 {
            let mut norm2 = 0.0f64;
            for i in 0..k {
                g[i] += 2.0 * c2 * w[i];
                norm2 += w[i] * w[i];
            }
            return f + c2 * norm2;
        }

        f
    };

    // Progress callback
    let mut progress = |prgr: &lbfgs::ProgressReport| -> bool {
        (log)(&format!(
            "***** Iteration #{} *****\nLoss: {:.6}\nFeature norm: {:.6}\nError norm: {:.6}\nActive features: {}\nLine search trials: {}\nLine search step: {:.6}\nSeconds required for this iteration: 0.000\n\n",
            prgr.k, prgr.fx, prgr.xnorm, prgr.gnorm, num_features, prgr.ls, prgr.step
        ));
        true // continue (true = keep going)
    };

    let result = lbfgs::lbfgs(&mut w, &mut evaluate, Some(&mut progress), &param);

    match &result {
        Ok(r) => match r.convergence {
            lbfgs::Convergence::AlreadyMinimized => (log)("L-BFGS stopped (already minimum)\n"),
            _ => (log)("L-BFGS resulted in convergence\n"),
        },
        Err(lbfgs::LbfgsError::MaximumIteration) => (log)("L-BFGS stopped (maximum iterations)\n"),
        Err(e) => (log)(&format!("L-BFGS finished ({:?})\n", e)),
    };

    w
}
