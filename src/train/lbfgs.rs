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
    let encoded_instances = encoder.encode_instances(instances);
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
        let f = encoder.objective_and_gradients_batch_encoded(&encoded_instances, w, g);

        add_l2_regularization(f, g, w, c2)
    };

    // Progress callback
    let mut progress = |prgr: &lbfgs::ProgressReport| -> bool {
        let active_features = prgr.x.iter().filter(|&&x| x != 0.0).count();
        (log)(&format!(
            "***** Iteration #{} *****\nLoss: {:.6}\nFeature norm: {:.6}\nError norm: {:.6}\nActive features: {}\nLine search trials: {}\nLine search step: {:.6}\nSeconds required for this iteration: 0.000\n\n",
            prgr.k, prgr.fx, prgr.xnorm, prgr.gnorm, active_features, prgr.ls, prgr.step
        ));
        true // continue (true = keep going)
    };

    let result = lbfgs::lbfgs(&mut w, &mut evaluate, Some(&mut progress), &param);

    match &result {
        Ok(r) => match r.convergence {
            lbfgs::Convergence::AlreadyMinimized => {
                (log)("L-BFGS terminated with error code (2)\n");
            }
            lbfgs::Convergence::Delta => {
                (log)("L-BFGS terminated with the stopping criteria\n");
            }
            _ => (log)("L-BFGS resulted in convergence\n"),
        },
        Err(lbfgs::LbfgsError::MaximumIteration) => {
            (log)("L-BFGS terminated with the maximum number of iterations\n");
        }
        Err(e) => (log)(&format!(
            "L-BFGS terminated with error code ({})\n",
            lbfgs_error_code(e)
        )),
    };
    (log)("Total seconds required for training: 0.000\n\n");

    w
}

fn lbfgs_error_code(error: &lbfgs::LbfgsError) -> i32 {
    match error {
        lbfgs::LbfgsError::UnknownError => -1024,
        lbfgs::LbfgsError::LogicError => -1023,
        lbfgs::LbfgsError::OutOfMemory => -1022,
        lbfgs::LbfgsError::Canceled => -1021,
        lbfgs::LbfgsError::InvalidN => -1020,
        lbfgs::LbfgsError::InvalidEpsilon => -1017,
        lbfgs::LbfgsError::InvalidTestPeriod => -1016,
        lbfgs::LbfgsError::InvalidDelta => -1015,
        lbfgs::LbfgsError::InvalidLineSearch => -1014,
        lbfgs::LbfgsError::InvalidMinStep => -1013,
        lbfgs::LbfgsError::InvalidMaxStep => -1012,
        lbfgs::LbfgsError::InvalidFtol => -1011,
        lbfgs::LbfgsError::InvalidWolfe => -1010,
        lbfgs::LbfgsError::InvalidGtol => -1009,
        lbfgs::LbfgsError::InvalidXtol => -1008,
        lbfgs::LbfgsError::InvalidMaxLineSearch => -1007,
        lbfgs::LbfgsError::InvalidOrthantwise => -1006,
        lbfgs::LbfgsError::InvalidOrthantwiseStart => -1005,
        lbfgs::LbfgsError::InvalidOrthantwiseEnd => -1004,
        lbfgs::LbfgsError::OutOfInterval => -1003,
        lbfgs::LbfgsError::IncorrectTMinMax => -1002,
        lbfgs::LbfgsError::RoundingError => -1001,
        lbfgs::LbfgsError::MinimumStep => -1000,
        lbfgs::LbfgsError::MaximumStep => -999,
        lbfgs::LbfgsError::MaximumLineSearch => -998,
        lbfgs::LbfgsError::MaximumIteration => -997,
        lbfgs::LbfgsError::WidthTooSmall => -996,
        lbfgs::LbfgsError::InvalidParameters => -995,
        lbfgs::LbfgsError::IncreaseGradient => -994,
    }
}

fn add_l2_regularization(mut objective: f64, g: &mut [f64], w: &[f64], c2: f64) -> f64 {
    if c2 > 0.0 {
        let mut norm2 = 0.0f64;
        for i in 0..w.len() {
            g[i] += 2.0 * c2 * w[i];
            norm2 += w[i] * w[i];
        }
        objective += c2 * norm2;
    }

    objective
}

#[cfg(test)]
mod tests {
    use super::{add_l2_regularization, lbfgs_error_code};
    use liblbfgs_compliant_rs::LbfgsError;

    #[test]
    fn l2_regularization_matches_c_objective_and_gradient_formula() {
        let mut gradient = vec![-0.5, -0.5, -0.75];
        let weights = vec![1.0, -2.0, 0.5];
        let objective = add_l2_regularization(10.0, &mut gradient, &weights, 0.25);

        assert_eq!(objective, 10.0 + 0.25 * (1.0 + 4.0 + 0.25));
        assert_eq!(gradient, vec![0.0, -1.5, -0.5]);
    }

    #[test]
    fn lbfgs_error_codes_match_c_constants() {
        assert_eq!(lbfgs_error_code(&LbfgsError::InvalidN), -1020);
        assert_eq!(lbfgs_error_code(&LbfgsError::MaximumLineSearch), -998);
        assert_eq!(lbfgs_error_code(&LbfgsError::IncreaseGradient), -994);
    }
}
