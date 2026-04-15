//! L2-regularized Stochastic Gradient Descent (Pegasos-style).
//!
//! Simplified implementation: uses batch gradient per epoch.
//! For exact C parity, the per-instance decay/gain logic would need
//! to be replicated exactly.

use crate::crf1d::encode::Crf1dEncoder;
use crate::train::LogFn;

pub struct L2SgdOptions {
    pub c2: f64,
    pub max_iterations: i32,
    pub period: i32,
    pub delta: f64,
    pub calibration_eta: f64,
}

pub fn train_l2sgd(
    encoder: &mut Crf1dEncoder,
    instances: &[crate::types::Instance],
    options: &L2SgdOptions,
    log: &mut LogFn,
) -> Vec<f64> {
    let k = encoder.num_features;
    let n = instances.len() as f64;
    let lambda = 2.0 * options.c2 / n;

    // Simplified calibration: CRFsuite defaults the initial eta to 0.1 and
    // derives t0 from it after calibration.
    let eta0 = options.calibration_eta;
    let t0 = 1.0 / (lambda * eta0);

    let mut w = vec![0.0f64; k];
    let mut g = vec![0.0f64; k];
    let mut best_loss = f64::MAX;
    let mut total_t = t0;

    for epoch in 1..=options.max_iterations {
        // Compute batch gradient
        let f = encoder.objective_and_gradients_batch(instances, &w, &mut g);

        // Add L2 regularization to gradient
        let mut norm2 = 0.0f64;
        for i in 0..k {
            g[i] += lambda * w[i];
            norm2 += w[i] * w[i];
        }
        let loss = f + 0.5 * lambda * norm2 * n;

        // Learning rate
        let eta = 1.0 / (lambda * total_t);
        total_t += n; // advance by N steps

        // SGD update
        for i in 0..k {
            w[i] -= eta * g[i];
        }

        (log)(&format!(
            "***** Iteration #{} *****\nLoss: {:.6}\n\n",
            epoch, loss
        ));

        if epoch % options.period == 0 {
            let improvement = (best_loss - loss) / loss.abs().max(1.0);
            if loss < best_loss && improvement < options.delta {
                (log)("Converged.\n");
                break;
            }
            best_loss = best_loss.min(loss);
        }
    }

    w
}
