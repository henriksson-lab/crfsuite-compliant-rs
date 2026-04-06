/// L2-regularized Stochastic Gradient Descent (Pegasos-style).
///
/// Simplified implementation: uses batch gradient per epoch.
/// For exact C parity, the per-instance decay/gain logic would need
/// to be replicated exactly.

use crate::crf1d::encode::Crf1dEncoder;
use crate::train::LogFn;

pub fn train_l2sgd(
    encoder: &mut Crf1dEncoder,
    instances: &[crate::types::Instance],
    c2: f64,
    max_iterations: i32,
    period: i32,
    delta: f64,
    log: &mut LogFn,
) -> Vec<f64> {
    let k = encoder.num_features;
    let n = instances.len() as f64;
    let lambda = 2.0 * c2 / n;

    // Calibration: simple initial learning rate
    let eta0 = 0.1f64;
    let t0 = (1.0 / (lambda * eta0)) as f64;

    let mut w = vec![0.0f64; k];
    let mut g = vec![0.0f64; k];
    let mut best_loss = f64::MAX;
    let mut total_t = t0;

    for epoch in 1..=max_iterations {
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

        (log)(&format!("***** Iteration #{} *****\nLoss: {:.6}\n\n", epoch, loss));

        if epoch % period == 0 {
            let improvement = (best_loss - loss) / loss.abs().max(1.0);
            if loss < best_loss {
                if improvement < delta {
                    (log)("Converged.\n");
                    break;
                }
            }
            best_loss = best_loss.min(loss);
        }
    }

    w
}
