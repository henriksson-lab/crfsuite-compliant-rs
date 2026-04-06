/// AROW (Adaptive Regularization of Weights) training.

use crate::crf1d::encode::Crf1dEncoder;
use crate::train::LogFn;

pub fn train_arow(
    encoder: &mut Crf1dEncoder,
    instances: &mut [crate::types::Instance],
    variance: f64,
    gamma: f64,
    max_iterations: i32,
    epsilon: f64,
    log: &mut LogFn,
) -> Vec<f64> {
    let k = encoder.num_features;
    let n = instances.len();
    let mut mean = vec![0.0f64; k];
    let mut cov = vec![variance; k]; // diagonal covariance

    for epoch in 1..=max_iterations {
        for i in 0..n {
            let j = (unsafe { libc::rand() } as usize) % n;
            instances.swap(i, j);
        }

        let mut sum_loss = 0.0f64;

        for inst in instances.iter() {
            encoder.set_weights(&mean, 1.0);
            encoder.set_instance(inst);

            let t_max = inst.num_items();
            let mut pred = vec![0i32; t_max];
            let sv = encoder.viterbi(&mut pred);

            if pred != inst.labels {
                let d = pred.iter().zip(inst.labels.iter())
                    .filter(|(p, g)| p != g).count() as f64;

                let sc = encoder.score(&inst.labels);
                let cost = (sv - sc) + d;

                // Compute delta = F(y) - F(y_pred)
                let mut delta = vec![0.0f64; k];
                encoder.features_on_path(inst, &inst.labels, |fid, val| {
                    delta[fid as usize] += inst.weight * val;
                });
                encoder.features_on_path(inst, &pred, |fid, val| {
                    delta[fid as usize] -= inst.weight * val;
                });

                // Compute alpha = cost / (gamma + sum(delta[k]^2 * cov[k]))
                let mut frac = gamma;
                for i in 0..k {
                    if delta[i] != 0.0 {
                        frac += delta[i] * delta[i] * cov[i];
                    }
                }
                let alpha = cost / frac;

                // Update mean and covariance
                for i in 0..k {
                    if delta[i] != 0.0 {
                        mean[i] += alpha * cov[i] * delta[i];
                        cov[i] = 1.0 / (1.0 / cov[i] + delta[i] * delta[i] / gamma);
                    }
                }

                sum_loss += cost * inst.weight;
            }
        }

        (log)(&format!("***** Iteration #{} *****\nLoss: {:.6}\n\n", epoch, sum_loss));

        if (sum_loss / n as f64) <= epsilon {
            break;
        }
    }

    mean
}
