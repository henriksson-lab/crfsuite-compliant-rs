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
    let mut cov = vec![variance; k];

    // Pre-allocate reusable buffers
    let max_t = instances.iter().map(|i| i.num_items()).max().unwrap_or(0);
    let mut pred = vec![0i32; max_t];
    let mut delta = vec![0.0f64; k];
    let mut active_indices: Vec<usize> = Vec::with_capacity(k / 4);

    for epoch in 1..=max_iterations {
        for i in 0..n {
            let j = crate::rng::rand_int() % n;
            instances.swap(i, j);
        }

        let mut sum_loss = 0.0f64;

        for inst in instances.iter() {
            encoder.set_weights(&mean, 1.0);
            encoder.set_instance(inst);

            let t_max = inst.num_items();
            let sv = encoder.viterbi(&mut pred[..t_max]);

            if pred[..t_max] != inst.labels[..] {
                let d = pred[..t_max].iter().zip(inst.labels.iter())
                    .filter(|(p, g)| p != g).count() as f64;

                let sc = encoder.score(&inst.labels);
                let cost = (sv - sc) + d;

                // Compute delta = F(y) - F(y_pred), track active indices
                active_indices.clear();
                encoder.features_on_path(inst, &inst.labels, |fid, val| {
                    let idx = fid as usize;
                    if delta[idx] == 0.0 { active_indices.push(idx); }
                    delta[idx] += inst.weight * val;
                });
                encoder.features_on_path(inst, &pred[..t_max], |fid, val| {
                    let idx = fid as usize;
                    if delta[idx] == 0.0 { active_indices.push(idx); }
                    delta[idx] -= inst.weight * val;
                });

                // Compute alpha using only active features
                let mut frac = gamma;
                for &i in &active_indices {
                    if delta[i] != 0.0 {
                        frac += delta[i] * delta[i] * cov[i];
                    }
                }
                let alpha = cost / frac;

                // Update mean and covariance (only active features)
                for &i in &active_indices {
                    if delta[i] != 0.0 {
                        mean[i] += alpha * cov[i] * delta[i];
                        cov[i] = 1.0 / (1.0 / cov[i] + delta[i] * delta[i] / gamma);
                    }
                }

                // Clear delta (only active entries)
                for &i in &active_indices {
                    delta[i] = 0.0;
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
