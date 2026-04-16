//! AROW (Adaptive Regularization of Weights) training.

use crate::crf1d::encode::Crf1dEncoder;
use crate::train::{HoldoutFn, LogFn};

#[allow(clippy::too_many_arguments)]
pub fn train_arow(
    encoder: &mut Crf1dEncoder,
    instances: &mut [crate::types::Instance],
    variance: f64,
    gamma: f64,
    max_iterations: i32,
    epsilon: f64,
    log: &mut LogFn,
    mut holdout: Option<&mut HoldoutFn<'_>>,
) -> Vec<f64> {
    let k = encoder.num_features;
    let n = instances.len();
    let mut mean = vec![0.0f64; k];
    let mut cov = vec![variance; k];
    let mut encoded_instances = encoder.encode_instances(instances);

    // Pre-allocate reusable buffers
    let max_t = instances.iter().map(|i| i.num_items()).max().unwrap_or(0);
    let mut pred = vec![0i32; max_t];
    let mut delta = vec![0.0f64; k];
    let mut used = vec![false; k];
    let mut active_indices: Vec<usize> = Vec::with_capacity(k / 4);
    let mut transitions_dirty = true;

    (log)("Adaptive Regularization of Weights (AROW)\n");
    (log)(&format!("variance: {:.6}\n", variance));
    (log)(&format!("gamma: {:.6}\n", gamma));
    (log)(&format!("max_iterations: {}\n", max_iterations));
    (log)(&format!("epsilon: {:.6}\n\n", epsilon));

    for epoch in 1..=max_iterations {
        for i in 0..n {
            let j = crate::rng::rand_int() % n;
            instances.swap(i, j);
            encoded_instances.swap(i, j);
        }

        let mut sum_loss = 0.0f64;

        for inst in &encoded_instances {
            if transitions_dirty {
                encoder.set_transitions_from_weights(&mean, 1.0);
                transitions_dirty = false;
            }
            encoder.set_encoded_instance_from_weights(inst, &mean, 1.0);

            let t_max = inst.items.len();
            let sv = encoder.viterbi(&mut pred[..t_max]);

            if pred[..t_max] != inst.labels[..] {
                let d = pred[..t_max].iter().zip(inst.labels.iter())
                    .filter(|(p, g)| p != g).count() as f64;

                let sc = encoder.score(&inst.labels);
                let cost = (sv - sc) + d;

                // Compute delta = F(y) - F(y_pred), track active indices
                active_indices.clear();
                encoder.features_on_path_encoded(inst, &inst.labels, |fid, val| {
                    let idx = fid as usize;
                    if !used[idx] {
                        used[idx] = true;
                        active_indices.push(idx);
                    }
                    delta[idx] += inst.weight * val;
                });
                encoder.features_on_path_encoded(inst, &pred[..t_max], |fid, val| {
                    let idx = fid as usize;
                    if !used[idx] {
                        used[idx] = true;
                        active_indices.push(idx);
                    }
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
                    used[i] = false;
                }

                transitions_dirty = true;
                sum_loss += cost * inst.weight;
            }
        }

        let feature_norm = mean.iter().map(|value| value * value).sum::<f64>().sqrt();
        (log)(&format!("***** Iteration #{} *****\n", epoch));
        (log)(&format!("Loss: {:.6}\n", sum_loss));
        (log)(&format!("Feature norm: {:.6}\n", feature_norm));
        (log)("Seconds required for this iteration: 0.000\n");

        if let Some(eval) = holdout.as_deref_mut() {
            eval(encoder, &mean, log);
            transitions_dirty = true;
        }

        (log)("\n");

        if (sum_loss / n as f64) <= epsilon {
            (log)("Terminated with the stopping criterion\n\n");
            break;
        }
    }

    (log)("Total seconds required for training: 0.000\n\n");

    mean
}
