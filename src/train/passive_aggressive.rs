//! Passive-Aggressive training (PA, PA-I, PA-II variants).

use crate::crf1d::encode::Crf1dEncoder;
use crate::train::{HoldoutFn, LogFn};

#[allow(clippy::too_many_arguments)]
pub fn train_passive_aggressive(
    encoder: &mut Crf1dEncoder,
    instances: &mut [crate::types::Instance],
    pa_type: i32,
    c: f64,
    error_sensitive: bool,
    averaging: bool,
    max_iterations: i32,
    epsilon: f64,
    log: &mut LogFn,
    mut holdout: Option<&mut HoldoutFn<'_>>,
) -> Vec<f64> {
    let k = encoder.num_features;
    let n = instances.len();
    let mut w = vec![0.0f64; k];
    let mut ws = vec![0.0f64; k];
    let mut wa = vec![0.0f64; k];
    let mut u = 1u64;

    let max_t = instances.iter().map(|i| i.num_items()).max().unwrap_or(0);
    let mut pred = vec![0i32; max_t];
    let mut delta = vec![0.0f64; k];
    let mut used = vec![false; k];
    let mut active_indices: Vec<usize> = Vec::with_capacity(k / 4);

    (log)("Passive Aggressive\n");
    (log)(&format!("type: {}\n", pa_type));
    (log)(&format!("c: {:.6}\n", c));
    (log)(&format!("error_sensitive: {}\n", error_sensitive as i32));
    (log)(&format!("averaging: {}\n", averaging as i32));
    (log)(&format!("max_iterations: {}\n", max_iterations));
    (log)(&format!("epsilon: {:.6}\n\n", epsilon));

    for epoch in 1..=max_iterations {
        for i in 0..n {
            let j = crate::rng::rand_int() % n;
            instances.swap(i, j);
        }

        let mut sum_loss = 0.0f64;

        for inst in instances.iter() {
            encoder.set_weights(&w, 1.0);
            encoder.set_instance(inst);

            let t_max = inst.num_items();
            let sv = encoder.viterbi(&mut pred[..t_max]);

            if pred[..t_max] != inst.labels[..] {
                let d = pred[..t_max].iter().zip(inst.labels.iter())
                    .filter(|(p, g)| p != g).count() as f64;

                let sc = encoder.score(&inst.labels);
                let cost = if error_sensitive {
                    (sv - sc) + d.sqrt()
                } else {
                    (sv - sc) + 1.0
                };

                active_indices.clear();
                encoder.features_on_path(inst, &inst.labels, |fid, val| {
                    let idx = fid as usize;
                    if !used[idx] {
                        used[idx] = true;
                        active_indices.push(idx);
                    }
                    delta[idx] += val;
                });
                encoder.features_on_path(inst, &pred[..t_max], |fid, val| {
                    let idx = fid as usize;
                    if !used[idx] {
                        used[idx] = true;
                        active_indices.push(idx);
                    }
                    delta[idx] -= val;
                });

                let norm2: f64 = active_indices.iter().map(|&i| delta[i] * delta[i]).sum();

                let tau = if norm2 == 0.0 {
                    0.0
                } else {
                    match pa_type {
                        1 => (cost / norm2).min(c),
                        2 => cost / (norm2 + 0.5 / c),
                        _ => cost / norm2,
                    }
                };

                let tw = tau * inst.weight;
                for &i in &active_indices {
                    if delta[i] != 0.0 {
                        w[i] += tw * delta[i];
                        ws[i] += tw * u as f64 * delta[i];
                    }
                }

                for &i in &active_indices {
                    delta[i] = 0.0;
                    used[i] = false;
                }

                sum_loss += cost * inst.weight;
            }

            u += 1;
        }

        if averaging {
            let inv_u = 1.0 / u as f64;
            for i in 0..k {
                wa[i] = w[i] - inv_u * ws[i];
            }
        } else {
            wa.copy_from_slice(&w);
        }

        let feature_norm = w.iter().map(|value| value * value).sum::<f64>().sqrt();
        (log)(&format!("***** Iteration #{} *****\n", epoch));
        (log)(&format!("Loss: {:.6}\n", sum_loss));
        (log)(&format!("Feature norm: {:.6}\n", feature_norm));
        (log)("Seconds required for this iteration: 0.000\n");

        if let Some(eval) = holdout.as_deref_mut() {
            eval(encoder, &wa, log);
        }

        (log)("\n");

        if (sum_loss / n as f64) < epsilon {
            (log)("Terminated with the stopping criterion\n\n");
            break;
        }
    }

    (log)("Total seconds required for training: 0.000\n\n");

    wa
}
