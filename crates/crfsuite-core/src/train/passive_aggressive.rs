/// Passive-Aggressive training (PA, PA-I, PA-II variants).

use crate::crf1d::encode::Crf1dEncoder;
use crate::train::LogFn;

pub fn train_passive_aggressive(
    encoder: &mut Crf1dEncoder,
    instances: &mut [crate::types::Instance],
    pa_type: i32,    // 0=PA, 1=PA-I, 2=PA-II
    c: f64,
    error_sensitive: bool,
    averaging: bool,
    max_iterations: i32,
    epsilon: f64,
    log: &mut LogFn,
) -> Vec<f64> {
    let k = encoder.num_features;
    let n = instances.len();
    let mut w = vec![0.0f64; k];
    let mut ws = vec![0.0f64; k];
    let mut u = 1u64;

    for epoch in 1..=max_iterations {
        for i in 0..n {
            let j = (unsafe { libc::rand() } as usize) % n;
            instances.swap(i, j);
        }

        let mut sum_loss = 0.0f64;

        for inst in instances.iter() {
            encoder.set_weights(&w, 1.0);
            encoder.set_instance(inst);

            let t_max = inst.num_items();
            let mut pred = vec![0i32; t_max];
            let sv = encoder.viterbi(&mut pred);

            if pred != inst.labels {
                let d = pred.iter().zip(inst.labels.iter())
                    .filter(|(p, g)| p != g).count() as f64;

                let sc = encoder.score(&inst.labels);
                let cost = if error_sensitive {
                    (sv - sc) + d.sqrt()
                } else {
                    (sv - sc) + 1.0
                };

                // Compute delta = F(y) - F(y_pred)
                let mut delta = vec![0.0f64; k];
                encoder.features_on_path(inst, &inst.labels, |fid, val| {
                    delta[fid as usize] += val;
                });
                encoder.features_on_path(inst, &pred, |fid, val| {
                    delta[fid as usize] -= val;
                });

                let norm2: f64 = delta.iter().map(|x| x * x).sum();

                let tau = if norm2 == 0.0 {
                    0.0
                } else {
                    match pa_type {
                        1 => (cost / norm2).min(c),     // PA-I
                        2 => cost / (norm2 + 0.5 / c),  // PA-II
                        _ => cost / norm2,               // PA
                    }
                };

                let tw = tau * inst.weight;
                for i in 0..k {
                    if delta[i] != 0.0 {
                        w[i] += tw * delta[i];
                        ws[i] += tw * u as f64 * delta[i];
                    }
                }

                sum_loss += cost * inst.weight;
            }

            u += 1;
        }

        (log)(&format!("***** Iteration #{} *****\nLoss: {:.6}\n\n", epoch, sum_loss));

        if (sum_loss / n as f64) < epsilon {
            break;
        }
    }

    if averaging {
        let inv_u = 1.0 / u as f64;
        for i in 0..k {
            w[i] -= inv_u * ws[i];
        }
    }

    w
}
