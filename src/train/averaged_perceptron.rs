//! Averaged Perceptron training.

use crate::crf1d::encode::Crf1dEncoder;
use crate::train::{HoldoutFn, LogFn};

pub fn train_averaged_perceptron(
    encoder: &mut Crf1dEncoder,
    instances: &mut [crate::types::Instance],
    max_iterations: i32,
    epsilon: f64,
    log: &mut LogFn,
    mut holdout: Option<&mut HoldoutFn<'_>>,
) -> Vec<f64> {
    let k = encoder.num_features;
    let n = instances.len();
    let mut w = vec![0.0f64; k];
    let mut ws = vec![0.0f64; k];
    let mut c = 1u64;
    let mut wa = vec![0.0f64; k];

    let max_t = instances.iter().map(|i| i.num_items()).max().unwrap_or(0);
    let mut pred = vec![0i32; max_t];

    (log)("Averaged perceptron\n");
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
            encoder.viterbi(&mut pred[..t_max]);

            if pred[..t_max] != inst.labels[..] {
                let d = pred[..t_max].iter().zip(inst.labels.iter())
                    .filter(|(p, g)| p != g).count();

                let cf = c as f64;
                encoder.features_on_path(inst, &inst.labels, |fid, val| {
                    let v = inst.weight * val;
                    w[fid as usize] += v;
                    ws[fid as usize] += cf * v;
                });

                encoder.features_on_path(inst, &pred[..t_max], |fid, val| {
                    let v = inst.weight * val;
                    w[fid as usize] -= v;
                    ws[fid as usize] -= cf * v;
                });

                sum_loss += d as f64 / t_max as f64 * inst.weight;
            }

            c += 1;
        }

        let avg_loss = sum_loss / n as f64;
        let inv_c = 1.0 / c as f64;
        for i in 0..k {
            wa[i] = w[i] - inv_c * ws[i];
        }
        let feature_norm = wa.iter().map(|value| value * value).sum::<f64>().sqrt();
        (log)(&format!("***** Iteration #{} *****\n", epoch));
        (log)(&format!("Loss: {:.6}\n", sum_loss));
        (log)(&format!("Feature norm: {:.6}\n", feature_norm));
        (log)("Seconds required for this iteration: 0.000\n");

        if let Some(eval) = holdout.as_deref_mut() {
            eval(encoder, &wa, log);
        }

        (log)("\n");

        if avg_loss < epsilon {
            (log)("Terminated with the stopping criterion\n\n");
            break;
        }
    }

    (log)("Total seconds required for training: 0.000\n\n");

    wa
}
