/// Averaged Perceptron training.

use crate::crf1d::encode::Crf1dEncoder;
use crate::train::LogFn;

pub fn train_averaged_perceptron(
    encoder: &mut Crf1dEncoder,
    instances: &mut [crate::types::Instance],
    max_iterations: i32,
    epsilon: f64,
    log: &mut LogFn,
) -> Vec<f64> {
    let k = encoder.num_features;
    let n = instances.len();
    let mut w = vec![0.0f64; k];
    let mut ws = vec![0.0f64; k]; // sum for averaging
    let mut c = 1u64;

    for epoch in 1..=max_iterations {
        // Shuffle
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
            encoder.viterbi(&mut pred);

            if pred != inst.labels {
                let d = pred.iter().zip(inst.labels.iter())
                    .filter(|(p, g)| p != g).count();

                // Update on correct path (+)
                encoder.features_on_path(inst, &inst.labels, |fid, val| {
                    let v = inst.weight * val;
                    w[fid as usize] += v;
                    ws[fid as usize] += c as f64 * v;
                });

                // Update on predicted path (-)
                encoder.features_on_path(inst, &pred, |fid, val| {
                    let v = inst.weight * val;
                    w[fid as usize] -= v;
                    ws[fid as usize] -= c as f64 * v;
                });

                sum_loss += d as f64 / t_max as f64 * inst.weight;
            }

            c += 1;
        }

        let avg_loss = sum_loss / n as f64;
        (log)(&format!("***** Iteration #{} *****\nLoss: {:.6}\n\n", epoch, sum_loss));

        if avg_loss < epsilon {
            break;
        }
    }

    // Compute averaged weights: wa = w - (1/c) * ws
    let inv_c = 1.0 / c as f64;
    for i in 0..k {
        w[i] -= inv_c * ws[i];
    }

    w
}
