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
    let mut ws = vec![0.0f64; k];
    let mut c = 1u64;

    let max_t = instances.iter().map(|i| i.num_items()).max().unwrap_or(0);
    let mut pred = vec![0i32; max_t];

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
        (log)(&format!("***** Iteration #{} *****\nLoss: {:.6}\n\n", epoch, sum_loss));

        if avg_loss < epsilon {
            break;
        }
    }

    let inv_c = 1.0 / c as f64;
    for i in 0..k {
        w[i] -= inv_c * ws[i];
    }

    w
}
