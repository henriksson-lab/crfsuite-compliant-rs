//! L2-regularized stochastic gradient descent.

use std::os::raw::c_int;

use crate::crf1d::encode::{Crf1dEncoder, EncodedInstance};
use crate::train::LogFn;
use crate::types::Instance;

unsafe extern "C" {
    fn rand() -> c_int;
}

pub struct L2SgdOptions {
    pub c2: f64,
    pub max_iterations: i32,
    pub period: i32,
    pub delta: f64,
    pub calibration_eta: f64,
    pub calibration_rate: f64,
    pub calibration_samples: i32,
    pub calibration_candidates: i32,
    pub calibration_max_trials: i32,
}

pub fn train_l2sgd(
    encoder: &mut Crf1dEncoder,
    instances: &[Instance],
    options: &L2SgdOptions,
    log: &mut LogFn,
) -> Vec<f64> {
    let k = encoder.num_features;
    let n = instances.len();
    let lambda = 2.0 * options.c2 / n as f64;
    let mut w = vec![0.0f64; k];
    let mut perm: Vec<usize> = (0..n).collect();
    let encoded_instances = encoder.encode_instances(instances);

    (log)("Stochastic Gradient Descent (SGD)\n");
    (log)(&format!("c2: {:.6}\n", options.c2));
    (log)(&format!("max_iterations: {}\n", options.max_iterations));
    (log)(&format!("period: {}\n", options.period));
    (log)(&format!("delta: {:.6}\n\n", options.delta));

    let t0 = l2sgd_calibration(
        encoder,
        &encoded_instances,
        &mut perm,
        &mut w,
        lambda,
        options,
        log,
    );
    let loss = l2sgd_epochs(
        encoder,
        &encoded_instances,
        &mut perm,
        &mut w,
        n,
        t0,
        lambda,
        options.max_iterations,
        false,
        options.period,
        options.delta,
        log,
    );

    (log)(&format!("Loss: {:.6}\n", loss));
    (log)("Total seconds required for training: 0.000\n\n");

    w
}

#[allow(clippy::too_many_arguments)]
fn l2sgd_epochs(
    encoder: &mut Crf1dEncoder,
    instances: &[EncodedInstance],
    perm: &mut [usize],
    w: &mut [f64],
    n: usize,
    t0: f64,
    lambda: f64,
    num_epochs: i32,
    calibration: bool,
    period: i32,
    epsilon: f64,
    log: &mut LogFn,
) -> f64 {
    let k = encoder.num_features;
    let mut t = 0.0f64;
    let mut decay = 1.0f64;
    let mut sum_loss = 0.0f64;
    let mut best_sum_loss = f64::MAX;
    let mut best_w = vec![0.0f64; k];
    let period = period.max(1) as usize;
    let mut pf = vec![0.0f64; period];
    let mut stopped = false;

    w.fill(0.0);

    for epoch in 1..=num_epochs {
        if !calibration {
            (log)(&format!("***** Epoch #{} *****\n", epoch));
            shuffle_perm_like_c(perm);
        }

        sum_loss = 0.0;
        let mut loss = 0.0;
        for &idx in perm.iter().take(n) {
            let eta = 1.0 / (lambda * (t0 + t));
            decay *= 1.0 - eta * lambda;
            let gain = eta / decay;
            loss = encoder.objective_and_gradients_online_encoded(&instances[idx], w, decay, gain);
            sum_loss += loss;
            t += 1.0;
        }

        if !loss.is_finite() {
            (log)("ERROR: overflow loss\n");
            sum_loss = loss;
            break;
        }

        for value in w.iter_mut() {
            *value *= decay;
        }
        decay = 1.0;

        let norm2 = w.iter().map(|value| value * value).sum::<f64>();
        sum_loss += 0.5 * lambda * norm2 * n as f64;

        if !calibration {
            if sum_loss < best_sum_loss {
                best_sum_loss = sum_loss;
                best_w.copy_from_slice(w);
            }

            let improvement = if period < epoch as usize {
                (pf[(epoch as usize - 1) % period] - sum_loss) / sum_loss
            } else {
                epsilon
            };
            pf[(epoch as usize - 1) % period] = sum_loss;

            (log)(&format!("Loss: {:.6}\n", sum_loss));
            if period < epoch as usize {
                (log)(&format!("Improvement ratio: {:.6}\n", improvement));
            }
            (log)(&format!("Feature L2-norm: {:.6}\n", norm2.sqrt()));
            let eta = 1.0 / (lambda * (t0 + t - 1.0));
            (log)(&format!("Learning rate (eta): {:.6}\n", eta));
            (log)(&format!("Total number of feature updates: {:.0}\n", t));
            (log)("Seconds required for this iteration: 0.000\n\n");

            if improvement < epsilon {
                stopped = true;
                break;
            }
        }
    }

    if !calibration {
        if stopped {
            (log)("SGD terminated with the stopping criteria\n");
        } else {
            (log)("SGD terminated with the maximum number of iterations\n");
        }
        if best_sum_loss < f64::MAX {
            sum_loss = best_sum_loss;
            w.copy_from_slice(&best_w);
        }
    }

    sum_loss
}

fn l2sgd_calibration(
    encoder: &mut Crf1dEncoder,
    instances: &[EncodedInstance],
    perm: &mut [usize],
    w: &mut [f64],
    lambda: f64,
    options: &L2SgdOptions,
    log: &mut LogFn,
) -> f64 {
    let mut dec = false;
    let mut trials = 1;
    let mut num = options.calibration_candidates;
    let mut eta = options.calibration_eta;
    let mut best_eta = options.calibration_eta;
    let mut best_loss = f64::MAX;
    let init_eta = options.calibration_eta;
    let rate = options.calibration_rate;
    let samples = instances.len().min(options.calibration_samples.max(0) as usize);

    (log)("Calibrating the learning rate (eta)\n");
    (log)(&format!("calibration.eta: {:.6}\n", eta));
    (log)(&format!("calibration.rate: {:.6}\n", rate));
    (log)(&format!("calibration.samples: {}\n", samples));
    (log)(&format!(
        "calibration.candidates: {}\n",
        options.calibration_candidates
    ));
    (log)(&format!(
        "calibration.max_trials: {}\n",
        options.calibration_max_trials
    ));

    shuffle_perm_like_c(perm);
    w.fill(0.0);

    let mut init_loss = 0.0;
    for &idx in perm.iter().take(samples) {
        init_loss += encoder.objective_and_gradients_online_encoded(&instances[idx], w, 1.0, 0.0);
    }
    (log)(&format!("Initial loss: {:.6}\n", init_loss));

    while num > 0 || !dec {
        (log)(&format!("Trial #{} (eta = {:.6}): ", trials, eta));
        let loss = l2sgd_epochs(
            encoder,
            instances,
            perm,
            w,
            samples,
            1.0 / (lambda * eta),
            lambda,
            1,
            true,
            1,
            0.0,
            log,
        );

        let ok = loss.is_finite() && loss < init_loss;
        if ok {
            (log)(&format!("{:.6}\n", loss));
            num -= 1;
        } else {
            (log)(&format!("{:.6} (worse)\n", loss));
        }

        if loss.is_finite() && loss < best_loss {
            best_loss = loss;
            best_eta = eta;
        }

        if !dec {
            if ok && 0 < num {
                eta *= rate;
            } else {
                dec = true;
                num = options.calibration_candidates;
                eta = init_eta / rate;
            }
        } else {
            eta /= rate;
        }

        trials += 1;
        if options.calibration_max_trials <= trials {
            break;
        }
    }

    (log)(&format!("Best learning rate (eta): {:.6}\n", best_eta));
    (log)("Seconds required: 0.000\n\n");

    1.0 / (lambda * best_eta)
}

fn shuffle_perm_like_c(perm: &mut [usize]) {
    let n = perm.len();
    for i in 0..n {
        let j = unsafe { rand() as usize } % n;
        perm.swap(i, j);
    }
}
