use crfsuite_compliant_rs::crf1d::encode::Crf1dEncoder;
use crfsuite_compliant_rs::train;
use crfsuite_compliant_rs::types::Instance;

use crate::learn_params::{
    get_float_param, get_int_param, get_str_param, ALGO_AROW, ALGO_AVERAGED_PERCEPTRON, ALGO_L2SGD,
    ALGO_LBFGS, ALGO_PASSIVE_AGGRESSIVE,
};

pub struct TrainOnce<'a> {
    pub algorithm: &'a str,
    pub params: &'a [(String, String)],
    pub instances: &'a [Instance],
    pub holdout: i32,
    pub num_labels: usize,
    pub num_attrs: usize,
    pub label_strings: &'a [String],
    pub attr_strings: &'a [String],
    pub model_path: &'a str,
}

pub fn train_once(
    config: TrainOnce<'_>,
    log: &mut train::LogFn,
) -> Result<(), Box<dyn std::error::Error>> {
    let train_instances: Vec<Instance> = config
        .instances
        .iter()
        .filter(|inst| inst.group != config.holdout)
        .cloned()
        .collect();
    let test_instances: Vec<Instance> = config
        .instances
        .iter()
        .filter(|inst| inst.group == config.holdout)
        .cloned()
        .collect();

    let algorithm = config.algorithm;
    let min_freq = get_float_param(algorithm, config.params, "feature.minfreq");
    let possible_states = get_int_param(algorithm, config.params, "feature.possible_states") != 0;
    let possible_transitions =
        get_int_param(algorithm, config.params, "feature.possible_transitions") != 0;

    let mut encoder = Crf1dEncoder::new(
        &train_instances,
        config.num_labels,
        config.num_attrs,
        min_freq,
        possible_states,
        possible_transitions,
    );

    if config.holdout >= 0 {
        (log)(&format!("Holdout group: {}\n\n", config.holdout + 1));
    }

    (log)(&format!("Feature generation\ntype: CRF1d\nfeature.minfreq: {:.6}\nfeature.possible_states: {}\nfeature.possible_transitions: {}\n",
        min_freq, possible_states as i32, possible_transitions as i32));
    (log)(&format!("Number of features: {}\n\n", encoder.num_features));

    let weights = match config.algorithm {
        ALGO_LBFGS => {
            let c1 = get_float_param(algorithm, config.params, "c1");
            let c2 = get_float_param(algorithm, config.params, "c2");
            let max_iter = get_int_param(algorithm, config.params, "max_iterations");
            let num_mem = get_int_param(algorithm, config.params, "num_memories");
            let epsilon = get_float_param(algorithm, config.params, "epsilon");
            let period = get_int_param(algorithm, config.params, "period");
            let delta = get_float_param(algorithm, config.params, "delta");
            let ls = get_str_param(algorithm, config.params, "linesearch");
            let max_ls = get_int_param(algorithm, config.params, "max_linesearch");

            (log)(&format!("L-BFGS optimization\nc1: {:.6}\nc2: {:.6}\nnum_memories: {}\nmax_iterations: {}\nepsilon: {:.6}\nstop: {}\ndelta: {:.6}\nlinesearch: {}\nlinesearch.max_iterations: {}\n\n",
                c1, c2, num_mem, max_iter, epsilon, period, delta, ls, max_ls));

            train::lbfgs::train_lbfgs(
                &mut encoder,
                &train_instances,
                c1,
                c2,
                max_iter,
                num_mem,
                epsilon,
                period,
                delta,
                &ls,
                max_ls,
                log,
            )
        }
        ALGO_L2SGD => {
            let c2 = get_float_param(algorithm, config.params, "c2");
            let max_iter = get_int_param(algorithm, config.params, "max_iterations");
            let period = get_int_param(algorithm, config.params, "period");
            let delta = get_float_param(algorithm, config.params, "delta");
            let calibration_eta = get_float_param(algorithm, config.params, "calibration.eta");
            let calibration_rate = get_float_param(algorithm, config.params, "calibration.rate");
            let calibration_samples =
                get_int_param(algorithm, config.params, "calibration.samples");
            let calibration_candidates =
                get_int_param(algorithm, config.params, "calibration.candidates");
            let calibration_max_trials =
                get_int_param(algorithm, config.params, "calibration.max_trials");
            train::l2sgd::train_l2sgd(
                &mut encoder,
                &train_instances,
                &train::l2sgd::L2SgdOptions {
                    c2,
                    max_iterations: max_iter,
                    period,
                    delta,
                    calibration_eta,
                    calibration_rate,
                    calibration_samples,
                    calibration_candidates,
                    calibration_max_trials,
                },
                log,
            )
        }
        ALGO_AVERAGED_PERCEPTRON => {
            let max_iter = get_int_param(algorithm, config.params, "max_iterations");
            let epsilon = get_float_param(algorithm, config.params, "epsilon");
            let mut holdout_eval =
                |encoder: &mut Crf1dEncoder, weights: &[f64], log: &mut train::LogFn| {
                    log_holdout_evaluation(
                        encoder,
                        weights,
                        &test_instances,
                        config.label_strings,
                        log,
                    );
                };
            let holdout = if config.holdout >= 0 {
                Some(&mut holdout_eval as &mut train::HoldoutFn<'_>)
            } else {
                None
            };
            train::averaged_perceptron::train_averaged_perceptron(
                &mut encoder,
                &mut train_instances.clone(),
                max_iter,
                epsilon,
                log,
                holdout,
            )
        }
        ALGO_PASSIVE_AGGRESSIVE => {
            let pa_type = get_int_param(algorithm, config.params, "type");
            let c = get_float_param(algorithm, config.params, "c");
            let error_sensitive = get_int_param(algorithm, config.params, "error_sensitive") != 0;
            let averaging = get_int_param(algorithm, config.params, "averaging") != 0;
            let max_iter = get_int_param(algorithm, config.params, "max_iterations");
            let epsilon = get_float_param(algorithm, config.params, "epsilon");
            let mut holdout_eval =
                |encoder: &mut Crf1dEncoder, weights: &[f64], log: &mut train::LogFn| {
                    log_holdout_evaluation(
                        encoder,
                        weights,
                        &test_instances,
                        config.label_strings,
                        log,
                    );
                };
            let holdout = if config.holdout >= 0 {
                Some(&mut holdout_eval as &mut train::HoldoutFn<'_>)
            } else {
                None
            };
            train::passive_aggressive::train_passive_aggressive(
                &mut encoder,
                &mut train_instances.clone(),
                pa_type,
                c,
                error_sensitive,
                averaging,
                max_iter,
                epsilon,
                log,
                holdout,
            )
        }
        ALGO_AROW => {
            let variance = get_float_param(algorithm, config.params, "variance");
            let gamma = get_float_param(algorithm, config.params, "gamma");
            let max_iter = get_int_param(algorithm, config.params, "max_iterations");
            let epsilon = get_float_param(algorithm, config.params, "epsilon");
            let mut holdout_eval =
                |encoder: &mut Crf1dEncoder, weights: &[f64], log: &mut train::LogFn| {
                    log_holdout_evaluation(
                        encoder,
                        weights,
                        &test_instances,
                        config.label_strings,
                        log,
                    );
                };
            let holdout = if config.holdout >= 0 {
                Some(&mut holdout_eval as &mut train::HoldoutFn<'_>)
            } else {
                None
            };
            train::arow::train_arow(
                &mut encoder,
                &mut train_instances.clone(),
                variance,
                gamma,
                max_iter,
                epsilon,
                log,
                holdout,
            )
        }
        _ => return Err(format!("Unknown algorithm: {}", config.algorithm).into()),
    };

    if !config.model_path.is_empty() {
        let model_bytes = encoder.save_model(&weights, config.label_strings, config.attr_strings);
        std::fs::write(config.model_path, &model_bytes)?;
    }

    Ok(())
}

fn log_holdout_evaluation(
    encoder: &mut Crf1dEncoder,
    weights: &[f64],
    instances: &[Instance],
    label_strings: &[String],
    log: &mut train::LogFn,
) {
    let num_labels = label_strings.len();
    let mut correct = vec![0i32; num_labels];
    let mut model = vec![0i32; num_labels];
    let mut obs = vec![0i32; num_labels];
    let mut item_total = 0i32;
    let mut item_correct = 0i32;
    let mut inst_total = 0i32;
    let mut inst_correct = 0i32;
    let max_t = instances
        .iter()
        .map(|inst| inst.num_items())
        .max()
        .unwrap_or(0);
    let mut pred = vec![0i32; max_t];

    encoder.set_weights(weights, 1.0);
    for inst in instances {
        let t_max = inst.num_items();
        encoder.set_instance(inst);
        encoder.viterbi(&mut pred[..t_max]);

        let mut sequence_correct = 0;
        for (&reference, &prediction) in inst.labels.iter().zip(&pred[..t_max]) {
            let reference = reference as usize;
            let prediction = prediction as usize;
            if reference < num_labels {
                obs[reference] += 1;
            }
            if prediction < num_labels {
                model[prediction] += 1;
            }
            if reference == prediction && reference < num_labels {
                correct[reference] += 1;
                item_correct += 1;
                sequence_correct += 1;
            }
            item_total += 1;
        }
        if sequence_correct == t_max {
            inst_correct += 1;
        }
        inst_total += 1;
    }

    (log)("Performance by label (#match, #model, #ref) (precision, recall, F1):\n");
    let mut macro_p = 0.0;
    let mut macro_r = 0.0;
    let mut macro_f = 0.0;
    for l in 0..num_labels {
        let p = if model[l] > 0 {
            correct[l] as f64 / model[l] as f64
        } else {
            0.0
        };
        let r = if obs[l] > 0 {
            correct[l] as f64 / obs[l] as f64
        } else {
            0.0
        };
        let f = if p + r > 0.0 {
            2.0 * p * r / (p + r)
        } else {
            0.0
        };
        macro_p += p;
        macro_r += r;
        macro_f += f;

        if obs[l] == 0 {
            (log)(&format!(
                "    {}: ({}, {}, {}) (******, ******, ******)\n",
                label_strings[l], correct[l], model[l], obs[l]
            ));
        } else {
            (log)(&format!(
                "    {}: ({}, {}, {}) ({:.4}, {:.4}, {:.4})\n",
                label_strings[l], correct[l], model[l], obs[l], p, r, f
            ));
        }
    }

    let num_labels = num_labels as f64;
    (log)(&format!(
        "Macro-average precision, recall, F1: ({:.6}, {:.6}, {:.6})\n",
        macro_p / num_labels,
        macro_r / num_labels,
        macro_f / num_labels
    ));
    let item_accuracy = if item_total > 0 {
        item_correct as f64 / item_total as f64
    } else {
        0.0
    };
    (log)(&format!(
        "Item accuracy: {} / {} ({:.4})\n",
        item_correct, item_total, item_accuracy
    ));
    let inst_accuracy = if inst_total > 0 {
        inst_correct as f64 / inst_total as f64
    } else {
        0.0
    };
    (log)(&format!(
        "Instance accuracy: {} / {} ({:.4})\n",
        inst_correct, inst_total, inst_accuracy
    ));
}
