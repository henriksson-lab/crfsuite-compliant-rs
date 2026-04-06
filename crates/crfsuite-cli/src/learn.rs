use std::fs::File;
use std::io::{self, BufReader};

use crfsuite_core::crf1d::encode::Crf1dEncoder;
use crfsuite_core::quark::Quark;
use crfsuite_core::train;
use crfsuite_core::types::{Attribute, Item, Instance};

use crate::iwa::{IwaReader, TokenType};

pub struct LearnArgs {
    pub model_type: String,
    pub algorithm: String,
    pub model_path: String,
    pub params: Vec<(String, String)>,
    pub split: i32,
    pub holdout: i32,
    pub cross_validate: bool,
    pub help_params: bool,
    pub input_files: Vec<String>,
}

pub fn run_learn(args: LearnArgs) -> Result<(), Box<dyn std::error::Error>> {
    let algorithm = match args.algorithm.as_str() {
        "ap" | "averaged-perceptron" => "averaged-perceptron",
        "pa" | "passive-aggressive" => "passive-aggressive",
        other => other,
    };

    let mut log = train::stdout_logger();

    // Read data
    let mut label_quark = Quark::new();
    let mut attr_quark = Quark::new();
    let mut instances: Vec<Instance> = Vec::new();

    let input_files = if args.input_files.is_empty() {
        vec!["-".to_string()]
    } else {
        args.input_files.clone()
    };

    for (fi, filename) in input_files.iter().enumerate() {
        let group = if args.split > 0 { 0 } else { fi as i32 };
        let input: Box<dyn io::BufRead> = if filename == "-" {
            Box::new(BufReader::new(io::stdin()))
        } else {
            Box::new(BufReader::new(File::open(filename)?))
        };

        let mut iwa = IwaReader::new(input);
        let mut inst = Instance::new();
        inst.group = group;
        let mut current_item = Item { contents: Vec::new() };
        let mut label_id: i32 = -1;
        let mut is_first_field = true;

        loop {
            let token = iwa.read();
            match token.token_type {
                TokenType::Boi => {
                    label_id = -1;
                    is_first_field = true;
                    current_item.contents.clear();
                }
                TokenType::Item => {
                    if is_first_field {
                        is_first_field = false;
                        if token.attr.starts_with('@') {
                            if token.attr == "@weight" {
                                if let Ok(w) = token.value.parse::<f64>() {
                                    inst.weight = w;
                                }
                            }
                            label_id = -2;
                        } else {
                            label_id = label_quark.get(&token.attr);
                        }
                    } else if label_id != -2 {
                        let aid = attr_quark.get(&token.attr);
                        let value = if token.value.is_empty() {
                            1.0
                        } else {
                            token.value.parse::<f64>().unwrap_or(1.0)
                        };
                        current_item.contents.push(Attribute { aid, value });
                    }
                }
                TokenType::Eoi => {
                    if label_id >= 0 {
                        inst.items.push(current_item.clone());
                        inst.labels.push(label_id);
                    }
                }
                TokenType::None | TokenType::Eof => {
                    if !inst.items.is_empty() {
                        instances.push(inst.clone());
                        inst = Instance::new();
                        inst.group = group;
                    }
                    if token.token_type == TokenType::Eof {
                        break;
                    }
                }
            }
        }
    }

    let num_labels = label_quark.num();
    let num_attrs = attr_quark.num();

    let min_freq = get_float_param(&args.params, "feature.minfreq", 0.0);
    let possible_states = get_int_param(&args.params, "feature.possible_states", 0) != 0;
    let possible_transitions = get_int_param(&args.params, "feature.possible_transitions", 0) != 0;

    let mut encoder = Crf1dEncoder::new(
        &instances, num_labels, num_attrs, min_freq,
        possible_states, possible_transitions,
    );

    (log)(&format!("Feature generation\ntype: CRF1d\nfeature.minfreq: {:.6}\nfeature.possible_states: {}\nfeature.possible_transitions: {}\n",
        min_freq, possible_states as i32, possible_transitions as i32));
    (log)(&format!("Number of features: {}\n\n", encoder.num_features));

    let label_strings: Vec<String> = (0..num_labels)
        .map(|i| label_quark.to_string(i as i32).unwrap().to_string())
        .collect();
    let attr_strings: Vec<String> = (0..num_attrs)
        .map(|i| attr_quark.to_string(i as i32).unwrap().to_string())
        .collect();

    // Train
    let weights = match algorithm {
        "lbfgs" => {
            let c1 = get_float_param(&args.params, "c1", 0.0);
            let c2 = get_float_param(&args.params, "c2", 1.0);
            let max_iter = get_int_param(&args.params, "max_iterations", i32::MAX);
            let num_mem = get_int_param(&args.params, "num_memories", 6);
            let epsilon = get_float_param(&args.params, "epsilon", 1e-5);
            let period = get_int_param(&args.params, "period", 10);
            let delta = get_float_param(&args.params, "delta", 1e-5);
            let ls = get_str_param(&args.params, "linesearch", "MoreThuente");
            let max_ls = get_int_param(&args.params, "max_linesearch", 20);

            (log)(&format!("L-BFGS optimization\nc1: {:.6}\nc2: {:.6}\nnum_memories: {}\nmax_iterations: {}\nepsilon: {:.6}\nstop: {}\ndelta: {:.6}\nlinesearch: {}\nlinesearch.max_iterations: {}\n\n",
                c1, c2, num_mem, max_iter, epsilon, period, delta, ls, max_ls));

            train::lbfgs::train_lbfgs(
                &mut encoder, &instances,
                c1, c2, max_iter, num_mem, epsilon, period, delta, &ls, max_ls,
                &mut log,
            )
        }
        "l2sgd" => {
            let c2 = get_float_param(&args.params, "c2", 1.0);
            let max_iter = get_int_param(&args.params, "max_iterations", 1000);
            let period = get_int_param(&args.params, "period", 10);
            let delta = get_float_param(&args.params, "delta", 1e-6);
            train::l2sgd::train_l2sgd(
                &mut encoder, &instances, c2, max_iter, period, delta, &mut log,
            )
        }
        "averaged-perceptron" => {
            let max_iter = get_int_param(&args.params, "max_iterations", 100);
            let epsilon = get_float_param(&args.params, "epsilon", 0.0);
            train::averaged_perceptron::train_averaged_perceptron(
                &mut encoder, &mut instances.clone(), max_iter, epsilon, &mut log,
            )
        }
        "passive-aggressive" => {
            let pa_type = get_int_param(&args.params, "type", 1);
            let c = get_float_param(&args.params, "c", 1.0);
            let error_sensitive = get_int_param(&args.params, "error_sensitive", 1) != 0;
            let averaging = get_int_param(&args.params, "averaging", 1) != 0;
            let max_iter = get_int_param(&args.params, "max_iterations", 100);
            let epsilon = get_float_param(&args.params, "epsilon", 0.0);
            train::passive_aggressive::train_passive_aggressive(
                &mut encoder, &mut instances.clone(),
                pa_type, c, error_sensitive, averaging, max_iter, epsilon, &mut log,
            )
        }
        "arow" => {
            let variance = get_float_param(&args.params, "variance", 1.0);
            let gamma = get_float_param(&args.params, "gamma", 1.0);
            let max_iter = get_int_param(&args.params, "max_iterations", 100);
            let epsilon = get_float_param(&args.params, "epsilon", 0.0);
            train::arow::train_arow(
                &mut encoder, &mut instances.clone(),
                variance, gamma, max_iter, epsilon, &mut log,
            )
        }
        _ => return Err(format!("Unknown algorithm: {}", algorithm).into()),
    };

    if !args.model_path.is_empty() {
        let model_bytes = encoder.save_model(&weights, &label_strings, &attr_strings);
        std::fs::write(&args.model_path, &model_bytes)?;
    }

    Ok(())
}

fn get_float_param(params: &[(String, String)], name: &str, default: f64) -> f64 {
    params.iter().find(|(k, _)| k == name).and_then(|(_, v)| v.parse().ok()).unwrap_or(default)
}

fn get_int_param(params: &[(String, String)], name: &str, default: i32) -> i32 {
    params.iter().find(|(k, _)| k == name).and_then(|(_, v)| v.parse().ok()).unwrap_or(default)
}

fn get_str_param(params: &[(String, String)], name: &str, default: &str) -> String {
    params.iter().find(|(k, _)| k == name).map(|(_, v)| v.clone()).unwrap_or(default.to_string())
}
