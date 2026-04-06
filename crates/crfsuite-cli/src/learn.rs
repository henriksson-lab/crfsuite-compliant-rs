use std::fs::File;
use std::io::{self, BufReader, Write};

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
    // Map short algorithm names to full names
    let algorithm = match args.algorithm.as_str() {
        "ap" | "averaged-perceptron" => "averaged-perceptron".to_string(),
        "pa" | "passive-aggressive" => "passive-aggressive".to_string(),
        other => other.to_string(),
    };

    #[cfg(feature = "pure-rust-train")]
    return run_learn_pure_rust(args, &algorithm);

    #[cfg(all(not(feature = "pure-rust-train"), feature = "ffi"))]
    return run_learn_ffi(args, &algorithm);

    #[cfg(all(not(feature = "pure-rust-train"), not(feature = "ffi")))]
    return Err("Enable either 'pure-rust-train' or 'ffi' feature".into());
}

// ── Pure Rust training ──────────────────────────────────────────────────────

#[cfg(feature = "pure-rust-train")]
fn run_learn_pure_rust(args: LearnArgs, algorithm: &str) -> Result<(), Box<dyn std::error::Error>> {
    use crfsuite_core::crf1d::encode::Crf1dEncoder;
    use crfsuite_core::quark::Quark;
    use crfsuite_core::train;
    use crfsuite_core::types::{Attribute, Item, Instance};

    let mut fpo = io::stdout();
    let mut log = train::stdout_logger();

    // Read data into Rust types
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
    let total_items: usize = instances.iter().map(|i| i.num_items()).sum();

    // Get parameters
    let min_freq = args.params.iter()
        .find(|(k, _)| k == "feature.minfreq")
        .and_then(|(_, v)| v.parse::<f64>().ok())
        .unwrap_or(0.0);
    let possible_states = args.params.iter()
        .find(|(k, _)| k == "feature.possible_states")
        .and_then(|(_, v)| v.parse::<i32>().ok())
        .unwrap_or(0) != 0;
    let possible_transitions = args.params.iter()
        .find(|(k, _)| k == "feature.possible_transitions")
        .and_then(|(_, v)| v.parse::<i32>().ok())
        .unwrap_or(0) != 0;

    // Initialize encoder
    let mut encoder = Crf1dEncoder::new(
        &instances, num_labels, num_attrs, min_freq,
        possible_states, possible_transitions,
    );

    (log)(&format!("Feature generation\ntype: CRF1d\nfeature.minfreq: {:.6}\nfeature.possible_states: {}\nfeature.possible_transitions: {}\n",
        min_freq, possible_states as i32, possible_transitions as i32));
    (log)(&format!("Number of features: {}\n\n", encoder.num_features));

    // Build label/attr string arrays for model saving
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

    // Save model
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

// ── FFI training (original) ─────────────────────────────────────────────────

#[cfg(all(not(feature = "pure-rust-train"), feature = "ffi"))]
fn run_learn_ffi(args: LearnArgs, algorithm: &str) -> Result<(), Box<dyn std::error::Error>> {
    use crfsuite_ffi::data::Data;
    use crfsuite_ffi::dictionary::Dictionary;
    use crfsuite_ffi::trainer::Trainer;
    use crate::reader::read_data;

    let mut fpo = io::stdout();

    let trainer = Trainer::new(&args.model_type, algorithm)?;

    if args.help_params {
        let params = trainer.params()?;
        let n = params.num();
        for i in 0..n {
            let name = params.name(i)?;
            if let Ok((ptype, help)) = params.help(&name) {
                writeln!(fpo, "{} ({}): {}", name, ptype, help)?;
            }
        }
        return Ok(());
    }

    {
        let params = trainer.params()?;
        for (name, value) in &args.params {
            params.set(name, value)?;
        }
    }

    trainer.set_stdout_logging();

    let mut attrs_ptr: *mut std::os::raw::c_void = std::ptr::null_mut();
    let mut labels_ptr: *mut std::os::raw::c_void = std::ptr::null_mut();

    let ret = unsafe {
        crfsuite_sys::crfsuite_create_instance(
            b"dictionary\0".as_ptr() as *const _,
            &mut attrs_ptr,
        )
    };
    if ret == 0 || attrs_ptr.is_null() {
        return Err("Failed to create attribute dictionary".into());
    }

    let ret = unsafe {
        crfsuite_sys::crfsuite_create_instance(
            b"dictionary\0".as_ptr() as *const _,
            &mut labels_ptr,
        )
    };
    if ret == 0 || labels_ptr.is_null() {
        return Err("Failed to create label dictionary".into());
    }

    let attrs = unsafe { Dictionary::from_raw(attrs_ptr as *mut crfsuite_sys::crfsuite_dictionary_t)? };
    let labels = unsafe { Dictionary::from_raw(labels_ptr as *mut crfsuite_sys::crfsuite_dictionary_t)? };

    let mut data = Data::new();
    data.set_dictionaries(attrs.ptr.as_ptr(), labels.ptr.as_ptr());

    let input_files = if args.input_files.is_empty() {
        vec!["-".to_string()]
    } else {
        args.input_files.clone()
    };

    let groups = input_files.len() as i32;
    for (i, filename) in input_files.iter().enumerate() {
        let group = if args.split > 0 { 0 } else { i as i32 };
        writeln!(fpo, "[{}] {}", i + 1, filename)?;

        let n = if filename == "-" {
            let stdin = io::stdin();
            read_data(BufReader::new(stdin.lock()), &mut data, &attrs, &labels, group)?
        } else {
            let f = File::open(filename)?;
            read_data(BufReader::new(f), &mut data, &attrs, &labels, group)?
        };

        writeln!(fpo, "Number of instances: {}", n)?;
    }

    writeln!(fpo, "Statistics the data set(s)")?;
    writeln!(fpo, "Number of data sets (groups): {}", groups)?;
    writeln!(fpo, "Number of instances: {}", data.num_instances())?;
    let total_items = unsafe { crfsuite_sys::crfsuite_data_totalitems(data.as_ptr() as *mut _) };
    writeln!(fpo, "Number of items: {}", total_items)?;
    writeln!(fpo, "Number of attributes: {}", attrs.num())?;
    writeln!(fpo, "Number of labels: {}", labels.num())?;
    writeln!(fpo)?;

    if args.cross_validate {
        let num_groups = if args.split > 0 { args.split } else { groups };
        for i in 0..num_groups {
            writeln!(fpo, "===== Cross validation ({}/{}) =====", i + 1, num_groups)?;
            trainer.train(data.as_ptr(), "", i)?;
        }
    } else {
        trainer.train(data.as_ptr(), &args.model_path, args.holdout)?;
    }

    Ok(())
}
