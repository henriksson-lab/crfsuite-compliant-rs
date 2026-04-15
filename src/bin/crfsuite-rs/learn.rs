use std::fs::File;
use std::io::{self, Write};

use crfsuite_compliant_rs::train;

use crate::cli_meta::DEFAULT_MODEL_TYPE;
use crate::learn_data::load_training_data;
use crate::learn_params::{
    is_known_algorithm, log_help_params, normalize_algorithm, validate_params,
};
use crate::learn_train::{train_once, TrainOnce};

#[allow(dead_code)]
pub struct LearnArgs {
    pub model_type: String,
    pub algorithm: String,
    pub model_path: String,
    pub log_to_file: bool,
    pub logbase: String,
    pub param_strings: Vec<String>,
    pub params: Vec<(String, String)>,
    pub split: i32,
    pub holdout: i32,
    pub cross_validate: bool,
    pub help_params: bool,
    pub input_files: Vec<String>,
}

pub fn run_learn(args: LearnArgs) -> Result<(), Box<dyn std::error::Error>> {
    if args.model_type != DEFAULT_MODEL_TYPE {
        return Err(format!("Unknown graphical model: {}", args.model_type).into());
    }

    let algorithm = normalize_algorithm(&args.algorithm);
    if !is_known_algorithm(algorithm) {
        return Err(format!("Unknown algorithm: {}", args.algorithm).into());
    }

    let mut log = create_logger(&args, algorithm)?;
    if args.help_params {
        log_help_params(algorithm, &mut log);
        return Ok(());
    }
    validate_params(algorithm, &args.params)?;

    let data = load_training_data(&args.input_files, args.split)?;

    if args.cross_validate {
        for holdout in 0..data.groups {
            (log)(&format!(
                "===== Cross validation ({}/{}) =====\n",
                holdout + 1,
                data.groups
            ));
            train_once(
                TrainOnce {
                    algorithm,
                    params: &args.params,
                    instances: &data.instances,
                    holdout,
                    num_labels: data.num_labels,
                    num_attrs: data.num_attrs,
                    label_strings: &data.label_strings,
                    attr_strings: &data.attr_strings,
                    model_path: "",
                },
                &mut log,
            )?;
            (log)("\n");
        }
    } else {
        train_once(
                TrainOnce {
                    algorithm,
                    params: &args.params,
                    instances: &data.instances,
                    holdout: args.holdout,
                    num_labels: data.num_labels,
                    num_attrs: data.num_attrs,
                    label_strings: &data.label_strings,
                    attr_strings: &data.attr_strings,
                    model_path: &args.model_path,
                },
            &mut log,
        )?;
    }

    Ok(())
}

fn create_logger(
    args: &LearnArgs,
    algorithm: &str,
) -> Result<train::LogFn, Box<dyn std::error::Error>> {
    if !args.log_to_file {
        return Ok(train::stdout_logger());
    }

    let mut path = format!("{}_{}", args.logbase, algorithm);
    for param in &args.param_strings {
        path.push('_');
        path.push_str(param);
    }

    let mut writer = io::BufWriter::new(File::create(path)?);
    Ok(Box::new(move |msg| {
        let _ = writer.write_all(msg.as_bytes());
        let _ = writer.flush();
    }))
}
