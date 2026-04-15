use std::fs::File;
use std::io::{self, BufReader, Write};

use crfsuite_compliant_rs::crf1d::encode::Crf1dEncoder;
use crfsuite_compliant_rs::quark::Quark;
use crfsuite_compliant_rs::train;
use crfsuite_compliant_rs::types::{Attribute, Instance, Item};

use crate::iwa::{atof, atoi, IwaReader, TokenType};

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

struct TrainOnce<'a> {
    args: &'a LearnArgs,
    algorithm: &'a str,
    instances: &'a [Instance],
    holdout: i32,
    num_labels: usize,
    num_attrs: usize,
    label_strings: &'a [String],
    attr_strings: &'a [String],
    model_path: &'a str,
}

pub fn run_learn(args: LearnArgs) -> Result<(), Box<dyn std::error::Error>> {
    if args.model_type != "1d" {
        return Err(format!("Unknown graphical model: {}", args.model_type).into());
    }

    let algorithm = match args.algorithm.as_str() {
        "ap" | "averaged-perceptron" => "averaged-perceptron",
        "pa" | "passive-aggressive" => "passive-aggressive",
        other => other,
    };
    if algorithm_param_specs(algorithm).is_empty() {
        return Err(format!("Unknown algorithm: {}", args.algorithm).into());
    }

    let mut log = create_logger(&args, algorithm)?;
    if args.help_params {
        log_help_params(algorithm, &mut log);
        return Ok(());
    }
    validate_params(algorithm, &args.params)?;

    // Read data
    let mut label_quark = Quark::new();
    let mut attr_quark = Quark::new();
    let mut instances: Vec<Instance> = Vec::new();

    let input_files = args.input_files.clone();

    for (fi, filename) in input_files.iter().enumerate() {
        let group = if args.split > 0 { 0 } else { fi as i32 };
        let input: Box<dyn io::BufRead> = if filename == "-" {
            Box::new(BufReader::new(io::stdin()))
        } else {
            let file = File::open(filename)
                .map_err(|_| format!("Failed to open the data set: {}", filename))?;
            Box::new(BufReader::new(file))
        };

        let mut iwa = IwaReader::new(input);
        let mut inst = Instance::new();
        inst.group = group;
        let mut current_item = Item {
            contents: Vec::new(),
        };
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
                                inst.weight = atof(&token.value);
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
                            atof(&token.value)
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

    if args.split > 0 {
        let n = instances.len();
        let mut rng_state = 12345u64;
        for i in 0..n {
            rng_state ^= rng_state << 13;
            rng_state ^= rng_state >> 7;
            rng_state ^= rng_state << 17;
            let j = ((rng_state >> 1) as usize) % n;
            instances.swap(i, j);
        }
        for (i, inst) in instances.iter_mut().enumerate() {
            inst.group = i as i32 % args.split;
        }
    }

    let num_labels = label_quark.num();
    let num_attrs = attr_quark.num();

    let label_strings: Vec<String> = (0..num_labels)
        .map(|i| label_quark.to_string(i as i32).unwrap().to_string())
        .collect();
    let attr_strings: Vec<String> = (0..num_attrs)
        .map(|i| attr_quark.to_string(i as i32).unwrap().to_string())
        .collect();

    let groups = if args.split > 0 {
        args.split
    } else {
        input_files.len() as i32
    };

    if args.cross_validate {
        for holdout in 0..groups {
            (log)(&format!(
                "===== Cross validation ({}/{}) =====\n",
                holdout + 1,
                groups
            ));
            train_once(
                TrainOnce {
                    args: &args,
                    algorithm,
                    instances: &instances,
                    holdout,
                    num_labels,
                    num_attrs,
                    label_strings: &label_strings,
                    attr_strings: &attr_strings,
                    model_path: "",
                },
                &mut log,
            )?;
            (log)("\n");
        }
    } else {
        train_once(
            TrainOnce {
                args: &args,
                algorithm,
                instances: &instances,
                holdout: args.holdout,
                num_labels,
                num_attrs,
                label_strings: &label_strings,
                attr_strings: &attr_strings,
                model_path: &args.model_path,
            },
            &mut log,
        )?;
    }

    Ok(())
}

fn train_once(
    config: TrainOnce<'_>,
    log: &mut train::LogFn,
) -> Result<(), Box<dyn std::error::Error>> {
    let train_instances: Vec<Instance> = config
        .instances
        .iter()
        .filter(|inst| inst.group != config.holdout)
        .cloned()
        .collect();

    let args = config.args;
    let min_freq = get_float_param(&args.params, "feature.minfreq", 0.0);
    let possible_states = get_int_param(&args.params, "feature.possible_states", 0) != 0;
    let possible_transitions = get_int_param(&args.params, "feature.possible_transitions", 0) != 0;

    let mut encoder = Crf1dEncoder::new(
        &train_instances,
        config.num_labels,
        config.num_attrs,
        min_freq,
        possible_states,
        possible_transitions,
    );

    (log)(&format!("Feature generation\ntype: CRF1d\nfeature.minfreq: {:.6}\nfeature.possible_states: {}\nfeature.possible_transitions: {}\n",
        min_freq, possible_states as i32, possible_transitions as i32));
    (log)(&format!("Number of features: {}\n\n", encoder.num_features));

    let weights = match config.algorithm {
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
        "l2sgd" => {
            let c2 = get_float_param(&args.params, "c2", 1.0);
            let max_iter = get_int_param(&args.params, "max_iterations", 1000);
            let period = get_int_param(&args.params, "period", 10);
            let delta = get_float_param(&args.params, "delta", 1e-6);
            let calibration_eta = get_float_param(&args.params, "calibration.eta", 0.1);
            train::l2sgd::train_l2sgd(
                &mut encoder,
                &train_instances,
                &train::l2sgd::L2SgdOptions {
                    c2,
                    max_iterations: max_iter,
                    period,
                    delta,
                    calibration_eta,
                },
                log,
            )
        }
        "averaged-perceptron" => {
            let max_iter = get_int_param(&args.params, "max_iterations", 100);
            let epsilon = get_float_param(&args.params, "epsilon", 0.0);
            train::averaged_perceptron::train_averaged_perceptron(
                &mut encoder,
                &mut train_instances.clone(),
                max_iter,
                epsilon,
                log,
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
                &mut encoder,
                &mut train_instances.clone(),
                pa_type,
                c,
                error_sensitive,
                averaging,
                max_iter,
                epsilon,
                log,
            )
        }
        "arow" => {
            let variance = get_float_param(&args.params, "variance", 1.0);
            let gamma = get_float_param(&args.params, "gamma", 1.0);
            let max_iter = get_int_param(&args.params, "max_iterations", 100);
            let epsilon = get_float_param(&args.params, "epsilon", 0.0);
            train::arow::train_arow(
                &mut encoder,
                &mut train_instances.clone(),
                variance,
                gamma,
                max_iter,
                epsilon,
                log,
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

fn get_float_param(params: &[(String, String)], name: &str, default: f64) -> f64 {
    params
        .iter()
        .find(|(k, _)| k == name)
        .map(|(_, v)| atof(v))
        .unwrap_or(default)
}

fn get_int_param(params: &[(String, String)], name: &str, default: i32) -> i32 {
    params
        .iter()
        .find(|(k, _)| k == name)
        .map(|(_, v)| atoi(v))
        .unwrap_or(default)
}

fn get_str_param(params: &[(String, String)], name: &str, default: &str) -> String {
    params
        .iter()
        .find(|(k, _)| k == name)
        .map(|(_, v)| v.clone())
        .unwrap_or(default.to_string())
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

fn validate_params(
    algorithm: &str,
    params: &[(String, String)],
) -> Result<(), Box<dyn std::error::Error>> {
    for (name, _) in params {
        if !is_known_param(algorithm, name) {
            return Err(format!("paraneter not found: {}", name).into());
        }
    }
    Ok(())
}

#[derive(Clone, Copy)]
enum ParamType {
    Int,
    Float,
    String,
}

struct ParamSpec {
    ty: ParamType,
    name: &'static str,
    default: &'static str,
    help: &'static str,
}

const FEATURE_PARAM_SPECS: &[ParamSpec] = &[
    ParamSpec {
        ty: ParamType::Float,
        name: "feature.minfreq",
        default: "0.000000",
        help: "The minimum frequency of features.",
    },
    ParamSpec {
        ty: ParamType::Int,
        name: "feature.possible_states",
        default: "0",
        help: "Force to generate possible state features.",
    },
    ParamSpec {
        ty: ParamType::Int,
        name: "feature.possible_transitions",
        default: "0",
        help: "Force to generate possible transition features.",
    },
];

const LBFGS_PARAM_SPECS: &[ParamSpec] = &[
    ParamSpec {
        ty: ParamType::Float,
        name: "c1",
        default: "0.000000",
        help: "Coefficient for L1 regularization.",
    },
    ParamSpec {
        ty: ParamType::Float,
        name: "c2",
        default: "1.000000",
        help: "Coefficient for L2 regularization.",
    },
    ParamSpec {
        ty: ParamType::Int,
        name: "max_iterations",
        default: "2147483647",
        help: "The maximum number of iterations for L-BFGS optimization.",
    },
    ParamSpec {
        ty: ParamType::Int,
        name: "num_memories",
        default: "6",
        help: "The number of limited memories for approximating the inverse hessian matrix.",
    },
    ParamSpec {
        ty: ParamType::Float,
        name: "epsilon",
        default: "0.000010",
        help: "Epsilon for testing the convergence of the objective.",
    },
    ParamSpec {
        ty: ParamType::Int,
        name: "period",
        default: "10",
        help: "The duration of iterations to test the stopping criterion.",
    },
    ParamSpec {
        ty: ParamType::Float,
        name: "delta",
        default: "0.000010",
        help: "The threshold for the stopping criterion; an L-BFGS iteration stops when the\nimprovement of the log likelihood over the last ${period} iterations is no\ngreater than this threshold.",
    },
    ParamSpec {
        ty: ParamType::String,
        name: "linesearch",
        default: "MoreThuente",
        help: "The line search algorithm used in L-BFGS updates:\n{   'MoreThuente': More and Thuente's method,\n    'Backtracking': Backtracking method with regular Wolfe condition,\n    'StrongBacktracking': Backtracking method with strong Wolfe condition\n}\n",
    },
    ParamSpec {
        ty: ParamType::Int,
        name: "max_linesearch",
        default: "20",
        help: "The maximum number of trials for the line search algorithm.",
    },
];

const L2SGD_PARAM_SPECS: &[ParamSpec] = &[
    ParamSpec {
        ty: ParamType::Float,
        name: "c2",
        default: "1.000000",
        help: "Coefficient for L2 regularization.",
    },
    ParamSpec {
        ty: ParamType::Int,
        name: "max_iterations",
        default: "1000",
        help: "The maximum number of iterations (epochs) for SGD optimization.",
    },
    ParamSpec {
        ty: ParamType::Int,
        name: "period",
        default: "10",
        help: "The duration of iterations to test the stopping criterion.",
    },
    ParamSpec {
        ty: ParamType::Float,
        name: "delta",
        default: "0.000001",
        help: "The threshold for the stopping criterion; an optimization process stops when\nthe improvement of the log likelihood over the last ${period} iterations is no\ngreater than this threshold.",
    },
    ParamSpec {
        ty: ParamType::Float,
        name: "calibration.eta",
        default: "0.100000",
        help: "The initial value of learning rate (eta) used for calibration.",
    },
    ParamSpec {
        ty: ParamType::Float,
        name: "calibration.rate",
        default: "2.000000",
        help: "The rate of increase/decrease of learning rate for calibration.",
    },
    ParamSpec {
        ty: ParamType::Int,
        name: "calibration.samples",
        default: "1000",
        help: "The number of instances used for calibration.",
    },
    ParamSpec {
        ty: ParamType::Int,
        name: "calibration.candidates",
        default: "10",
        help: "The number of candidates of learning rate.",
    },
    ParamSpec {
        ty: ParamType::Int,
        name: "calibration.max_trials",
        default: "20",
        help: "The maximum number of trials of learning rates for calibration.",
    },
];

const AVERAGED_PERCEPTRON_PARAM_SPECS: &[ParamSpec] = &[
    ParamSpec {
        ty: ParamType::Int,
        name: "max_iterations",
        default: "100",
        help: "The maximum number of iterations.",
    },
    ParamSpec {
        ty: ParamType::Float,
        name: "epsilon",
        default: "0.000000",
        help: "The stopping criterion (the ratio of incorrect label predictions).",
    },
];

const PASSIVE_AGGRESSIVE_PARAM_SPECS: &[ParamSpec] = &[
    ParamSpec {
        ty: ParamType::Int,
        name: "type",
        default: "1",
        help: "The strategy for updating feature weights: {\n    0: PA without slack variables,\n    1: PA type I,\n    2: PA type II\n}.\n",
    },
    ParamSpec {
        ty: ParamType::Float,
        name: "c",
        default: "1.000000",
        help: "The aggressiveness parameter.",
    },
    ParamSpec {
        ty: ParamType::Int,
        name: "error_sensitive",
        default: "1",
        help: "Consider the number of incorrect labels to the cost function.",
    },
    ParamSpec {
        ty: ParamType::Int,
        name: "averaging",
        default: "1",
        help: "Compute the average of feature weights (similarly to Averaged Perceptron).",
    },
    ParamSpec {
        ty: ParamType::Int,
        name: "max_iterations",
        default: "100",
        help: "The maximum number of iterations.",
    },
    ParamSpec {
        ty: ParamType::Float,
        name: "epsilon",
        default: "0.000000",
        help: "The stopping criterion (the mean loss).",
    },
];

const AROW_PARAM_SPECS: &[ParamSpec] = &[
    ParamSpec {
        ty: ParamType::Float,
        name: "variance",
        default: "1.000000",
        help: "The initial variance of every feature weight.",
    },
    ParamSpec {
        ty: ParamType::Float,
        name: "gamma",
        default: "1.000000",
        help: "Tradeoff parameter.",
    },
    ParamSpec {
        ty: ParamType::Int,
        name: "max_iterations",
        default: "100",
        help: "The maximum number of iterations.",
    },
    ParamSpec {
        ty: ParamType::Float,
        name: "epsilon",
        default: "0.000000",
        help: "The stopping criterion (the mean loss).",
    },
];

fn log_help_params(algorithm: &str, log: &mut train::LogFn) {
    (log)(&format!("PARAMETERS for {} (crf1d):\n", algorithm));
    (log)("\n");

    for spec in FEATURE_PARAM_SPECS
        .iter()
        .chain(algorithm_param_specs(algorithm).iter())
    {
        (log)(&format!(
            "{} {} = {};\n",
            spec.ty.as_str(),
            spec.name,
            spec.default
        ));
        (log)(spec.help);
        (log)("\n");
        (log)("\n");
    }
}

impl ParamType {
    fn as_str(self) -> &'static str {
        match self {
            ParamType::Int => "int",
            ParamType::Float => "float",
            ParamType::String => "string",
        }
    }
}

fn algorithm_param_specs(algorithm: &str) -> &'static [ParamSpec] {
    match algorithm {
        "lbfgs" => LBFGS_PARAM_SPECS,
        "l2sgd" => L2SGD_PARAM_SPECS,
        "averaged-perceptron" => AVERAGED_PERCEPTRON_PARAM_SPECS,
        "passive-aggressive" => PASSIVE_AGGRESSIVE_PARAM_SPECS,
        "arow" => AROW_PARAM_SPECS,
        _ => &[],
    }
}

fn is_known_param(algorithm: &str, name: &str) -> bool {
    if FEATURE_PARAM_SPECS.iter().any(|spec| spec.name == name) {
        return true;
    }

    algorithm_param_specs(algorithm)
        .iter()
        .any(|spec| spec.name == name)
}
