use crfsuite_compliant_rs::train;

use crate::iwa::{atof, atoi};

pub const ALGO_LBFGS: &str = "lbfgs";
pub const ALGO_L2SGD: &str = "l2sgd";
pub const ALGO_AP_SHORT: &str = "ap";
pub const ALGO_AVERAGED_PERCEPTRON: &str = "averaged-perceptron";
pub const ALGO_PA_SHORT: &str = "pa";
pub const ALGO_PASSIVE_AGGRESSIVE: &str = "passive-aggressive";
pub const ALGO_AROW: &str = "arow";

pub fn normalize_algorithm(algorithm: &str) -> &str {
    match algorithm {
        ALGO_AP_SHORT | ALGO_AVERAGED_PERCEPTRON => ALGO_AVERAGED_PERCEPTRON,
        ALGO_PA_SHORT | ALGO_PASSIVE_AGGRESSIVE => ALGO_PASSIVE_AGGRESSIVE,
        other => other,
    }
}

pub fn is_known_algorithm(algorithm: &str) -> bool {
    !algorithm_param_specs(algorithm).is_empty()
}

pub fn get_float_param(algorithm: &str, params: &[(String, String)], name: &str) -> f64 {
    atof(param_value_or_default(algorithm, params, name))
}

pub fn get_int_param(algorithm: &str, params: &[(String, String)], name: &str) -> i32 {
    atoi(param_value_or_default(algorithm, params, name))
}

pub fn get_str_param(algorithm: &str, params: &[(String, String)], name: &str) -> String {
    param_value_or_default(algorithm, params, name).to_string()
}

pub fn validate_params(
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

pub fn log_help_params(algorithm: &str, log: &mut train::LogFn) {
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

fn param_value_or_default<'a>(
    algorithm: &str,
    params: &'a [(String, String)],
    name: &str,
) -> &'a str {
    params
        .iter()
        .find(|(k, _)| k == name)
        .map(|(_, v)| v.as_str())
        .unwrap_or_else(|| {
            find_param_spec(algorithm, name)
                .map(|spec| spec.default)
                .unwrap_or("")
        })
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
        ALGO_LBFGS => LBFGS_PARAM_SPECS,
        ALGO_L2SGD => L2SGD_PARAM_SPECS,
        ALGO_AVERAGED_PERCEPTRON => AVERAGED_PERCEPTRON_PARAM_SPECS,
        ALGO_PASSIVE_AGGRESSIVE => PASSIVE_AGGRESSIVE_PARAM_SPECS,
        ALGO_AROW => AROW_PARAM_SPECS,
        _ => &[],
    }
}

fn is_known_param(algorithm: &str, name: &str) -> bool {
    find_param_spec(algorithm, name).is_some()
}

fn find_param_spec(algorithm: &str, name: &str) -> Option<&'static ParamSpec> {
    FEATURE_PARAM_SPECS
        .iter()
        .chain(algorithm_param_specs(algorithm).iter())
        .find(|spec| spec.name == name)
}
