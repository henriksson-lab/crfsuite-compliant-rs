mod cli_meta;
mod cli_dump;
mod iwa;
mod learn;
mod learn_data;
mod learn_params;
mod learn_train;
mod tag;

use clap::{Parser, Subcommand};

#[derive(Parser)]
#[command(
    name = cli_meta::APP_NAME,
    version = cli_meta::APP_VERSION,
    about = cli_meta::APP_ABOUT,
    disable_version_flag = true,
    disable_help_subcommand = true
)]
struct Cli {
    #[command(subcommand)]
    command: Option<Commands>,
}

#[derive(Subcommand)]
enum Commands {
    /// Train a CRF model
    Learn {
        /// Model type (default: 1d)
        #[arg(short = 't', long = "type", default_value = cli_meta::DEFAULT_MODEL_TYPE)]
        model_type: String,

        /// Training algorithm: lbfgs, l2sgd, ap, pa, arow
        #[arg(short = 'a', long = "algorithm", default_value = cli_meta::DEFAULT_LEARN_ALGORITHM)]
        algorithm: String,

        /// Model file path
        #[arg(short = 'm', long = "model", default_value = "")]
        model_path: String,

        /// Write the training log to a generated file
        #[arg(short = 'l', long = "log-to-file")]
        log_to_file: bool,

        /// Base name for a generated log file
        #[arg(short = 'L', long = "logbase", default_value = cli_meta::DEFAULT_LOGBASE)]
        logbase: String,

        /// Set algorithm parameters (NAME=VALUE)
        #[arg(short = 'p', long = "set", num_args = 1)]
        params: Vec<String>,

        /// Split instances into N groups
        #[arg(short = 'g', long = "split", default_value = "0")]
        split: String,

        /// Use M-th group for holdout evaluation
        #[arg(short = 'e', long = "holdout")]
        holdout: Option<String>,

        /// N-fold cross validation
        #[arg(short = 'x', long = "cross-validate")]
        cross_validate: bool,

        /// Show algorithm-specific parameters
        #[arg(short = 'H', long = cli_meta::HELP_PARAMS_LONG, alias = cli_meta::HELP_PARAMS_ALIAS)]
        help_params: bool,

        /// Input data files
        files: Vec<String>,
    },

    /// Tag sequences using a trained model
    Tag {
        /// Model file path
        #[arg(short = 'm', long = "model", default_value = "")]
        model_path: String,

        /// Evaluate performance against gold labels
        #[arg(short = 't', long = "test")]
        test: bool,

        /// Output reference labels alongside predictions
        #[arg(short = 'r', long = "reference")]
        reference: bool,

        /// Output sequence probability
        #[arg(short = 'p', long = "probability")]
        probability: bool,

        /// Output marginal probability of predicted label
        #[arg(short = 'i', long = "marginal")]
        marginal: bool,

        /// Output marginal probabilities for all labels
        #[arg(short = 'l', long = "marginal-all")]
        marginal_all: bool,

        /// Suppress tagging output
        #[arg(short = 'q', long = "quiet")]
        quiet: bool,

        /// Input data file
        file: Option<String>,
    },

    /// Dump a model in human-readable format
        Dump {
            /// Model file path
            model: Option<String>,

            #[arg(hide = true)]
            extra: Vec<String>,
        },

    #[command(external_subcommand)]
    External(Vec<String>),
}

fn main() {
    handle_c_top_level_options();
    let cli = Cli::parse();

    let result = match cli.command {
        None => Err(cli_meta::NO_COMMAND_ERROR.into()),
        Some(Commands::Learn {
            model_type,
            algorithm,
            model_path,
            log_to_file,
            logbase,
            params,
            split,
            holdout,
            cross_validate,
            help_params,
            files,
        }) => {
            let parsed_params: Vec<(String, String)> = params
                .iter()
                .filter_map(|p| {
                    let mut parts = p.splitn(2, '=');
                    let name = parts.next()?.to_string();
                    let value = parts.next().unwrap_or("").to_string();
                    Some((name, value))
                })
                .collect();

            learn::run_learn(learn::LearnArgs {
                model_type,
                algorithm,
                model_path,
                log_to_file,
                logbase,
                param_strings: params,
                params: parsed_params,
                split: iwa::atoi(&split),
                holdout: holdout
                    .as_deref()
                    .map(|value| iwa::atoi(value) - 1)
                    .unwrap_or(-1),
                cross_validate,
                help_params,
                input_files: files,
            })
        }
        Some(Commands::Tag {
            model_path,
            test,
            reference,
            probability,
            marginal,
            marginal_all,
            quiet,
            file,
        }) => tag::run_tag(tag::TagArgs {
            model_path,
            test,
            reference,
            probability,
            marginal,
            marginal_all,
            quiet,
            input_file: file,
        }),
        Some(Commands::Dump { model, extra: _ }) => {
            cli_dump::run_dump(cli_dump::DumpArgs { model_path: model })
        }
        Some(Commands::External(args)) => {
            Err(format!("Unrecognized command ({}) specified.", args[0]).into())
        }
    };

    if let Err(e) = result {
        eprintln!("ERROR: {}", e);
        std::process::exit(1);
    }
}

fn handle_c_top_level_options() {
    let args: Vec<String> = std::env::args().collect();
    let program = args
        .first()
        .cloned()
        .unwrap_or_else(|| cli_meta::APP_NAME.to_string());
    let Some(first) = args.get(1) else {
        return;
    };

    if first == "-h" || first == "--help" {
        print!("{}", cli_meta::top_level_help(&program));
        std::process::exit(0);
    }

    if let Some(help) = c_subcommand_help(&program, &args) {
        print!("{help}");
        std::process::exit(0);
    }

    if args.get(1).is_some_and(|command| command == "dump")
        && args.get(2).is_some_and(|option| option.starts_with('-'))
    {
        eprintln!("Unrecognized option {}", args[2]);
        std::process::exit(1);
    }

    if first.starts_with('-') {
        eprintln!("Unrecognized option {first}");
        std::process::exit(1);
    }
}

fn c_subcommand_help(program: &str, args: &[String]) -> Option<String> {
    if args.len() != 3 || (args[2] != "-h" && args[2] != "--help") {
        return None;
    }
    match args[1].as_str() {
        "learn" => Some(cli_meta::learn_help(program)),
        "tag" => Some(cli_meta::tag_help(program)),
        "dump" => Some(cli_meta::dump_help(program)),
        _ => None,
    }
}
