mod cli_dump;
mod iwa;
mod learn;
mod tag;

use clap::{Parser, Subcommand};

#[derive(Parser)]
#[command(
    name = "crfsuite-rs",
    version = "0.12.2",
    about = "CRFsuite - Conditional Random Fields"
)]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Train a CRF model
    Learn {
        /// Model type (default: 1d)
        #[arg(short = 't', long = "type", default_value = "1d")]
        model_type: String,

        /// Training algorithm: lbfgs, l2sgd, ap, pa, arow
        #[arg(short = 'a', long = "algorithm", default_value = "lbfgs")]
        algorithm: String,

        /// Model file path
        #[arg(short = 'm', long = "model", default_value = "")]
        model_path: String,

        /// Write the training log to a generated file
        #[arg(short = 'l', long = "log-to-file")]
        log_to_file: bool,

        /// Base name for a generated log file
        #[arg(short = 'L', long = "logbase", default_value = "log.crfsuite")]
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
        #[arg(short = 'H', long = "help-parameters", alias = "help-params")]
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
        model: String,
    },
}

fn main() {
    let cli = Cli::parse();

    let result = match cli.command {
        Commands::Learn {
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
        } => {
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
        Commands::Tag {
            model_path,
            test,
            reference,
            probability,
            marginal,
            marginal_all,
            quiet,
            file,
        } => tag::run_tag(tag::TagArgs {
            model_path,
            test,
            reference,
            probability,
            marginal,
            marginal_all,
            quiet,
            input_file: file,
        }),
        Commands::Dump { model } => cli_dump::run_dump(cli_dump::DumpArgs { model_path: model }),
    };

    if let Err(e) = result {
        eprintln!("ERROR: {}", e);
        std::process::exit(1);
    }
}
