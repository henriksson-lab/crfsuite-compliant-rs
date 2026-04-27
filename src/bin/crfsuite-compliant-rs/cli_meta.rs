pub const APP_NAME: &str = "crfsuite-compliant-rs";
pub const APP_VERSION: &str = "0.12.2";
pub const APP_ABOUT: &str = "CRFsuite - Conditional Random Fields";

pub const DEFAULT_MODEL_TYPE: &str = "1d";
pub const DEFAULT_LEARN_ALGORITHM: &str = crate::learn_params::ALGO_LBFGS;
pub const DEFAULT_LOGBASE: &str = "log.crfsuite";

pub const NO_COMMAND_ERROR: &str = "No command specified. See help (-h) for the usage.";
pub const HELP_PARAMS_LONG: &str = "help-parameters";
pub const HELP_PARAMS_ALIAS: &str = "help-params";

pub fn top_level_help(program: &str) -> String {
    format!(
        concat!(
            "CRFSuite 0.12.2  Copyright (c) 2007-2013 Naoaki Okazaki\n\n",
            "USAGE: {program} <COMMAND> [OPTIONS]\n",
            "    COMMAND     Command name to specify the processing\n",
            "    OPTIONS     Arguments for the command (optional; command-specific)\n\n",
            "COMMAND:\n",
            "    learn       Obtain a model from a training set of instances\n",
            "    tag         Assign suitable labels to given instances by using a model\n",
            "    dump        Output a model in a plain-text format\n\n",
            "For the usage of each command, specify -h option in the command argument.\n"
        ),
        program = program
    )
}

pub fn learn_help(program: &str) -> String {
    format!(
        concat!(
            "CRFSuite 0.12.2  Copyright (c) 2007-2013 Naoaki Okazaki\n\n",
            "USAGE: {program} learn [OPTIONS] [DATA1] [DATA2] ...\n",
            "Trains a model using training data set(s).\n\n",
            "  DATA    file(s) corresponding to data set(s) for training; if multiple N files\n",
            "          are specified, this utility assigns a group number (1...N) to the\n",
            "          instances in each file; if a file name is '-', the utility reads a\n",
            "          data set from STDIN\n\n",
            "OPTIONS:\n",
            "  -t, --type=TYPE       specify a graphical model (DEFAULT='1d'):\n",
            "                        (this option is reserved for the future use)\n",
            "      1d                    1st-order Markov CRF with state and transition\n",
            "                            features; transition features are not conditioned\n",
            "                            on observations\n",
            "  -a, --algorithm=NAME  specify a training algorithm (DEFAULT='lbfgs')\n",
            "      lbfgs                 L-BFGS with L1/L2 regularization\n",
            "      l2sgd                 SGD with L2-regularization\n",
            "      ap                    Averaged Perceptron\n",
            "      pa                    Passive Aggressive\n",
            "      arow                  Adaptive Regularization of Weights (AROW)\n",
            "  -p, --set=NAME=VALUE  set the algorithm-specific parameter NAME to VALUE;\n",
            "                        use '-H' or '--help-parameters' with the algorithm name\n",
            "                        specified by '-a' or '--algorithm' and the graphical\n",
            "                        model specified by '-t' or '--type' to see the list of\n",
            "                        algorithm-specific parameters\n",
            "  -m, --model=FILE      store the model to FILE (DEFAULT=''); if the value is\n",
            "                        empty, this utility does not store the model\n",
            "  -g, --split=N         split the instances into N groups; this option is\n",
            "                        useful for holdout evaluation and cross validation\n",
            "  -e, --holdout=M       use the M-th data for holdout evaluation and the rest\n",
            "                        for training\n",
            "  -x, --cross-validate  repeat holdout evaluations for #i in {{1, ..., N}} groups\n",
            "                        (N-fold cross validation)\n",
            "  -l, --log-to-file     write the training log to a file instead of to STDOUT;\n",
            "                        The filename is determined automatically by the training\n",
            "                        algorithm, parameters, and source files\n",
            "  -L, --logbase=BASE    set the base name for a log file (used with -l option)\n",
            "  -h, --help            show the usage of this command and exit\n",
            "  -H, --help-parameters show the help message of algorithm-specific parameters;\n",
            "                        specify an algorithm with '-a' or '--algorithm' option,\n",
            "                        and specify a graphical model with '-t' or '--type' option\n"
        ),
        program = program
    )
}

pub fn tag_help(program: &str) -> String {
    format!(
        concat!(
            "CRFSuite 0.12.2  Copyright (c) 2007-2013 Naoaki Okazaki\n\n",
            "USAGE: {program} tag [OPTIONS] [DATA]\n",
            "Assign suitable labels to the instances in the data set given by a file (DATA).\n",
            "If the argument DATA is omitted or '-', this utility reads a data from STDIN.\n",
            "Evaluate the performance of the model on labeled instances (with -t option).\n\n",
            "OPTIONS:\n",
            "    -m, --model=MODEL   Read a model from a file (MODEL)\n",
            "    -t, --test          Report the performance of the model on the data\n",
            "    -r, --reference     Output the reference labels in the input data\n",
            "    -p, --probability   Output the probability of the label sequences\n",
            "    -i, --marginal      Output the marginal probabilitiy of items for their predicted label\n",
            "    -l, --marginal-all  Output the marginal probabilities of items for all labels\n",
            "    -q, --quiet         Suppress tagging results (useful for test mode)\n",
            "    -h, --help          Show the usage of this command and exit\n"
        ),
        program = program
    )
}

pub fn dump_help(program: &str) -> String {
    format!(
        concat!(
            "USAGE: {program} dump [OPTIONS] <MODEL>\n",
            "Output the model stored in the file (MODEL) in a plain-text format\n\n",
            "OPTIONS:\n",
            "    -h, --help      Show the usage of this command and exit\n"
        ),
        program = program
    )
}
