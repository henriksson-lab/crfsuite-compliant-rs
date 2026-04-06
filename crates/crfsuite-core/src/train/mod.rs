pub mod lbfgs;
pub mod l2sgd;
pub mod averaged_perceptron;
pub mod passive_aggressive;
pub mod arow;

/// Logging callback type for training progress.
pub type LogFn = Box<dyn FnMut(&str)>;

/// Convenience: no-op logger.
pub fn noop_logger() -> LogFn {
    Box::new(|_| {})
}

/// Logger that prints to stdout.
pub fn stdout_logger() -> LogFn {
    Box::new(|msg| { print!("{}", msg); })
}
