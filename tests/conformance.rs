//! Conformance tests: Rust CLI vs original C CLI.
//!
//! Focus: output *files* (model binaries) and tagging *results* (labels,
//! scores, marginals) must be identical.  CLI chrome (banners, progress bars,
//! timing lines) is intentionally ignored.
//!
//! Prerequisites:
//!   cargo build -p crfsuite-cli
//!   (cd crfsuite && make)          # builds the C binary

use std::io::Write;
use std::path::PathBuf;
use std::process::{Command, Output, Stdio};

// ── helpers ─────────────────────────────────────────────────────────────────

/// Skip test if C binary is not available (e.g. on crates.io).
macro_rules! skip_without_c {
    () => {
        if !c_bin().exists() {
            eprintln!("SKIP: C crfsuite binary not found");
            return;
        }
    };
}

fn project_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).to_path_buf()
}

fn rust_bin() -> PathBuf {
    if let Some(bin) = option_env!("CARGO_BIN_EXE_crfsuite-rs") {
        return PathBuf::from(bin);
    }
    let root = project_root();
    root.join("target/debug/crfsuite-rs")
}

fn c_bin() -> PathBuf {
    project_root().join("crfsuite/frontend/.libs/crfsuite")
}
fn c_lib_path() -> PathBuf {
    project_root().join("crfsuite/lib/crf/.libs")
}
fn test_data(name: &str) -> String {
    project_root()
        .join("test_data")
        .join(name)
        .to_str()
        .unwrap()
        .to_string()
}

/// Run the Rust CLI; return stdout.  Panics on non-zero exit.
fn run_rust(args: &[&str]) -> String {
    let bin = rust_bin();
    assert!(
        bin.exists(),
        "Rust binary missing – run `cargo build -p crfsuite-cli`"
    );
    let o = Command::new(&bin)
        .args(args)
        .output()
        .expect("spawn crfsuite-rs");
    assert!(
        o.status.success(),
        "crfsuite-rs {args:?} exit={}\n{}",
        o.status,
        String::from_utf8_lossy(&o.stderr)
    );
    String::from_utf8(o.stdout).unwrap()
}

fn run_rust_raw(args: &[&str]) -> std::process::Output {
    let bin = rust_bin();
    Command::new(&bin)
        .args(args)
        .output()
        .expect("spawn crfsuite-rs")
}

/// Run the C CLI; return stdout.
fn run_c_raw(args: &[&str]) -> std::process::Output {
    let bin = c_bin();
    Command::new(&bin)
        .args(args)
        .env("LD_LIBRARY_PATH", c_lib_path())
        .output()
        .expect("spawn C crfsuite")
}

fn run_c(args: &[&str]) -> String {
    let o = run_c_raw(args);
    assert!(
        o.status.success(),
        "C crfsuite {args:?} exit={}\n{}",
        o.status,
        String::from_utf8_lossy(&o.stderr)
    );
    String::from_utf8(o.stdout).unwrap()
}

fn run_with_stdin(bin: &PathBuf, args: &[&str], input: &[u8], ld_env: Option<&PathBuf>) -> String {
    let mut cmd = Command::new(bin);
    cmd.args(args).stdin(Stdio::piped()).stdout(Stdio::piped());
    if let Some(ld) = ld_env {
        cmd.env("LD_LIBRARY_PATH", ld);
    }
    let mut child = cmd.spawn().expect("spawn stdin command");
    child.stdin.as_mut().unwrap().write_all(input).unwrap();
    let o = child.wait_with_output().expect("wait stdin command");
    assert!(
        o.status.success(),
        "stdin command {args:?} exit={}\n{}",
        o.status,
        String::from_utf8_lossy(&o.stderr)
    );
    String::from_utf8(o.stdout).unwrap()
}

/// Train a model with given binary + args; return path to model file.
fn train_model(bin: &PathBuf, algo: &str, input: &str, out_model: &str, ld_env: Option<&PathBuf>) {
    let mut cmd = Command::new(bin);
    cmd.args(["learn", "-a", algo, "-m", out_model, input]);
    if let Some(ld) = ld_env {
        cmd.env("LD_LIBRARY_PATH", ld);
    }
    let o = cmd.output().expect("spawn train");
    assert!(
        o.status.success(),
        "train {algo} failed: {}",
        String::from_utf8_lossy(&o.stdout)
    );
}

fn assert_identical(a: &str, b: &str, ctx: &str) {
    if a == b {
        return;
    }
    let ab = a.as_bytes();
    let bb = b.as_bytes();
    for i in 0..ab.len().max(bb.len()) {
        if ab.get(i) != bb.get(i) {
            let line = a[..i.min(a.len())].matches('\n').count() + 1;
            panic!(
                "{ctx}: first diff at byte {i} (line ~{line})\n  C:    {:?}\n  Rust: {:?}",
                ab.get(i),
                bb.get(i)
            );
        }
    }
}

fn assert_tag_test_output_matches(c: &str, rs: &str, ctx: &str) {
    assert_identical(
        &normalize_tag_elapsed(c),
        &normalize_tag_elapsed(rs),
        ctx,
    );
}

fn normalize_tag_elapsed(output: &str) -> String {
    output
        .lines()
        .map(|line| {
            if line.starts_with("Elapsed time: ") {
                "Elapsed time: <normalized>"
            } else {
                line
            }
        })
        .collect::<Vec<_>>()
        .join("\n")
        + if output.ends_with('\n') { "\n" } else { "" }
}

fn assert_same_output(c: &Output, rs: &Output, ctx: &str) {
    assert_eq!(
        c.status.code(),
        rs.status.code(),
        "{ctx}: exit code mismatch"
    );
    let c_stdout = String::from_utf8(c.stdout.clone()).unwrap();
    let rs_stdout = String::from_utf8(rs.stdout.clone()).unwrap();
    assert_identical(
        strip_c_cli_chrome(&c_stdout),
        &rs_stdout,
        &format!("{ctx}: stdout"),
    );
    assert_identical(
        &String::from_utf8(c.stderr.clone()).unwrap(),
        &String::from_utf8(rs.stderr.clone()).unwrap(),
        &format!("{ctx}: stderr"),
    );
}

fn assert_same_failure(c: Output, rs: Output, ctx: &str) {
    assert!(!c.status.success(), "C should fail for {ctx}");
    assert!(!rs.status.success(), "Rust should fail for {ctx}");
    assert_same_output(&c, &rs, ctx);
}

fn c_stdout_without_banner(output: Output) -> String {
    let stdout = String::from_utf8(output.stdout).unwrap();
    strip_c_cli_chrome(&stdout).to_string()
}

fn strip_c_cli_chrome(stdout: &str) -> &str {
    // C `learn` prints banner/progress chrome before some semantic output and
    // even before some failures. Rust intentionally keeps those paths quiet.
    let mut rest = stdout;
    if rest.starts_with("CRFSuite ") {
        rest = rest.split_once("\n\n").map(|(_, tail)| tail).unwrap_or(rest);
    }
    loop {
        if rest.starts_with("Start time of the training: ") {
            rest = rest.split_once('\n').map(|(_, tail)| tail).unwrap_or("");
        } else if rest.starts_with('\n') {
            rest = &rest[1..];
        } else if let Some(tail) = rest.strip_prefix("Reading the data set(s)\n") {
            rest = tail;
        } else {
            return rest;
        }
    }
}

fn assert_same_cli_help(c: Output, rs: Output, ctx: &str) {
    assert!(c.status.success(), "C should print help for {ctx}");
    assert!(rs.status.success(), "Rust should print help for {ctx}");
    assert_eq!(c.status.code(), rs.status.code(), "{ctx}: exit code mismatch");
    let c_stdout = String::from_utf8(c.stdout).unwrap();
    let rs_stdout = String::from_utf8(rs.stdout).unwrap();
    let c_usage = c_bin().to_string_lossy().to_string();
    let rs_usage = rust_bin().to_string_lossy().to_string();
    assert_identical(
        &c_stdout.replace(&c_usage, "<BIN>"),
        &rs_stdout.replace(&rs_usage, "<BIN>"),
        &format!("{ctx}: stdout"),
    );
    assert_identical(
        &String::from_utf8(c.stderr).unwrap(),
        &String::from_utf8(rs.stderr).unwrap(),
        &format!("{ctx}: stderr"),
    );
}

fn train_iwa_with_params(
    bin: &PathBuf,
    input: &std::path::Path,
    model: &std::path::Path,
    params: &[&str],
    ld_env: Option<&PathBuf>,
) {
    let mut cmd = Command::new(bin);
    cmd.args(["learn", "-a", "lbfgs"]);
    for param in params {
        cmd.args(["-p", param]);
    }
    cmd.args(["-m", model.to_str().unwrap(), input.to_str().unwrap()]);
    if let Some(ld) = ld_env {
        cmd.env("LD_LIBRARY_PATH", ld);
    }
    let o = cmd.output().expect("spawn IWA train");
    assert!(
        o.status.success(),
        "IWA train failed: status={}\nstdout={}\nstderr={}",
        o.status,
        String::from_utf8_lossy(&o.stdout),
        String::from_utf8_lossy(&o.stderr)
    );
}

fn assert_iwa_dump_matches_c(data: &str, params: &[&str], ctx: &str) {
    let tmp = tempfile::tempdir().unwrap();
    let input = tmp.path().join("train.iwa");
    std::fs::write(&input, data).unwrap();
    assert_iwa_file_dump_matches_c(&input, params, ctx);
}

fn assert_iwa_file_dump_matches_c(input: &std::path::Path, params: &[&str], ctx: &str) {
    let tmp = tempfile::tempdir().unwrap();
    let c_model = tmp.path().join("c.bin");
    let rs_model = tmp.path().join("rs.bin");

    train_iwa_with_params(&c_bin(), input, &c_model, params, Some(&c_lib_path()));
    train_iwa_with_params(&rust_bin(), input, &rs_model, params, None);

    let c = run_c(&["dump", c_model.to_str().unwrap()]);
    let rs = run_rust(&["dump", rs_model.to_str().unwrap()]);
    assert_identical(&c, &rs, ctx);
}

// ── model equivalence ───────────────────────────────────────────────────────
//
// Train with Rust, verify that both C and Rust produce the same tag output.
// (Models may not be byte-identical due to floating point ordering, but must
// be functionally equivalent.)

fn model_equivalent(algo: &str) {
    skip_without_c!();
    let input = test_data("train.txt");
    let tmp = tempfile::tempdir().unwrap();
    let rm = tmp.path().join("rs.bin");
    let rms = rm.to_str().unwrap();

    train_model(&rust_bin(), algo, &input, rms, None);

    // Tag with both C and Rust, compare
    let c = run_c(&["tag", "-m", rms, "-p", "-l", &input]);
    let rs = run_rust(&["tag", "-m", rms, "-p", "-l", &input]);
    assert_identical(&c, &rs, &format!("model equiv {algo}: tag -p -l"));
}

#[test]
fn model_equiv_lbfgs() {
    model_equivalent("lbfgs");
}
#[test]
fn model_equiv_l2sgd() {
    model_equivalent("l2sgd");
}
#[test]
fn model_equiv_arow() {
    model_equivalent("arow");
}

// ── tagging output identity ─────────────────────────────────────────────────
//
// Given the *same* model, both CLIs must produce byte-identical tag output
// (labels, scores, marginals).

fn tag_identical(flags: &[&str], input_file: &str, ctx: &str) {
    skip_without_c!();
    let m = test_data("model_c.bin");
    let input = test_data(input_file);
    let mut args = vec!["tag", "-m", &m];
    args.extend_from_slice(flags);
    args.push(&input);

    let c_out = run_c(&args);
    let rs_out = run_rust(&args);
    assert_identical(&c_out, &rs_out, ctx);
}

// basic labels
#[test]
fn tag_labels_train() {
    tag_identical(&[], "train.txt", "tag train.txt");
}
#[test]
fn tag_labels_test() {
    tag_identical(&[], "test.txt", "tag test.txt");
}
#[test]
fn tag_labels_single() {
    tag_identical(&[], "single_item.txt", "tag single_item");
}
#[test]
fn tag_labels_multi() {
    tag_identical(&[], "multi_sequence.txt", "tag multi_seq");
}

// reference labels
#[test]
fn tag_ref_train() {
    tag_identical(&["-r"], "train.txt", "tag -r train");
}
#[test]
fn tag_ref_test() {
    tag_identical(&["-r"], "test.txt", "tag -r test");
}

// probability + score
#[test]
fn tag_prob_train() {
    tag_identical(&["-p"], "train.txt", "tag -p train");
}
#[test]
fn tag_prob_test() {
    tag_identical(&["-p"], "test.txt", "tag -p test");
}

// marginal of predicted label
#[test]
fn tag_marginal_train() {
    tag_identical(&["-i"], "train.txt", "tag -i train");
}
#[test]
fn tag_marginal_test() {
    tag_identical(&["-i"], "test.txt", "tag -i test");
}
#[test]
fn tag_marginal_single() {
    tag_identical(&["-i"], "single_item.txt", "tag -i single");
}

// all-label marginals
#[test]
fn tag_margall_train() {
    tag_identical(&["-l"], "train.txt", "tag -l train");
}
#[test]
fn tag_margall_test() {
    tag_identical(&["-l"], "test.txt", "tag -l test");
}
#[test]
fn tag_margall_single() {
    tag_identical(&["-l"], "single_item.txt", "tag -l single");
}
#[test]
fn tag_margall_multi() {
    tag_identical(&["-l"], "multi_sequence.txt", "tag -l multi");
}

// combined flags
#[test]
fn tag_rpi_train() {
    tag_identical(&["-r", "-p", "-i"], "train.txt", "tag -rpi train");
}
#[test]
fn tag_rpi_test() {
    tag_identical(&["-r", "-p", "-i"], "test.txt", "tag -rpi test");
}
#[test]
fn tag_rpl_test() {
    tag_identical(&["-r", "-p", "-l"], "test.txt", "tag -rpl test");
}
#[test]
fn tag_q_suppresses_prob_marginals() {
    tag_identical(&["-q", "-p", "-l"], "test.txt", "tag -qpl test");
}
#[test]
fn tag_q_suppresses_reference() {
    tag_identical(&["-q", "-r"], "test.txt", "tag -qr test");
}

#[test]
fn tag_test_output_matches_c_except_elapsed_time() {
    skip_without_c!();
    let model = test_data("model_c.bin");
    let input = test_data("test.txt");
    let cases: &[(&[&str], &str)] = &[
        (&["-t"], "tag -t"),
        (&["-q", "-t"], "tag -qt"),
        (&["-q", "-t", "-p", "-i", "-l", "-r"], "tag -qtpilr"),
        (
            &[
                "--test",
                "--reference",
                "--probability",
                "--marginal",
                "--marginal-all",
            ],
            "tag long aliases with -t",
        ),
    ];

    for (flags, ctx) in cases {
        let mut args = vec!["tag", "-m", &model];
        args.extend_from_slice(flags);
        args.push(&input);
        let c = run_c(&args);
        let rs = run_rust(&args);
        assert_tag_test_output_matches(&c, &rs, ctx);
    }
}

#[test]
fn tag_dash_reads_stdin_like_c() {
    skip_without_c!();
    let model = test_data("model_c.bin");
    let input = std::fs::read(test_data("test.txt")).unwrap();
    let args = ["tag", "-m", &model, "-p", "-l", "-"];

    let c = run_with_stdin(&c_bin(), &args, &input, Some(&c_lib_path()));
    let rs = run_with_stdin(&rust_bin(), &args, &input, None);
    assert_identical(&c, &rs, "tag - reads stdin");
}

#[test]
fn tag_unknown_reference_label_matches_c() {
    skip_without_c!();
    let tmp = tempfile::tempdir().unwrap();
    let input = tmp.path().join("unknown_label.txt");
    std::fs::write(&input, "UNKNOWN\tw[0]=the\tpos[0]=DT\n").unwrap();
    let input = input.to_str().unwrap();
    let model = test_data("model_c.bin");

    let c = run_c(&["tag", "-m", &model, "-r", input]);
    let rs = run_rust(&["tag", "-m", &model, "-r", input]);
    assert_identical(&c, &rs, "tag -r unknown reference label");
}

#[test]
fn tag_invalid_numeric_value_matches_c_atof() {
    skip_without_c!();
    let tmp = tempfile::tempdir().unwrap();
    let input = tmp.path().join("invalid_value.txt");
    std::fs::write(
        &input,
        "B-NP\tw[0]=the:2abc\tpos[0]=DT:not-a-number\nB-NP\tw[0]=the:0x1p2\tpos[0]=DT:0x1.8p+1\n",
    )
    .unwrap();
    let input = input.to_str().unwrap();
    let model = test_data("model_c.bin");

    let c = run_c(&["tag", "-m", &model, "-p", "-l", input]);
    let rs = run_rust(&["tag", "-m", &model, "-p", "-l", input]);
    assert_identical(&c, &rs, "tag invalid numeric value");
}

#[test]
fn learn_holdout_excludes_group_like_c() {
    skip_without_c!();
    let train = test_data("train.txt");
    let test = test_data("test.txt");
    let tmp = tempfile::tempdir().unwrap();
    let c_model = tmp.path().join("c_holdout.bin");
    let rs_model = tmp.path().join("rs_holdout.bin");
    let c_model = c_model.to_str().unwrap();
    let rs_model = rs_model.to_str().unwrap();

    let mut c_cmd = Command::new(c_bin());
    c_cmd
        .args([
            "learn", "-a", "lbfgs", "-e", "2", "-m", c_model, &train, &test,
        ])
        .env("LD_LIBRARY_PATH", c_lib_path());
    let c_status = c_cmd.output().expect("spawn C holdout train");
    assert!(c_status.status.success(), "C holdout train failed");

    let rs_status = Command::new(rust_bin())
        .args([
            "learn", "-a", "lbfgs", "-e", "2", "-m", rs_model, &train, &test,
        ])
        .output()
        .expect("spawn Rust holdout train");
    assert!(rs_status.status.success(), "Rust holdout train failed");

    let c = run_c(&["tag", "-m", c_model, "-p", "-l", &test]);
    let rs = run_rust(&["tag", "-m", rs_model, "-p", "-l", &test]);
    assert_identical(&c, &rs, "learn -e holdout model");
}

#[test]
fn learn_split_holdout_uses_c_rand_shuffle() {
    skip_without_c!();
    let tmp = tempfile::tempdir().unwrap();
    let input = tmp.path().join("split.iwa");
    std::fs::write(
        &input,
        "\
A\ta0\n\n\
B\ta1\n\n\
C\ta2\n\n\
A\ta3\n\n\
B\ta4\n\n\
C\ta5\n\n",
    )
    .unwrap();
    let c_model = tmp.path().join("c_split.bin");
    let rs_model = tmp.path().join("rs_split.bin");

    let c = Command::new(c_bin())
        .args([
            "learn",
            "-a",
            "lbfgs",
            "-g",
            "3",
            "-e",
            "2",
            "-p",
            "max_iterations=0",
            "-m",
            c_model.to_str().unwrap(),
            input.to_str().unwrap(),
        ])
        .env("LD_LIBRARY_PATH", c_lib_path())
        .output()
        .expect("spawn C split holdout train");
    assert!(c.status.success(), "C split holdout train failed");

    let rs = Command::new(rust_bin())
        .args([
            "learn",
            "-a",
            "lbfgs",
            "-g",
            "3",
            "-e",
            "2",
            "-p",
            "max_iterations=0",
            "-m",
            rs_model.to_str().unwrap(),
            input.to_str().unwrap(),
        ])
        .output()
        .expect("spawn Rust split holdout train");
    assert!(rs.status.success(), "Rust split holdout train failed");

    let c = run_c(&["dump", c_model.to_str().unwrap()]);
    let rs = run_rust(&["dump", rs_model.to_str().unwrap()]);
    assert_identical(&c, &rs, "learn -g/-e C rand shuffle");
}

#[test]
fn learn_split_cross_validation_uses_c_rand_shuffle() {
    skip_without_c!();
    let input = test_data("train.txt");

    let c = run_c_raw(&[
        "learn",
        "-a",
        "lbfgs",
        "-x",
        "-g",
        "3",
        "-p",
        "max_iterations=1",
        &input,
    ]);
    let rs = run_rust_raw(&[
        "learn",
        "-a",
        "lbfgs",
        "-x",
        "-g",
        "3",
        "-p",
        "max_iterations=1",
        &input,
    ]);
    assert!(c.status.success(), "C split cross validation failed");
    assert!(rs.status.success(), "Rust split cross validation failed");

    let c_stdout = String::from_utf8(c.stdout).unwrap();
    let rs_stdout = String::from_utf8(rs.stdout).unwrap();
    let c_lines: Vec<&str> = c_stdout
        .lines()
        .filter(|line| line.starts_with("Number of features: ") || line.starts_with("Loss: "))
        .collect();
    let rs_lines: Vec<&str> = rs_stdout
        .lines()
        .filter(|line| line.starts_with("Number of features: ") || line.starts_with("Loss: "))
        .collect();
    assert_eq!(c_lines, rs_lines, "learn -g/-x split metrics differ");
}

#[test]
fn learn_cross_validate_does_not_write_model_like_c() {
    skip_without_c!();
    let train = test_data("train.txt");
    let tmp = tempfile::tempdir().unwrap();
    let model = tmp.path().join("crossval.bin");
    let model = model.to_str().unwrap();

    let rs_status = Command::new(rust_bin())
        .args([
            "learn",
            "-a",
            "lbfgs",
            "-x",
            "-g",
            "2",
            "-p",
            "max_iterations=1",
            "-m",
            model,
            &train,
        ])
        .output()
        .expect("spawn Rust cross-validation train");
    assert!(
        rs_status.status.success(),
        "Rust cross-validation train failed"
    );
    assert!(
        !std::path::Path::new(model).exists(),
        "cross-validation should not write the -m model, matching C"
    );
}

#[test]
fn learn_unknown_parameter_fails_like_c() {
    skip_without_c!();
    let train = test_data("train.txt");

    let c = run_c_raw(&["learn", "-a", "lbfgs", "-p", "does_not_exist=1", &train]);
    let rs = run_rust_raw(&["learn", "-a", "lbfgs", "-p", "does_not_exist=1", &train]);
    assert!(
        String::from_utf8_lossy(&rs.stderr).contains("ERROR: paraneter not found: does_not_exist"),
        "Rust should preserve C's unknown-parameter diagnostic spelling"
    );
    assert_same_failure(c, rs, "learn unknown parameter");
}

#[test]
fn learn_error_messages_match_c() {
    skip_without_c!();

    assert_same_failure(
        run_c_raw(&["learn", "-t", "nope"]),
        run_rust_raw(&["learn", "-t", "nope"]),
        "learn unknown graphical model",
    );
    assert_same_failure(
        run_c_raw(&["learn", "-a", "nope"]),
        run_rust_raw(&["learn", "-a", "nope"]),
        "learn unknown algorithm",
    );
    assert_same_failure(
        run_c_raw(&["learn", "/no/such/file"]),
        run_rust_raw(&["learn", "/no/such/file"]),
        "learn missing input file",
    );
}

#[test]
fn learn_empty_training_data_matches_c_lbfgs_error_log() {
    skip_without_c!();
    let tmp = tempfile::tempdir().unwrap();
    let input = tmp.path().join("empty.iwa");
    std::fs::write(&input, "").unwrap();

    let c = run_c_raw(&["learn", input.to_str().unwrap()]);
    let rs = run_rust_raw(&["learn", input.to_str().unwrap()]);
    assert!(c.status.success(), "C empty training should exit successfully");
    assert!(
        rs.status.success(),
        "Rust empty training should exit successfully"
    );

    let c_stdout = String::from_utf8(c.stdout).unwrap();
    let rs_stdout = String::from_utf8(rs.stdout).unwrap();
    assert!(
        c_stdout.contains("L-BFGS terminated with error code (-1020)\n"),
        "C empty training should log LBFGS invalid-N code"
    );
    assert!(
        rs_stdout.contains("L-BFGS terminated with error code (-1020)\n"),
        "Rust empty training should log LBFGS invalid-N code"
    );
}

#[test]
fn tag_missing_input_error_matches_c() {
    skip_without_c!();
    let model = test_data("model_c.bin");
    assert_same_failure(
        run_c_raw(&["tag", "-m", &model, "/no/such/input"]),
        run_rust_raw(&["tag", "-m", &model, "/no/such/input"]),
        "tag missing input file",
    );
}

#[test]
fn tag_model_open_failures_match_c() {
    skip_without_c!();
    let tmp = tempfile::tempdir().unwrap();
    let bad_model = tmp.path().join("bad_model.bin");
    std::fs::write(&bad_model, "not a model").unwrap();
    let bad_model = bad_model.to_str().unwrap();

    assert_same_failure(
        run_c_raw(&["tag", "-m", "/no/such/model", &test_data("test.txt")]),
        run_rust_raw(&["tag", "-m", "/no/such/model", &test_data("test.txt")]),
        "tag missing model file",
    );
    assert_same_failure(
        run_c_raw(&["tag", "-m", bad_model, &test_data("test.txt")]),
        run_rust_raw(&["tag", "-m", bad_model, &test_data("test.txt")]),
        "tag malformed model file",
    );
    assert_same_failure(
        run_c_raw(&["tag", &test_data("test.txt")]),
        run_rust_raw(&["tag", &test_data("test.txt")]),
        "tag default empty model file",
    );
}

#[test]
fn dump_model_open_failures_match_c() {
    skip_without_c!();
    let model = test_data("model_c.bin");
    let tmp = tempfile::tempdir().unwrap();
    let bad_model = tmp.path().join("bad_model.bin");
    std::fs::write(&bad_model, "not a model").unwrap();
    let bad_model = bad_model.to_str().unwrap();

    assert_same_failure(
        run_c_raw(&["dump", "/no/such/model"]),
        run_rust_raw(&["dump", "/no/such/model"]),
        "dump missing model file",
    );
    assert_same_failure(
        run_c_raw(&["dump", bad_model]),
        run_rust_raw(&["dump", bad_model]),
        "dump malformed model file",
    );
    assert_same_failure(
        run_c_raw(&["dump"]),
        run_rust_raw(&["dump"]),
        "dump missing model argument",
    );
    assert_same_failure(
        run_c_raw(&["dump", "--bad", &model]),
        run_rust_raw(&["dump", "--bad", &model]),
        "dump unknown option",
    );
}

#[test]
fn dump_ignores_extra_args_like_c() {
    skip_without_c!();
    let model = test_data("model_c.bin");
    let c = run_c(&["dump", &model, "extra", "ignored"]);
    let rs = run_rust(&["dump", &model, "extra", "ignored"]);
    assert_identical(&c, &rs, "dump ignores extra args");
}

#[test]
fn dump_unusual_symbols_match_c() {
    skip_without_c!();
    assert_iwa_dump_matches_c(
        "L\\:one\ta\\:b\tc\\\\d\nL two\tattr with space\t@attr:2\tempty:\n\n",
        &["max_iterations=0"],
        "dump unusual labels and attributes",
    );
}

#[test]
fn top_level_command_errors_match_c() {
    skip_without_c!();
    assert_same_failure(
        run_c_raw(&[]),
        run_rust_raw(&[]),
        "top-level missing command",
    );
    assert_same_failure(
        run_c_raw(&["nope"]),
        run_rust_raw(&["nope"]),
        "top-level unknown command",
    );
    assert_same_failure(
        run_c_raw(&["help"]),
        run_rust_raw(&["help"]),
        "top-level help command is unknown",
    );
    assert_same_failure(
        run_c_raw(&["--version"]),
        run_rust_raw(&["--version"]),
        "top-level long version option is unknown",
    );
    assert_same_failure(
        run_c_raw(&["-V"]),
        run_rust_raw(&["-V"]),
        "top-level short version option is unknown",
    );
    assert_same_failure(
        run_c_raw(&["--bad"]),
        run_rust_raw(&["--bad"]),
        "top-level unknown option",
    );
}

#[test]
fn top_level_help_matches_c() {
    skip_without_c!();
    assert_same_cli_help(
        run_c_raw(&["--help"]),
        run_rust_raw(&["--help"]),
        "top-level --help",
    );
    assert_same_cli_help(run_c_raw(&["-h"]), run_rust_raw(&["-h"]), "top-level -h");
}

#[test]
fn subcommand_help_matches_c() {
    skip_without_c!();
    for command in ["learn", "tag", "dump"] {
        assert_same_cli_help(
            run_c_raw(&[command, "-h"]),
            run_rust_raw(&[command, "-h"]),
            &format!("{command} -h"),
        );
        assert_same_cli_help(
            run_c_raw(&[command, "--help"]),
            run_rust_raw(&[command, "--help"]),
            &format!("{command} --help"),
        );
    }
}

#[test]
fn learn_help_parameters_matches_c() {
    skip_without_c!();

    for algorithm in [
        "lbfgs",
        "l2sgd",
        "ap",
        "averaged-perceptron",
        "pa",
        "passive-aggressive",
        "arow",
    ] {
        let c_help = c_stdout_without_banner(run_c_raw(&[
            "learn",
            "-a",
            algorithm,
            "--help-params",
        ]));
        let rs_help = run_rust(&["learn", "-a", algorithm, "--help-params"]);
        assert_identical(
            &c_help,
            &rs_help,
            &format!("learn --help-params {algorithm}"),
        );
    }
}

#[test]
fn learn_parameter_parsing_matches_c_atoi_atof() {
    skip_without_c!();
    assert_iwa_dump_matches_c(
        "A\tx\nB\ty\n\n",
        &[
            "feature.minfreq=1junk",
            "feature.possible_states=2tail",
            "feature.possible_transitions=1tail",
            "max_iterations=0",
        ],
        "parameter prefix parsing",
    );
    assert_iwa_dump_matches_c(
        "A\tx\nB\ty\n\n",
        &[
            "feature.minfreq",
            "feature.possible_states",
            "feature.possible_transitions",
            "max_iterations=0",
        ],
        "parameter empty values become zero",
    );
}

#[test]
fn learn_string_parameter_is_copied_literally_like_c() {
    skip_without_c!();
    let tmp = tempfile::tempdir().unwrap();
    let input = tmp.path().join("train.iwa");
    std::fs::write(&input, "A\tx\n\n").unwrap();
    let c_model = tmp.path().join("c_string_param.bin");
    let rs_model = tmp.path().join("rs_string_param.bin");

    let c = Command::new(c_bin())
        .args([
            "learn",
            "-a",
            "lbfgs",
            "-p",
            "linesearch=BacktrackingJunk",
            "-p",
            "max_iterations=0",
            "-m",
            c_model.to_str().unwrap(),
            input.to_str().unwrap(),
        ])
        .env("LD_LIBRARY_PATH", c_lib_path())
        .output()
        .expect("spawn C learn string parameter");
    assert!(c.status.success(), "C learn string parameter failed");

    let rs = Command::new(rust_bin())
        .args([
            "learn",
            "-a",
            "lbfgs",
            "-p",
            "linesearch=BacktrackingJunk",
            "-p",
            "max_iterations=0",
            "-m",
            rs_model.to_str().unwrap(),
            input.to_str().unwrap(),
        ])
        .output()
        .expect("spawn Rust learn string parameter");
    assert!(rs.status.success(), "Rust learn string parameter failed");

    let c_out = String::from_utf8(c.stdout).unwrap();
    let rs_out = String::from_utf8(rs.stdout).unwrap();
    let c_line = c_out
        .lines()
        .find(|line| line.starts_with("linesearch: "))
        .unwrap();
    let rs_line = rs_out
        .lines()
        .find(|line| line.starts_with("linesearch: "))
        .unwrap();
    assert_identical(c_line, rs_line, "string parameter copied literally");
}

fn lbfgs_core_log(output: &Output) -> Vec<String> {
    String::from_utf8(output.stdout.clone())
        .unwrap()
        .lines()
        .filter(|line| {
            line.starts_with("L-BFGS optimization")
                || line.starts_with("c1: ")
                || line.starts_with("c2: ")
                || line.starts_with("num_memories: ")
                || line.starts_with("max_iterations: ")
                || line.starts_with("epsilon: ")
                || line.starts_with("stop: ")
                || line.starts_with("delta: ")
                || line.starts_with("linesearch: ")
                || line.starts_with("linesearch.max_iterations: ")
                || line.starts_with("***** Iteration #")
                || line.starts_with("Loss: ")
                || line.starts_with("Feature norm: ")
                || line.starts_with("Error norm: ")
                || line.starts_with("Active features: ")
                || line.starts_with("Line search trials: ")
                || line.starts_with("Line search step: ")
                || line.starts_with("L-BFGS resulted")
                || line.starts_with("L-BFGS terminated")
        })
        .map(str::to_string)
        .collect()
}

#[test]
fn learn_lbfgs_parameter_matrix_core_logs_match_c() {
    skip_without_c!();
    let train = test_data("train.txt");
    let cases: &[(&str, &[&str])] = &[
        (
            "baseline",
            &[
                "max_iterations=1",
                "num_memories=4",
                "epsilon=0.00001",
                "period=3",
                "delta=0.00001",
                "max_linesearch=20",
            ],
        ),
        (
            "backtracking",
            &[
                "max_iterations=1",
                "linesearch=Backtracking",
                "max_linesearch=20",
            ],
        ),
        (
            "strong-backtracking",
            &[
                "max_iterations=1",
                "linesearch=StrongBacktracking",
                "max_linesearch=20",
            ],
        ),
        (
            "l1-overrides-linesearch",
            &[
                "c1=0.1",
                "c2=0.5",
                "max_iterations=1",
                "linesearch=MoreThuente",
                "max_linesearch=20",
            ],
        ),
    ];

    for (name, params) in cases {
        let mut c_cmd = Command::new(c_bin());
        c_cmd.args(["learn", "-a", "lbfgs"]);
        for param in *params {
            c_cmd.args(["-p", param]);
        }
        let c = c_cmd
            .arg(&train)
            .env("LD_LIBRARY_PATH", c_lib_path())
            .output()
            .expect("spawn C lbfgs parameter matrix");
        assert!(
            c.status.success(),
            "C lbfgs parameter matrix failed for {name}: {}",
            String::from_utf8_lossy(&c.stdout)
        );

        let mut rs_cmd = Command::new(rust_bin());
        rs_cmd.args(["learn", "-a", "lbfgs"]);
        for param in *params {
            rs_cmd.args(["-p", param]);
        }
        let rs = rs_cmd
            .arg(&train)
            .output()
            .expect("spawn Rust lbfgs parameter matrix");
        assert!(
            rs.status.success(),
            "Rust lbfgs parameter matrix failed for {name}: {}",
            String::from_utf8_lossy(&rs.stdout)
        );

        assert_eq!(
            lbfgs_core_log(&c),
            lbfgs_core_log(&rs),
            "lbfgs parameter matrix {name}"
        );
    }
}

#[test]
fn learn_l2sgd_calibration_parameters_are_accepted_like_c() {
    skip_without_c!();
    let train = test_data("train.txt");

    let c = run_c_raw(&[
        "learn",
        "-a",
        "l2sgd",
        "-p",
        "calibration.samples=1",
        "-p",
        "calibration.max_trials=1",
        "-p",
        "calibration.candidates=1",
        "-p",
        "calibration.eta=0.1",
        "-p",
        "calibration.rate=2",
        "-p",
        "max_iterations=1",
        &train,
    ]);
    assert!(
        c.status.success(),
        "C should accept l2sgd calibration parameters"
    );

    let rs = Command::new(rust_bin())
        .args([
            "learn",
            "-a",
            "l2sgd",
            "-p",
            "calibration.samples=1",
            "-p",
            "calibration.max_trials=1",
            "-p",
            "calibration.candidates=1",
            "-p",
            "calibration.eta=0.1",
            "-p",
            "calibration.rate=2",
            "-p",
            "max_iterations=1",
            &train,
        ])
        .output()
        .expect("spawn Rust l2sgd calibration train");
    assert!(
        rs.status.success(),
        "Rust should accept l2sgd calibration parameters: {}",
        String::from_utf8_lossy(&rs.stderr)
    );
}

fn l2sgd_core_log(output: &Output) -> Vec<String> {
    String::from_utf8(output.stdout.clone())
        .unwrap()
        .lines()
        .filter(|line| {
            line.starts_with("Stochastic Gradient Descent")
                || line.starts_with("c2: ")
                || line.starts_with("max_iterations: ")
                || line.starts_with("period: ")
                || line.starts_with("delta: ")
                || line.starts_with("Calibrating the learning rate")
                || line.starts_with("calibration.")
                || line.starts_with("Initial loss: ")
                || line.starts_with("Trial #")
                || line.starts_with("Best learning rate")
                || line.starts_with("***** Epoch #")
                || line.starts_with("Loss: ")
                || line.starts_with("Improvement ratio: ")
                || line.starts_with("Feature L2-norm: ")
                || line.starts_with("Learning rate (eta): ")
                || line.starts_with("Total number of feature updates: ")
                || line.starts_with("SGD terminated")
        })
        .map(str::to_string)
        .collect()
}

#[test]
fn learn_l2sgd_calibration_variant_logs_match_c() {
    skip_without_c!();
    let train = test_data("train.txt");
    let cases: &[(&str, &[&str])] = &[
        (
            "samples",
            &[
                "calibration.samples=1",
                "calibration.candidates=2",
                "calibration.max_trials=4",
                "calibration.eta=0.1",
                "calibration.rate=2",
                "max_iterations=1",
            ],
        ),
        (
            "candidates",
            &[
                "calibration.samples=2",
                "calibration.candidates=1",
                "calibration.max_trials=4",
                "calibration.eta=0.1",
                "calibration.rate=2",
                "max_iterations=1",
            ],
        ),
        (
            "eta-rate-trials",
            &[
                "calibration.samples=2",
                "calibration.candidates=2",
                "calibration.max_trials=3",
                "calibration.eta=0.05",
                "calibration.rate=3",
                "max_iterations=1",
            ],
        ),
    ];

    for (name, params) in cases {
        let mut c_cmd = Command::new(c_bin());
        c_cmd.args(["learn", "-a", "l2sgd"]);
        for param in *params {
            c_cmd.args(["-p", param]);
        }
        let c = c_cmd
            .arg(&train)
            .env("LD_LIBRARY_PATH", c_lib_path())
            .output()
            .expect("spawn C l2sgd calibration variant");
        assert!(c.status.success(), "C l2sgd calibration variant failed");

        let mut rs_cmd = Command::new(rust_bin());
        rs_cmd.args(["learn", "-a", "l2sgd"]);
        for param in *params {
            rs_cmd.args(["-p", param]);
        }
        let rs = rs_cmd
            .arg(&train)
            .output()
            .expect("spawn Rust l2sgd calibration variant");
        assert!(rs.status.success(), "Rust l2sgd calibration variant failed");

        assert_eq!(
            l2sgd_core_log(&c),
            l2sgd_core_log(&rs),
            "l2sgd calibration variant {name}"
        );
    }
}

#[test]
fn learn_l2sgd_fixed_parameters_match_c_tagging() {
    skip_without_c!();
    let train = test_data("train.txt");
    let test = test_data("test.txt");
    let tmp = tempfile::tempdir().unwrap();
    let c_model = tmp.path().join("c_l2sgd_fixed.bin");
    let rs_model = tmp.path().join("rs_l2sgd_fixed.bin");
    let params = [
        "calibration.samples=2",
        "calibration.candidates=2",
        "calibration.max_trials=4",
        "calibration.eta=0.1",
        "calibration.rate=2",
        "max_iterations=2",
        "period=1",
        "delta=0",
    ];

    let mut c_cmd = Command::new(c_bin());
    c_cmd.args(["learn", "-a", "l2sgd"]);
    for param in params {
        c_cmd.args(["-p", param]);
    }
    let c = c_cmd
        .args(["-m", c_model.to_str().unwrap(), &train])
        .env("LD_LIBRARY_PATH", c_lib_path())
        .output()
        .expect("spawn C fixed l2sgd train");
    assert!(
        c.status.success(),
        "C fixed l2sgd failed: {}",
        String::from_utf8_lossy(&c.stdout)
    );

    let mut rs_cmd = Command::new(rust_bin());
    rs_cmd.args(["learn", "-a", "l2sgd"]);
    for param in params {
        rs_cmd.args(["-p", param]);
    }
    let rs = rs_cmd
        .args(["-m", rs_model.to_str().unwrap(), &train])
        .output()
        .expect("spawn Rust fixed l2sgd train");
    assert!(
        rs.status.success(),
        "Rust fixed l2sgd failed: {}",
        String::from_utf8_lossy(&rs.stdout)
    );

    let c = run_c(&["tag", "-m", c_model.to_str().unwrap(), "-p", "-l", &test]);
    let rs = run_rust(&["tag", "-m", rs_model.to_str().unwrap(), "-p", "-l", &test]);
    assert_identical(&c, &rs, "fixed-parameter l2sgd C/Rust tagging");
}

fn assert_fixed_online_trainer_matches_c(algorithm: &str, params: &[&str]) {
    let train = test_data("train.txt");
    let test = test_data("test.txt");
    let tmp = tempfile::tempdir().unwrap();
    let c_model = tmp.path().join("c_online.bin");
    let rs_model = tmp.path().join("rs_online.bin");

    let mut c_cmd = Command::new(c_bin());
    c_cmd.args(["learn", "-a", algorithm]);
    for param in params {
        c_cmd.args(["-p", param]);
    }
    let c = c_cmd
        .args(["-m", c_model.to_str().unwrap(), &train])
        .env("LD_LIBRARY_PATH", c_lib_path())
        .output()
        .expect("spawn C fixed online train");
    assert!(
        c.status.success(),
        "C fixed {algorithm} failed: {}",
        String::from_utf8_lossy(&c.stdout)
    );

    let mut rs_cmd = Command::new(rust_bin());
    rs_cmd.args(["learn", "-a", algorithm]);
    for param in params {
        rs_cmd.args(["-p", param]);
    }
    let rs = rs_cmd
        .args(["-m", rs_model.to_str().unwrap(), &train])
        .output()
        .expect("spawn Rust fixed online train");
    assert!(
        rs.status.success(),
        "Rust fixed {algorithm} failed: {}",
        String::from_utf8_lossy(&rs.stdout)
    );

    let c = run_c(&["tag", "-m", c_model.to_str().unwrap(), "-p", "-l", &test]);
    let rs = run_rust(&["tag", "-m", rs_model.to_str().unwrap(), "-p", "-l", &test]);
    assert_identical(&c, &rs, &format!("fixed {algorithm} C/Rust tagging"));
}

fn online_core_log(output: &Output) -> Vec<String> {
    String::from_utf8(output.stdout.clone())
        .unwrap()
        .lines()
        .filter(|line| {
            line.starts_with("Averaged perceptron")
                || line.starts_with("Passive Aggressive")
                || line.starts_with("Adaptive Regularization")
                || line.starts_with("type: ")
                || line.starts_with("c: ")
                || line.starts_with("error_sensitive: ")
                || line.starts_with("averaging: ")
                || line.starts_with("variance: ")
                || line.starts_with("gamma: ")
                || line.starts_with("max_iterations: ")
                || line.starts_with("epsilon: ")
                || line.starts_with("***** Iteration #")
                || line.starts_with("Loss: ")
                || line.starts_with("Feature norm: ")
                || line.starts_with("Terminated with the stopping criterion")
                || line.starts_with("Holdout group: ")
                || line.starts_with("Performance by label")
                || line.starts_with("    ")
                || line.starts_with("Macro-average precision")
                || line.starts_with("Item accuracy: ")
                || line.starts_with("Instance accuracy: ")
        })
        .map(str::to_string)
        .collect()
}

fn assert_fixed_online_trainer_log_matches_c(algorithm: &str, params: &[&str]) {
    let train = test_data("train.txt");

    let mut c_cmd = Command::new(c_bin());
    c_cmd.args(["learn", "-a", algorithm]);
    for param in params {
        c_cmd.args(["-p", param]);
    }
    let c = c_cmd
        .arg(&train)
        .env("LD_LIBRARY_PATH", c_lib_path())
        .output()
        .expect("spawn C fixed online log train");
    assert!(
        c.status.success(),
        "C fixed {algorithm} log train failed: {}",
        String::from_utf8_lossy(&c.stdout)
    );

    let mut rs_cmd = Command::new(rust_bin());
    rs_cmd.args(["learn", "-a", algorithm]);
    for param in params {
        rs_cmd.args(["-p", param]);
    }
    let rs = rs_cmd
        .arg(&train)
        .output()
        .expect("spawn Rust fixed online log train");
    assert!(
        rs.status.success(),
        "Rust fixed {algorithm} log train failed: {}",
        String::from_utf8_lossy(&rs.stdout)
    );

    assert_eq!(
        online_core_log(&c),
        online_core_log(&rs),
        "fixed {algorithm} core log"
    );
}

fn assert_online_holdout_log_matches_c(algorithm: &str, params: &[&str]) {
    let train = test_data("train.txt");
    let test = test_data("test.txt");

    let mut c_cmd = Command::new(c_bin());
    c_cmd.args(["learn", "-a", algorithm, "-e", "2"]);
    for param in params {
        c_cmd.args(["-p", param]);
    }
    let c = c_cmd
        .args([&train, &test])
        .env("LD_LIBRARY_PATH", c_lib_path())
        .output()
        .expect("spawn C online holdout train");
    assert!(
        c.status.success(),
        "C online holdout failed for {algorithm}: {}",
        String::from_utf8_lossy(&c.stdout)
    );

    let mut rs_cmd = Command::new(rust_bin());
    rs_cmd.args(["learn", "-a", algorithm, "-e", "2"]);
    for param in params {
        rs_cmd.args(["-p", param]);
    }
    let rs = rs_cmd
        .args([&train, &test])
        .output()
        .expect("spawn Rust online holdout train");
    assert!(
        rs.status.success(),
        "Rust online holdout failed for {algorithm}: {}",
        String::from_utf8_lossy(&rs.stdout)
    );

    assert_eq!(
        online_core_log(&c),
        online_core_log(&rs),
        "online holdout log {algorithm}"
    );
}

#[test]
fn learn_online_holdout_logs_match_c() {
    skip_without_c!();
    assert_online_holdout_log_matches_c(
        "averaged-perceptron",
        &["max_iterations=1", "epsilon=0"],
    );
    assert_online_holdout_log_matches_c(
        "passive-aggressive",
        &[
            "type=2",
            "c=0.5",
            "error_sensitive=1",
            "averaging=1",
            "max_iterations=1",
            "epsilon=0",
        ],
    );
    assert_online_holdout_log_matches_c(
        "arow",
        &["variance=0.5", "gamma=1.5", "max_iterations=1", "epsilon=0"],
    );
}

#[test]
fn learn_averaged_perceptron_fixed_parameters_match_c_tagging() {
    skip_without_c!();
    assert_fixed_online_trainer_matches_c(
        "averaged-perceptron",
        &["max_iterations=2", "epsilon=0"],
    );
}

#[test]
fn learn_averaged_perceptron_fixed_core_log_matches_c() {
    skip_without_c!();
    assert_fixed_online_trainer_log_matches_c(
        "averaged-perceptron",
        &["max_iterations=2", "epsilon=0"],
    );
}

#[test]
fn learn_passive_aggressive_fixed_parameters_match_c_tagging() {
    skip_without_c!();
    assert_fixed_online_trainer_matches_c(
        "passive-aggressive",
        &[
            "type=2",
            "c=0.5",
            "error_sensitive=1",
            "averaging=1",
            "max_iterations=2",
            "epsilon=0",
        ],
    );
}

#[test]
fn learn_passive_aggressive_fixed_core_log_matches_c() {
    skip_without_c!();
    assert_fixed_online_trainer_log_matches_c(
        "passive-aggressive",
        &[
            "type=2",
            "c=0.5",
            "error_sensitive=1",
            "averaging=1",
            "max_iterations=2",
            "epsilon=0",
        ],
    );
}

#[test]
fn learn_arow_fixed_parameters_match_c_tagging() {
    skip_without_c!();
    assert_fixed_online_trainer_matches_c(
        "arow",
        &["variance=0.5", "gamma=1.5", "max_iterations=2", "epsilon=0"],
    );
}

#[test]
fn learn_arow_fixed_core_log_matches_c() {
    skip_without_c!();
    assert_fixed_online_trainer_log_matches_c(
        "arow",
        &["variance=0.5", "gamma=1.5", "max_iterations=2", "epsilon=0"],
    );
}

#[test]
fn learn_without_files_ignores_stdin_like_c() {
    skip_without_c!();
    let tmp = tempfile::tempdir().unwrap();
    let c_model = tmp.path().join("c_noinput.bin");
    let rs_model = tmp.path().join("rs_noinput.bin");
    let c_model = c_model.to_str().unwrap();
    let rs_model = rs_model.to_str().unwrap();
    let input = b"A\tx\n\n";

    run_with_stdin(
        &c_bin(),
        &[
            "learn",
            "-a",
            "lbfgs",
            "-p",
            "max_iterations=1",
            "-m",
            c_model,
        ],
        input,
        Some(&c_lib_path()),
    );
    run_with_stdin(
        &rust_bin(),
        &[
            "learn",
            "-a",
            "lbfgs",
            "-p",
            "max_iterations=1",
            "-m",
            rs_model,
        ],
        input,
        None,
    );

    let c = run_c(&["dump", c_model]);
    let rs = run_rust(&["dump", rs_model]);
    assert_identical(&c, &rs, "learn without files ignores stdin");
}

#[test]
fn learn_dash_reads_stdin_like_c() {
    skip_without_c!();
    let tmp = tempfile::tempdir().unwrap();
    let c_model = tmp.path().join("c_stdin.bin");
    let rs_model = tmp.path().join("rs_stdin.bin");
    let c_model = c_model.to_str().unwrap();
    let rs_model = rs_model.to_str().unwrap();
    let input = std::fs::read(test_data("train.txt")).unwrap();

    run_with_stdin(
        &c_bin(),
        &[
            "learn",
            "-a",
            "lbfgs",
            "-p",
            "max_iterations=1",
            "-m",
            c_model,
            "-",
        ],
        &input,
        Some(&c_lib_path()),
    );
    run_with_stdin(
        &rust_bin(),
        &[
            "learn",
            "-a",
            "lbfgs",
            "-p",
            "max_iterations=1",
            "-m",
            rs_model,
            "-",
        ],
        &input,
        None,
    );

    let c = run_c(&["tag", "-m", c_model, "-p", "-l", &test_data("test.txt")]);
    let rs = run_rust(&["tag", "-m", rs_model, "-p", "-l", &test_data("test.txt")]);
    assert_identical(&c, &rs, "learn - reads stdin");
}

#[test]
fn learn_file_and_dash_reads_stdin_like_c() {
    skip_without_c!();
    let train = test_data("train.txt");
    let tmp = tempfile::tempdir().unwrap();
    let c_model = tmp.path().join("c_mixed_stdin.bin");
    let rs_model = tmp.path().join("rs_mixed_stdin.bin");
    let c_model = c_model.to_str().unwrap();
    let rs_model = rs_model.to_str().unwrap();
    let input = std::fs::read(test_data("test.txt")).unwrap();

    run_with_stdin(
        &c_bin(),
        &[
            "learn",
            "-a",
            "lbfgs",
            "-p",
            "max_iterations=1",
            "-m",
            c_model,
            &train,
            "-",
        ],
        &input,
        Some(&c_lib_path()),
    );
    run_with_stdin(
        &rust_bin(),
        &[
            "learn",
            "-a",
            "lbfgs",
            "-p",
            "max_iterations=1",
            "-m",
            rs_model,
            &train,
            "-",
        ],
        &input,
        None,
    );

    let c = run_c(&["tag", "-m", c_model, "-p", "-l", &test_data("test.txt")]);
    let rs = run_rust(&["tag", "-m", rs_model, "-p", "-l", &test_data("test.txt")]);
    assert_identical(&c, &rs, "learn file - reads stdin");
}

#[test]
fn learn_log_to_file_help_matches_c() {
    skip_without_c!();
    let tmp = tempfile::tempdir().unwrap();
    let c_base = tmp.path().join("c_log");
    let rs_base = tmp.path().join("rs_log");
    let c_base = c_base.to_str().unwrap();
    let rs_base = rs_base.to_str().unwrap();

    let _c = run_c_raw(&[
        "learn",
        "-a",
        "l2sgd",
        "-l",
        "-L",
        c_base,
        "--help-params",
        "-p",
        "does_not_exist=1",
    ]);
    let c_log_path = format!("{}_l2sgd_does_not_exist=1", c_base);
    assert!(
        std::path::Path::new(&c_log_path).exists(),
        "C should write the -H log file"
    );

    let rs = Command::new(rust_bin())
        .args([
            "learn",
            "-a",
            "l2sgd",
            "-l",
            "-L",
            rs_base,
            "--help-params",
            "-p",
            "does_not_exist=1",
        ])
        .output()
        .expect("spawn Rust learn -l -H");
    assert!(rs.status.success(), "Rust learn -l -H failed");

    let c_log = std::fs::read_to_string(c_log_path).unwrap();
    let rs_log = std::fs::read_to_string(format!("{}_l2sgd_does_not_exist=1", rs_base)).unwrap();
    assert_identical(&c_log, &rs_log, "learn -l -H log file");
}

#[test]
fn learn_log_to_file_uses_c_filename_pattern() {
    skip_without_c!();
    let train = test_data("train.txt");
    let tmp = tempfile::tempdir().unwrap();
    let base = tmp.path().join("rs_log");
    let base = base.to_str().unwrap();

    let rs = Command::new(rust_bin())
        .args([
            "learn",
            "-a",
            "lbfgs",
            "-l",
            "-L",
            base,
            "-p",
            "max_iterations=1",
            &train,
        ])
        .output()
        .expect("spawn Rust learn -l");
    assert!(rs.status.success(), "Rust learn -l failed");

    let log_path = format!("{}_lbfgs_max_iterations=1", base);
    let log = std::fs::read_to_string(&log_path).expect("read generated Rust log");
    assert!(
        log.contains("Feature generation"),
        "generated log should contain training output"
    );
    assert!(
        String::from_utf8(rs.stdout).unwrap().is_empty(),
        "learn -l should redirect training output away from stdout"
    );
}

#[test]
fn learn_empty_model_path_does_not_write_model_like_c() {
    skip_without_c!();
    let train = test_data("train.txt");
    let tmp = tempfile::tempdir().unwrap();

    let c = Command::new(c_bin())
        .current_dir(tmp.path())
        .args([
            "learn",
            "-a",
            "lbfgs",
            "-p",
            "max_iterations=1",
            "-m",
            "",
            &train,
        ])
        .env("LD_LIBRARY_PATH", c_lib_path())
        .output()
        .expect("spawn C learn empty model");
    assert!(c.status.success(), "C learn with empty model path failed");
    assert_eq!(std::fs::read_dir(tmp.path()).unwrap().count(), 0);

    let rs = Command::new(rust_bin())
        .current_dir(tmp.path())
        .args([
            "learn",
            "-a",
            "lbfgs",
            "-p",
            "max_iterations=1",
            "-m",
            "",
            &train,
        ])
        .output()
        .expect("spawn Rust learn empty model");
    assert!(
        rs.status.success(),
        "Rust learn with empty model path failed"
    );
    assert_eq!(std::fs::read_dir(tmp.path()).unwrap().count(), 0);
}

#[test]
fn learn_iwa_escaping_matches_c() {
    skip_without_c!();
    let tmp = tempfile::tempdir().unwrap();
    let input = tmp.path().join("escaped.iwa");
    std::fs::write(&input, "A\ta\\:b\tc\\\\d\n\n").unwrap();
    let c_model = tmp.path().join("c_escaped.bin");
    let rs_model = tmp.path().join("rs_escaped.bin");

    let c = Command::new(c_bin())
        .args([
            "learn",
            "-a",
            "lbfgs",
            "-p",
            "max_iterations=0",
            "-m",
            c_model.to_str().unwrap(),
            input.to_str().unwrap(),
        ])
        .env("LD_LIBRARY_PATH", c_lib_path())
        .output()
        .expect("spawn C learn escaped IWA");
    assert!(c.status.success(), "C learn escaped IWA failed");

    let rs = Command::new(rust_bin())
        .args([
            "learn",
            "-a",
            "lbfgs",
            "-p",
            "max_iterations=0",
            "-m",
            rs_model.to_str().unwrap(),
            input.to_str().unwrap(),
        ])
        .output()
        .expect("spawn Rust learn escaped IWA");
    assert!(rs.status.success(), "Rust learn escaped IWA failed");

    let c = run_c(&["dump", c_model.to_str().unwrap()]);
    let rs = run_rust(&["dump", rs_model.to_str().unwrap()]);
    assert_identical(&c, &rs, "IWA escaped attribute parsing");
}

#[test]
fn learn_iwa_crlf_matches_c() {
    skip_without_c!();
    let tmp = tempfile::tempdir().unwrap();
    let input = tmp.path().join("crlf.iwa");
    std::fs::write(&input, b"A\tx\r\n\r\n").unwrap();
    let c_model = tmp.path().join("c_crlf.bin");
    let rs_model = tmp.path().join("rs_crlf.bin");

    let c = Command::new(c_bin())
        .args([
            "learn",
            "-a",
            "lbfgs",
            "-p",
            "max_iterations=0",
            "-m",
            c_model.to_str().unwrap(),
            input.to_str().unwrap(),
        ])
        .env("LD_LIBRARY_PATH", c_lib_path())
        .output()
        .expect("spawn C learn CRLF IWA");
    assert!(c.status.success(), "C learn CRLF IWA failed");

    let rs = Command::new(rust_bin())
        .args([
            "learn",
            "-a",
            "lbfgs",
            "-p",
            "max_iterations=0",
            "-m",
            rs_model.to_str().unwrap(),
            input.to_str().unwrap(),
        ])
        .output()
        .expect("spawn Rust learn CRLF IWA");
    assert!(rs.status.success(), "Rust learn CRLF IWA failed");

    let c = run_c(&["dump", c_model.to_str().unwrap()]);
    let rs = run_rust(&["dump", rs_model.to_str().unwrap()]);
    assert_identical(&c, &rs, "IWA CRLF parsing");
}

#[test]
fn learn_iwa_weight_declarations_match_c() {
    skip_without_c!();
    let cases = [
        ("missing", "@weight\nA\tx\n\n"),
        ("invalid", "@weight:not-a-number\nA\tx\n\n"),
        ("multiple", "@weight:0\n@weight:2\nA\tx\n\n"),
        ("after_item", "A\tx\n@weight:2\n\n"),
    ];

    for (name, data) in cases {
        let tmp = tempfile::tempdir().unwrap();
        let input = tmp.path().join(format!("{name}.iwa"));
        std::fs::write(&input, data).unwrap();
        let c_model = tmp.path().join("c_weight.bin");
        let rs_model = tmp.path().join("rs_weight.bin");

        let c = Command::new(c_bin())
            .args([
                "learn",
                "-a",
                "lbfgs",
                "-p",
                "feature.minfreq=1.5",
                "-p",
                "max_iterations=0",
                "-m",
                c_model.to_str().unwrap(),
                input.to_str().unwrap(),
            ])
            .env("LD_LIBRARY_PATH", c_lib_path())
            .output()
            .expect("spawn C learn @weight IWA");
        assert!(c.status.success(), "C learn @weight IWA failed for {name}");

        let rs = Command::new(rust_bin())
            .args([
                "learn",
                "-a",
                "lbfgs",
                "-p",
                "feature.minfreq=1.5",
                "-p",
                "max_iterations=0",
                "-m",
                rs_model.to_str().unwrap(),
                input.to_str().unwrap(),
            ])
            .output()
            .expect("spawn Rust learn @weight IWA");
        assert!(
            rs.status.success(),
            "Rust learn @weight IWA failed for {name}"
        );

        let c = run_c(&["dump", c_model.to_str().unwrap()]);
        let rs = run_rust(&["dump", rs_model.to_str().unwrap()]);
        assert_identical(&c, &rs, &format!("IWA @weight parsing {name}"));
    }
}

#[test]
fn learn_iwa_unknown_declaration_fails_like_c() {
    skip_without_c!();
    let tmp = tempfile::tempdir().unwrap();
    let input = tmp.path().join("unknown_declaration.iwa");
    std::fs::write(&input, "@unknown:1\n\n").unwrap();

    let c = Command::new(c_bin())
        .args(["learn", "-a", "lbfgs", input.to_str().unwrap()])
        .env("LD_LIBRARY_PATH", c_lib_path())
        .output()
        .expect("spawn C learn unknown declaration");
    assert!(!c.status.success(), "C should reject unknown declarations");

    let rs = Command::new(rust_bin())
        .args(["learn", "-a", "lbfgs", input.to_str().unwrap()])
        .output()
        .expect("spawn Rust learn unknown declaration");
    assert!(
        !rs.status.success(),
        "Rust should reject unknown declarations"
    );
    assert_eq!(c.status.code(), rs.status.code());
}

#[test]
fn learn_iwa_field_edge_cases_match_c() {
    skip_without_c!();
    let fixtures = [
        ("iwa_edge_fields.txt", "IWA field edge fixture"),
        ("iwa_edge_weights.txt", "IWA weight edge fixture"),
        ("iwa_escaped_fields.txt", "IWA escaped field fixture"),
    ];
    for (fixture, ctx) in fixtures {
        let input = PathBuf::from(test_data(fixture));
        assert_iwa_file_dump_matches_c(&input, &["max_iterations=0"], ctx);
    }
}

#[test]
fn learn_generated_small_iwa_cases_match_c() {
    skip_without_c!();
    let labels = ["A", "B"];
    let attrs = ["x", "y"];
    for length in 1..=3 {
        let mut data = String::new();
        for pos in 0..length {
            let label = labels[pos % labels.len()];
            let attr = attrs[(pos + length) % attrs.len()];
            data.push_str(label);
            data.push('\t');
            data.push_str(attr);
            data.push(':');
            data.push_str(&(pos as i32 - 1).to_string());
            if pos % 2 == 0 {
                data.push('\t');
                data.push_str(attr);
                data.push_str(":2");
            }
            data.push('\n');
        }
        data.push('\n');
        assert_iwa_dump_matches_c(
            &data,
            &["feature.possible_states=1", "max_iterations=0"],
            &format!("generated small IWA length {length}"),
        );
    }
}

#[test]
fn learn_holdout_only_symbols_match_c_dictionary_order() {
    skip_without_c!();
    let tmp = tempfile::tempdir().unwrap();
    let train = tmp.path().join("train.iwa");
    let holdout = tmp.path().join("holdout.iwa");
    std::fs::write(&train, "A\tx\n\n").unwrap();
    std::fs::write(&holdout, "B\ty\n\n").unwrap();
    let c_model = tmp.path().join("c_holdout_symbols.bin");
    let rs_model = tmp.path().join("rs_holdout_symbols.bin");

    let c = Command::new(c_bin())
        .args([
            "learn",
            "-a",
            "lbfgs",
            "-e",
            "2",
            "-p",
            "max_iterations=0",
            "-m",
            c_model.to_str().unwrap(),
            train.to_str().unwrap(),
            holdout.to_str().unwrap(),
        ])
        .env("LD_LIBRARY_PATH", c_lib_path())
        .output()
        .expect("spawn C learn holdout-only symbols");
    assert!(c.status.success(), "C learn holdout-only symbols failed");

    let rs = Command::new(rust_bin())
        .args([
            "learn",
            "-a",
            "lbfgs",
            "-e",
            "2",
            "-p",
            "max_iterations=0",
            "-m",
            rs_model.to_str().unwrap(),
            train.to_str().unwrap(),
            holdout.to_str().unwrap(),
        ])
        .output()
        .expect("spawn Rust learn holdout-only symbols");
    assert!(
        rs.status.success(),
        "Rust learn holdout-only symbols failed"
    );

    let c = run_c(&["dump", c_model.to_str().unwrap()]);
    let rs = run_rust(&["dump", rs_model.to_str().unwrap()]);
    assert_identical(&c, &rs, "holdout-only label/attribute dictionary order");
}

#[test]
fn learn_feature_map_small_fixtures_match_c() {
    skip_without_c!();
    let cases = [
        (
            "one label one attribute",
            "A\tx\n\n",
            &["max_iterations=0"][..],
        ),
        (
            "two labels one transition",
            "A\tx\nB\ty\n\n",
            &["max_iterations=0"][..],
        ),
        (
            "duplicate attributes",
            "A\tx:1\tx:2\n\n",
            &["max_iterations=0"][..],
        ),
        (
            "zero and negative attributes",
            "A\tzero:0\tneg:-2\tpos:2\n\n",
            &["max_iterations=0"][..],
        ),
        (
            "rare features below minfreq",
            "A\tcommon\nA\tcommon\trare\n\n",
            &["feature.minfreq=1.5", "max_iterations=0"][..],
        ),
    ];

    for (ctx, data, params) in cases {
        assert_iwa_dump_matches_c(data, params, ctx);
    }
}

#[test]
fn learn_forced_possible_features_match_c() {
    skip_without_c!();
    let cases = [
        (
            "possible state features",
            "A\tx\nB\ty\n\n",
            &["feature.possible_states=1", "max_iterations=0"][..],
        ),
        (
            "possible transition features",
            "A\tx\nB\ty\n\n",
            &["feature.possible_transitions=1", "max_iterations=0"][..],
        ),
        (
            "possible state and transition features with minfreq",
            "A\tx\nB\ty\n\n",
            &[
                "feature.possible_states=1",
                "feature.possible_transitions=1",
                "feature.minfreq=1",
                "max_iterations=0",
            ][..],
        ),
    ];

    for (ctx, data, params) in cases {
        assert_iwa_dump_matches_c(data, params, ctx);
    }
}

#[test]
fn learn_possible_states_ignores_holdout_only_attributes_like_c() {
    skip_without_c!();
    let tmp = tempfile::tempdir().unwrap();
    let train = tmp.path().join("train.iwa");
    let holdout = tmp.path().join("holdout.iwa");
    std::fs::write(&train, "A\tx\n\n").unwrap();
    std::fs::write(&holdout, "B\ty\n\n").unwrap();
    let c_model = tmp.path().join("c_holdout_possible_states.bin");
    let rs_model = tmp.path().join("rs_holdout_possible_states.bin");

    let mut c_cmd = Command::new(c_bin());
    c_cmd
        .args([
            "learn",
            "-a",
            "lbfgs",
            "-e",
            "2",
            "-p",
            "feature.possible_states=1",
            "-p",
            "max_iterations=0",
            "-m",
            c_model.to_str().unwrap(),
            train.to_str().unwrap(),
            holdout.to_str().unwrap(),
        ])
        .env("LD_LIBRARY_PATH", c_lib_path());
    let c = c_cmd
        .output()
        .expect("spawn C learn holdout possible states");
    assert!(c.status.success(), "C learn holdout possible states failed");

    let rs = Command::new(rust_bin())
        .args([
            "learn",
            "-a",
            "lbfgs",
            "-e",
            "2",
            "-p",
            "feature.possible_states=1",
            "-p",
            "max_iterations=0",
            "-m",
            rs_model.to_str().unwrap(),
            train.to_str().unwrap(),
            holdout.to_str().unwrap(),
        ])
        .output()
        .expect("spawn Rust learn holdout possible states");
    assert!(
        rs.status.success(),
        "Rust learn holdout possible states failed"
    );

    let c = run_c(&["dump", c_model.to_str().unwrap()]);
    let rs = run_rust(&["dump", rs_model.to_str().unwrap()]);
    assert_identical(&c, &rs, "possible states ignore holdout-only attrs");
}

#[test]
fn tag_viterbi_ties_and_marginals_match_c() {
    skip_without_c!();
    let tmp = tempfile::tempdir().unwrap();
    let train = tmp.path().join("train.iwa");
    let input = tmp.path().join("tie.iwa");
    std::fs::write(&train, "A\tx\nB\ty\n\n").unwrap();
    std::fs::write(&input, "A\tunknown\n\n").unwrap();
    let c_model = tmp.path().join("c_tie.bin");
    let rs_model = tmp.path().join("rs_tie.bin");

    train_iwa_with_params(
        &c_bin(),
        &train,
        &c_model,
        &["max_iterations=0"],
        Some(&c_lib_path()),
    );
    train_iwa_with_params(
        &rust_bin(),
        &train,
        &rs_model,
        &["max_iterations=0"],
        None,
    );

    let c = run_c(&[
        "tag",
        "-m",
        c_model.to_str().unwrap(),
        "-p",
        "-i",
        "-l",
        input.to_str().unwrap(),
    ]);
    let rs = run_rust(&[
        "tag",
        "-m",
        rs_model.to_str().unwrap(),
        "-p",
        "-i",
        "-l",
        input.to_str().unwrap(),
    ]);
    assert_identical(&c, &rs, "viterbi ties and marginal formatting");
}

// ── cross-implementation: train with one, tag with the other ────────────────
//
// For *every* algorithm (including non-deterministic ones) we train once with
// C, then tag the same input with both C and Rust and demand identical output.
// This proves Rust reads C models correctly.  We also do the reverse for
// deterministic algorithms.

fn cross_c_train_both_tag(algo: &str) {
    skip_without_c!();
    let input = test_data("train.txt");
    let tmp = tempfile::tempdir().unwrap();
    let model = tmp.path().join("model.bin");
    let ms = model.to_str().unwrap();

    train_model(&c_bin(), algo, &input, ms, Some(&c_lib_path()));

    // plain labels
    let c = run_c(&["tag", "-m", ms, &input]);
    let rs = run_rust(&["tag", "-m", ms, &input]);
    assert_identical(&c, &rs, &format!("cross C→both tag {algo}"));

    // scores + marginals
    let c = run_c(&["tag", "-m", ms, "-p", "-i", &input]);
    let rs = run_rust(&["tag", "-m", ms, "-p", "-i", &input]);
    assert_identical(&c, &rs, &format!("cross C→both tag -pi {algo}"));

    // all-label marginals
    let c = run_c(&["tag", "-m", ms, "-l", &input]);
    let rs = run_rust(&["tag", "-m", ms, "-l", &input]);
    assert_identical(&c, &rs, &format!("cross C→both tag -l {algo}"));
}

fn cross_rust_train_both_tag(algo: &str) {
    skip_without_c!();
    let input = test_data("train.txt");
    let tmp = tempfile::tempdir().unwrap();
    let model = tmp.path().join("model.bin");
    let ms = model.to_str().unwrap();

    train_model(&rust_bin(), algo, &input, ms, None);

    let c = run_c(&["tag", "-m", ms, &input]);
    let rs = run_rust(&["tag", "-m", ms, &input]);
    assert_identical(&c, &rs, &format!("cross Rust→both tag {algo}"));

    let c = run_c(&["tag", "-m", ms, "-p", "-i", &input]);
    let rs = run_rust(&["tag", "-m", ms, "-p", "-i", &input]);
    assert_identical(&c, &rs, &format!("cross Rust→both tag -pi {algo}"));
}

// C-trained model, tagged by both
#[test]
fn cross_ctrain_lbfgs() {
    cross_c_train_both_tag("lbfgs");
}
#[test]
fn cross_ctrain_l2sgd() {
    cross_c_train_both_tag("l2sgd");
}
#[test]
fn cross_ctrain_ap() {
    cross_c_train_both_tag("averaged-perceptron");
}
#[test]
fn cross_ctrain_pa() {
    cross_c_train_both_tag("passive-aggressive");
}
#[test]
fn cross_ctrain_arow() {
    cross_c_train_both_tag("arow");
}

// Rust-trained model, tagged by both (deterministic only)
#[test]
fn cross_rstrain_lbfgs() {
    cross_rust_train_both_tag("lbfgs");
}
#[test]
fn cross_rstrain_l2sgd() {
    cross_rust_train_both_tag("l2sgd");
}
#[test]
fn cross_rstrain_arow() {
    cross_rust_train_both_tag("arow");
}

// ── dump identity ───────────────────────────────────────────────────────────
//
// Dump is pure model → text, so it must match byte-for-byte.

fn dump_identical(algo: &str) {
    skip_without_c!();
    let input = test_data("train.txt");
    let tmp = tempfile::tempdir().unwrap();
    let model = tmp.path().join("model.bin");
    let ms = model.to_str().unwrap();

    train_model(&c_bin(), algo, &input, ms, Some(&c_lib_path()));

    let c = run_c(&["dump", ms]);
    let rs = run_rust(&["dump", ms]);
    assert_identical(&c, &rs, &format!("dump {algo}"));
}

#[test]
fn dump_lbfgs() {
    dump_identical("lbfgs");
}
#[test]
fn dump_l2sgd() {
    dump_identical("l2sgd");
}
#[test]
fn dump_arow() {
    dump_identical("arow");
}
#[test]
fn dump_ap() {
    dump_identical("averaged-perceptron");
}
#[test]
fn dump_pa() {
    dump_identical("passive-aggressive");
}

// ── tag on unseen data with various models ──────────────────────────────────
//
// Train on train.txt, tag on test.txt (partially-unseen features).

fn tag_unseen_identical(algo: &str) {
    skip_without_c!();
    let train = test_data("train.txt");
    let test = test_data("test.txt");
    let tmp = tempfile::tempdir().unwrap();
    let model = tmp.path().join("model.bin");
    let ms = model.to_str().unwrap();

    train_model(&c_bin(), algo, &train, ms, Some(&c_lib_path()));

    // plain
    let c = run_c(&["tag", "-m", ms, &test]);
    let rs = run_rust(&["tag", "-m", ms, &test]);
    assert_identical(&c, &rs, &format!("unseen tag {algo}"));

    // scores + marginals
    let c = run_c(&["tag", "-m", ms, "-p", "-l", &test]);
    let rs = run_rust(&["tag", "-m", ms, "-p", "-l", &test]);
    assert_identical(&c, &rs, &format!("unseen tag -pl {algo}"));
}

#[test]
fn unseen_lbfgs() {
    tag_unseen_identical("lbfgs");
}
#[test]
fn unseen_l2sgd() {
    tag_unseen_identical("l2sgd");
}
#[test]
fn unseen_arow() {
    tag_unseen_identical("arow");
}
#[test]
fn unseen_ap() {
    tag_unseen_identical("averaged-perceptron");
}
#[test]
fn unseen_pa() {
    tag_unseen_identical("passive-aggressive");
}
