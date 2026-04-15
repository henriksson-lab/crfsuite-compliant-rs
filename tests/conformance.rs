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
use std::process::{Command, Stdio};

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
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .to_path_buf()
}

fn rust_bin() -> PathBuf {
    if let Some(bin) = option_env!("CARGO_BIN_EXE_crfsuite-rs") {
        return PathBuf::from(bin);
    }
    let root = project_root();
    root.join("target/debug/crfsuite-rs")
}

fn c_bin() -> PathBuf { project_root().join("crfsuite/frontend/.libs/crfsuite") }
fn c_lib_path() -> PathBuf { project_root().join("crfsuite/lib/crf/.libs") }
fn test_data(name: &str) -> String { project_root().join("test_data").join(name).to_str().unwrap().to_string() }

/// Run the Rust CLI; return stdout.  Panics on non-zero exit.
fn run_rust(args: &[&str]) -> String {
    let bin = rust_bin();
    assert!(bin.exists(), "Rust binary missing – run `cargo build -p crfsuite-cli`");
    let o = Command::new(&bin).args(args).output().expect("spawn crfsuite-rs");
    assert!(o.status.success(), "crfsuite-rs {args:?} exit={}\n{}", o.status, String::from_utf8_lossy(&o.stderr));
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
    assert!(o.status.success(), "C crfsuite {args:?} exit={}\n{}", o.status, String::from_utf8_lossy(&o.stderr));
    String::from_utf8(o.stdout).unwrap()
}

fn run_with_stdin(bin: &PathBuf, args: &[&str], input: &[u8], ld_env: Option<&PathBuf>) -> String {
    let mut cmd = Command::new(bin);
    cmd.args(args).stdin(Stdio::piped()).stdout(Stdio::piped());
    if let Some(ld) = ld_env { cmd.env("LD_LIBRARY_PATH", ld); }
    let mut child = cmd.spawn().expect("spawn stdin command");
    child.stdin.as_mut().unwrap().write_all(input).unwrap();
    let o = child.wait_with_output().expect("wait stdin command");
    assert!(o.status.success(), "stdin command {args:?} exit={}\n{}", o.status, String::from_utf8_lossy(&o.stderr));
    String::from_utf8(o.stdout).unwrap()
}

/// Train a model with given binary + args; return path to model file.
fn train_model(bin: &PathBuf, algo: &str, input: &str, out_model: &str, ld_env: Option<&PathBuf>) {
    let mut cmd = Command::new(bin);
    cmd.args(["learn", "-a", algo, "-m", out_model, input]);
    if let Some(ld) = ld_env { cmd.env("LD_LIBRARY_PATH", ld); }
    let o = cmd.output().expect("spawn train");
    assert!(o.status.success(), "train {algo} failed: {}", String::from_utf8_lossy(&o.stdout));
}

fn assert_identical(a: &str, b: &str, ctx: &str) {
    if a == b { return; }
    let ab = a.as_bytes();
    let bb = b.as_bytes();
    for i in 0..ab.len().max(bb.len()) {
        if ab.get(i) != bb.get(i) {
            let line = a[..i.min(a.len())].matches('\n').count() + 1;
            panic!("{ctx}: first diff at byte {i} (line ~{line})\n  C:    {:?}\n  Rust: {:?}",
                ab.get(i), bb.get(i));
        }
    }
}

fn assert_same_failure(c: std::process::Output, rs: std::process::Output, ctx: &str) {
    assert!(!c.status.success(), "C should fail for {ctx}");
    assert!(!rs.status.success(), "Rust should fail for {ctx}");
    assert_eq!(c.status.code(), rs.status.code(), "{ctx}: exit code mismatch");
    assert_identical(
        &String::from_utf8(c.stderr).unwrap(),
        &String::from_utf8(rs.stderr).unwrap(),
        ctx,
    );
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
    let c  = run_c(&["tag", "-m", rms, "-p", "-l", &input]);
    let rs = run_rust(&["tag", "-m", rms, "-p", "-l", &input]);
    assert_identical(&c, &rs, &format!("model equiv {algo}: tag -p -l"));
}

#[test] fn model_equiv_lbfgs()  { model_equivalent("lbfgs"); }
#[test] fn model_equiv_l2sgd()  { model_equivalent("l2sgd"); }
#[test] fn model_equiv_arow()   { model_equivalent("arow"); }

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

    let c_out  = run_c(&args);
    let rs_out = run_rust(&args);
    assert_identical(&c_out, &rs_out, ctx);
}

// basic labels
#[test] fn tag_labels_train()      { tag_identical(&[],             "train.txt",          "tag train.txt"); }
#[test] fn tag_labels_test()       { tag_identical(&[],             "test.txt",           "tag test.txt"); }
#[test] fn tag_labels_single()     { tag_identical(&[],             "single_item.txt",    "tag single_item"); }
#[test] fn tag_labels_multi()      { tag_identical(&[],             "multi_sequence.txt", "tag multi_seq"); }

// reference labels
#[test] fn tag_ref_train()         { tag_identical(&["-r"],         "train.txt",          "tag -r train"); }
#[test] fn tag_ref_test()          { tag_identical(&["-r"],         "test.txt",           "tag -r test"); }

// probability + score
#[test] fn tag_prob_train()        { tag_identical(&["-p"],         "train.txt",          "tag -p train"); }
#[test] fn tag_prob_test()         { tag_identical(&["-p"],         "test.txt",           "tag -p test"); }

// marginal of predicted label
#[test] fn tag_marginal_train()    { tag_identical(&["-i"],         "train.txt",          "tag -i train"); }
#[test] fn tag_marginal_test()     { tag_identical(&["-i"],         "test.txt",           "tag -i test"); }
#[test] fn tag_marginal_single()   { tag_identical(&["-i"],         "single_item.txt",    "tag -i single"); }

// all-label marginals
#[test] fn tag_margall_train()     { tag_identical(&["-l"],         "train.txt",          "tag -l train"); }
#[test] fn tag_margall_test()      { tag_identical(&["-l"],         "test.txt",           "tag -l test"); }
#[test] fn tag_margall_single()    { tag_identical(&["-l"],         "single_item.txt",    "tag -l single"); }
#[test] fn tag_margall_multi()     { tag_identical(&["-l"],         "multi_sequence.txt", "tag -l multi"); }

// combined flags
#[test] fn tag_rpi_train()         { tag_identical(&["-r","-p","-i"], "train.txt",        "tag -rpi train"); }
#[test] fn tag_rpi_test()          { tag_identical(&["-r","-p","-i"], "test.txt",         "tag -rpi test"); }
#[test] fn tag_rpl_test()          { tag_identical(&["-r","-p","-l"], "test.txt",         "tag -rpl test"); }
#[test] fn tag_q_suppresses_prob_marginals() { tag_identical(&["-q","-p","-l"], "test.txt", "tag -qpl test"); }
#[test] fn tag_q_suppresses_reference() { tag_identical(&["-q","-r"], "test.txt", "tag -qr test"); }

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
    std::fs::write(&input, "B-NP\tw[0]=the:2abc\tpos[0]=DT:not-a-number\n").unwrap();
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
        .args(["learn", "-a", "lbfgs", "-e", "2", "-m", c_model, &train, &test])
        .env("LD_LIBRARY_PATH", c_lib_path());
    let c_status = c_cmd.output().expect("spawn C holdout train");
    assert!(c_status.status.success(), "C holdout train failed");

    let rs_status = Command::new(rust_bin())
        .args(["learn", "-a", "lbfgs", "-e", "2", "-m", rs_model, &train, &test])
        .output()
        .expect("spawn Rust holdout train");
    assert!(rs_status.status.success(), "Rust holdout train failed");

    let c = run_c(&["tag", "-m", c_model, "-p", "-l", &test]);
    let rs = run_rust(&["tag", "-m", rs_model, "-p", "-l", &test]);
    assert_identical(&c, &rs, "learn -e holdout model");
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
    assert!(rs_status.status.success(), "Rust cross-validation train failed");
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
}

#[test]
fn learn_help_parameters_matches_c() {
    skip_without_c!();

    let c = run_c_raw(&["learn", "-a", "l2sgd", "--help-params"]);
    let c_out = String::from_utf8(c.stdout).unwrap();
    let c_help = c_out
        .split_once("\n\n")
        .map(|(_, help)| help)
        .unwrap_or(&c_out);
    let rs_help = run_rust(&["learn", "-a", "l2sgd", "--help-params"]);
    assert_identical(c_help, &rs_help, "learn --help-params l2sgd");
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
    assert!(c.status.success(), "C should accept l2sgd calibration parameters");

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
        &["learn", "-a", "lbfgs", "-p", "max_iterations=1", "-m", c_model],
        input,
        Some(&c_lib_path()),
    );
    run_with_stdin(
        &rust_bin(),
        &["learn", "-a", "lbfgs", "-p", "max_iterations=1", "-m", rs_model],
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
        &["learn", "-a", "lbfgs", "-p", "max_iterations=1", "-m", c_model, "-"],
        &input,
        Some(&c_lib_path()),
    );
    run_with_stdin(
        &rust_bin(),
        &["learn", "-a", "lbfgs", "-p", "max_iterations=1", "-m", rs_model, "-"],
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
        &["learn", "-a", "lbfgs", "-p", "max_iterations=1", "-m", c_model, &train, "-"],
        &input,
        Some(&c_lib_path()),
    );
    run_with_stdin(
        &rust_bin(),
        &["learn", "-a", "lbfgs", "-p", "max_iterations=1", "-m", rs_model, &train, "-"],
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
    assert!(std::path::Path::new(&c_log_path).exists(), "C should write the -H log file");

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
    assert!(log.contains("Feature generation"), "generated log should contain training output");
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
        .args(["learn", "-a", "lbfgs", "-p", "max_iterations=1", "-m", "", &train])
        .env("LD_LIBRARY_PATH", c_lib_path())
        .output()
        .expect("spawn C learn empty model");
    assert!(c.status.success(), "C learn with empty model path failed");
    assert_eq!(std::fs::read_dir(tmp.path()).unwrap().count(), 0);

    let rs = Command::new(rust_bin())
        .current_dir(tmp.path())
        .args(["learn", "-a", "lbfgs", "-p", "max_iterations=1", "-m", "", &train])
        .output()
        .expect("spawn Rust learn empty model");
    assert!(rs.status.success(), "Rust learn with empty model path failed");
    assert_eq!(std::fs::read_dir(tmp.path()).unwrap().count(), 0);
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
    let c  = run_c(&["tag", "-m", ms, &input]);
    let rs = run_rust(&["tag", "-m", ms, &input]);
    assert_identical(&c, &rs, &format!("cross C→both tag {algo}"));

    // scores + marginals
    let c  = run_c(&["tag", "-m", ms, "-p", "-i", &input]);
    let rs = run_rust(&["tag", "-m", ms, "-p", "-i", &input]);
    assert_identical(&c, &rs, &format!("cross C→both tag -pi {algo}"));

    // all-label marginals
    let c  = run_c(&["tag", "-m", ms, "-l", &input]);
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

    let c  = run_c(&["tag", "-m", ms, &input]);
    let rs = run_rust(&["tag", "-m", ms, &input]);
    assert_identical(&c, &rs, &format!("cross Rust→both tag {algo}"));

    let c  = run_c(&["tag", "-m", ms, "-p", "-i", &input]);
    let rs = run_rust(&["tag", "-m", ms, "-p", "-i", &input]);
    assert_identical(&c, &rs, &format!("cross Rust→both tag -pi {algo}"));
}

// C-trained model, tagged by both
#[test] fn cross_ctrain_lbfgs()  { cross_c_train_both_tag("lbfgs"); }
#[test] fn cross_ctrain_l2sgd()  { cross_c_train_both_tag("l2sgd"); }
#[test] fn cross_ctrain_ap()     { cross_c_train_both_tag("averaged-perceptron"); }
#[test] fn cross_ctrain_pa()     { cross_c_train_both_tag("passive-aggressive"); }
#[test] fn cross_ctrain_arow()   { cross_c_train_both_tag("arow"); }

// Rust-trained model, tagged by both (deterministic only)
#[test] fn cross_rstrain_lbfgs() { cross_rust_train_both_tag("lbfgs"); }
#[test] fn cross_rstrain_l2sgd() { cross_rust_train_both_tag("l2sgd"); }
#[test] fn cross_rstrain_arow()  { cross_rust_train_both_tag("arow"); }

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

    let c  = run_c(&["dump", ms]);
    let rs = run_rust(&["dump", ms]);
    assert_identical(&c, &rs, &format!("dump {algo}"));
}

#[test] fn dump_lbfgs() { dump_identical("lbfgs"); }
#[test] fn dump_l2sgd() { dump_identical("l2sgd"); }
#[test] fn dump_arow()  { dump_identical("arow"); }
#[test] fn dump_ap()    { dump_identical("averaged-perceptron"); }
#[test] fn dump_pa()    { dump_identical("passive-aggressive"); }

// ── tag on unseen data with various models ──────────────────────────────────
//
// Train on train.txt, tag on test.txt (partially-unseen features).

fn tag_unseen_identical(algo: &str) {
    skip_without_c!();
    let train = test_data("train.txt");
    let test  = test_data("test.txt");
    let tmp = tempfile::tempdir().unwrap();
    let model = tmp.path().join("model.bin");
    let ms = model.to_str().unwrap();

    train_model(&c_bin(), algo, &train, ms, Some(&c_lib_path()));

    // plain
    let c  = run_c(&["tag", "-m", ms, &test]);
    let rs = run_rust(&["tag", "-m", ms, &test]);
    assert_identical(&c, &rs, &format!("unseen tag {algo}"));

    // scores + marginals
    let c  = run_c(&["tag", "-m", ms, "-p", "-l", &test]);
    let rs = run_rust(&["tag", "-m", ms, "-p", "-l", &test]);
    assert_identical(&c, &rs, &format!("unseen tag -pl {algo}"));
}

#[test] fn unseen_lbfgs() { tag_unseen_identical("lbfgs"); }
#[test] fn unseen_l2sgd() { tag_unseen_identical("l2sgd"); }
#[test] fn unseen_arow()  { tag_unseen_identical("arow"); }
#[test] fn unseen_ap()    { tag_unseen_identical("averaged-perceptron"); }
#[test] fn unseen_pa()    { tag_unseen_identical("passive-aggressive"); }
