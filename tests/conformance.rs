//! Conformance tests: Rust CLI vs original C CLI.
//!
//! Focus: output *files* (model binaries) and tagging *results* (labels,
//! scores, marginals) must be identical.  CLI chrome (banners, progress bars,
//! timing lines) is intentionally ignored.
//!
//! Prerequisites:
//!   cargo build -p crfsuite-cli
//!   (cd crfsuite && make)          # builds the C binary

use std::path::PathBuf;
use std::process::Command;

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
    let root = project_root();
    let release = root.join("target/release/crfsuite-rs");
    if release.exists() { release } else { root.join("target/debug/crfsuite-rs") }
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
