//! Cross-validation: Rust model dump vs C crfsuite dump command.

use crfsuite_compliant_rs::dump::dump_model;
use crfsuite_compliant_rs::model::ModelReader;
use std::path::PathBuf;
use std::process::Command;

fn project_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .to_path_buf()
}

fn c_bin() -> PathBuf { project_root().join("crfsuite/frontend/.libs/crfsuite") }
fn c_lib_path() -> PathBuf { project_root().join("crfsuite/lib/crf/.libs") }

fn c_dump(model_path: &str) -> String {
    let o = Command::new(c_bin())
        .args(["dump", model_path])
        .env("LD_LIBRARY_PATH", c_lib_path())
        .output()
        .expect("failed to run C crfsuite dump");
    assert!(o.status.success());
    String::from_utf8(o.stdout).unwrap()
}

fn rust_dump(model_path: &str) -> String {
    let data = std::fs::read(model_path).unwrap();
    let model = ModelReader::open(&data).unwrap();
    let mut out = Vec::new();
    dump_model(&model, &mut out).unwrap();
    String::from_utf8(out).unwrap()
}

#[test]
fn dump_matches_c_for_lbfgs_model() {
    if !c_bin().exists() {
        eprintln!("SKIP: C binary not found");
        return;
    }
    let model = project_root().join("test_data/model_c.bin");
    let c_out = c_dump(model.to_str().unwrap());
    let rs_out = rust_dump(model.to_str().unwrap());

    if c_out != rs_out {
        // Find first difference
        let c_lines: Vec<&str> = c_out.lines().collect();
        let rs_lines: Vec<&str> = rs_out.lines().collect();
        for (i, (cl, rl)) in c_lines.iter().zip(rs_lines.iter()).enumerate() {
            if cl != rl {
                panic!(
                    "Dump differs at line {}:\n  C:    {:?}\n  Rust: {:?}",
                    i + 1, cl, rl
                );
            }
        }
        if c_lines.len() != rs_lines.len() {
            panic!(
                "Dump line count differs: C={}, Rust={}",
                c_lines.len(), rs_lines.len()
            );
        }
    }
}
