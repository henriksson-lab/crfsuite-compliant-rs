use std::env;
use std::path::PathBuf;

fn main() {
    let crfsuite_root = PathBuf::from(env::var("CARGO_MANIFEST_DIR").unwrap())
        .join("../../crfsuite");

    let include_dir = crfsuite_root.join("include");
    let cqdb_include = crfsuite_root.join("lib/cqdb/include");
    let crf_src = crfsuite_root.join("lib/crf/src");
    let cqdb_src = crfsuite_root.join("lib/cqdb/src");

    // Get liblbfgs include path from DEP_LBFGS_INCLUDE env var set by liblbfgs-sys
    let lbfgs_include = env::var("DEP_LBFGS_INCLUDE")
        .unwrap_or_else(|_| "/usr/include".to_string());

    // Compile CQDB sources
    cc::Build::new()
        .file(cqdb_src.join("cqdb.c"))
        .file(cqdb_src.join("lookup3.c"))
        .include(&include_dir)
        .include(&cqdb_include)
        .std("gnu99")
        .opt_level(3)
        .flag("-fomit-frame-pointer")
        .flag("-ffast-math")
        .warnings(false)
        .compile("cqdb");

    // Compile CRFsuite library sources
    let crf_sources = [
        "crf1d_context.c",
        "crf1d_encode.c",
        "crf1d_feature.c",
        "crf1d_model.c",
        "crf1d_tag.c",
        "crfsuite.c",
        "crfsuite_train.c",
        "dataset.c",
        "dictionary.c",
        "holdout.c",
        "logging.c",
        "params.c",
        "quark.c",
        "rumavl.c",
        "train_arow.c",
        "train_averaged_perceptron.c",
        "train_l2sgd.c",
        "train_lbfgs.c",
        "train_passive_aggressive.c",
    ];

    let mut build = cc::Build::new();
    for src in &crf_sources {
        build.file(crf_src.join(src));
    }
    build
        .include(&include_dir)
        .include(&cqdb_include)
        .include(&crf_src)
        .include(&lbfgs_include)
        .std("gnu99")
        .opt_level(3)
        .flag("-fomit-frame-pointer")
        .flag("-ffast-math")
        .flag("-msse2")
        .flag("-mfpmath=sse")
        .define("USE_SSE", None)
        .warnings(false)
        .compile("crfsuite");

    println!("cargo:rustc-link-lib=m");

    // Generate bindings
    let bindings = bindgen::Builder::default()
        .header(include_dir.join("crfsuite.h").to_str().unwrap())
        .clang_arg(format!("-I{}", include_dir.display()))
        .clang_arg(format!("-I{}", cqdb_include.display()))
        .allowlist_type("crfsuite_.*")
        .allowlist_type("tag_crfsuite_.*")
        .allowlist_function("crfsuite_.*")
        .allowlist_var("CRFSUITE.*")
        .allowlist_var("CRFSUITEERR.*")
        // Treat floatval_t as f64
        .blocklist_type("floatval_t")
        .raw_line("pub type floatval_t = f64;")
        .derive_debug(true)
        .derive_default(true)
        .generate()
        .expect("Unable to generate bindings");

    let out_path = PathBuf::from(env::var("OUT_DIR").unwrap());
    bindings
        .write_to_file(out_path.join("bindings.rs"))
        .expect("Couldn't write bindings!");
}
