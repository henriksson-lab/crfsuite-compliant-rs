fn main() {
    // Compile C helper for cross-validation testing of vecexp
    cc::Build::new()
        .file("c_helpers/vecexp_helper.c")
        .flag("-msse2")
        .flag("-mfpmath=sse")
        .flag("-O3")
        .warnings(false)
        .compile("vecexp_helper");
}
