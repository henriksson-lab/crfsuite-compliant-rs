fn main() {
    cc::Build::new()
        .file("src/logging_shim.c")
        .compile("logging_shim");
}
