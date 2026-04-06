use crfsuite_ffi::model::Model;

pub struct DumpArgs {
    pub model_path: String,
}

pub fn run_dump(args: DumpArgs) -> Result<(), Box<dyn std::error::Error>> {
    let model = Model::from_file(&args.model_path)?;

    // Get stdout as a C FILE*
    let stdout_file = unsafe { libc::fdopen(1, b"w\0".as_ptr() as *const _) };
    if stdout_file.is_null() {
        return Err("Failed to open stdout as FILE*".into());
    }

    model.dump_to_file(stdout_file)?;

    unsafe { libc::fflush(stdout_file) };

    Ok(())
}
