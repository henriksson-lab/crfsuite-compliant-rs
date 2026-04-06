pub struct DumpArgs {
    pub model_path: String,
}

pub fn run_dump(args: DumpArgs) -> Result<(), Box<dyn std::error::Error>> {
    #[cfg(feature = "pure-rust-dump")]
    {
        let data = std::fs::read(&args.model_path)?;
        let model = crfsuite_core::model::ModelReader::open(&data)
            .ok_or("Failed to open model")?;
        let mut stdout = std::io::stdout();
        crfsuite_core::dump::dump_model(&model, &mut stdout)?;
        return Ok(());
    }

    #[cfg(all(not(feature = "pure-rust-dump"), feature = "ffi"))]
    {
        let model = crfsuite_ffi::model::Model::from_file(&args.model_path)?;
        let stdout_file = unsafe { libc::fdopen(1, b"w\0".as_ptr() as *const _) };
        if stdout_file.is_null() {
            return Err("Failed to open stdout as FILE*".into());
        }
        model.dump_to_file(stdout_file)?;
        unsafe { libc::fflush(stdout_file) };
        return Ok(());
    }

    #[cfg(all(not(feature = "pure-rust-dump"), not(feature = "ffi")))]
    {
        Err("Enable either 'pure-rust-dump' or 'ffi' feature".into())
    }
}
