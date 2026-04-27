pub struct DumpArgs {
    pub model_path: Option<String>,
}

pub fn run_dump(args: DumpArgs) -> Result<(), Box<dyn std::error::Error>> {
    let Some(model_path) = args.model_path else {
        return Err("No model specified.".into());
    };

    let data = std::fs::read(&model_path).unwrap_or_else(|_| std::process::exit(3));
    let model = crfsuite_compliant_rs::model::ModelReader::open(&data)
        .unwrap_or_else(|| std::process::exit(3));
    let stdout = std::io::stdout();
    let mut stdout = std::io::BufWriter::new(stdout.lock());
    crfsuite_compliant_rs::dump::dump_model(&model, &mut stdout)?;
    Ok(())
}
