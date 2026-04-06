pub struct DumpArgs {
    pub model_path: String,
}

pub fn run_dump(args: DumpArgs) -> Result<(), Box<dyn std::error::Error>> {
    let data = std::fs::read(&args.model_path)?;
    let model = crfsuite_compliant_rs::model::ModelReader::open(&data)
        .ok_or("Failed to open model")?;
    let mut stdout = std::io::stdout();
    crfsuite_compliant_rs::dump::dump_model(&model, &mut stdout)?;
    Ok(())
}
