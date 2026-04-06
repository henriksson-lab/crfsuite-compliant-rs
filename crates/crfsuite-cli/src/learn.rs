use std::fs::File;
use std::io::{self, BufReader, Write};

use crfsuite_ffi::data::Data;
use crfsuite_ffi::dictionary::Dictionary;
use crfsuite_ffi::trainer::Trainer;
use crfsuite_sys;

use crate::reader::read_data;

pub struct LearnArgs {
    pub model_type: String,
    pub algorithm: String,
    pub model_path: String,
    pub params: Vec<(String, String)>,
    pub split: i32,
    pub holdout: i32,
    pub cross_validate: bool,
    pub help_params: bool,
    pub input_files: Vec<String>,
}

pub fn run_learn(args: LearnArgs) -> Result<(), Box<dyn std::error::Error>> {
    let mut fpo = io::stdout();

    // Map short algorithm names to full names
    let algorithm = match args.algorithm.as_str() {
        "ap" | "averaged-perceptron" => "averaged-perceptron",
        "pa" | "passive-aggressive" => "passive-aggressive",
        other => other,
    };

    // Create trainer
    let trainer = Trainer::new(&args.model_type, algorithm)?;

    // If help requested, show parameters and exit
    if args.help_params {
        let params = trainer.params()?;
        let n = params.num();
        for i in 0..n {
            let name = params.name(i)?;
            if let Ok((ptype, help)) = params.help(&name) {
                writeln!(fpo, "{} ({}): {}", name, ptype, help)?;
            }
        }
        return Ok(());
    }

    // Set trainer parameters
    {
        let params = trainer.params()?;
        for (name, value) in &args.params {
            params.set(name, value)?;
        }
    }

    // Set logging to stdout
    trainer.set_stdout_logging();

    // Create dictionaries
    // We need to create dictionaries via crfsuite_create_instance
    let mut attrs_ptr: *mut std::os::raw::c_void = std::ptr::null_mut();
    let mut labels_ptr: *mut std::os::raw::c_void = std::ptr::null_mut();

    let ret = unsafe {
        crfsuite_sys::crfsuite_create_instance(
            b"dictionary\0".as_ptr() as *const _,
            &mut attrs_ptr,
        )
    };
    if ret == 0 || attrs_ptr.is_null() {
        return Err("Failed to create attribute dictionary".into());
    }

    let ret = unsafe {
        crfsuite_sys::crfsuite_create_instance(
            b"dictionary\0".as_ptr() as *const _,
            &mut labels_ptr,
        )
    };
    if ret == 0 || labels_ptr.is_null() {
        return Err("Failed to create label dictionary".into());
    }

    let attrs = unsafe { Dictionary::from_raw(attrs_ptr as *mut crfsuite_sys::crfsuite_dictionary_t)? };
    let labels = unsafe { Dictionary::from_raw(labels_ptr as *mut crfsuite_sys::crfsuite_dictionary_t)? };

    // Create data set
    let mut data = Data::new();
    data.set_dictionaries(attrs.ptr.as_ptr(), labels.ptr.as_ptr());

    // Read input files
    let input_files = if args.input_files.is_empty() {
        vec!["-".to_string()]
    } else {
        args.input_files.clone()
    };

    let groups = input_files.len() as i32;
    for (i, filename) in input_files.iter().enumerate() {
        let group = if args.split > 0 { 0 } else { i as i32 };
        writeln!(fpo, "[{}] {}", i + 1, filename)?;

        let n = if filename == "-" {
            let stdin = io::stdin();
            read_data(BufReader::new(stdin.lock()), &mut data, &attrs, &labels, group)?
        } else {
            let f = File::open(filename)?;
            read_data(BufReader::new(f), &mut data, &attrs, &labels, group)?
        };

        writeln!(fpo, "Number of instances: {}", n)?;
    }

    writeln!(fpo, "Statistics the data set(s)")?;
    writeln!(fpo, "Number of data sets (groups): {}", groups)?;
    writeln!(fpo, "Number of instances: {}", data.num_instances())?;
    let total_items = unsafe { crfsuite_sys::crfsuite_data_totalitems(data.as_ptr() as *mut _) };
    writeln!(fpo, "Number of items: {}", total_items)?;
    writeln!(fpo, "Number of attributes: {}", attrs.num())?;
    writeln!(fpo, "Number of labels: {}", labels.num())?;
    writeln!(fpo)?;

    // Handle split/shuffle for cross-validation
    // TODO: implement random shuffle + group assignment for --split flag

    // Train
    if args.cross_validate {
        let num_groups = if args.split > 0 { args.split } else { groups };
        for i in 0..num_groups {
            writeln!(fpo, "===== Cross validation ({}/{}) =====", i + 1, num_groups)?;
            trainer.train(data.as_ptr(), "", i)?;
        }
    } else {
        trainer.train(data.as_ptr(), &args.model_path, args.holdout)?;
    }

    // Don't let Data drop the dictionaries since Dictionary wrappers own them
    // (they were set via set_dictionaries which just copies the pointer)
    // Actually, Data::drop calls crfsuite_data_finish which frees instances but not dicts.
    // The Dictionary wrappers will release via their own Drop.

    Ok(())
}
