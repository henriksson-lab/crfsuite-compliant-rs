use std::fs::File;
use std::io::{self, BufReader};
use std::os::raw::{c_int, c_uint};

use crfsuite_compliant_rs::quark::Quark;
use crfsuite_compliant_rs::types::{Attribute, Instance, Item};

use crate::iwa::{atof, IwaReader, TokenType};

unsafe extern "C" {
    fn srand(seed: c_uint);
    fn rand() -> c_int;
}

pub struct LoadedData {
    pub instances: Vec<Instance>,
    pub num_labels: usize,
    pub num_attrs: usize,
    pub label_strings: Vec<String>,
    pub attr_strings: Vec<String>,
    pub groups: i32,
}

pub fn load_training_data(
    input_files: &[String],
    split: i32,
) -> Result<LoadedData, Box<dyn std::error::Error>> {
    let mut label_quark = Quark::new();
    let mut attr_quark = Quark::new();
    let mut instances: Vec<Instance> = Vec::new();
    let input_files = input_files.to_vec();

    for (fi, filename) in input_files.iter().enumerate() {
        let group = if split > 0 { 0 } else { fi as i32 };
        let input: Box<dyn io::BufRead> = if filename == "-" {
            Box::new(BufReader::new(io::stdin()))
        } else {
            let file =
                File::open(filename).map_err(|_| format!("Failed to open the data set: {}", filename))?;
            Box::new(BufReader::new(file))
        };

        let mut iwa = IwaReader::new(input);
        let mut inst = Instance::new();
        inst.group = group;
        let mut current_item = Item {
            contents: Vec::new(),
        };
        let mut label_id: i32 = -1;
        let mut is_first_field = true;

        loop {
            let token = iwa.read();
            match token.token_type {
                TokenType::Boi => {
                    label_id = -1;
                    is_first_field = true;
                    current_item.contents.clear();
                }
                TokenType::Item => {
                    if is_first_field {
                        is_first_field = false;
                        if token.attr.starts_with('@') {
                            if token.attr == "@weight" {
                                inst.weight = atof(&token.value);
                            } else {
                                return Err(
                                    format!("unrecognized declaration: {}", token.attr).into()
                                );
                            }
                            label_id = -2;
                        } else {
                            label_id = label_quark.get(&token.attr);
                        }
                    } else if label_id != -2 {
                        let aid = attr_quark.get(&token.attr);
                        let value = if token.value.is_empty() {
                            1.0
                        } else {
                            atof(&token.value)
                        };
                        current_item.contents.push(Attribute { aid, value });
                    }
                }
                TokenType::Eoi => {
                    if label_id >= 0 {
                        inst.items.push(current_item.clone());
                        inst.labels.push(label_id);
                    }
                }
                TokenType::None | TokenType::Eof => {
                    if !inst.items.is_empty() {
                        instances.push(inst.clone());
                        inst = Instance::new();
                        inst.group = group;
                    }
                    if token.token_type == TokenType::Eof {
                        break;
                    }
                }
            }
        }
    }

    if split > 0 {
        let n = instances.len();
        unsafe {
            srand(0);
        }
        for i in 0..n {
            let j = unsafe { rand() as usize } % n;
            instances.swap(i, j);
        }
        for (i, inst) in instances.iter_mut().enumerate() {
            inst.group = i as i32 % split;
        }
    }

    let num_labels = label_quark.num();
    let num_attrs = attr_quark.num();
    let label_strings: Vec<String> = (0..num_labels)
        .map(|i| label_quark.to_string(i as i32).unwrap().to_string())
        .collect();
    let attr_strings: Vec<String> = (0..num_attrs)
        .map(|i| attr_quark.to_string(i as i32).unwrap().to_string())
        .collect();
    let groups = if split > 0 {
        split
    } else {
        input_files.len() as i32
    };

    Ok(LoadedData {
        instances,
        num_labels,
        num_attrs,
        label_strings,
        attr_strings,
        groups,
    })
}
