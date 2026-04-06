use std::io::BufRead;

use crfsuite_ffi::data::{Data, InstanceBuilder, ItemBuilder};
use crfsuite_ffi::dictionary::Dictionary;
use crfsuite_ffi::error::CrfError;

use crate::iwa::{IwaReader, TokenType};

/// Read training data in IWA format. Returns the number of instances read.
pub fn read_data<R: BufRead>(
    reader: R,
    data: &mut Data,
    attrs: &Dictionary,
    labels: &Dictionary,
    group: i32,
) -> Result<i32, CrfError> {
    let mut iwa = IwaReader::new(reader);
    let mut inst = InstanceBuilder::new();
    inst.set_group(group);
    let mut item = ItemBuilder::new();
    let mut label_id: i32 = -1;
    let mut count: i32 = 0;

    loop {
        let token = iwa.read();
        match token.token_type {
            TokenType::Boi => {
                label_id = -1;
                item.reset();
            }
            TokenType::Item => {
                if label_id == -1 {
                    // First field is the label
                    if token.attr.starts_with('@') {
                        // Special declarations
                        if token.attr == "@weight" {
                            if let Ok(w) = token.value.parse::<f64>() {
                                inst.set_weight(w);
                            }
                        }
                        label_id = -2; // Mark as declaration, skip further processing
                    } else {
                        label_id = labels.get(&token.attr)?;
                    }
                } else if label_id != -2 {
                    // Attribute field
                    let aid = attrs.get(&token.attr)?;
                    let value = if token.value.is_empty() {
                        1.0
                    } else {
                        token.value.parse::<f64>().unwrap_or(1.0)
                    };
                    item.append_attribute(aid, value)?;
                }
            }
            TokenType::Eoi => {
                // End of item: append to instance
                if label_id >= 0 {
                    inst.append(&item.inner, label_id)?;
                }
            }
            TokenType::None | TokenType::Eof => {
                // End of sequence: append instance to data
                if !inst.is_empty() {
                    data.append_instance(&inst.inner)?;
                    count += 1;
                    inst.reset();
                    inst.set_group(group);
                }
                if token.token_type == TokenType::Eof {
                    break;
                }
            }
        }
    }

    Ok(count)
}
