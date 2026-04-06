use std::fs::File;
use std::io::{self, BufRead, BufReader, Write};
use std::time::Instant;

use crfsuite_ffi::data::{InstanceBuilder, ItemBuilder};
use crfsuite_ffi::evaluation::Evaluation;
use crfsuite_ffi::model::Model;

use crate::iwa::{IwaReader, TokenType};

pub struct TagArgs {
    pub model_path: String,
    pub test: bool,
    pub reference: bool,
    pub probability: bool,
    pub marginal: bool,
    pub marginal_all: bool,
    pub quiet: bool,
    pub input_file: Option<String>,
}

pub fn run_tag(args: TagArgs) -> Result<(), Box<dyn std::error::Error>> {
    let mut fpo = io::stdout();

    let model = Model::from_file(&args.model_path)?;
    let tagger = model.get_tagger()?;
    let labels = model.get_labels()?;
    let attrs = model.get_attrs()?;

    let num_labels = labels.num();
    let mut eval = if args.test {
        Some(Evaluation::new(num_labels))
    } else {
        None
    };

    let input: Box<dyn BufRead> = match &args.input_file {
        Some(path) => Box::new(BufReader::new(File::open(path)?)),
        None => Box::new(BufReader::new(io::stdin())),
    };

    let start = Instant::now();
    let mut num_instances = 0i32;
    let mut _num_items_total = 0i32;

    let mut iwa = IwaReader::new(input);
    let mut inst = InstanceBuilder::new();
    let mut item = ItemBuilder::new();
    let mut ref_labels: Vec<i32> = Vec::new();
    let mut current_label: i32 = -1;
    let mut is_first_field = true;

    loop {
        let token = iwa.read();
        match token.token_type {
            TokenType::Boi => {
                current_label = -1;
                is_first_field = true;
                item.reset();
            }
            TokenType::Item => {
                if is_first_field {
                    is_first_field = false;
                    current_label = labels.to_id(&token.attr)?;
                } else {
                    let aid = attrs.to_id(&token.attr)?;
                    if aid >= 0 {
                        let value = if token.value.is_empty() {
                            1.0
                        } else {
                            token.value.parse::<f64>().unwrap_or(1.0)
                        };
                        item.append_attribute(aid, value)?;
                    }
                }
            }
            TokenType::Eoi => {
                ref_labels.push(current_label);
                inst.append(&item.inner, current_label)?;
            }
            TokenType::None | TokenType::Eof => {
                if !inst.is_empty() {
                    let n = inst.num_items() as usize;
                    _num_items_total += n as i32;
                    num_instances += 1;
                    tagger.set_instance(inst.as_mut_ptr())?;
                    let (output_labels, score) = tagger.viterbi(n)?;

                    if !args.quiet {
                        if args.probability {
                            let lognorm = tagger.lognorm()?;
                            writeln!(fpo, "@score\t{:.6}\t{:.6}", score, lognorm)?;
                            writeln!(fpo, "@probability\t{:.6}", (score - lognorm).exp())?;
                        }

                        for t in 0..n {
                            if args.reference && t < ref_labels.len() {
                                let ref_label = if ref_labels[t] >= 0 {
                                    labels.to_string(ref_labels[t]).unwrap_or_default()
                                } else {
                                    String::new()
                                };
                                write!(fpo, "{}\t", ref_label)?;
                            }

                            let pred_label = labels.to_string(output_labels[t])
                                .unwrap_or_else(|_| format!("{}", output_labels[t]));
                            write!(fpo, "{}", pred_label)?;

                            if args.marginal {
                                let prob = tagger.marginal_point(output_labels[t], t as i32)?;
                                write!(fpo, ":{:.6}", prob)?;
                            }

                            if args.marginal_all {
                                for l in 0..num_labels {
                                    let prob = tagger.marginal_point(l, t as i32)?;
                                    let lname = labels.to_string(l).unwrap_or_default();
                                    write!(fpo, "\t{}:{:.6}", lname, prob)?;
                                }
                            }

                            writeln!(fpo)?;
                        }
                        writeln!(fpo)?;
                    }

                    if let Some(ref mut ev) = eval {
                        if ref_labels.len() == n {
                            ev.accumulate(&ref_labels, &output_labels)?;
                        }
                    }

                    inst.reset();
                    ref_labels.clear();
                }

                if token.token_type == TokenType::Eof {
                    break;
                }
            }
        }
    }

    // Print evaluation results
    if let Some(ref mut ev) = eval {
        ev.finalize();
        writeln!(fpo, "Performance by label (#match, #model, #ref) (precision, recall, F1):")?;

        for l in 0..num_labels {
            if let Some(le) = ev.label_eval(l) {
                let lname = labels.to_string(l).unwrap_or_default();
                writeln!(
                    fpo,
                    "    {}: ({}, {}, {}) ({:.4}, {:.4}, {:.4})",
                    lname, le.num_correct, le.num_model, le.num_observation,
                    le.precision, le.recall, le.fmeasure
                )?;
            }
        }

        writeln!(
            fpo,
            "Macro-average precision, recall, F1: ({:.6}, {:.6}, {:.6})",
            ev.macro_precision(), ev.macro_recall(), ev.macro_fmeasure()
        )?;
        writeln!(
            fpo,
            "Item accuracy: {} / {} ({:.4})",
            ev.item_total_correct(), ev.item_total_num(), ev.item_accuracy()
        )?;
        writeln!(
            fpo,
            "Instance accuracy: {} / {} ({:.4})",
            ev.inst_total_correct(), ev.inst_total_num(), ev.inst_accuracy()
        )?;

        let elapsed = start.elapsed().as_secs_f64();
        writeln!(
            fpo,
            "Elapsed time: {:.6} [sec] ({:.1} [instance/sec])",
            elapsed,
            num_instances as f64 / elapsed
        )?;
    }

    Ok(())
}
