use std::fs::File;
use std::io::{self, BufRead, BufReader, Write};
use std::time::Instant;

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
    #[cfg(feature = "pure-rust-tag")]
    return run_tag_pure_rust(args);

    #[cfg(all(not(feature = "pure-rust-tag"), feature = "ffi"))]
    return run_tag_ffi(args);

    #[cfg(all(not(feature = "pure-rust-tag"), not(feature = "ffi")))]
    return Err("Enable either 'pure-rust-tag' or 'ffi' feature".into());
}

// ── Pure Rust implementation ──────────────────���─────────────────────────────

#[cfg(feature = "pure-rust-tag")]
fn run_tag_pure_rust(args: TagArgs) -> Result<(), Box<dyn std::error::Error>> {
    use crfsuite_core::model::ModelReader;
    use crfsuite_core::crf1d::tag::Crf1dTagger;
    use crfsuite_core::types::{Attribute, Item, Instance};

    let mut fpo = io::stdout();

    let model_data = std::fs::read(&args.model_path)?;
    let model = ModelReader::open(&model_data).ok_or("Failed to open model")?;
    let mut tagger = Crf1dTagger::new(&model);

    let num_labels = model.num_labels() as i32;

    // Simple evaluation state
    let mut eval_correct = vec![0i32; num_labels as usize];
    let mut eval_model = vec![0i32; num_labels as usize];
    let mut eval_obs = vec![0i32; num_labels as usize];
    let mut item_correct = 0i32;
    let mut item_total = 0i32;
    let mut inst_correct_count = 0i32;
    let mut inst_total = 0i32;

    let input: Box<dyn BufRead> = match &args.input_file {
        Some(path) => Box::new(BufReader::new(File::open(path)?)),
        None => Box::new(BufReader::new(io::stdin())),
    };

    let start = Instant::now();
    let mut num_instances = 0i32;

    let mut iwa = IwaReader::new(input);
    let mut inst = Instance::new();
    let mut current_item = Item { contents: Vec::new() };
    let mut ref_labels: Vec<i32> = Vec::new();
    let mut current_label: i32 = -1;
    let mut is_first_field = true;

    loop {
        let token = iwa.read();
        match token.token_type {
            TokenType::Boi => {
                current_label = -1;
                is_first_field = true;
                current_item.contents.clear();
            }
            TokenType::Item => {
                if is_first_field {
                    is_first_field = false;
                    current_label = model.to_lid(&token.attr).unwrap_or(-1);
                } else {
                    let aid = model.to_aid(&token.attr).unwrap_or(-1);
                    if aid >= 0 {
                        let value = if token.value.is_empty() {
                            1.0
                        } else {
                            token.value.parse::<f64>().unwrap_or(1.0)
                        };
                        current_item.contents.push(Attribute { aid, value });
                    }
                }
            }
            TokenType::Eoi => {
                ref_labels.push(current_label);
                inst.items.push(current_item.clone());
                inst.labels.push(current_label);
            }
            TokenType::None | TokenType::Eof => {
                if !inst.items.is_empty() {
                    let n = inst.num_items();
                    num_instances += 1;

                    tagger.set(&inst);
                    let (output_labels, score) = tagger.viterbi();

                    if !args.quiet {
                        if args.probability {
                            let lognorm = tagger.lognorm();
                            writeln!(fpo, "@score\t{:.6}\t{:.6}", score, lognorm)?;
                            writeln!(fpo, "@probability\t{:.6}", (score - lognorm).exp())?;
                        }

                        for t in 0..n {
                            if args.reference && t < ref_labels.len() {
                                let ref_label = if ref_labels[t] >= 0 {
                                    model.to_label(ref_labels[t]).unwrap_or("").to_string()
                                } else {
                                    String::new()
                                };
                                write!(fpo, "{}\t", ref_label)?;
                            }

                            let pred_label = model.to_label(output_labels[t])
                                .unwrap_or("?");
                            write!(fpo, "{}", pred_label)?;

                            if args.marginal {
                                let prob = tagger.marginal_point(output_labels[t], t as i32);
                                write!(fpo, ":{:.6}", prob)?;
                            }

                            if args.marginal_all {
                                for l in 0..num_labels {
                                    let prob = tagger.marginal_point(l, t as i32);
                                    let lname = model.to_label(l).unwrap_or("?");
                                    write!(fpo, "\t{}:{:.6}", lname, prob)?;
                                }
                            }

                            writeln!(fpo)?;
                        }
                        writeln!(fpo)?;
                    }

                    // Evaluation
                    if args.test && ref_labels.len() == n {
                        let mut all_correct = true;
                        for t in 0..n {
                            let pred = output_labels[t];
                            let gold = ref_labels[t];
                            if pred >= 0 && (pred as usize) < eval_model.len() {
                                eval_model[pred as usize] += 1;
                            }
                            if gold >= 0 && (gold as usize) < eval_obs.len() {
                                eval_obs[gold as usize] += 1;
                            }
                            item_total += 1;
                            if pred == gold {
                                item_correct += 1;
                                if pred >= 0 && (pred as usize) < eval_correct.len() {
                                    eval_correct[pred as usize] += 1;
                                }
                            } else {
                                all_correct = false;
                            }
                        }
                        inst_total += 1;
                        if all_correct {
                            inst_correct_count += 1;
                        }
                    }

                    inst.items.clear();
                    inst.labels.clear();
                    ref_labels.clear();
                }

                if token.token_type == TokenType::Eof {
                    break;
                }
            }
        }
    }

    // Print evaluation results
    if args.test {
        writeln!(fpo, "Performance by label (#match, #model, #ref) (precision, recall, F1):")?;

        let mut macro_p = 0.0f64;
        let mut macro_r = 0.0f64;
        let mut macro_f = 0.0f64;

        for l in 0..num_labels as usize {
            let lname = model.to_label(l as i32).unwrap_or("?");
            let c = eval_correct[l] as f64;
            let m = eval_model[l] as f64;
            let o = eval_obs[l] as f64;
            let p = if m > 0.0 { c / m } else { 0.0 };
            let r = if o > 0.0 { c / o } else { 0.0 };
            let f = if p + r > 0.0 { 2.0 * p * r / (p + r) } else { 0.0 };
            macro_p += p;
            macro_r += r;
            macro_f += f;

            if eval_obs[l] == 0 {
                writeln!(fpo, "    {}: ({}, {}, {}) (******, ******, ******)",
                    lname, eval_correct[l], eval_model[l], eval_obs[l])?;
            } else {
                writeln!(fpo, "    {}: ({}, {}, {}) ({:.4}, {:.4}, {:.4})",
                    lname, eval_correct[l], eval_model[l], eval_obs[l], p, r, f)?;
            }
        }

        let nl = num_labels as f64;
        macro_p /= nl;
        macro_r /= nl;
        macro_f /= nl;

        writeln!(fpo, "Macro-average precision, recall, F1: ({:.6}, {:.6}, {:.6})",
            macro_p, macro_r, macro_f)?;

        let item_acc = if item_total > 0 { item_correct as f64 / item_total as f64 } else { 0.0 };
        writeln!(fpo, "Item accuracy: {} / {} ({:.4})", item_correct, item_total, item_acc)?;

        let inst_acc = if inst_total > 0 { inst_correct_count as f64 / inst_total as f64 } else { 0.0 };
        writeln!(fpo, "Instance accuracy: {} / {} ({:.4})", inst_correct_count, inst_total, inst_acc)?;

        let elapsed = start.elapsed().as_secs_f64();
        writeln!(fpo, "Elapsed time: {:.6} [sec] ({:.1} [instance/sec])",
            elapsed, num_instances as f64 / elapsed)?;
    }

    Ok(())
}

// ── FFI implementation (original) ───────────────────────��───────────────────

#[cfg(all(not(feature = "pure-rust-tag"), feature = "ffi"))]
fn run_tag_ffi(args: TagArgs) -> Result<(), Box<dyn std::error::Error>> {
    use crfsuite_ffi::data::{InstanceBuilder, ItemBuilder};
    use crfsuite_ffi::evaluation::Evaluation;
    use crfsuite_ffi::model::Model;

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

    if let Some(ref mut ev) = eval {
        ev.finalize();
        writeln!(fpo, "Performance by label (#match, #model, #ref) (precision, recall, F1):")?;

        for l in 0..num_labels {
            if let Some(le) = ev.label_eval(l) {
                let lname = labels.to_string(l).unwrap_or_default();
                writeln!(fpo, "    {}: ({}, {}, {}) ({:.4}, {:.4}, {:.4})",
                    lname, le.num_correct, le.num_model, le.num_observation,
                    le.precision, le.recall, le.fmeasure)?;
            }
        }

        writeln!(fpo, "Macro-average precision, recall, F1: ({:.6}, {:.6}, {:.6})",
            ev.macro_precision(), ev.macro_recall(), ev.macro_fmeasure())?;
        writeln!(fpo, "Item accuracy: {} / {} ({:.4})",
            ev.item_total_correct(), ev.item_total_num(), ev.item_accuracy())?;
        writeln!(fpo, "Instance accuracy: {} / {} ({:.4})",
            ev.inst_total_correct(), ev.inst_total_num(), ev.inst_accuracy())?;

        let elapsed = start.elapsed().as_secs_f64();
        writeln!(fpo, "Elapsed time: {:.6} [sec] ({:.1} [instance/sec])",
            elapsed, num_instances as f64 / elapsed)?;
    }

    Ok(())
}
