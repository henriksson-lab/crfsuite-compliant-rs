# crfsuite-rs

A pure Rust implementation of [CRFsuite](http://www.chokkan.org/software/crfsuite/) — Conditional Random Fields for labeling sequential data.

This is a faithful port of Naoaki Okazaki's CRFsuite 0.12, producing **binary-compatible model files** and **identical inference results**. The Rust implementation is **1.1–1.8x faster** than the original C code.

Based on [chokkan/crfsuite](https://github.com/chokkan/crfsuite) version **0.12.2**, commit [`a2a1547`](https://github.com/chokkan/crfsuite/commit/a2a1547727985e3aff6a35cffe073f57f0223e9d).

Particular has been put into ensuring that this crate gives exactly the same results as the original version, but compiler differences, CPUs, etc, may still
affect this. Please report any differences!

As this implementation is pure Rust, it is suited for use in webassembly, and you can also use it directly as a library!

## Features

- **1st-order linear-chain CRF** with dyad (state + transition) features
- **5 training algorithms**: L-BFGS, L2-SGD, Averaged Perceptron, Passive-Aggressive, AROW
- **Binary-compatible** with C CRFsuite model files — models trained by either implementation can be used by the other
- **86 tests** including conformance tests that verify identical output against the original C implementation

## Installation

```bash
# As a library
cargo add crfsuite-compliant-rs

# As a CLI tool
cargo install crfsuite-compliant-rs
```

## Library usage

Add to your `Cargo.toml`:

```toml
[dependencies]
crfsuite-compliant-rs = { version = "0.1", default-features = false }
```

(Use `default-features = false` to avoid pulling in `clap` when you only need the library.)

### Training a model

```rust
use crfsuite_compliant_rs::crf1d::encode::Crf1dEncoder;
use crfsuite_compliant_rs::quark::Quark;
use crfsuite_compliant_rs::train;
use crfsuite_compliant_rs::types::{Attribute, Item, Instance};

// Build training data
let mut labels = Quark::new();
let mut attrs = Quark::new();

let mut instance = Instance::new();
instance.items.push(Item {
    contents: vec![
        Attribute { aid: attrs.get("w=the"), value: 1.0 },
        Attribute { aid: attrs.get("pos=DT"), value: 1.0 },
    ],
});
instance.labels.push(labels.get("B-NP"));

instance.items.push(Item {
    contents: vec![
        Attribute { aid: attrs.get("w=cat"), value: 1.0 },
        Attribute { aid: attrs.get("pos=NN"), value: 1.0 },
    ],
});
instance.labels.push(labels.get("I-NP"));

let instances = vec![instance];

// Initialize encoder and train
let mut encoder = Crf1dEncoder::new(
    &instances,
    labels.num(),    // num_labels
    attrs.num(),     // num_attrs
    0.0,             // min_freq
    false, false,    // possible_states, possible_transitions
);

let mut log = train::stdout_logger();
let weights = train::lbfgs::train_lbfgs(
    &mut encoder, &instances,
    0.0, 1.0,           // c1 (L1), c2 (L2)
    100, 6,              // max_iterations, num_memories
    1e-5, 10, 1e-5,      // epsilon, period, delta
    "MoreThuente", 20,   // linesearch, max_linesearch
    &mut log,
);

// Save model
let label_strings: Vec<String> = (0..labels.num())
    .map(|i| labels.to_string(i as i32).unwrap().to_string())
    .collect();
let attr_strings: Vec<String> = (0..attrs.num())
    .map(|i| attrs.to_string(i as i32).unwrap().to_string())
    .collect();
let model_bytes = encoder.save_model(&weights, &label_strings, &attr_strings);
std::fs::write("model.bin", &model_bytes).unwrap();
```

### Loading a model and tagging

```rust
use crfsuite_compliant_rs::model::ModelReader;
use crfsuite_compliant_rs::crf1d::tag::Crf1dTagger;
use crfsuite_compliant_rs::types::{Attribute, Item, Instance};

// Load model
let data = std::fs::read("model.bin").unwrap();
let model = ModelReader::open(&data).unwrap();
let mut tagger = Crf1dTagger::new(&model);

// Build an instance to tag
let mut inst = Instance::new();
inst.items.push(Item {
    contents: vec![
        Attribute { aid: model.to_aid("w=the").unwrap_or(-1), value: 1.0 },
        Attribute { aid: model.to_aid("pos=DT").unwrap_or(-1), value: 1.0 },
    ],
});
inst.labels.push(0); // placeholder

inst.items.push(Item {
    contents: vec![
        Attribute { aid: model.to_aid("w=cat").unwrap_or(-1), value: 1.0 },
        Attribute { aid: model.to_aid("pos=NN").unwrap_or(-1), value: 1.0 },
    ],
});
inst.labels.push(0);

// Tag
tagger.set(&inst);
let (labels, score) = tagger.viterbi();
for (t, label_id) in labels.iter().enumerate() {
    let label = model.to_label(*label_id).unwrap_or("?");
    println!("Item {}: {}", t, label);
}
```

## CLI usage

### Training

```bash
# Train with L-BFGS (default)
crfsuite-rs learn -a lbfgs -m model.bin train.txt

# Train with other algorithms
crfsuite-rs learn -a l2sgd -m model.bin train.txt
crfsuite-rs learn -a ap -m model.bin train.txt       # averaged perceptron
crfsuite-rs learn -a pa -m model.bin train.txt       # passive-aggressive
crfsuite-rs learn -a arow -m model.bin train.txt

# Set algorithm parameters
crfsuite-rs learn -a lbfgs -p c2=0.1 -p max_iterations=200 -m model.bin train.txt
```

### Tagging

```bash
# Basic tagging
crfsuite-rs tag -m model.bin test.txt

# With reference labels and evaluation
crfsuite-rs tag -m model.bin -t test.txt

# With probability scores
crfsuite-rs tag -m model.bin -p test.txt

# With marginal probabilities
crfsuite-rs tag -m model.bin -i test.txt      # predicted label marginal
crfsuite-rs tag -m model.bin -l test.txt      # all label marginals
```

### Model inspection

```bash
crfsuite-rs dump model.bin
```

## Data format

Input uses the IWA (Item With Attributes) format — tab-separated fields, one item per line, blank lines between sequences:

```
LABEL1	attr1:value1	attr2:value2	attr3
LABEL2	attr1:value1	attr4

LABEL1	attr2:value2	attr5:0.5
LABEL3	attr1:value1
```

- First field is the label
- Remaining fields are `attribute:value` pairs (value defaults to 1.0 if omitted)
- Blank lines separate sequences
- `\:` and `\\` escape colons and backslashes in attribute names

## Performance

Benchmarked on 10,000 sequences (~115k items), compiled with `target-cpu=native`:

| Task | C (original) | Rust | Speedup |
|---|---|---|---|
| Tag (plain labels) | 0.63s | 0.42s | **1.5x faster** |
| Tag (scores + marginals) | 0.53s | 0.30s | **1.8x faster** |
| Train L-BFGS | 3.84s | 3.55s | **1.08x faster** |
| Train AROW | 1.14s | 0.80s | **1.4x faster** |
| Train Avg Perceptron | 0.73s | 0.75s | ~parity |
| Train Passive-Aggressive | 0.86s | 0.83s | ~parity |

## Project structure

```
src/
  lib.rs             Library root
  types.rs           Attribute, Item, Instance types
  model.rs           Binary model reader
  model_writer.rs    Binary model writer
  dump.rs            Human-readable model dump
  quark.rs           String-to-ID dictionary
  vecmath.rs         Vector math with exact SSE2 exp() polynomial
  params.rs          Algorithm parameter store
  dataset.rs         Dataset with shuffle
  cqdb/              Constant Quark Database (model string lookup)
  crf1d/
    context.rs       Forward-backward algorithm, Viterbi decoding
    encode.rs        Training encoder (gradient computation)
    feature.rs       Feature extraction
    tag.rs           Tagger (inference)
  train/             Training algorithms (lbfgs, l2sgd, ap, pa, arow)
  bin/crfsuite-rs/   CLI binary
tests/               Conformance tests vs original C implementation
```

## Testing

```bash
# Run all tests
cargo test

# Conformance tests against the C binary (optional, requires building crfsuite/)
# These skip automatically if the C binary is not present.
```

The conformance test suite verifies against the original C binary:
- Tag output identical (labels, scores, marginals) across all flag combinations
- Dump output byte-identical
- Cross-implementation: models trained by one can be tagged by the other
- Tested with all 5 training algorithms

## Compatibility

- Reads and writes the same binary model format as CRFsuite 0.12
- Models are interchangeable between C and Rust implementations
- The `vecexp` polynomial approximation is reproduced exactly (bit-identical to the C SSE2 implementation) to ensure numerical consistency

## License

This is a port of CRFsuite by Naoaki Okazaki, licensed under the modified BSD license. See the original `crfsuite/` directory for the full license text.
