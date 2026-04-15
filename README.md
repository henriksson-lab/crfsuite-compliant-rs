# crfsuite-rs

A pure Rust implementation of [CRFsuite](http://www.chokkan.org/software/crfsuite/) — Conditional Random Fields for labeling sequential data.

This is a Rust port of Naoaki Okazaki's CRFsuite 0.12, focused on **binary-compatible model files** and **identical inference results** for the covered 1D CRF paths. The Rust implementation is **1.1-1.8x faster** than the original C code in the benchmarks used by this project.

Based on [chokkan/crfsuite](https://github.com/chokkan/crfsuite) version **0.12.2**, commit [`a2a1547`](https://github.com/chokkan/crfsuite/commit/a2a1547727985e3aff6a35cffe073f57f0223e9d).

Particular care has been put into matching the original version, and `TODO.md` records the completed fidelity audit. Compiler, CPU, or platform C-library differences may still affect some floating-point and `rand()`-dependent paths. Please report any differences.

As this implementation is pure Rust, it is suited for use in webassembly, and you can also use it directly as a library!

## Features

- **1st-order linear-chain CRF** with dyad (state + transition) features
- **5 training algorithms**: L-BFGS, L2-SGD, Averaged Perceptron, Passive-Aggressive, AROW
- **Binary-compatible** with C CRFsuite model files: models trained by either implementation can be used by the other on covered fixtures
- Conformance tests compare CLI behavior, model dumps, tagging output, parsing, and selected training paths against the original C implementation

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
  dataset.rs         Dataset with shuffle
  cqdb/              Constant Quark Database (model string lookup)
  crf1d/
    context.rs       Forward-backward algorithm, Viterbi decoding
    encode.rs        Training encoder (gradient computation)
    feature.rs       Feature extraction
    tag.rs           Tagger (inference)
  train/             Training algorithms (lbfgs, l2sgd, ap, pa, arow)
  bin/crfsuite-rs/   CLI binary, shared CLI metadata, learn data loading/training, and C-compatible parameter metadata
tests/               Conformance tests vs original C implementation
```

## Testing

```bash
# Run all tests
cargo test

# Lint all targets with warnings denied
cargo clippy --all-targets -- -D warnings
```

### C/Rust conformance tests

The conformance tests run automatically as part of `cargo test`. Tests that need
the original C executable skip themselves when the binary is not present at
`crfsuite/frontend/.libs/crfsuite`.

Build the bundled C implementation before running the full conformance suite:

```bash
(cd crfsuite && ./autogen.sh && ./configure && make)
cargo test --test conformance
```

The test harness sets `LD_LIBRARY_PATH` to `crfsuite/lib/crf/.libs` for C
commands, so no shell-level environment variable is normally required. If you run
the C binary manually, use:

```bash
LD_LIBRARY_PATH=crfsuite/lib/crf/.libs crfsuite/frontend/.libs/crfsuite --help
```

The conformance suite verifies:

- Tag output identity for labels, scores, probabilities, and marginals.
- Dump output identity for C and Rust models.
- Cross-implementation use: selected C-trained models can be tagged by Rust, and selected Rust-trained models can be tagged by C.
- Parser edge cases for IWA escaping, `@weight`, CRLF input, empty fields, and C-compatible `atoi`/`atof` behavior.
- CLI failures for selected missing files, malformed models, unknown parameters, and top-level command errors.

## Compatibility

- Reads and writes the CRFsuite 0.12 binary model format using explicit little-endian encoding.
- Rust can read C models, and C can read Rust models for the covered model layout and CQDB sections.
- Minimal deterministic L-BFGS training with `max_iterations=0` is covered by byte-for-byte model equality tests.
- General training output is treated as functionally compatible when tag/dump output matches, because floating-point ordering and trainer details can affect byte layout.
- The `vecexp` polynomial approximation is reproduced exactly for the covered SSE2 fixture to preserve inference consistency.

### Known deviations from C

- A final non-empty IWA line without a trailing newline is handled gracefully instead of preserving C's hang/wait behavior.
- CLI chrome is intentionally quieter in Rust. The conformance tests normalize C-only banners and training progress preambles when the semantic result is stderr, exit status, or parameter output.
- `learn -g` and online trainer shuffles call the platform C `srand(0)`/`rand()` stream for compatibility with the bundled C build on the same platform.
- Top-level help, subcommand help, and selected command errors are matched by conformance tests, while broader C banner/progress chrome remains intentionally quieter.

## License

This is a port of CRFsuite by Naoaki Okazaki, licensed under the modified BSD license. See the original `crfsuite/` directory for the full license text.
