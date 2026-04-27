# crfsuite-rs

A pure Rust implementation of [CRFsuite](http://www.chokkan.org/software/crfsuite/) — Conditional Random Fields for labeling sequential data.

This is a Rust port of Naoaki Okazaki's CRFsuite 0.12 [chokkan/crfsuite](https://github.com/chokkan/crfsuite), commit [`a2a1547`](https://github.com/chokkan/crfsuite/commit/a2a1547727985e3aff6a35cffe073f57f0223e9d).

* 2026-04-27: Translation now passes tests. Ready for use but be vigilant to bugs

## This is an LLM-mediated faithful (hopefully) translation, not the original code! 

Most users should probably first see if the existing original code works for them, unless they have reason otherwise. The original source
may have newer features and it has had more love in terms of fixing bugs. In fact, we aim to replicate bugs if they are present, for the
sake of reproducibility! (but then we might have added a few more in the process)

There are however cases when you might prefer this Rust version. We generally agree with [this manifesto](https://rewrites.bio/) but more specifically:
* We have had many issues with ensuring that our software works using existing containers (Docker, PodMan, Singularity). One size does not fit all and it eats our resources trying to keep up with every way of delivering software
* Common package managers do not work well. It was great when we had a few Linux distributions with stable procedures, but now there are just too many ecosystems (Homebrew, Conda). Conda has an NP-complete resolver which does not scale. Homebrew is only so-stable. And our dependencies in Python still break. These can no longer be considered professional serious options. Meanwhile, Cargo enables multiple versions of packages to be available, even within the same program(!)
* The future is the web. We deploy software in the web browser, and until now that has meant Javascript. This is a language where even the == operator is broken. Typescript is one step up, but a game changer is the ability to compile Rust code into webassembly, enabling performance and sharing of code with the backend. Translating code to Rust enables new ways of deployment and running code in the browser has especial benefits for science - researchers do not have deep pockets to run servers, so pushing compute to the user enables deployment that otherwise would be impossible
* Old CLI-based utilities are bad for the environment(!). A large amount of compute resources are spent creating and communicating via small files, which we can bypass by using code as libraries. Even better, we can avoid frequent reloading of databases by hoisting this stage, with up to 100x speedups in some cases. Less compute means faster compute and less electricity wasted
* LLM-mediated translations may actually be safer to use than the original code. This article shows that [running the same code on different operating systems can give somewhat different answers](https://doi.org/10.1038/nbt.3820). This is a gap that Rust+Cargo can reduce. Typesafe interfaces also reduce coding mistakes and error handling, as opposed to typical command-line scripting

But:

* **This approach should still be considered experimental**. The LLM technology is immature and has sharp corners. But there are opportunities to reap, and the genie is not going back into the bottle. This translation is as much aimed to learn how to improve the technology and get feedback on the results.
* Translations are not endorsed by the original authors unless otherwise noted. **Do not send bug reports to the original developers**. Use our Github issues page instead.
* **Do not trust the benchmarks on this page**. They are used to help evaluate the translation. If you want improved performance, you generally have to use this code as a library, and use the additional tricks it offers. We generally accept performance losses in order to reduce our dependency issues
* **Check the original Github pages for information about the package**. This README is kept sparse on purpose. It is not meant to be the primary source of information
* **If you are the author of the original code and wish to move to Rust, you can obtain ownership of this repository and crate**. Until then, our commitment is to offer an as-faithful-as-possible translation of a snapshot of your code. If we find serious bugs, we will report them to you. Otherwise we will just replicate them, to ensure comparability across studies that claim to use package XYZ v.666. Think of this like a fancy Ubuntu .deb-package of your software - that is how we treat it

This blurb might be out of date. Go to [this page](https://github.com/henriksson-lab/rustification) for the latest information and further information about how we approach translation


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
crfsuite-compliant-rs = { version = "0.2.1", default-features = false }
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

These are single-run `bench.sh` timings on `test_data/bench_10k.txt`
(8.3 MB, 125,183 lines, 10,000 sequences, ~115k items), comparing
`target/release/crfsuite-rs` against `crfsuite/frontend/.libs/crfsuite`.
Treat them as smoke-test measurements, not a portable benchmark claim.

| Task | C (original) | Rust | Result |
|---|---:|---:|---|
| Train L-BFGS | 4.029s | 4.191s | Rust 4.0% slower |
| Train L2-SGD | 110.437s | 113.511s | Rust 2.8% slower |
| Train Averaged Perceptron | 7.080s | 7.368s | Rust 4.1% slower |
| Train Passive-Aggressive | 8.542s | 8.853s | Rust 3.6% slower |
| Train AROW | 9.784s | 9.068s | Rust 7.3% faster |
| Tag plain labels | 0.370s | 0.193s | Rust 1.9x faster |
| Tag scores + marginals (`-p -i`) | 0.446s | 0.298s | Rust 1.5x faster |
| Tag all marginals (`-l`) | 0.828s | 0.585s | Rust 1.4x faster |
| Tag quiet eval (`-t -q`) | 0.303s | 0.265s | Rust 1.1x faster |
| Dump | 0.019s | 0.010s | Rust 1.9x faster |

On the same realistic fixture, output parity was checked explicitly:

- C-trained L-BFGS model, plain tagging: byte-identical.
- C-trained L-BFGS model, `tag -p -l`: byte-identical.
- C-trained L-BFGS model, `dump`: byte-identical.
- C-trained and Rust-trained L-BFGS model behavior on `tag -p -l`: byte-identical.
- C-trained and Rust-trained L-BFGS dump output: byte-identical.

Observed SHA-256 hashes for the shared outputs:

| Output | SHA-256 |
|---|---|
| `tag -p -l` | `fdecd00ab5f3a8ea62c7892b71f7d4dc64c8c31ee8746248c053bf0fbb400a5e` |
| `dump` | `83361a6dad62b7ee1dad85d9c727bafd8a1a4c62d2109a640bf8f269ead42200` |

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

This is a port of CRFsuite by Naoaki Okazaki, licensed under the modified BSD license.
