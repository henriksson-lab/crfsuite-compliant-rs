# crfsuite-rs

A pure Rust implementation of [CRFsuite](http://www.chokkan.org/software/crfsuite/) — Conditional Random Fields for labeling sequential data.

This is a faithful port of Naoaki Okazaki's CRFsuite 0.12, producing **binary-compatible model files** and **identical inference results**. The Rust implementation is **1.1–1.8x faster** than the original C code.

Based on [chokkan/crfsuite](https://github.com/chokkan/crfsuite) version **0.12.2**, commit [`a2a1547`](https://github.com/chokkan/crfsuite/commit/a2a1547727985e3aff6a35cffe073f57f0223e9d).

## Features

- **1st-order linear-chain CRF** with dyad (state + transition) features
- **5 training algorithms**: L-BFGS, L2-SGD, Averaged Perceptron, Passive-Aggressive, AROW
- **Binary-compatible** with C CRFsuite model files — models trained by either implementation can be used by the other
- **86 tests** including conformance tests that verify identical output against the original C implementation

## Installation

```bash
cargo build --release
```

The binary is at `target/release/crfsuite-rs`. The only C dependency is `liblbfgs` (bundled automatically via `liblbfgs-sys`).

## Usage

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
crates/
  crfsuite-core/     Pure Rust CRF library
    src/
      vecmath.rs       Vector math with exact SSE2 exp() polynomial
      cqdb/            Constant Quark Database (model string lookup)
      crf1d/
        context.rs     Forward-backward algorithm, Viterbi decoding
        encode.rs      Training encoder (gradient computation)
        feature.rs     Feature extraction
        tag.rs         Tagger (inference)
      model.rs         Binary model reader
      model_writer.rs  Binary model writer
      train/           Training algorithms (lbfgs, l2sgd, ap, pa, arow)
      dump.rs          Human-readable model dump
      quark.rs         String-to-ID dictionary
      params.rs        Algorithm parameter store
      dataset.rs       Dataset with shuffle
      types.rs         Attribute, Item, Instance types
  crfsuite-cli/      Command-line interface
  crfsuite-tests/    Conformance tests vs original C implementation
```

## Testing

```bash
# Run all tests (requires the original C crfsuite binary for conformance tests)
cargo test --workspace

# Run only the pure Rust unit tests
cargo test -p crfsuite-core
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
