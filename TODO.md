# TODO: Faithful CRFsuite Translation

This project is intended to behave like the original C CRFsuite implementation
in `crfsuite/`. The items below are ordered by likely correctness impact and by
how directly they affect byte-for-byte CLI/model parity.

## P0: High-Impact Correctness Gaps

- [x] Port `l2sgd` training faithfully from `crfsuite/lib/crf/src/train_l2sgd.c`.
  - Previously, Rust `src/train/l2sgd.rs` was explicitly simplified.
  - Port the per-instance update loop, `decay`/`gain` handling, lazy L2
    regularization, and stopping behavior.
  - Port the calibration loop in `l2sgd_calibration`, including
    `calibration.rate`, `calibration.samples`, `calibration.candidates`, and
    `calibration.max_trials`.
  - Add conformance tests that train C and Rust `l2sgd` with fixed parameters
    and compare tagging output on several datasets.
  - Add parameter-sensitivity tests proving each calibration parameter affects
    Rust behavior the same way it affects C behavior.
  - Covered: Rust now uses the C per-instance lazy `decay`/`gain` SGD update,
    zeroes weights at the same helper boundaries, uses C's calibration loop and
    sample-count regularization behavior, and preserves the C RNG stream for
    dataset shuffles. Conformance now compares fixed-parameter C/Rust-trained
    tagging output and calibration-variant core logs for sample, candidate,
    eta, rate, and max-trial settings.

- [x] Replace the Rust `learn -g` shuffle with the original C shuffle semantics.
  - C uses `srand(0)` and `rand()` in `crfsuite_data_shuffle`.
  - Current Rust uses a local xorshift approximation.
  - Port the exact shuffle algorithm and libc-compatible RNG behavior or call
    the platform C RNG behind a small compatibility layer.
  - Add conformance tests for `learn -g N -x` and `learn -g N -e M` that compare
    C/Rust model behavior.
  - Covered: Rust now calls platform `srand(0)`/`rand()` for the split shuffle,
    and conformance tests cover `learn -g N -e M` model dump parity plus
    `learn -g N -x` shuffle-sensitive fold feature/loss metrics.

- [x] Audit and port all online trainers for exact update order and stopping.
  - `averaged-perceptron`
  - `passive-aggressive`
  - `arow`
  - Compare every log line, stopping criterion, weight averaging rule, and
    instance iteration order against the C files in `crfsuite/lib/crf/src/`.
  - Add small deterministic fixtures where expected weights can be compared
    directly or indirectly through identical tag output.
  - Covered so far: AP/PA/AROW now use the C `rand()` stream for per-epoch
    dataset shuffles; PA/AROW sparse delta collection now uses explicit C-style
    active-feature de-duplication instead of `delta == 0` detection; fixed
    short-run C/Rust-trained models now produce identical tagging output for
    AP, PA-II, and AROW; fixed core progress logs now match C for algorithm
    parameters, losses, feature norms, and stopping lines; `learn -e` holdout
    evaluation logs now match C for AP, PA-II, and AROW.

- [x] Audit `lbfgs` against `train_lbfgs.c` and bundled L-BFGS behavior.
  - Confirm line-search names, defaults, error handling, max line-search
    behavior, convergence checks, and regularization terms.
  - Verify C/Rust behavior for `c1`, `c2`, `epsilon`, `period`, `delta`,
    `num_memories`, `linesearch`, and `max_linesearch`.
  - Add negative tests for invalid line-search strings if C rejects them.
  - Covered so far: non-maximum-iteration L-BFGS errors are logged with C numeric
    error codes.
  - Covered: progress now reports C-style active nonzero feature counts and C
    termination wording for convergence, stopping criteria, max iterations, and
    already-minimized returns; total-time footer is emitted. Parameter matrix
    conformance compares core C/Rust logs for `c1`, `c2`, `epsilon`, `period`,
    `delta`, `num_memories`, `linesearch`, and `max_linesearch`. C accepts
    unknown line-search strings by logging the provided string and falling back
    to More-Thuente; Rust preserves that behavior.

## P1: CLI Frontend Fidelity

- [x] Add C-compatible top-level command behavior.
  - Original `crfsuite` has a top-level banner and command dispatcher.
  - Rust currently uses `clap` subcommands and does not reproduce all C chrome.
  - Decide whether CLI chrome is in scope; if yes, match stdout/stderr and exit
  codes for `--help`, no command, unknown command, and subcommand help.
  - Covered: no command, unknown command, top-level `-h`/`--help`,
    subcommand `-h`/`--help`, disabled clap `help` pseudo-command, disabled
    version flags, and unknown top-level options now match C output and status.

- [x] Match `learn` option aliases exactly.
  - C accepts `--help-params`, while usage text mentions `--help-parameters`.
  - Rust currently exposes `--help-parameters`.
  - Add aliases for every C long option, including historical names and spelling.

- [x] Implement `learn -l/--log-to-file` and `-L/--logbase`.
  - C writes training logs to a generated filename based on algorithm,
    parameters, and source files.
  - Rust currently logs to stdout only.
  - Add conformance tests for generated file names and file contents.

- [x] Verify `learn -m ""` behavior.
  - C does not write a model when model path is empty.
  - Rust appears to match, but add a direct conformance test.

- [x] Verify `learn` stdin behavior.
  - C reads from stdin for an explicit file name `-`.
  - C ignores stdin when no data files are provided.
  - Add tests for explicit stdin input, multiple files with `-`, and holdout grouping.

- [x] Match C error formatting and exit codes.
  - Unknown graphical model.
  - Unknown algorithm.
  - Unknown parameter, including C typo: `paraneter not found`.
  - Missing model file.
  - Missing input file.
  - Malformed model.
  - Empty training data.
  - Invalid parameter values accepted through `atoi`/`atof`.
  - Covered so far: `learn` unknown model/algorithm/parameter and missing input;
    `tag` missing input; `tag`/`dump` missing or malformed model files; top-level
    missing command, unknown command, help pseudo-command, version flags, and
    unknown options; empty `learn` data success status and L-BFGS numeric error
    log.
  - Covered: conformance assertions compare exit codes, stdout, and stderr for
    these failure paths; the unknown-parameter test explicitly preserves C's
    `paraneter not found` diagnostic spelling.

- [x] Audit `tag` option aliases and incompatible flag combinations.
  - Compare C behavior for `-t`, `-r`, `-p`, `-i`, `-l`, `-q`.
  - Add tests for quiet mode combined with probability and marginal flags.
  - Covered: short and long aliases match exposed flags; `-q` suppresses `-p`,
    `-l`, `-i`, and `-r` tagging output while preserving `-t` evaluation
    output; `-t` performance output matches C after normalizing elapsed time.

- [x] Audit `dump` CLI behavior.
  - Compare stdout/stderr, exit codes, and malformed model handling.
  - Add fixtures for models containing unusual labels or attributes.
  - Covered: missing argument, missing model file, malformed model file, unknown
    option, ignored extra positional arguments, normal dump parity for existing
    fixtures, and unusual label/attribute dump fixtures.

## P1: IWA/Input Parsing Fidelity

- [x] Decide whether to reproduce C behavior for final line without trailing
  newline.
  - C appears to hang or wait in some cases.
  - Rust currently handles EOF more gracefully.
  - If preserving C bugs is not desired, document this intentional deviation.
  - Decision: do not reproduce the C hang for a final non-empty line without
    trailing `\n`; keep Rust's graceful EOF handling as an intentional
    deviation.

- [x] Match CRLF behavior exactly or document the deviation.
  - C treats `\r` as data in some contexts.
  - Rust currently trims or normalizes in ways that may differ.
  - Add fixtures with CRLF and bare CR line endings.
  - Covered so far: CRLF is treated C-style in IWA training fixtures.

- [x] Audit escaping rules in IWA fields.
  - `:`
  - `\`
  - tabs
  - spaces
  - empty attribute names
  - empty values
  - labels beginning with `@`
  - attributes beginning with `@`
  - Add C/Rust conformance tests for every case.
  - Covered so far: escaped `:` and `\` in attributes; escaped `:` and `\` in
    values at parser unit-test level.
  - Covered: tabs, spaces, empty attribute names, empty values, labels beginning
    with `@`, and attributes beginning with `@`.

- [x] Audit `@weight` behavior.
  - Missing value.
  - Invalid numeric value.
  - Multiple `@weight` lines per instance.
  - `@weight` after normal items.
  - Unknown `@` directives.
  - Covered: missing value, invalid value, multiple declarations, declaration
    after normal items, and unknown declaration failure.

- [x] Audit label and attribute dictionary insertion order.
  - Ensure quark ID assignment matches C for all parser edge cases.
  - Add tests where labels/attributes first appear only in holdout data.
  - Covered: labels/attributes first appearing only in held-out data and parser
    edge-case symbols via C/Rust dump comparisons.

- [x] Confirm C-compatible `atoi` and `atof` edge cases.
  - Overflow.
  - `NaN`, `Inf`, `Infinity`.
  - Hex floats.
  - Locale-sensitive decimal separators.
  - Signed values with whitespace.
  - Sign-only strings.
  - Covered so far: signed values with whitespace, sign-only strings, invalid
    prefixes, `NaN`, `nan(...)`, `Inf`/`Infinity`, hex floats, and glibc-style
    positive and negative integer overflow.

## P1: Model Format Fidelity

- [x] Audit model writer section order and byte layout.
  - Header.
  - Labels CQDB.
  - Attributes CQDB.
  - Feature references.
  - Feature weights.
  - Attribute references.
  - Label references.
  - Footer or padding, if any.
  - Covered so far: minimal deterministic C/Rust trained model is byte-identical,
    which checks header, chunk order, CQDBs, refs, feature serialization, and
    padding for that path.
  - Covered: raw writer-layout test checks header fields, C chunk order, chunk
    sizes, active feature/attribute counts, CQDB boundaries, reference chunk
    counts, final file size, and zero padding before aligned ref chunks.

- [x] Add byte-level writer parity tests where feasible.
  - For small deterministic datasets and trainer settings.
  - For manually constructed model structures independent of training.
  - Covered: `lbfgs max_iterations=0` minimal model byte parity against C.

- [x] Audit model reader behavior on malformed input.
  - Bad magic.
  - Truncated sections.
  - Bad offsets.
  - Unknown section types.
  - Invalid CQDB data.
  - Covered: unit tests reject short headers, bad magic/type, malformed offsets,
    unknown feature/label-ref/attr-ref chunks, truncated feature/ref sections,
    bad CQDB magic, bad CQDB byte-order checks, and truncated CQDB payloads.
  - Non-UTF8 strings.
  - Covered so far: bad magic, too-small files, invalid offsets, unknown FEAT
    and LFRF chunks, truncated feature arrays, and CLI missing/malformed model
    failures.

- [x] Verify endianness assumptions.
  - C model format is little-endian in practice.
  - Rust should explicitly encode/decode endian instead of relying on host
    layout.
  - Covered: reader and writer use explicit little-endian primitives; minimal
    whole-model byte parity verifies the path.

- [x] Audit floating-point serialization.
  - Confirm `f64`/`floatval_t` width assumptions.
  - Confirm byte order and alignment.
  - Add fixtures with unusual weights: zero, negative zero, tiny values, large
    values, NaN if C can produce or read it.
  - Covered: zero and negative zero pruning, exact bit preservation for tiny,
    large, and NaN `f64` weights through writer/reader roundtrip.

## P1: Feature Generation Fidelity

- [x] Audit `crf1d_encode.c` feature generation line-by-line.
  - State features.
  - Transition features.
  - Possible state features.
  - Possible transition features.
  - Frequency thresholding.
  - Attribute value handling.
  - Duplicate attributes in the same item.
  - Covered: Rust now mirrors C's state/transition feature keys, weighted
    frequency accumulation, minfreq pruning, duplicate accumulation, and forced
    feature insertion. Important nuance: `feature.possible_states` only forces
    labels for attributes observed in the training subset, not every dictionary
    attribute.

- [x] Add small feature-map fixtures.
  - One label, one attribute.
  - Two labels, one transition.
  - Duplicate attributes.
  - Zero-valued attributes.
  - Negative-valued attributes.
  - Rare features below `feature.minfreq`.
  - Covered: C-vs-Rust dump identity fixtures train tiny IWA inputs with
    `max_iterations=0` so the generated model reflects feature construction.

- [x] Verify holdout filtering before feature generation.
  - Features appearing only in holdout data should be excluded.
  - Label/attribute dictionaries may still include all parsed symbols; confirm
    C behavior and match it.
  - Covered: holdout-only symbols stay in dictionaries but are excluded from
    feature generation; `feature.possible_states=1` also ignores holdout-only
    attributes, matching C.

- [x] Verify forced possible features.
  - `feature.possible_states=1`
  - `feature.possible_transitions=1`
  - Both flags together.
  - Interaction with `feature.minfreq`.
  - Covered: C-vs-Rust dump identity fixtures for possible states, possible
    transitions, and both flags with `feature.minfreq`.

## P1: Inference/Math Fidelity

- [x] Audit forward/backward scaling against C.
  - Scale factor initialization.
  - Underflow handling.
  - Empty sequence behavior.
  - Single-item sequence behavior.
  - Covered: Rust mirrors C's scaled alpha/beta formulas, single-item
    initialization, and zero-sum underflow fallback (`scale=1`). Empty
    sequences are skipped by the CLI parsers before reaching the CRF context;
    C context routines assume `T > 0`.

- [x] Audit Viterbi behavior.
  - Tie-breaking order.
  - Sequence length 0 and 1.
  - Labels with equal scores.
  - Large negative and positive scores.
  - Covered: tests verify one-item paths, score/path consistency, equal-score
    ties choosing the lowest label via C's strict `<` comparison, and CLI
    C-vs-Rust behavior on a zero-weight tie fixture. Empty sequences are
    parser-skipped as above.

- [x] Audit marginal probability output.
  - State marginals.
  - Transition or path marginals if exposed.
  - Formatting precision and rounding.
  - Behavior for impossible labels or unknown labels.
  - Covered: context tests verify state and transition marginal accumulation;
    conformance tests compare C and Rust `tag -i`, `tag -l`, `tag -p`, and
    combined tie/marginal formatting on normal, single-item, GECCO, and
    unknown-attribute inputs.

- [x] Audit `fast_exp` approximation.
  - Confirm whether C uses exact `exp`, approximate exp, or lookup behavior.
  - Add numeric comparison tests around clipping thresholds.
  - Covered: the bundled C build uses `-DUSE_SSE`, so Rust keeps the SSE
    polynomial approximation. Bit fixtures generated from C cover the clamp
    endpoints and representative negative/zero/positive values.

- [x] Verify objective and gradient calculations.
  - Compare per-instance objective values with C instrumentation or small
    hand-computed fixtures.
  - Compare gradients before regularization.
  - Compare gradients after regularization.
  - Covered: hand-computed two-label/two-token fixtures verify
    `objective_and_gradients_batch` objective, observed-minus-model gradient
    signs, transition expectations, and instance weighting. L-BFGS L2
    regularization is factored into a helper and tested against C's
    `f += c2 * ||w||^2`, `g += 2 * c2 * w` formula.

## P2: Parameter System

- [x] Move CLI parameter metadata into a single source of truth.
  - Names.
  - Types.
  - Defaults.
  - Help text.
  - Parser behavior.
  - Runtime trainer mapping.
  - Covered: `learn.rs` `ParamSpec` tables now drive help output,
    validation, and runtime defaults. Parameter values are parsed at use time
    with C-compatible `atoi`/`atof` helpers or literal string copying.

- [x] Add tests for `learn -H` for every algorithm.
  - `lbfgs`
  - `l2sgd`
  - `ap`
  - `averaged-perceptron`
  - `pa`
  - `passive-aggressive`
  - `arow`
  - Covered: conformance tests compare C and Rust help-parameter output for
    all listed algorithm names and aliases.

- [x] Validate all parameters use C parsing semantics.
  - Int parameters use `atoi`.
  - Float parameters use `atof`.
  - String parameters are copied literally.
  - Empty values become C-equivalent defaults or zero values depending on path.
  - Covered: conformance fixtures exercise prefix parsing, empty numeric
    values, and literal string parameter logging against C.

- [x] Decide what to do with `src/params.rs`.
  - If unused, remove it or wire it into CLI/training.
  - If used internally later, make it match C `params.c`.
  - Covered: removed the unused module because it was not wired into training
    and used Rust parse semantics instead of C `atoi`/`atof`.

## P2: Test Harness Improvements

- [x] Ensure conformance tests always use the current test binary.
  - Prefer `CARGO_BIN_EXE_crfsuite-rs`.
  - Avoid stale `target/release` binaries.
  - Build the C binary automatically or skip clearly.
  - Covered: conformance helpers prefer `CARGO_BIN_EXE_crfsuite-rs` and fall
    back only to `target/debug/crfsuite-rs`; C-dependent tests skip clearly
    when `crfsuite/frontend/.libs/crfsuite` is absent.

- [x] Add a helper for running C with normalized banner handling.
  - Some C commands print the CRFSuite banner.
  - Tests should strip it only where appropriate and document why.
  - Covered: `strip_c_cli_chrome` removes C banner/training progress chrome for
    CLI comparisons where Rust intentionally omits it.

- [x] Add stdout/stderr/exit-code comparison helpers.
  - Current tests mostly compare stdout or success/failure.
  - Full CLI fidelity needs all three.
  - Covered: `assert_same_output` compares exit code, stdout, and stderr;
    failure comparisons now use it.

- [x] Add golden fixtures for parser edge cases.
  - Keep each fixture tiny and named for the behavior it tests.
  - Compare C and Rust command outputs byte-for-byte.
  - Covered: `test_data/iwa_edge_fields.txt`,
    `test_data/iwa_edge_weights.txt`, and `test_data/iwa_escaped_fields.txt`
    are trained by both CLIs and compared through byte-identical dumps.

- [x] Add property-style differential tests where deterministic.
  - Generate small IWA inputs.
  - Run both CLIs.
  - Compare outputs for commands that should be deterministic.
  - Keep failing cases as minimized fixtures.
  - Covered: `learn_generated_small_iwa_cases_match_c` deterministically
    generates short IWA sequences, trains C/Rust with deterministic settings,
    and compares dump output.

- [x] Avoid broad `rustfmt` churn in legacy-formatted tests.
  - Format only touched files where local style allows it.
  - If formatting the repo is desired, do it as a separate commit.
  - Covered: local edits have avoided whole-file or whole-repo rustfmt churn;
    remaining rustfmt diffs are reported separately instead of being folded
    into correctness work.

## P2: Documentation

- [x] Document known intentional deviations from C.
  - Better EOF handling, if kept.
  - CRLF normalization, if kept.
  - CLI chrome differences, if out of scope.
  - Any training implementations that are approximate rather than faithful.
  - Covered: README documents graceful EOF behavior, CLI chrome normalization,
    platform C `rand()` usage for shuffle parity, and the conformance-tested
    scope of top-level help/error matching.

- [x] Document how to run C/Rust conformance tests.
  - Required C build command.
  - Required environment variables.
  - Expected skips when C binary is missing.
  - Covered: README documents the bundled C build command, test command,
    harness-managed `LD_LIBRARY_PATH`, manual C invocation environment, and
    skip behavior when the C binary is absent.

- [x] Document model compatibility guarantees.
  - Rust reads C models.
  - C reads Rust models.
  - Byte-level model equality expectations.
  - Functional equality expectations.
  - Covered: README documents read/write compatibility, byte-level equality
    scope, functional equality expectations, and `vecexp` parity coverage.

## P3: Cleanup After Correctness

- [x] Reduce duplicated CLI metadata once behavior is locked.
  - Covered: app/default/error strings now live in
    `src/bin/crfsuite-rs/cli_meta.rs`, and learn algorithm names/aliases now
    live in `src/bin/crfsuite-rs/learn_params.rs` for validation, help, and
    trainer dispatch.
- [x] Split large `learn.rs` into parser, parameter metadata, data loading, and
  training orchestration modules.
  - Partial: C-compatible learn parameter metadata, defaults, help output, and
    validation now live in `src/bin/crfsuite-rs/learn_params.rs`.
  - Partial: IWA training data loading and split grouping now live in
    `src/bin/crfsuite-rs/learn_data.rs`.
  - Covered: per-holdout feature generation/training/model writing now lives in
    `src/bin/crfsuite-rs/learn_train.rs`; `learn.rs` coordinates validation,
    logging, data loading, and cross-validation flow.
- [x] Add focused comments only where the Rust differs from straightforward C.
  - Covered: comments call out C-only CLI chrome normalization and intentionally
    quieter Rust behavior without broadly annotating straightforward code.
- [x] Review untracked generated files and decide what belongs in version
  control.
  - Covered: `.gitignore` now ignores local build/agent output (`target/`,
    `.codex`, `.codex/`) while leaving source fixtures, `TODO.md`, `Cargo.lock`,
    `CLAUDE.md`, and the original `crfsuite/` comparison tree visible for an
    explicit commit decision.
- [x] Consider CI jobs that build C CRFsuite and run conformance tests.
  - Covered: `.github/workflows/ci.yml` runs Rust tests/clippy and includes a
    Linux C conformance job that installs autotools/liblbfgs, builds bundled
    CRFsuite when present, and skips clearly when `crfsuite/` is absent.

## Current Verification Baseline

Last known passing checks:

```sh
cargo clippy --all-targets -- -D warnings
cargo test
```

Current high-risk known deviation:

No unchecked fidelity TODOs remain. The remaining intentional deviation is that
Rust handles a final non-empty IWA line without a trailing newline gracefully
instead of preserving C's hang/wait behavior.

## Performance Follow-Up

Latest single-run `bench.sh` baseline on `test_data/bench_10k.txt`
(10,000 sequences, ~115k items), after updating `liblbfgs-compliant-rs` to
0.1.4:

| Task | C (original) | Rust | Result |
|---|---:|---:|---|
| Train L-BFGS | 3.281s | 3.459s | Rust 1.05x slower |
| Train L2-SGD | 89.296s | 93.098s | Rust 1.04x slower |
| Train Averaged Perceptron | 5.892s | 6.045s | Rust 1.03x slower |
| Train Passive-Aggressive | 6.969s | 7.176s | Rust 1.03x slower |
| Train AROW | 7.030s | 7.144s | Rust 1.02x slower |
| Tag plain labels | 0.244s | 0.186s | Rust 1.31x faster |
| Tag scores + marginals (`-p -i`) | 0.325s | 0.225s | Rust 1.44x faster |
| Tag all marginals (`-l`) | 0.609s | 0.369s | Rust 1.65x faster |
| Tag quiet eval (`-t -q`) | 0.233s | 0.168s | Rust 1.39x faster |
| Dump | 0.016s | 0.013s | Rust 1.23x faster |

- [ ] Profile training hot paths before optimizing.
  - Use `perf record --call-graph=dwarf -- target/release/crfsuite-rs learn
    -a arow -m /tmp/arow.bin test_data/bench_10k.txt`.
  - Also profile L-BFGS separately because it uses full forward/backward and
    gradient accumulation rather than Viterbi-only online updates.
  - If available, capture flamegraphs for `lbfgs`, `l2sgd`, `ap`, `pa`, and
    `arow`.

- [x] Remove per-instance full weight copies in online trainers.
  - Current online paths call `encoder.set_weights(&w, 1.0)` per instance,
    which copies the entire weight vector into `encoder.weights`.
  - Add encoder methods that fill transition/state scores directly from
    `&[f64]` without copying weights into the encoder.
  - Candidate shape:
    - `set_transitions_with_weights(&mut self, w: &[f64], scale: f64)`
    - `set_instance_with_weights(&mut self, inst: &Instance, w: &[f64],
      scale: f64)`
  - Update AP, PA, AROW, and L2-SGD paths to use the no-copy entry points where
    behavior remains identical.
  - Rerun `cargo test`, `cargo build --release`, and `bash bench.sh`.
  - Covered: `Crf1dEncoder` now has borrowed-weight setup methods; AP, PA,
    AROW, and L2-SGD no longer copy the full weight vector per instance.
    `cargo test`, `cargo clippy --all-targets -- -D warnings`,
    `cargo build --release`, and `bash bench.sh` passed after the change.

- [x] Avoid recomputing transition scores when weights have not changed.
  - Transition scores are instance-independent.
  - Online trainers currently recompute them for every instance.
  - Track a `transitions_dirty` flag and recompute only after a weight update.
  - This is especially relevant for AP, where correctly predicted instances do
    not update weights.
  - Preserve exact C-compatible update order and log behavior.
  - Covered: AP, PA, and AROW now keep a `transitions_dirty` flag and recompute
    transition scores only after updates or after holdout evaluation has loaded
    evaluation weights into the shared encoder.

- [x] Pre-encode training instances after feature generation.
  - Current scoring repeatedly walks `Instance -> Item -> Attribute ->
    attr_refs[aid] -> feature id -> feature_dst[fid]`.
  - Build a training-only encoded representation once, after feature generation,
    containing direct `(fid, dst, value)` entries per item.
  - Use the encoded representation for state scoring, observation expectation,
    and model expectation accumulation.
  - Keep the original `Instance` representation for public APIs and I/O.
  - Verify model/tagging parity against the existing C conformance tests.
  - Covered: `Crf1dEncoder` can now build training-only encoded instances.
    L-BFGS, L2-SGD, AP, PA, and AROW use encoded state features and direct
    transition feature lookup in their hot loops while public and holdout paths
    keep using `Instance`. Full conformance tests and clippy passed.

- [ ] Specialize encoder contexts by trainer needs.
  - `Crf1dEncoder::new` currently creates a context with
    `CTXF_MARGINALS | CTXF_VITERBI`.
  - Viterbi-only online trainers do not need marginal buffers.
  - Add a way to construct or borrow a Viterbi-only context for AP, PA, and AROW.
  - Keep marginal-enabled contexts for L-BFGS and L2-SGD objective/gradient
    paths.

- [ ] Consider deterministic parallel L-BFGS objective evaluation.
  - Objective and gradient evaluation is parallelizable across instances.
  - Floating-point reduction order can change results, so this should not
    silently replace the serial fidelity path.
  - If implemented, gate it behind an opt-in feature or parameter and reduce
    chunk results in deterministic order.
  - Document that bit-for-bit parity may differ in parallel mode.

- [ ] Evaluate release build tuning separately from algorithmic changes.
  - Compare current release builds with
    `RUSTFLAGS="-C target-cpu=native" cargo build --release`.
  - Consider:
    - `lto = "thin"`
    - `codegen-units = 1`
  - Keep these as separate commits because they affect build time and binary
    characteristics but do not address the likely training hot-path overheads.
