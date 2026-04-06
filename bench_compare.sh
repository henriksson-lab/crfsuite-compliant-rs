#!/bin/bash
# Performance comparison: C original vs Rust-FFI vs Pure Rust
set -e

ROOT="$(cd "$(dirname "$0")" && pwd)"
C_BIN="$ROOT/crfsuite/frontend/.libs/crfsuite"
FFI_BIN="/tmp/crfsuite-rs-ffi"
PURE_BIN="/tmp/crfsuite-rs-pure"
export LD_LIBRARY_PATH="$ROOT/crfsuite/lib/crf/.libs:$LD_LIBRARY_PATH"

DATA="$ROOT/test_data/bench_10k.txt"
TMPDIR=$(mktemp -d)
trap "rm -rf $TMPDIR" EXIT

echo "============================================================"
echo "  CRFsuite Performance: C vs Rust-FFI vs Pure Rust"
echo "============================================================"
echo ""
echo "Dataset: 10,000 sequences, ~115k items"
echo ""

timeit() {
    local label="$1"; shift
    printf "  %-12s " "$label"
    { time "$@" > /dev/null 2>&1 ; } 2>&1 | grep real | awk '{print $2}'
}

# ── Training: L-BFGS ────────────────────────────────────────
echo "── TRAINING: L-BFGS ─────────────────────────────────────"
echo ""
timeit "C:" $C_BIN learn -a lbfgs -m "$TMPDIR/c.bin" "$DATA"
timeit "Rust-FFI:" $FFI_BIN learn -a lbfgs -m "$TMPDIR/ffi.bin" "$DATA"
timeit "Pure-Rust:" $PURE_BIN learn -a lbfgs -m "$TMPDIR/pure.bin" "$DATA"
echo ""

# ── Training: AROW ──────────────────────────────────────────
echo "── TRAINING: AROW ───────────────────────────────────────"
echo ""
timeit "C:" $C_BIN learn -a arow -m "$TMPDIR/c_arow.bin" "$DATA"
timeit "Rust-FFI:" $FFI_BIN learn -a arow -m "$TMPDIR/ffi_arow.bin" "$DATA"
timeit "Pure-Rust:" $PURE_BIN learn -a arow -m "$TMPDIR/pure_arow.bin" "$DATA"
echo ""

# ── Training: Averaged Perceptron ────────────────────────────
echo "── TRAINING: Averaged Perceptron ────────────────────────"
echo ""
timeit "C:" $C_BIN learn -a averaged-perceptron -m "$TMPDIR/c_ap.bin" "$DATA"
timeit "Rust-FFI:" $FFI_BIN learn -a averaged-perceptron -m "$TMPDIR/ffi_ap.bin" "$DATA"
timeit "Pure-Rust:" $PURE_BIN learn -a ap -m "$TMPDIR/pure_ap.bin" "$DATA"
echo ""

# Use C-trained lbfgs model for tagging benchmarks
MODEL="$TMPDIR/c.bin"

# ── Tagging: plain labels ───────────────────────────────────
echo "── TAGGING: plain labels ────────────────────────────────"
echo ""
timeit "C:" $C_BIN tag -m "$MODEL" "$DATA"
timeit "Rust-FFI:" $FFI_BIN tag -m "$MODEL" "$DATA"
timeit "Pure-Rust:" $PURE_BIN tag -m "$MODEL" "$DATA"
echo ""

# ── Tagging: with scores + marginals ────────────────────────
echo "── TAGGING: scores + marginals (-p -i) ──────────────────"
echo ""
timeit "C:" $C_BIN tag -m "$MODEL" -p -i "$DATA"
timeit "Rust-FFI:" $FFI_BIN tag -m "$MODEL" -p -i "$DATA"
timeit "Pure-Rust:" $PURE_BIN tag -m "$MODEL" -p -i "$DATA"
echo ""

# ── Tagging: all marginals ──────────────────────────────────
echo "── TAGGING: all marginals (-l) ──────────────────────────"
echo ""
timeit "C:" $C_BIN tag -m "$MODEL" -l "$DATA"
timeit "Rust-FFI:" $FFI_BIN tag -m "$MODEL" -l "$DATA"
timeit "Pure-Rust:" $PURE_BIN tag -m "$MODEL" -l "$DATA"
echo ""

# ── Tagging: quiet eval ─────────────────────────────────────
echo "── TAGGING: quiet eval (-t -q) ──────────────────────────"
echo ""
timeit "C:" $C_BIN tag -m "$MODEL" -t -q "$DATA"
timeit "Rust-FFI:" $FFI_BIN tag -m "$MODEL" -t -q "$DATA"
timeit "Pure-Rust:" $PURE_BIN tag -m "$MODEL" -t -q "$DATA"
echo ""

# ── Dump ────────────────────────────────────────────────────
echo "── DUMP ─────────────────────────────────────────────────"
echo ""
timeit "C:" $C_BIN dump "$MODEL"
timeit "Rust-FFI:" $FFI_BIN dump "$MODEL"
timeit "Pure-Rust:" $PURE_BIN dump "$MODEL"
echo ""

# ── Verify correctness ──────────────────────────────────────
echo "── CORRECTNESS CHECK ────────────────────────────────────"
echo ""
$C_BIN tag -m "$MODEL" "$DATA" > "$TMPDIR/c_out.txt" 2>/dev/null
$PURE_BIN tag -m "$MODEL" "$DATA" > "$TMPDIR/pure_out.txt" 2>/dev/null
if diff -q "$TMPDIR/c_out.txt" "$TMPDIR/pure_out.txt" > /dev/null 2>&1; then
    echo "  Tag output: C == Pure Rust  ✓"
else
    echo "  Tag output: DIFFERS"
fi

$C_BIN dump "$MODEL" > "$TMPDIR/c_dump.txt" 2>/dev/null
$PURE_BIN dump "$MODEL" > "$TMPDIR/pure_dump.txt" 2>/dev/null
if diff -q "$TMPDIR/c_dump.txt" "$TMPDIR/pure_dump.txt" > /dev/null 2>&1; then
    echo "  Dump output: C == Pure Rust  ✓"
else
    echo "  Dump output: DIFFERS"
fi
echo ""

echo "============================================================"
echo "  Done."
echo "============================================================"
