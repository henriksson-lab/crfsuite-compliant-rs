#!/bin/bash
# Performance comparison: Rust crfsuite-rs vs original C crfsuite
set -e

ROOT="$(cd "$(dirname "$0")" && pwd)"
C_BIN="$ROOT/crfsuite/frontend/.libs/crfsuite"
RS_BIN="$ROOT/target/release/crfsuite-rs"
export LD_LIBRARY_PATH="$ROOT/crfsuite/lib/crf/.libs:$LD_LIBRARY_PATH"

DATA="$ROOT/test_data/bench_10k.txt"
TRAIN_DATA="$ROOT/test_data/train.txt"
TMPDIR=$(mktemp -d)
trap "rm -rf $TMPDIR" EXIT

echo "============================================================"
echo "  CRFsuite Performance Benchmark: C vs Rust"
echo "============================================================"
echo ""
echo "Dataset: $(wc -l < "$DATA") lines, 10000 sequences, ~115k items"
echo ""

# ── Training benchmarks ─────────────────────────────────────────
echo "── TRAINING (bench_10k.txt) ─────────────────────────────────"
echo ""

for algo in lbfgs l2sgd averaged-perceptron passive-aggressive arow; do
    short=$(echo $algo | sed 's/averaged-perceptron/ap/;s/passive-aggressive/pa/')
    printf "%-6s  C:    " "$short"
    { time $C_BIN learn -a $algo -m "$TMPDIR/c_${short}.bin" "$DATA" > /dev/null 2>&1 ; } 2>&1 | grep real | awk '{print $2}'
    printf "%-6s  Rust: " "$short"
    { time $RS_BIN learn -a $algo -m "$TMPDIR/rs_${short}.bin" "$DATA" > /dev/null 2>&1 ; } 2>&1 | grep real | awk '{print $2}'
    echo ""
done

# ── Tagging benchmarks ──────────────────────────────────────────
# First train a model on the bench data
echo "── TAGGING (bench_10k.txt, lbfgs model) ─────────────────────"
echo ""
$C_BIN learn -a lbfgs -m "$TMPDIR/bench_model.bin" "$DATA" > /dev/null 2>&1

echo "  Plain labels:"
printf "    C:    "
{ time $C_BIN tag -m "$TMPDIR/bench_model.bin" "$DATA" > /dev/null 2>&1 ; } 2>&1 | grep real | awk '{print $2}'
printf "    Rust: "
{ time $RS_BIN tag -m "$TMPDIR/bench_model.bin" "$DATA" > /dev/null 2>&1 ; } 2>&1 | grep real | awk '{print $2}'
echo ""

echo "  With scores + marginals (-p -i):"
printf "    C:    "
{ time $C_BIN tag -m "$TMPDIR/bench_model.bin" -p -i "$DATA" > /dev/null 2>&1 ; } 2>&1 | grep real | awk '{print $2}'
printf "    Rust: "
{ time $RS_BIN tag -m "$TMPDIR/bench_model.bin" -p -i "$DATA" > /dev/null 2>&1 ; } 2>&1 | grep real | awk '{print $2}'
echo ""

echo "  With all marginals (-l):"
printf "    C:    "
{ time $C_BIN tag -m "$TMPDIR/bench_model.bin" -l "$DATA" > /dev/null 2>&1 ; } 2>&1 | grep real | awk '{print $2}'
printf "    Rust: "
{ time $RS_BIN tag -m "$TMPDIR/bench_model.bin" -l "$DATA" > /dev/null 2>&1 ; } 2>&1 | grep real | awk '{print $2}'
echo ""

echo "  Quiet evaluation (-t -q):"
printf "    C:    "
{ time $C_BIN tag -m "$TMPDIR/bench_model.bin" -t -q "$DATA" > /dev/null 2>&1 ; } 2>&1 | grep real | awk '{print $2}'
printf "    Rust: "
{ time $RS_BIN tag -m "$TMPDIR/bench_model.bin" -t -q "$DATA" > /dev/null 2>&1 ; } 2>&1 | grep real | awk '{print $2}'
echo ""

# ── Model I/O ───────────────────────────────────────────────────
echo "── DUMP (lbfgs model from bench_10k) ────────────────────────"
echo ""
printf "    C:    "
{ time $C_BIN dump "$TMPDIR/bench_model.bin" > /dev/null 2>&1 ; } 2>&1 | grep real | awk '{print $2}'
printf "    Rust: "
{ time $RS_BIN dump "$TMPDIR/bench_model.bin" > /dev/null 2>&1 ; } 2>&1 | grep real | awk '{print $2}'
echo ""

echo "============================================================"
echo "  Done."
echo "============================================================"
