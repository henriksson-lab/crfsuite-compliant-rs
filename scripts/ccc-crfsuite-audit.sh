#!/usr/bin/env bash
set -euo pipefail

# Generate focused code-complexity-comparator reports for CRFSuite translation
# audits. The raw comparator is intentionally language-agnostic; these filters
# keep frontend Rust code out of core C comparisons and remove Rust test helpers.

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CCC="${CCC:-/data/henriksson/github/claude/code-complexity-comparator/target/release/ccc-rs}"
OUT="${OUT:-$ROOT/target/ccc-audit}"
MAP="${MAP:-$ROOT/ccc_mapping.toml}"

mkdir -p "$OUT"

"$CCC" analyze "$ROOT/crfsuite/lib/crf/src" --recurse -l c -o "$OUT/c-core.json"
"$CCC" analyze "$ROOT/crfsuite/frontend" --recurse -l c -o "$OUT/c-frontend.json"
"$CCC" analyze "$ROOT/src" --recurse -l rust -o "$OUT/rust-all.json"

jq '
  .source_file = "src (filtered core)" |
  .functions |= map(select(
    (.location.file | contains("/src/bin/") | not) and
    (.name | startswith("test_") | not) and
    (.name | test("_test$") | not) and
    (.name | test("^(tiny_instance|make_instance)$") | not)
  )) |
  .structs |= map(select(
    (.location.file | contains("/src/bin/") | not)
  ))
' "$OUT/rust-all.json" > "$OUT/rust-core.json"

jq '
  .source_file = "src/bin (filtered frontend)" |
  .functions |= map(select(.location.file | contains("/src/bin/"))) |
  .structs |= map(select(.location.file | contains("/src/bin/")))
' "$OUT/rust-all.json" > "$OUT/rust-frontend.json"

"$CCC" compare "$OUT/rust-core.json" "$OUT/c-core.json" --mapping "$MAP" --top 50 \
  > "$OUT/core-compare.txt"
"$CCC" missing "$OUT/rust-core.json" "$OUT/c-core.json" --mapping "$MAP" \
  > "$OUT/core-missing.txt"
"$CCC" constants-diff "$OUT/rust-core.json" "$OUT/c-core.json" --mapping "$MAP" --top 50 \
  > "$OUT/core-constants-diff.txt"

"$CCC" compare "$OUT/rust-frontend.json" "$OUT/c-frontend.json" --mapping "$MAP" --top 50 \
  > "$OUT/frontend-compare.txt"
"$CCC" missing "$OUT/rust-frontend.json" "$OUT/c-frontend.json" --mapping "$MAP" \
  > "$OUT/frontend-missing.txt"
"$CCC" constants-diff "$OUT/rust-frontend.json" "$OUT/c-frontend.json" --mapping "$MAP" --top 50 \
  > "$OUT/frontend-constants-diff.txt"

printf 'Wrote CRFSuite CCC audit reports to %s\n' "$OUT"
printf 'Core compare: %s\n' "$OUT/core-compare.txt"
printf 'Frontend compare: %s\n' "$OUT/frontend-compare.txt"
