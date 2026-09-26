#!/bin/sh
set -eu
cd "$(dirname "$0")/../.."
exp=exp4
model="$exp/output-wikitext-103-raw-100mb.model"
mkdir -p "$exp"
if [ -e "$model" ]; then
    echo "Model already exists: $model. Preserve this experiment before running again." >&2
    exit 1
fi
test -f data/wikitext-103-raw-100mb.jsonl
test -f data/wikitext-103-raw-validation.jsonl
date -u '+%Y-%m-%dT%H:%M:%SZ' > "$exp/training-started-utc.txt"
status=0
/usr/bin/time -p -o "$exp/training.time.txt" \
    cargo run --manifest-path exp1/Cargo.toml -- train --input data/wikitext-103-raw-100mb.jsonl \
    --validation data/wikitext-103-raw-validation.jsonl --output "$model" \
    --rounds 12 --max-nodes 250000 --no-match > "$exp/training.log" 2>&1 || status=$?
printf '%s\n' "$status" > "$exp/training-exit-code.txt"
date -u '+%Y-%m-%dT%H:%M:%SZ' > "$exp/training-finished-utc.txt"
test "$status" -eq 0
cargo run --manifest-path exp1/Cargo.toml -- benchmark --model "$model" --input data/benchmarks/blimp/data \
    --reference exp1/benchmarks/references/blimp-models-summary.jsonl \
    --report "$exp/blimp.report.json" --samples "$exp/blimp.samples.jsonl" > "$exp/blimp.log" 2>&1
cargo run --manifest-path exp1/Cargo.toml -- infer --model "$model" --prompt "Once upon a time," --no-match --seed 42 --n 400 \
    > "$exp/sample.txt" 2> "$exp/inference.log"
echo "Training and BLiMP complete; results in $exp/"
