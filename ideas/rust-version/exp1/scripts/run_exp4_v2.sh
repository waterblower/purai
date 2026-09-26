#!/bin/sh
set -eu
cd "$(dirname "$0")/../.."
exp=exp4/v2
model="$exp/output-wikitext-103-raw-100mb-v2.model"
mkdir -p "$exp"
if [ -e "$model" ]; then
    echo "Model already exists: $model. Preserve this experiment before running again." >&2
    exit 1
fi
test -f data/wikitext-103-raw-100mb.jsonl
test -f data/wikitext-103-raw-validation.jsonl
# Check the exact same prepared datasets used by v1.
shasum -a 256 -c <<'CHECKSUMS' > "$exp/data-checksums.log"
dd55d166111c0a83ee800c12ef022199d2be9d9fb5ff71528b5ef28300c04359  data/wikitext-103-raw-100mb.jsonl
0fa288a976f8785bc1c129d91f752e74548dcab006af5f8b0a8f0019760d6aef  data/wikitext-103-raw-validation.jsonl
CHECKSUMS
cargo build --offline --manifest-path exp1/Cargo.toml > "$exp/build.log" 2>&1
date -u '+%Y-%m-%dT%H:%M:%SZ' > "$exp/training-started-utc.txt"
status=0
/usr/bin/time -p -o "$exp/training.time.txt" \
    cargo run --offline --manifest-path exp1/Cargo.toml -- train --input data/wikitext-103-raw-100mb.jsonl \
    --validation data/wikitext-103-raw-validation.jsonl --output "$model" \
    --rounds 36 --max-nodes 250000 --no-match --stop-on-bpb-increase \
    > "$exp/training.log" 2>&1 || status=$?
printf '%s\n' "$status" > "$exp/training-exit-code.txt"
date -u '+%Y-%m-%dT%H:%M:%SZ' > "$exp/training-finished-utc.txt"
test "$status" -eq 0
cargo run --offline --manifest-path exp1/Cargo.toml -- benchmark --model "$model" --input data/benchmarks/blimp/data \
    --reference exp1/benchmarks/references/blimp-models-summary.jsonl \
    --report "$exp/blimp.report.json" --samples "$exp/blimp.samples.jsonl" > "$exp/blimp.log" 2>&1
cargo run --offline --manifest-path exp1/Cargo.toml -- infer --model "$model" --prompt "Once upon a time," --no-match --seed 42 --n 400 \
    > "$exp/sample.txt" 2> "$exp/inference.log"
echo "Training and BLiMP complete; results in $exp/"
