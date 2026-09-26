#!/bin/sh
# Download official raw WikiText and create article-aligned JSONL with Rust.
set -eu
cd "$(dirname "$0")/../.."
exp=exp4
source_dir=data/wikitext-103-raw-source
revision=b08601e04326c79dfdd32d625aee71d232d685c3
base="https://huggingface.co/datasets/Salesforce/wikitext/resolve/$revision/wikitext-103-raw-v1"
mkdir -p "$exp" "$source_dir"
fetch() {
    if [ ! -f "$2" ]; then
        curl --fail --location --retry 2 --max-time 300 "$1" --output "$2.part" > "$3" 2>&1
        mv "$2.part" "$2"
    fi
}
# The first shard already contains enough complete articles for the 100 MB prefix.
fetch "$base/train-00000-of-00002.parquet" "$source_dir/train-00000-of-00002.parquet" "$exp/download-train.log"
fetch "$base/validation-00000-of-00001.parquet" "$source_dir/validation-00000-of-00001.parquet" "$exp/download-validation.log"
shasum -a 256 -c <<'CHECKSUMS'
74da360f23826045b3e6ac6375411fdb15f003030aa74f2596ed08b857cb9212  data/wikitext-103-raw-source/train-00000-of-00002.parquet
204929b7ff9d6184953f867dedb860e40aa69c078fc1e54b3baaa8fb28511c4c  data/wikitext-103-raw-source/validation-00000-of-00001.parquet
CHECKSUMS
if [ ! -f data/wikitext-103-raw-100mb.jsonl ]; then
    cargo run --locked --manifest-path exp1/tools/wikitext-prep/Cargo.toml --target-dir exp1/target/wikitext-prep -- \
        "$source_dir/train-00000-of-00002.parquet" data/wikitext-103-raw-100mb.jsonl 100000000 "$exp/data-train.json" \
        > "$exp/preparation-train.log" 2>&1
fi
if [ ! -f data/wikitext-103-raw-validation.jsonl ]; then
    cargo run --locked --manifest-path exp1/tools/wikitext-prep/Cargo.toml --target-dir exp1/target/wikitext-prep -- \
        "$source_dir/validation-00000-of-00001.parquet" data/wikitext-103-raw-validation.jsonl 0 "$exp/data-validation.json" \
        > "$exp/preparation-validation.log" 2>&1
fi
shasum -a 256 -c <<'CHECKSUMS'
dd55d166111c0a83ee800c12ef022199d2be9d9fb5ff71528b5ef28300c04359  data/wikitext-103-raw-100mb.jsonl
0fa288a976f8785bc1c129d91f752e74548dcab006af5f8b0a8f0019760d6aef  data/wikitext-103-raw-validation.jsonl
CHECKSUMS
echo 'WikiText-103 raw training and validation data are ready.'
