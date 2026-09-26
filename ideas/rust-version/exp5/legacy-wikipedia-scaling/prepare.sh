#!/bin/sh
set -eu
cd "$(dirname "$0")/../.."
revision=b04c8d1ceb2f5cd4588862100d08de323dccfbaa
mkdir -p data/wikipedia-20231101-en-source
for shard in 00000 00001; do
source="data/wikipedia-20231101-en-source/train-$shard-of-00041.parquet"
if [ ! -f "$source" ]; then
  curl --fail --location --retry 3 --max-time 1200 \
    "https://huggingface.co/datasets/wikimedia/wikipedia/resolve/$revision/20231101.en/train-$shard-of-00041.parquet" \
    --output "$source.part" > "exp5/legacy-wikipedia-scaling/download-$shard.log" 2>&1
  mv "$source.part" "$source"
fi
done
cargo test --offline --locked --manifest-path exp5/legacy-wikipedia-scaling/prep/Cargo.toml --target-dir exp5/legacy-wikipedia-scaling/build > exp5/legacy-wikipedia-scaling/prep-tests.log 2>&1
cargo run --offline --locked --manifest-path exp5/legacy-wikipedia-scaling/prep/Cargo.toml --target-dir exp5/legacy-wikipedia-scaling/build -- \
  data exp5/legacy-wikipedia-scaling/data-preparation.json \
  data/wikipedia-20231101-en-source/train-00000-of-00041.parquet \
  data/wikipedia-20231101-en-source/train-00001-of-00041.parquet > exp5/legacy-wikipedia-scaling/preparation.log 2>&1
