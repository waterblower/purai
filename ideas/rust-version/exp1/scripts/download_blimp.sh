#!/bin/sh
# Public benchmark data and published scores only; no Python or model weights.
set -eu
cd "$(dirname "$0")/.."
revision=3e56b06fcabca9b30822fc66435fca6b1aa40bb1
base="https://raw.githubusercontent.com/alexwarstadt/blimp/$revision"
mkdir -p ../data/benchmarks benchmarks/references benchmarks/results
fetch() {
    if [ ! -f "$2" ]; then
        curl --fail --location --retry 2 --max-time 120 "$1" --output "$2.part"
        mv "$2.part" "$2"
    fi
}
fetch "$base/BLiMP.zip" ../data/benchmarks/BLiMP.zip
fetch "$base/raw_results/summary/models_summary.jsonl" benchmarks/references/blimp-models-summary.jsonl
shasum -a 256 -c <<'CHECKSUMS'
172a1a714d72e72a89e8384dd531483f3d0f8a9aee3d33dccaf7992d8cecbb73  ../data/benchmarks/BLiMP.zip
d54b1b361559eec9182f89930d5012af89d1a35d8e2ff435637b3166e2e3037a  benchmarks/references/blimp-models-summary.jsonl
CHECKSUMS
unzip -oq ../data/benchmarks/BLiMP.zip 'data/*.jsonl' -d ../data/benchmarks/blimp
