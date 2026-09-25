#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
cargo fmt --check
cargo clippy --offline --locked --all-targets -- -D warnings
cargo test --offline --locked
scripts/roc.sh fmt --check platform
scripts/roc.sh fmt --check examples
scripts/roc.sh fmt --check tests/io.roc
python3 tests/integration.py
