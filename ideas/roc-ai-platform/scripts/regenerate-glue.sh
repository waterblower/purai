#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
scripts/roc.sh glue scripts/RustGlue.roc src/ platform/main.roc --no-cache
