#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
case "$(uname -s)-$(uname -m)" in
    Darwin-arm64) target=arm64mac ;;
    Darwin-x86_64) target=x64mac ;;
    *) echo 'The current build script supports native macOS (arm64 and x86_64).' >&2; exit 1 ;;
esac
cargo build --release --locked --offline --lib
mkdir -p "platform/targets/$target"
cp target/release/libhost.a "platform/targets/$target/libhost.a"
echo "Built platform/targets/$target/libhost.a"
