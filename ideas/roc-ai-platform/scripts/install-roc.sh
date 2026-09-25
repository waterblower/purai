#!/usr/bin/env bash
# Install only into this project; does not change PATH or the system toolchain.
set -euo pipefail
cd "$(dirname "$0")/.."
version="$(cat roc-version)"
case "$(uname -s)-$(uname -m)" in
    Darwin-arm64)
        arch=macos_apple_silicon
        sha=e1fdb4ab75a4df278922a374c9a6e2e177992afb1b6fcd728d836fd68e3ae893
        ;;
    Darwin-x86_64)
        arch=macos_x86_64
        sha=0805b91ffb6d6e3acb7cfd6e8385986e738d7e5cbcde061422f42cead98de8a8
        ;;
    *) echo 'This platform currently provides macOS build targets.' >&2; exit 1 ;;
esac
archive="roc_nightly-$arch-${version#nightly-}.tar.gz"
mkdir -p .tools/roc
download="$(mktemp "$PWD/.tools/roc-download.XXXXXX")"
trap 'rm -f "$download"' EXIT
curl --fail --location --silent --show-error --retry 3 \
    "https://github.com/roc-lang/nightlies/releases/download/$version/$archive" -o "$download"
actual="$(shasum -a 256 "$download" | cut -d ' ' -f 1)"
if [[ "$actual" != "$sha" ]]; then
    echo 'Compiler checksum mismatch; refusing to extract.' >&2
    exit 1
fi
tar -xzf "$download" -C .tools/roc
".tools/roc/${archive%.tar.gz}/roc" version
