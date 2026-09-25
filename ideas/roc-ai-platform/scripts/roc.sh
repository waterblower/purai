#!/usr/bin/env bash
set -euo pipefail
root="$(cd "$(dirname "$0")/.." && pwd)"
version="$(cat "$root/roc-version")"
if [[ -n "${ROC:-}" ]]; then
    compiler="$ROC"
elif command -v roc >/dev/null 2>&1; then
    compiler="$(command -v roc)"
else
    case "$(uname -s)-$(uname -m)" in
        Darwin-arm64) arch=macos_apple_silicon ;;
        Darwin-x86_64) arch=macos_x86_64 ;;
        *) echo 'Set ROC to the pinned compiler executable.' >&2; exit 1 ;;
    esac
    compiler="$root/.tools/roc/roc_nightly-$arch-${version#nightly-}/roc"
fi
if [[ ! -x "$compiler" ]]; then
    echo "Install $version (see README.md), or set ROC=/path/to/roc." >&2
    exit 1
fi
if [[ "$("$compiler" version)" != "Roc compiler version $version" ]]; then
    echo "This platform requires Roc $version." >&2
    exit 1
fi
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-$root/.tools/cache}"
exec "$compiler" "$@"
