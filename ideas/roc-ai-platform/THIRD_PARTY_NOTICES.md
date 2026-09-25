# Third-party notices

This project is distributed under the Universal Permissive License 1.0; see
`LICENSE` for the complete terms.

## Roc Rust glue generator and generated ABI support

`scripts/RustGlue.roc` is vendored unchanged from
https://github.com/roc-lang/roc/blob/f45bfbe/src/glue/src/RustGlue.roc.
`src/roc_platform_abi.rs` is generated from it with compiler
`nightly-2026-09-24-f45bfbe` and this project's `platform/main.roc`.

Copyright © 2019 Richard Feldman and subsequent Roc authors <roc-lang.org/authors>

Licensed under the Universal Permissive License (UPL), Version 1.0.

## Rust platform template

The host architecture, build target declarations, and wrapper pattern were
adapted from https://github.com/lukewilliamboswell/roc-platform-template-rust
at commit `dfa40a0dd1a87a3ada2cf2ee3c45230206e8c459`.

Copyright © 2024 Luke Boswell and subsequent authors

Licensed under the Universal Permissive License (UPL), Version 1.0.

## Compiler download

The optional installer fetches official Roc nightly binaries into the ignored
`.tools/` directory and checks SHA-256 digests recorded by the release API for
https://github.com/roc-lang/nightlies/releases/tag/nightly-2026-09-24-f45bfbe.
The compiler distribution includes its own `LICENSE` and `legal_details` files.
