//! Synchronous I/O for Roc applications. See platform/*.roc for the public API.

mod effects;
mod host;
// Compiler-generated code is checked by its own ABI layout assertions.
#[allow(clippy::all, unexpected_cfgs)]
mod roc_platform_abi;
