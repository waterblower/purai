//! Native compiler prototype. No subprocesses and no third-party dependencies.
mod aarch64;
mod frontend;
mod macho;
mod sha256;

/// Byte offset in the original UTF-8 source, plus a user-facing explanation.
#[derive(Debug, Clone)]
pub struct Diagnostic {
    pub offset: usize,
    pub message: String,
}

impl Diagnostic {
    fn new(offset: usize, message: impl Into<String>) -> Self {
        Self {
            offset,
            message: message.into(),
        }
    }

    pub fn render(&self, path: &str, source: &str) -> String {
        let offset = self.offset.min(source.len());
        let before = &source[..offset];
        let line = before.bytes().filter(|&b| b == b'\n').count() + 1;
        let start = before.rfind('\n').map_or(0, |i| i + 1);
        let column = source[start..offset].chars().count() + 1;
        let text = source[start..].lines().next().unwrap_or("");
        format!(
            "{path}:{line}:{column}: error: {}\n  {text}\n  {}^",
            self.message,
            " ".repeat(column - 1)
        )
    }
}

pub fn check(source: &str) -> Result<(), Diagnostic> {
    frontend::lower(source).map(|_| ())
}

/// Return a complete, ad-hoc signed, position-independent ARM64 Mach-O image.
pub fn compile(source: &str) -> Result<Vec<u8>, Diagnostic> {
    let program = frontend::lower(source)?;
    let code = aarch64::generate(&program).map_err(|e| Diagnostic::new(0, e))?;
    macho::executable(code).map_err(|e| Diagnostic::new(0, e))
}
