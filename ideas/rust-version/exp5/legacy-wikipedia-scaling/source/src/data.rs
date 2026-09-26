use crate::Result;
use std::fs;
use std::path::{Path, PathBuf};

pub type Documents = Vec<Vec<u8>>;

pub fn read_docs(path: impl AsRef<Path>) -> Result<Vec<Vec<u8>>> {
    let path = path.as_ref();
    let bytes = fs::read(path)?;
    if path.extension().is_some_and(|ext| ext == "jsonl") {
        let text = std::str::from_utf8(&bytes)?;
        let mut docs = Vec::new();
        for (i, line) in text.lines().enumerate() {
            if line.trim().is_empty() {
                continue;
            }
            let value: serde_json::Value = serde_json::from_str(line)
                .map_err(|e| format!("{}:{}: {e}", path.display(), i + 1))?;
            let s = value
                .as_str()
                .or_else(|| value.get("text").and_then(serde_json::Value::as_str))
                .ok_or_else(|| {
                    format!(
                        "{}:{}: expected a JSON string or {{\"text\": \"...\"}}",
                        path.display(),
                        i + 1
                    )
                })?;
            if !s.is_empty() {
                docs.push(s.as_bytes().to_vec());
            }
        }
        if docs.is_empty() {
            return Err("input contains no nonempty documents".into());
        }
        Ok(docs)
    } else {
        if bytes.is_empty() {
            return Err("input is empty".into());
        }
        Ok(vec![bytes])
    }
}

pub fn byte_count(docs: &[Vec<u8>]) -> usize {
    docs.iter().map(Vec::len).sum()
}

pub fn default_validation(input: &Path) -> Option<PathBuf> {
    if input.file_name().is_some_and(|n| n == "train.jsonl") {
        let sibling = input.with_file_name("validation.jsonl");
        if sibling.is_file() {
            return Some(sibling);
        }
    }
    None
}

/// Separate complete documents when possible. For a single raw file, split its
/// bytes and reset all history at the boundary. Both partitions must be nonempty.
pub fn split_tail(mut docs: Documents, fraction: f64) -> Result<(Documents, Documents)> {
    if !fraction.is_finite() || fraction <= 0.0 || fraction >= 1.0 {
        return Err("split fraction must be between 0 and 1".into());
    }
    if docs.len() > 1 {
        let tail = ((docs.len() as f64 * fraction).ceil() as usize).clamp(1, docs.len() - 1);
        let cut = docs.len() - tail;
        let held_out = docs.split_off(cut);
        Ok((docs, held_out))
    } else {
        let mut text = docs.pop().ok_or("no documents")?;
        if text.len() < 2 {
            return Err("at least two bytes are needed for a held-out split".into());
        }
        let cut =
            ((text.len() as f64 * (1.0 - fraction)).floor() as usize).clamp(1, text.len() - 1);
        let held_out = text.split_off(cut);
        Ok((vec![text], vec![held_out]))
    }
}
