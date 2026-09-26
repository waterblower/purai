use crate::{Result, fingerprint};
use serde::{Deserialize, Serialize};
use std::{collections::BTreeSet, fs, path::Path};

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Document {
    pub id: String,
    pub bytes: Vec<u8>,
}
impl Document {
    pub fn new(bytes: Vec<u8>) -> Self {
        Self {
            id: fingerprint(&bytes),
            bytes,
        }
    }
}

/// An explicit byte cap fails rather than silently training on a prefix.
pub fn read(path: &Path, max_bytes: usize) -> Result<Vec<Document>> {
    let metadata = fs::metadata(path)?;
    if metadata.len() > max_bytes as u64 {
        return Err(format!(
            "{} exceeds --max-data-bytes {} (file bytes)",
            path.display(),
            max_bytes
        )
        .into());
    }
    let raw = fs::read(path)?;
    if raw.len() > max_bytes {
        return Err("input grew beyond byte budget".into());
    }
    let mut docs = Vec::new();
    if path.extension().is_some_and(|s| s == "jsonl") {
        for (i, line) in raw.split(|b| *b == b'\n').enumerate() {
            if line.iter().all(u8::is_ascii_whitespace) {
                continue;
            }
            let v: serde_json::Value =
                serde_json::from_slice(line).map_err(|e| format!("JSONL line {}: {e}", i + 1))?;
            let bytes = if let Some(s) = v
                .as_str()
                .or_else(|| v.get("text").and_then(|v| v.as_str()))
            {
                s.as_bytes().to_vec()
            } else if let Some(a) = v.get("bytes").and_then(|v| v.as_array()) {
                a.iter()
                    .map(|v| {
                        v.as_u64()
                            .filter(|n| *n < 256)
                            .map(|n| n as u8)
                            .ok_or("bytes must be integers 0..255")
                    })
                    .collect::<std::result::Result<Vec<_>, _>>()?
            } else {
                return Err(format!(
                    "JSONL line {} needs a string, text field, or bytes array",
                    i + 1
                )
                .into());
            };
            if !bytes.is_empty() {
                docs.push(Document::new(bytes));
            }
        }
    } else if !raw.is_empty() {
        docs.push(Document::new(raw));
    }
    let mut seen = BTreeSet::new();
    docs.retain(|d| seen.insert(d.id.clone()));
    if docs.is_empty() {
        return Err("empty corpus".into());
    }
    Ok(docs)
}

pub struct Split {
    pub train: Vec<Document>,
    pub calibration: Vec<Document>,
    pub development: Vec<Document>,
}

pub fn split(mut docs: Vec<Document>, explicit_dev: Option<Vec<Document>>) -> Result<Split> {
    docs.sort_by(|a, b| a.id.cmp(&b.id));
    if docs.len() == 1 {
        let raw = &docs[0].bytes;
        if raw.len() < 30 {
            return Err("need >= 30 bytes, or >= 3 documents, for independent splits".into());
        }
        let a = raw.len() * 8 / 10;
        let b = raw.len() * 9 / 10;
        docs = vec![
            Document::new(raw[..a].to_vec()),
            Document::new(raw[a..b].to_vec()),
            Document::new(raw[b..].to_vec()),
        ];
        // Single-file byte boundaries reset all runtime memory.
    }
    let holdout = (docs.len() / 10).max(1);
    let calibration = docs.split_off(docs.len() - holdout);
    let development = if let Some(dev) = explicit_dev {
        dev
    } else {
        if docs.len() < 2 {
            return Err("need >= 3 documents without explicit validation".into());
        }
        docs.split_off(docs.len() - holdout.min(docs.len() - 1))
    };
    let out = Split {
        train: docs,
        calibration,
        development,
    };
    let mut seen = BTreeSet::new();
    for doc in out
        .train
        .iter()
        .chain(&out.calibration)
        .chain(&out.development)
    {
        if !seen.insert(doc.id.clone()) {
            return Err("identical document across training/calibration/development; supply independent documents".into());
        }
    }
    Ok(out)
}

pub fn byte_count(docs: &[Document]) -> usize {
    docs.iter().map(|d| d.bytes.len()).sum()
}
