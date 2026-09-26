mod sha256;
use parquet::{
    file::reader::{FileReader, SerializedFileReader},
    record::RowAccessor,
};
use serde_json::json;
use std::{
    collections::HashSet,
    env,
    fs::{self, File, OpenOptions},
    io::{BufWriter, Write},
    path::PathBuf,
};
type Result<T> = std::result::Result<T, Box<dyn std::error::Error>>;

fn split(hash: &[u8; 32]) -> usize {
    match u64::from_be_bytes(hash[..8].try_into().unwrap()) % 100 {
        0 => 2, // development, never used for checkpoint selection in this experiment
        1 => 3, // independent held-out test
        _ => 0,
    }
}

fn main() -> Result<()> {
    let args: Vec<_> = env::args().skip(1).collect();
    if args.len() < 3 {
        return Err("usage: exp5-prep DATA_DIR REPORT.json SHARD.parquet [...]".into());
    }
    let dir = PathBuf::from(&args[0]);
    let names = [
        "wikipedia-20231101-en-1gb.jsonl",
        "wikipedia-20231101-en-100mb.jsonl",
        "wikipedia-20231101-en-dev.jsonl",
        "wikipedia-20231101-en-test.jsonl",
    ];
    let targets = [1_000_000_000usize, 100_000_000, 5_000_000, 5_000_000];
    for name in names {
        if dir.join(name).exists() {
            return Err(format!("refusing to overwrite {name}").into());
        }
    }
    let mut outs = Vec::new();
    for name in names {
        outs.push(BufWriter::new(
            OpenOptions::new()
                .write(true)
                .create_new(true)
                .open(dir.join(format!("{name}.part")))?,
        ));
    }
    let mut manifest = BufWriter::new(
        OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(dir.join("wikipedia-20231101-en-documents.jsonl.part"))?,
    );
    let mut seen = HashSet::new();
    let mut counts = [0usize; 4];
    let mut bytes = [0usize; 4];
    let mut duplicates = 0;
    let mut rows = 0;
    let mut used_shards = Vec::new();
    'shards: for shard in &args[2..] {
        let reader = SerializedFileReader::new(File::open(shard)?)?;
        used_shards.push(shard);
        for row in reader.get_row_iter(None)? {
            let row = row?;
            rows += 1;
            let columns: Vec<_> = row.get_column_iter().map(|(n, _)| n.as_str()).collect();
            let text_col = columns
                .iter()
                .position(|&n| n == "text")
                .ok_or("missing text column")?;
            let id_col = columns
                .iter()
                .position(|&n| n == "id")
                .ok_or("missing id column")?;
            let title_col = columns
                .iter()
                .position(|&n| n == "title")
                .ok_or("missing title column")?;
            let text = row.get_string(text_col)?.trim();
            if text.is_empty() {
                continue;
            }
            let hash = sha256::digest(text.as_bytes());
            if !seen.insert(hash) {
                duplicates += 1;
                continue;
            }
            let part = split(&hash);
            if bytes[part] >= targets[part] {
                continue;
            }
            serde_json::to_writer(&mut outs[part], text)?;
            writeln!(outs[part])?;
            bytes[part] += text.len();
            counts[part] += 1;
            let in_small = part == 0 && bytes[1] < targets[1];
            if in_small {
                serde_json::to_writer(&mut outs[1], text)?;
                writeln!(outs[1])?;
                bytes[1] += text.len();
                counts[1] += 1;
            }
            let hex: String = hash.iter().map(|b| format!("{b:02x}")).collect();
            serde_json::to_writer(
                &mut manifest,
                &json!({"id":row.get_string(id_col)?, "title":row.get_string(title_col)?, "text_sha256":hex,"bytes":text.len(),"split":names[part],"in_100mb":in_small,"source":shard}),
            )?;
            writeln!(manifest)?;
            if bytes.iter().zip(targets).all(|(&b, t)| b >= t) {
                break 'shards;
            }
        }
    }
    for out in &mut outs {
        out.flush()?;
    }
    manifest.flush()?;
    if !bytes.iter().zip(targets).all(|(&b, t)| b >= t) {
        return Err(
            format!("insufficient source: {bytes:?}; incomplete .part outputs retained").into(),
        );
    }
    for name in names {
        fs::rename(dir.join(format!("{name}.part")), dir.join(name))?;
    }
    fs::rename(
        dir.join("wikipedia-20231101-en-documents.jsonl.part"),
        dir.join("wikipedia-20231101-en-documents.jsonl"),
    )?;
    let splits: Vec<_> = names.iter().enumerate().map(|(i,n)| json!({"file":n,"documents":counts[i],"decoded_bytes":bytes[i],"target_bytes":targets[i]})).collect();
    let report = json!({"dataset":"wikimedia/wikipedia","config":"20231101.en","revision":"b04c8d1ceb2f5cd4588862100d08de323dccfbaa","splits":splits,"source_shards":used_shards,"rows_read":rows,"exact_duplicates_skipped":duplicates,"protocol":"Read shards and articles in source order. Strip only outer whitespace from text; no title prefix or normalization. Deduplicate SHA-256 of trimmed text before split. Big-endian first 8 SHA bytes modulo 100: 0 dev, 1 test, 2..99 train. Keep complete articles until each target is met. 100MB train is a strict prefix of 1GB train. Dev/test independently capped at 5MB. This is a source-prefix sample, not a random sample of all Wikipedia. No near-duplicate or BLiMP contamination audit."});
    fs::write(&args[1], serde_json::to_string_pretty(&report)? + "\n")?;
    println!("{}", serde_json::to_string_pretty(&report)?);
    Ok(())
}

#[cfg(test)]
mod tests {
    #[test]
    fn deterministic_partition() {
        for n in 0u64..300 {
            let mut h = [0; 32];
            h[..8].copy_from_slice(&n.to_be_bytes());
            assert_eq!(
                super::split(&h),
                match n % 100 {
                    0 => 2,
                    1 => 3,
                    _ => 0,
                }
            );
        }
    }
}
