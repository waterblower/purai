use parquet::{
    file::reader::{FileReader, SerializedFileReader},
    record::RowAccessor,
};
use serde_json::json;
use std::{
    env,
    fs::{self, File},
    io::{BufWriter, Write},
    path::Path,
    time::Instant,
};

type Result<T> = std::result::Result<T, Box<dyn std::error::Error>>;

// WikiText article titles use one separated '=' at either end. Section titles
// such as '= = History = =' are deliberately not treated as new articles.
fn is_article_title(text: &str) -> bool {
    text.trim()
        .strip_prefix("= ")
        .and_then(|s| s.strip_suffix(" ="))
        .is_some_and(|s| {
            !s.trim().is_empty() && !s.trim().starts_with('=') && !s.trim().ends_with('=')
        })
}

struct Articles<W> {
    out: W,
    pending: String,
    target: usize,
    bytes: usize,
    documents: usize,
    rows: usize,
    titles: Vec<String>,
}
impl<W: Write> Articles<W> {
    fn new(out: W, target: usize) -> Self {
        Self {
            out,
            pending: String::new(),
            target,
            bytes: 0,
            documents: 0,
            rows: 0,
            titles: Vec::new(),
        }
    }
    fn flush_article(&mut self) -> Result<()> {
        let article = self.pending.trim();
        if !article.is_empty() {
            serde_json::to_writer(&mut self.out, article)?;
            writeln!(self.out)?;
            self.bytes += article.len();
            self.documents += 1;
            self.titles
                .push(article.lines().next().unwrap_or("").to_owned());
        }
        self.pending.clear();
        Ok(())
    }
    // False means the desired complete-article prefix has been emitted.
    fn push(&mut self, text: &str) -> Result<bool> {
        self.rows += 1;
        if is_article_title(text) {
            self.flush_article()?;
            if self.target != 0 && self.bytes >= self.target {
                return Ok(false);
            }
        }
        if self.pending.is_empty() && text.trim().is_empty() {
            return Ok(true);
        }
        if self.pending.is_empty() && !is_article_title(text) {
            return Err(
                "source starts mid-article; refusing to manufacture a document boundary".into(),
            );
        }
        self.pending.push_str(text);
        if !text.ends_with('\n') {
            self.pending.push('\n');
        }
        Ok(true)
    }
}

fn run() -> Result<()> {
    let args: Vec<String> = env::args().skip(1).collect();
    if args.len() != 4 {
        return Err(
            "usage: wikitext-prep INPUT.parquet OUTPUT.jsonl TARGET_BYTES_OR_0 REPORT.json".into(),
        );
    }
    let target: usize = args[2].parse()?;
    let output = Path::new(&args[1]);
    let report = Path::new(&args[3]);
    if output.exists() {
        return Err("refusing to overwrite existing dataset output".into());
    }
    if report.exists() && report.canonicalize()? == Path::new(&args[0]).canonicalize()? {
        return Err("report must not overwrite the source".into());
    }
    if output == report {
        return Err("dataset and report must have different paths".into());
    }
    let start = Instant::now();
    let reader = SerializedFileReader::new(File::open(&args[0])?)?;
    let temp = output.with_extension("jsonl.part");
    let mut articles = Articles::new(BufWriter::new(File::create(&temp)?), target);
    let mut reached_target = false;
    for row in reader.get_row_iter(None)? {
        let row = row?;
        if !articles.push(row.get_string(0)?)? {
            reached_target = true;
            break;
        }
    }
    if target == 0 {
        articles.flush_article()?;
    } else if !reached_target {
        return Err("training shard ended before a complete-article boundary beyond target".into());
    }
    articles.out.flush()?;
    let metadata = json!({"input":args[0],"output":args[1],"target_bytes":target,
        "decoded_bytes":articles.bytes,"documents":articles.documents,"source_rows_read":articles.rows,
        "article_titles":articles.titles,"source_total_rows":reader.metadata().file_metadata().num_rows(),
        "protocol":"Source-order complete article prefix. Single '=' title lines mark articles; subsection headings remain inside articles. Preserve text/case/punctuation; add newline between rows when absent, strip only outer article whitespace. No detokenization or vocabulary replacement. JSONL string per article. Target 0 emits the whole source (use only for complete validation split).",
        "seconds":start.elapsed().as_secs_f64()});
    fs::rename(temp, output)?;
    fs::write(report, serde_json::to_string_pretty(&metadata)? + "\n")?;
    println!(
        "{} documents, {} decoded bytes -> {}",
        articles.documents,
        articles.bytes,
        output.display()
    );
    Ok(())
}
fn main() {
    if let Err(e) = run() {
        eprintln!("error: {e}");
        std::process::exit(1);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn titles_and_article_boundaries_are_preserved() {
        assert!(is_article_title(" = Article title = \n"));
        assert!(!is_article_title(" = = Section = = \n"));
        let mut a = Articles::new(Vec::new(), 1);
        assert!(a.push("").unwrap());
        assert!(a.push(" = Article = \n").unwrap());
        assert!(a.push("Text without a trailing newline.").unwrap());
        assert!(a.push(" = = Section = = \n").unwrap());
        assert!(!a.push(" = Next article = \n").unwrap());
        assert_eq!(a.documents, 1);
        let text: String = serde_json::from_slice(&a.out).unwrap();
        assert_eq!(
            text,
            "= Article = \nText without a trailing newline.\n = = Section = ="
        );
        assert_eq!(a.bytes, text.len());
        assert!(a.pending.is_empty());
    }
    #[test]
    fn validation_flushes_last_article_and_rejects_mid_article_start() {
        let mut a = Articles::new(Vec::new(), 0);
        assert!(a.push("stray paragraph").is_err());
        a.push("= First =").unwrap();
        a.push("A.").unwrap();
        a.push("= Second =").unwrap();
        a.push("B.").unwrap();
        a.flush_article().unwrap();
        assert_eq!(a.documents, 2);
        assert_eq!(String::from_utf8(a.out).unwrap().lines().count(), 2);
    }
}
