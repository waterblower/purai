//! BLiMP full-sentence forced-choice evaluation. No fitting or sampling occurs.
use crate::{Result, predict::Predictor};
use serde_json::{Value, json};
use std::{
    collections::{BTreeMap, BTreeSet},
    fs::{self, File},
    io::{BufRead, BufReader, Write},
    path::Path,
};

/// Negative log probability of the entire UTF-8 byte string, without EOS,
/// length normalization, sampling temperature, or the match-copy predictor.
pub fn sentence_bits(predictor: &Predictor<'_>, text: &[u8]) -> f64 {
    text.iter()
        .enumerate()
        .map(|(i, &b)| {
            let (node, _, _) = predictor.compiled.context(text, i, usize::MAX);
            -predictor.compiled.distributions[node][b as usize].log2()
        })
        .sum()
}

#[derive(Default)]
struct Counts {
    pairs: u64,
    correct: u64,
    ties: u64,
    near_ties: u64,
}
impl Counts {
    fn record(&mut self, good: f64, bad: f64) {
        self.pairs += 1;
        self.correct += u64::from(good < bad);
        self.ties += u64::from(good == bad);
        self.near_ties += u64::from((good - bad).abs() < 1e-10);
    }
    fn value(&self) -> Value {
        json!({"pairs":self.pairs,"correct":self.correct,"ties":self.ties,
            "near_ties_1e_minus_10_bits":self.near_ties,
            "accuracy":self.correct as f64 / self.pairs as f64})
    }
}

fn string<'a>(row: &'a Value, key: &str) -> Result<&'a str> {
    row.get(key)
        .and_then(Value::as_str)
        .filter(|s| !s.is_empty())
        .ok_or_else(|| format!("missing or empty string field {key}").into())
}

pub fn blimp(
    predictor: &Predictor<'_>,
    input: &Path,
    mut samples: Option<&mut dyn Write>,
) -> Result<Value> {
    let mut files = fs::read_dir(input)?
        .map(|entry| entry.map(|e| e.path()))
        .collect::<std::io::Result<Vec<_>>>()?;
    files.retain(|p| p.extension().is_some_and(|e| e == "jsonl"));
    files.sort();
    if files.is_empty() {
        return Err("BLiMP input must contain .jsonl files".into());
    }
    let mut total = Counts::default();
    let mut groups = BTreeMap::<String, Counts>::new();
    let mut paradigms = BTreeMap::<String, Value>::new();
    let mut lengths = BTreeMap::<String, Counts>::new();
    let mut all_ids_complete = true;
    for path in files {
        let mut count = Counts::default();
        let mut ids = BTreeSet::new();
        let mut uid = String::new();
        let mut category = String::new();
        for (line, row) in BufReader::new(File::open(&path)?).lines().enumerate() {
            let row = row?;
            if row.trim().is_empty() {
                continue;
            }
            let parsed: Value = serde_json::from_str(&row)
                .map_err(|e| format!("{}:{}: {e}", path.display(), line + 1))?;
            let current_uid = string(&parsed, "UID")?;
            // Match the authors' published category grouping for the animacy tasks.
            let raw_category = string(&parsed, "linguistics_term")?;
            let current_category = if raw_category == "s-selection" {
                "argument_structure"
            } else {
                raw_category
            };
            if uid.is_empty() {
                uid = current_uid.to_owned();
                category = current_category.to_owned();
            } else if uid != current_uid || category != current_category {
                return Err(format!("{}: mixed paradigms/categories", path.display()).into());
            }
            let id = parsed["pairID"]
                .as_u64()
                .or_else(|| parsed["pairID"].as_str().and_then(|s| s.parse().ok()))
                .ok_or("invalid BLiMP pairID")?;
            if !ids.insert(id) {
                return Err(format!("duplicate pairID {id} in {uid}").into());
            }
            let good = string(&parsed, "sentence_good")?;
            let bad = string(&parsed, "sentence_bad")?;
            let good_bits = sentence_bits(predictor, good.as_bytes());
            let bad_bits = sentence_bits(predictor, bad.as_bytes());
            if !good_bits.is_finite() || !bad_bits.is_finite() {
                return Err("non-finite sentence probability".into());
            }
            count.record(good_bits, bad_bits);
            total.record(good_bits, bad_bits);
            groups
                .entry(category.clone())
                .or_default()
                .record(good_bits, bad_bits);
            let length_group = match good.len().cmp(&bad.len()) {
                std::cmp::Ordering::Less => "good_shorter_bytes",
                std::cmp::Ordering::Equal => "same_bytes",
                std::cmp::Ordering::Greater => "good_longer_bytes",
            };
            lengths
                .entry(length_group.into())
                .or_default()
                .record(good_bits, bad_bits);
            if let Some(out) = samples.as_deref_mut() {
                serde_json::to_writer(
                    &mut *out,
                    &json!({"uid":uid,"pair_id":id,
                    "category":category,"source_category":raw_category,"sentence_good":good,"sentence_bad":bad,
                    "good_bits":good_bits,"bad_bits":bad_bits,"correct":good_bits < bad_bits}),
                )?;
                writeln!(out)?;
            }
        }
        if count.pairs == 0 {
            return Err(format!("empty BLiMP file {}", path.display()).into());
        }
        all_ids_complete &=
            ids.len() == 1000 && ids.first() == Some(&0) && ids.last() == Some(&999);
        let mut value = count.value();
        value["category"] = json!(category);
        if paradigms.insert(uid.clone(), value).is_some() {
            return Err(format!("duplicate paradigm {uid}").into());
        }
    }
    let full_suite = paradigms.len() == 67 && all_ids_complete;
    let macro_accuracy = paradigms
        .values()
        .map(|v| v["accuracy"].as_f64().unwrap())
        .sum::<f64>()
        / paradigms.len() as f64;
    Ok(
        json!({"benchmark":"BLiMP", "full_suite":full_suite,"overall":total.value(),
        "paradigm_macro_accuracy":macro_accuracy,"paradigms":paradigms,
        "categories":groups.into_iter().map(|(k,v)|(k,v.value())).collect::<BTreeMap<_,_>>(),
        "byte_length_diagnostics":lengths.into_iter().map(|(k,v)|(k,v.value())).collect::<BTreeMap<_,_>>(),
        "category_mapping":{"s-selection":"argument_structure"},
        "protocol":"Full-sentence byte log probability including first byte; no added BOS/EOS; no length normalization; strict P(good)>P(bad), ties count as incorrect. Each sentence starts with empty history. Graph only, no match-copy, no fitting on benchmark."}),
    )
}

/// Join reference rows by UID, rather than averaging the already-aggregated rows.
pub fn add_references(report: &mut Value, reference: &Path) -> Result<()> {
    let mut refs = BTreeMap::new();
    for row in BufReader::new(File::open(reference)?).lines() {
        let row = row?;
        if row.trim().is_empty() {
            continue;
        }
        let value: Value = serde_json::from_str(&row)?;
        let uid = string(&value, "UID")?.to_owned();
        if uid == "overall" {
            continue;
        }
        if refs.insert(uid.clone(), value).is_some() {
            return Err(format!("duplicate reference UID {uid}").into());
        }
    }
    let paradigms = report["paradigms"]
        .as_object_mut()
        .ok_or("missing paradigms")?;
    let mut totals = BTreeMap::<String, f64>::new();
    let mut groups = BTreeMap::<String, BTreeMap<String, f64>>::new();
    let mut group_weights = BTreeMap::<String, f64>::new();
    let mut weight = 0.0;
    for (uid, result) in paradigms {
        let row = refs
            .get(uid)
            .ok_or_else(|| format!("no reference for {uid}"))?;
        let category = result["category"]
            .as_str()
            .ok_or("missing category")?
            .to_owned();
        if row["linguistics_term"].as_str() != Some(&category) {
            return Err(format!("reference category mismatch for {uid}").into());
        }
        let n = result["pairs"].as_u64().ok_or("missing pair count")? as f64;
        weight += n;
        *group_weights.entry(category.clone()).or_default() += n;
        let mut scores = BTreeMap::new();
        for model in ["ngram", "lstm", "txl", "gpt2"] {
            let score = row[model]
                .as_f64()
                .filter(|p| (0.0..=1.0).contains(p))
                .ok_or_else(|| format!("invalid reference score for {uid}/{model}"))?;
            *totals.entry(model.into()).or_default() += score * n;
            *groups
                .entry(category.clone())
                .or_default()
                .entry(model.into())
                .or_default() += score * n;
            scores.insert(model, score);
        }
        result["published_reference_accuracy"] = json!(scores);
    }
    for score in totals.values_mut() {
        *score /= weight;
    }
    for (category, scores) in &mut groups {
        for score in scores.values_mut() {
            *score /= group_weights[category];
        }
    }
    report["published_references"] = json!({"overall_accuracy":totals,"categories":groups,
        "source_file":reference,"source_url":"https://github.com/alexwarstadt/blimp/blob/3e56b06fcabca9b30822fc66435fca6b1aa40bb1/raw_results/summary/models_summary.jsonl",
        "labels":{"ngram":"Published word 5-gram (Gigaword)","lstm":"Published Gulordava LSTM (Wikipedia)",
            "txl":"Published Transformer-XL Large (WikiText-103)","gpt2":"Published GPT-2 Large, 774M parameters (WebText)"},
        "comparison":"External published baselines, not rerun locally. Training corpora, budgets, tokenization and boundary handling differ. Reference rows describe 1000 pairs each; only a full-suite run is comparable to the full published aggregate."});
    Ok(())
}
