use growing_byte_model::{
    benchmark,
    graph::{Model, Params},
    model_io,
    predict::Predictor,
};
use serde_json::{Value, json};
use std::{
    fs,
    path::PathBuf,
    process::Command,
    time::{SystemTime, UNIX_EPOCH},
};

struct Fixture(PathBuf);
impl Fixture {
    fn new() -> Self {
        let dir = std::env::temp_dir().join(format!(
            "blimp-test-{}-{}",
            std::process::id(),
            SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        fs::create_dir_all(dir.join("data")).unwrap();
        Self(dir)
    }
    fn rows(&self, uid: &str, pairs: &[(&str, &str)]) {
        let rows: Vec<_> = pairs
            .iter()
            .enumerate()
            .map(|(i, (good, bad))| {
                json!({"UID":uid,"linguistics_term":"syntax","pairID":i.to_string(),
                "sentence_good":good,"sentence_bad":bad})
                .to_string()
            })
            .collect();
        fs::write(
            self.0.join("data").join(format!("{uid}.jsonl")),
            rows.join("\n"),
        )
        .unwrap();
    }
}
impl Drop for Fixture {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.0);
    }
}
fn model() -> Model {
    Model::root(Params::default(), &[b"aaaaaabbbb".to_vec()])
}

#[test]
fn source_animacy_label_uses_the_published_argument_structure_group() {
    let dir = Fixture::new();
    let row = json!({"UID":"animate_subject_passive","linguistics_term":"s-selection",
        "pairID":0,"sentence_good":"a","sentence_bad":"b"});
    fs::write(
        dir.0.join("data/animate_subject_passive.jsonl"),
        row.to_string(),
    )
    .unwrap();
    let model = model();
    let report = benchmark::blimp(&Predictor::new(&model), &dir.0.join("data"), None).unwrap();
    assert_eq!(report["categories"]["argument_structure"]["pairs"], 1);
    assert!(report["categories"].get("s-selection").is_none());
}

#[test]
fn sentence_score_matches_existing_evaluator_and_keeps_total_length_cost() {
    let model = model();
    let predictor = Predictor::new(&model);
    let text = b"aabb\x00\xff".to_vec();
    let expected = predictor
        .evaluate(std::slice::from_ref(&text), false, usize::MAX)
        .bits;
    assert!((benchmark::sentence_bits(&predictor, &text) - expected).abs() < 1e-12);
    // Per-byte normalization would incorrectly favor 'aa' here.
    assert!(
        benchmark::sentence_bits(&predictor, b"aa") > benchmark::sentence_bits(&predictor, b"b")
    );
    assert!(
        benchmark::sentence_bits(&predictor, b"a") < benchmark::sentence_bits(&predictor, b"b")
    );
}

#[test]
fn aggregation_counts_ties_and_joins_only_matching_reference_paradigms() {
    let dir = Fixture::new();
    dir.rows("a", &[("a", "b"), ("aa", "b")]);
    dir.rows("b", &[("aa", "aa")]);
    let model = model();
    let mut samples = Vec::new();
    let mut report = benchmark::blimp(
        &Predictor::new(&model),
        &dir.0.join("data"),
        Some(&mut samples),
    )
    .unwrap();
    assert_eq!(report["overall"]["pairs"], 3);
    assert_eq!(report["overall"]["correct"], 1);
    assert_eq!(report["overall"]["ties"], 1);
    assert_eq!(report["full_suite"], false);
    assert_eq!(report["paradigm_macro_accuracy"], 0.25);
    let parsed: Vec<Value> = String::from_utf8(samples)
        .unwrap()
        .lines()
        .map(|s| serde_json::from_str(s).unwrap())
        .collect();
    assert_eq!(parsed.len(), 3);
    assert_eq!(parsed[0]["correct"], true);
    let refs = dir.0.join("reference.jsonl");
    let rows: Vec<_> = [("a",0.3),("b",0.9),("overall",0.99)].iter().map(|(uid,score)|
        json!({"UID":uid,"linguistics_term":"syntax","ngram":score,"lstm":score,"txl":score,"gpt2":score}).to_string()).collect();
    fs::write(&refs, rows.join("\n")).unwrap();
    benchmark::add_references(&mut report, &refs).unwrap();
    assert!(
        (report["published_references"]["overall_accuracy"]["lstm"]
            .as_f64()
            .unwrap()
            - 0.5)
            .abs()
            < 1e-12
    );
    dir.rows("a", &[("a", "b"), ("aa", "b")]);
    let path = dir.0.join("data/a.jsonl");
    let line = fs::read_to_string(&path)
        .unwrap()
        .lines()
        .next()
        .unwrap()
        .to_owned();
    fs::write(&path, format!("{line}\n{line}\n")).unwrap();
    assert!(benchmark::blimp(&Predictor::new(&model), &dir.0.join("data"), None).is_err());
}

#[test]
fn cli_rejects_partial_suites_by_default_and_protects_inputs() {
    let dir = Fixture::new();
    dir.rows("fixture", &[("a", "b")]);
    let path = dir.0.join("test.model");
    model_io::save(&path, &model()).unwrap();
    let exe = env!("CARGO_BIN_EXE_growing-byte-model");
    let run = |args: &[&str]| {
        Command::new(exe)
            .current_dir(&dir.0)
            .args(["benchmark", "--model", "test.model", "--input", "data"])
            .args(args)
            .output()
            .unwrap()
    };
    assert!(!run(&["--report", "report.json"]).status.success());
    let ok = run(&[
        "--report",
        "report.json",
        "--samples",
        "samples.jsonl",
        "--allow-partial",
    ]);
    assert!(
        ok.status.success(),
        "{}",
        String::from_utf8_lossy(&ok.stderr)
    );
    let report: Value =
        serde_json::from_slice(&fs::read(dir.0.join("report.json")).unwrap()).unwrap();
    assert_eq!(report["overall"]["accuracy"], 1.0);
    assert!(
        !run(&["--report", "test.model", "--allow-partial"])
            .status
            .success()
    );
    assert!(
        !run(&["--report", "data/fixture.jsonl", "--allow-partial"])
            .status
            .success()
    );
    assert!(
        !run(&[
            "--report",
            "report.json",
            "--samples",
            "report.json",
            "--allow-partial"
        ])
        .status
        .success()
    );
    assert!(model_io::load(path).is_ok());
}
