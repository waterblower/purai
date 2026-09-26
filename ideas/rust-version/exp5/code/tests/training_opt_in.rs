//! Compiled but deliberately not executed by default: code only, no training run.
use afrn_v5::{
    data::{Document, Split},
    model::Model,
    train::{self, Options},
    vm::Limits,
};

#[test]
#[ignore = "runs a miniature program-search experiment; opt in when training is authorized"]
fn learns_a_predictive_program() {
    fn docs(start: usize) -> Vec<Document> {
        (start..start + 8)
            .map(|i| Document::new(format!("k{i}:R;x:B;k{i}?R;{}", "abR;".repeat(30)).into_bytes()))
            .collect()
    }
    let split = Split {
        train: docs(0),
        calibration: docs(20),
        development: docs(40),
    };
    let opts = Options {
        rounds: 4,
        candidates_per_round: 32,
        max_seconds: 60.0,
        max_work: 500_000_000,
        mdl_weight: 0.0,
        work_weight: 0.0,
        ..Options::default()
    };
    let (m, r) = train::train(Model::new(Limits::default(), 64), split, opts, |_| {}).unwrap();
    assert!(!m.experts.is_empty());
    assert!(r.final_development.bpb().is_finite());
    m.validate().unwrap();
}
