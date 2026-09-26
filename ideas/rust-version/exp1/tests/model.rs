use growing_byte_model::{
    data,
    graph::{ByteSet, Compiled, Histogram, Model, Params},
    math::{self, Rng},
    model_io,
    predict::{self, GenerateOptions, MatchState, Predictor},
    train::{self, Options},
};

fn corpus(seed: u64, n: usize) -> Vec<Vec<u8>> {
    let mut rng = Rng(seed);
    let subjects = ["the king", "the queen", "a small girl", "the little dog"];
    let verbs = [" likes ", " sees ", " finds "];
    let objects = ["the red ball", "the big tree", "a little bird"];
    (0..n)
        .map(|_| {
            format!(
                "{}{}{}.\n",
                subjects[(rng.uniform() * 4.0) as usize],
                verbs[(rng.uniform() * 3.0) as usize],
                objects[(rng.uniform() * 3.0) as usize]
            )
            .into_bytes()
        })
        .collect()
}
fn options() -> Options {
    Options {
        rounds: 6,
        max_depth: 6,
        max_nodes: 3000,
        min_count: 2,
        structure_weight: 0.1,
        merges_per_round: 12,
        ..Options::default()
    }
}
fn learned() -> Model {
    train::train(&corpus(1, 120), Params::default(), &options(), |_, _| {}).unwrap()
}

#[test]
fn controlled_training_stops_without_running_another_round() {
    use std::ops::ControlFlow;
    let docs = corpus(1, 120);
    let opts = Options {
        rounds: 36,
        ..options()
    };
    let mut visited = Vec::new();
    let mut checkpoint = Vec::new();
    let model = train::train_controlled(&docs, Params::default(), &opts, |model, report| {
        visited.push(report.round);
        if report.round == 2 {
            checkpoint = model_io::encode(model).unwrap();
            ControlFlow::Break(())
        } else {
            ControlFlow::Continue(())
        }
    })
    .unwrap();
    assert_eq!(visited, [1, 2]);
    assert_eq!(model_io::encode(&model).unwrap(), checkpoint);
}

#[test]
fn topology_grows_and_generalizes_on_new_sentences() {
    let docs = corpus(1, 120);
    let valid = corpus(2, 50);
    let root = Model::root(Params::default(), &docs);
    let model = train::train(&docs, Params::default(), &options(), |_, _| {}).unwrap();
    assert!(model.nodes.len() > 10);
    assert!(model.max_depth() > 1);
    let before = Predictor::new(&root)
        .evaluate(&valid, false, usize::MAX)
        .bpb();
    let after = Predictor::new(&model)
        .evaluate(&valid, false, usize::MAX)
        .bpb();
    assert!(after < before * 0.65, "before={before}, after={after}");
}

#[test]
fn every_round_has_exact_recounted_evidence() {
    let docs = corpus(7, 80);
    train::train(&docs, Params::default(), &options(), |model, _| {
        model.validate().unwrap();
        let counts = train::recount(model, &docs);
        for (node, count) in model.nodes.iter().zip(counts) {
            assert_eq!(node.counts, count);
        }
    })
    .unwrap();
}

#[test]
fn training_and_model_encoding_are_deterministic() {
    assert_eq!(
        model_io::encode(&learned()).unwrap(),
        model_io::encode(&learned()).unwrap()
    );
}

#[test]
fn binary_round_trip_preserves_predictions_and_calibration() {
    let mut model = learned();
    predict::calibrate(&mut model, &corpus(8, 40));
    let encoded = model_io::encode(&model).unwrap();
    let decoded = model_io::decode(&encoded).unwrap();
    assert_eq!(model, decoded);
    assert_eq!(encoded, model_io::encode(&decoded).unwrap());
    let valid = corpus(9, 30);
    assert_eq!(
        Predictor::new(&model)
            .evaluate(&valid, true, usize::MAX)
            .bits,
        Predictor::new(&decoded)
            .evaluate(&valid, true, usize::MAX)
            .bits
    );
}

#[test]
fn corrupt_truncated_and_unknown_version_models_are_rejected() {
    let bytes = model_io::encode(&learned()).unwrap();
    for n in [0, 7, 20, bytes.len() - 1] {
        assert!(model_io::decode(&bytes[..n]).is_err());
    }
    let mut bad = bytes.clone();
    bad[30] ^= 255;
    assert!(model_io::decode(&bad).is_err());
    let mut bad = bytes;
    bad[8] = 99;
    let end = bad.len() - 8;
    let sum = model_io::checksum(&bad[..end]);
    bad[end..].copy_from_slice(&sum.to_le_bytes());
    assert!(model_io::decode(&bad).is_err());
}

#[test]
fn malformed_topology_is_rejected() {
    let mut m = learned();
    m.nodes[1].parent = Some(1);
    assert!(m.validate().is_err());
    let mut m = learned();
    let c = m.nodes[0].children[0];
    m.nodes[0].children.push(c);
    assert!(m.validate().is_err());
}

#[test]
fn count_overflow_is_an_error_not_a_panic() {
    let mut m = Model::root(Params::default(), &[b"ab".to_vec()]);
    m.training_bytes = u64::MAX;
    m.nodes[0].counts = Histogram(vec![(b'a', u64::MAX), (b'b', 1)]);
    assert!(m.validate().is_err());
    assert!(model_io::encode(&m).is_err());
}

#[test]
fn graph_and_match_never_read_future_answers() {
    let model = learned();
    let predictor = Predictor::new(&model);
    let a = b"the king sees the king sees the red ball";
    for end in 0..a.len() {
        let mut b = a.to_vec();
        for x in &mut b[end..] {
            *x ^= 255;
        }
        let mut ma = MatchState::default();
        let mut mb = MatchState::default();
        for i in 0..end {
            ma.observe(a, i);
            mb.observe(&b, i);
        }
        let (pa, sa) = predictor.distribution(a, end, &ma, true, usize::MAX);
        let (pb, sb) = predictor.distribution(&b, end, &mb, true, usize::MAX);
        assert_eq!(pa, pb);
        assert_eq!(sa.node, sb.node);
        assert_eq!(sa.match_byte, sb.match_byte);
    }
}

#[test]
fn document_boundaries_reset_all_runtime_state() {
    let mut model = learned();
    predict::calibrate(&mut model, &corpus(6, 20));
    let p = Predictor::new(&model);
    let docs = corpus(4, 3);
    let together = p.evaluate(&docs, true, usize::MAX).bits;
    let separate: f64 = docs
        .iter()
        .map(|d| p.evaluate(std::slice::from_ref(d), true, usize::MAX).bits)
        .sum();
    assert!((together - separate).abs() < 1e-9);
    let docs = vec![b"a".to_vec(), b"b".to_vec()];
    let model = train::train(&docs, Params::default(), &options(), |_, _| {}).unwrap();
    assert_eq!(
        model.nodes.len(),
        1,
        "must not learn a cross-document context"
    );
}

#[test]
fn all_distributions_are_positive_and_normalized() {
    let mut model = learned();
    predict::calibrate(&mut model, &corpus(9, 40));
    let p = Predictor::new(&model);
    let mut mm = MatchState::default();
    let text = b"the king sees the king sees the king sees ";
    for i in 0..text.len() {
        let (out, _) = p.distribution(text, i, &mm, true, usize::MAX);
        assert!(out.iter().all(|v| v.is_finite() && *v > 0.0));
        assert!((out.iter().sum::<f64>() - 1.0).abs() < 1e-10);
        mm.observe(text, i);
    }
}

#[test]
fn budget_limits_internal_traversal_without_changing_output_length() {
    let model = learned();
    let p = Predictor::new(&model);
    let prompt = b"the king sees ";
    let mut counts = Vec::new();
    let small = p
        .generate(
            prompt,
            &GenerateOptions {
                bytes: 40,
                budget: 1,
                ..GenerateOptions::default()
            },
            |_, s, _| counts.push(s.steps),
        )
        .unwrap();
    assert!(counts.iter().all(|&s| s == 1));
    let mut long_steps = 0;
    let big = p
        .generate(
            prompt,
            &GenerateOptions {
                bytes: 40,
                ..GenerateOptions::default()
            },
            |_, s, _| long_steps += s.steps,
        )
        .unwrap();
    assert_eq!(small.len(), big.len());
    assert!(long_steps > 40);
}

#[test]
fn seeded_sampling_and_extreme_temperatures_work() {
    let model = learned();
    let p = Predictor::new(&model);
    for temperature in [0.0, 1e-12, 0.85, 1000.0] {
        let opts = GenerateOptions {
            bytes: 30,
            temperature,
            top_p: 0.9,
            ..GenerateOptions::default()
        };
        let a = p.generate(b"the king", &opts, |_, _, _| {}).unwrap();
        let b = p.generate(b"the king", &opts, |_, _, _| {}).unwrap();
        assert_eq!(a, b);
    }
    assert!(
        p.generate(
            b"",
            &GenerateOptions {
                temperature: f64::NAN,
                ..GenerateOptions::default()
            },
            |_, _, _| {}
        )
        .is_err()
    );
}

#[test]
fn exact_binary_bytes_are_supported() {
    let text: Vec<u8> = (0..=255).cycle().take(4096).collect();
    let model = train::train(
        &[text],
        Params::default(),
        &Options {
            rounds: 3,
            merges_per_round: 0,
            ..options()
        },
        |_, _| {},
    )
    .unwrap();
    let p = Predictor::new(&model);
    let bytes = p
        .generate(
            &[0, 1, 2, 3],
            &GenerateOptions {
                bytes: 32,
                temperature: 0.0,
                ..GenerateOptions::default()
            },
            |_, _, _| {},
        )
        .unwrap();
    assert_eq!(bytes, (4..36).collect::<Vec<u8>>());
}

#[test]
fn learned_merging_reduces_topology_and_preserves_evidence() {
    let docs = vec![b"ax".to_vec(), b"bx".to_vec()]
        .into_iter()
        .cycle()
        .take(200)
        .collect::<Vec<_>>();
    let mut m = Model::root(Params::default(), &docs);
    let mut c = Histogram::default();
    c.add(b'x', 100);
    m.add_child(0, ByteSet::singleton(b'a'), c.clone());
    m.add_child(0, ByteSet::singleton(b'b'), c);
    let (merged, _) = train::merge_leaves(&mut m, &options());
    assert_eq!(merged, 1);
    assert_eq!(m.nodes.len(), 2);
    assert_eq!(m.classes(), 1);
    m.validate().unwrap();
    assert_eq!(m.nodes[1].counts, train::recount(&m, &docs)[1]);
    let c = Compiled::new(&m);
    assert_eq!(
        c.context(b"ax", 1, usize::MAX).0,
        c.context(b"bx", 1, usize::MAX).0
    );
}

#[test]
fn pruning_removes_unjustified_leaf() {
    let docs = vec![b"aa".to_vec(); 100];
    let mut m = Model::root(Params::default(), &docs);
    let mut c = Histogram::default();
    c.add(b'a', 100);
    m.add_child(0, ByteSet::singleton(b'a'), c);
    assert_eq!(train::prune(&mut m, 1e6), 1);
    m.validate().unwrap();
    assert_eq!(m.nodes.len(), 1);
}

#[test]
fn calibration_does_not_change_graph_or_training_evidence() {
    let mut m = learned();
    let before = m.nodes.clone();
    predict::calibrate(&mut m, &corpus(55, 10));
    assert_eq!(m.nodes, before);
    m.validate().unwrap();
}

#[test]
fn max_nodes_is_a_hard_growth_limit() {
    let m = train::train(
        &corpus(11, 100),
        Params::default(),
        &Options {
            max_nodes: 8,
            ..options()
        },
        |m, _| assert!(m.nodes.len() <= 8),
    )
    .unwrap();
    assert!(m.nodes.len() <= 8);
    m.validate().unwrap();
}

#[test]
fn split_keeps_documents_separate() {
    let docs = vec![b"one".to_vec(), b"two".to_vec(), b"three".to_vec()];
    let (a, b) = data::split_tail(docs.clone(), 0.1).unwrap();
    assert_eq!(a, docs[..2]);
    assert_eq!(b, docs[2..]);
    let (a, b) = data::split_tail(vec![b"abcdef".to_vec()], 0.5).unwrap();
    assert_eq!(a[0], b"abc");
    assert_eq!(b[0], b"def");
}

#[test]
fn math_is_consistent() {
    assert!(math::lgamma(1.0).abs() < 1e-12);
    assert!((math::lgamma(5.0) - 24f64.ln()).abs() < 1e-12);
    let mut h = Histogram::default();
    h.add(65, 3);
    h.add(66, 7);
    let p = math::smooth(&h, &math::UNIFORM, 0.75, 2.0);
    assert!((p.iter().sum::<f64>() - 1.0).abs() < 1e-12);
    assert!(math::py_bits(&h, &math::UNIFORM, 0.75, 2.0) > 0.0);
}
