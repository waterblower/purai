//! Assembled-program tests verify executable capability, not learned competence.
//! No test in this file runs the training search.
use afrn_v5::{
    data::Document,
    ir::{Arithmetic, Op, Program, Registry, Value},
    model::{self, Expert, Histogram, Model},
    proposal,
    vm::{Limits, Prediction, Runtime},
};
use std::{
    collections::BTreeMap,
    fs,
    time::{SystemTime, UNIX_EPOCH},
};

fn observe(vm: &mut Runtime, p: &Program, r: &Registry, text: &[u8]) {
    for &b in text {
        vm.observe(b, p, r, &Limits::default());
    }
}
fn byte(vm: &mut Runtime, p: &Program, r: &Registry) -> Option<u8> {
    match vm.predict(p, r, &Limits::default(), 32768) {
        Some(Prediction::Byte(b)) => Some(b),
        _ => None,
    }
}
fn temporary() -> std::path::PathBuf {
    let p = std::env::temp_dir().join(format!(
        "afrn-v5-test-{}-{}",
        std::process::id(),
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    ));
    fs::create_dir(&p).unwrap();
    p
}

#[test]
fn far_history_changes_answer_despite_identical_suffix_and_new_entity() {
    let mut r = Registry::default();
    let p = proposal::associative(b':', b';', b'?', &mut r);
    p.validate(&r).unwrap();
    let mut a = Runtime::default();
    let mut b = Runtime::default();
    observe(&mut a, &p, &r, b"unseen_entity:R;");
    observe(&mut b, &p, &r, b"unseen_entity:B;");
    let suffix = format!("{}unseen_entity?", "noise:x;".repeat(4096));
    observe(&mut a, &p, &r, suffix.as_bytes());
    observe(&mut b, &p, &r, suffix.as_bytes());
    assert_eq!(byte(&mut a, &p, &r), Some(b'R'));
    assert_eq!(byte(&mut b, &p, &r), Some(b'B'));
    assert!(a.valid && b.valid);
    assert!(a.counters.peak_accounted_bytes < 100_000);
}

#[test]
fn latest_assignment_wins_and_documents_do_not_share_runtime_memory() {
    let mut r = Registry::default();
    let p = proposal::associative(b':', b';', b'?', &mut r);
    let mut vm = Runtime::default();
    observe(&mut vm, &p, &r, b"key:red;key:blue;key?");
    for expected in b"blue" {
        assert_eq!(byte(&mut vm, &p, &r), Some(*expected));
        vm.observe(*expected, &p, &r, &Limits::default());
    }
    let mut empty = Runtime::default();
    observe(&mut empty, &p, &r, b"key?");
    assert_eq!(byte(&mut empty, &p, &r), None);
}

#[test]
fn arithmetic_uses_bound_operands_not_precomputed_answer_lookup() {
    let mut r = Registry::default();
    let p = proposal::arithmetic(b'*', b'=', b'\n', Arithmetic::Multiply, &mut r);
    let mut vm = Runtime::default();
    observe(&mut vm, &p, &r, b"123*45=");
    for expected in b"5535" {
        assert_eq!(byte(&mut vm, &p, &r), Some(*expected));
        vm.observe(*expected, &p, &r, &Limits::default());
    }
    assert_eq!(byte(&mut vm, &p, &r), None);
    let mut overflow = Runtime::default();
    observe(&mut overflow, &p, &r, b"9223372036854775807*9=");
    assert_eq!(byte(&mut overflow, &p, &r), None);
}

fn loop_program() -> Program {
    Program {
        observe: vec![Op::Input { dst: 1 }],
        predict: vec![
            Op::Set {
                dst: 2,
                value: Value::Int(0),
            },
            Op::Set {
                dst: 3,
                value: Value::Int(1),
            },
            Op::WhileNonzero {
                reg: 1,
                body: vec![
                    Op::Arithmetic {
                        dst: 2,
                        a: 2,
                        b: 3,
                        kind: Arithmetic::Add,
                    },
                    Op::Arithmetic {
                        dst: 1,
                        a: 1,
                        b: 3,
                        kind: Arithmetic::Subtract,
                    },
                ],
            },
            Op::FormatDecimal { dst: 6, src: 2 },
            Op::Set {
                dst: 7,
                value: Value::Int(0),
            },
            Op::Emit { bytes: 6, index: 7 },
        ],
    }
}

#[test]
fn internal_loop_work_varies_independently_of_single_byte_output() {
    let p = loop_program();
    let r = Registry::default();
    p.validate(&r).unwrap();
    let mut short = Runtime::default();
    let mut long = Runtime::default();
    observe(&mut short, &p, &r, &[3]);
    observe(&mut long, &p, &r, &[9]);
    assert_eq!(byte(&mut short, &p, &r), Some(b'3'));
    assert_eq!(byte(&mut long, &p, &r), Some(b'9'));
    assert!(long.counters.instructions > short.counters.instructions);
    assert_eq!(long.regs[1], Value::Int(9));
}

#[test]
fn fuel_exhaustion_abstains_and_more_compute_can_recover_without_extra_output() {
    let p = loop_program();
    let r = Registry::default();
    let mut vm = Runtime::default();
    observe(&mut vm, &p, &r, &[9]);
    let before = vm.counters.work;
    assert_eq!(vm.predict(&p, &r, &Limits::default(), 130), None);
    assert!(vm.counters.work - before <= 130);
    assert_eq!(vm.counters.exhaustions, 1);
    assert_eq!(byte(&mut vm, &p, &r), Some(b'9'));
}

#[test]
fn malicious_nonterminating_loop_is_bounded() {
    let p = Program {
        observe: vec![],
        predict: vec![
            Op::Set {
                dst: 1,
                value: Value::Int(1),
            },
            Op::WhileNonzero {
                reg: 1,
                body: vec![],
            },
        ],
    };
    let mut vm = Runtime::default();
    assert_eq!(
        vm.predict(&p, &Registry::default(), &Limits::default(), 1000),
        None
    );
    assert_eq!(vm.counters.work, 1000);
    assert_eq!(vm.counters.exhaustions, 1);
}

#[test]
fn prediction_cannot_read_future_input_or_commit_memory_writes() {
    for op in [Op::Input { dst: 1 }, Op::Put { key: 1, value: 2 }] {
        let p = Program {
            observe: vec![],
            predict: vec![op],
        };
        let mut vm = Runtime::default();
        assert_eq!(
            vm.predict(&p, &Registry::default(), &Limits::default(), 1000),
            None
        );
        assert_eq!(vm.counters.exhaustions, 1);
        assert!(vm.valid);
    }
}

#[test]
fn observation_failure_invalidates_partial_state() {
    let p = proposal::context(10);
    let mut vm = Runtime::default();
    let limits = Limits {
        observe_fuel: 1,
        ..Limits::default()
    };
    vm.observe(b'x', &p, &Registry::default(), &limits);
    assert!(!vm.valid);
    assert_eq!(
        vm.predict(&p, &Registry::default(), &Limits::default(), 1000),
        None
    );
}

#[test]
fn bounded_memory_evicts_without_growing_with_document_length() {
    let mut r = Registry::default();
    let p = proposal::associative(b':', b';', b'?', &mut r);
    let limits = Limits {
        memory_entries: 1,
        ..Limits::default()
    };
    let mut vm = Runtime::default();
    for &b in b"a:R;b:B;a?" {
        vm.observe(b, &p, &r, &limits);
    }
    assert_eq!(byte(&mut vm, &p, &r), None);
    assert_eq!(vm.counters.evictions, 1);
}

#[test]
fn registry_union_is_associative_commutative_idempotent_and_checks_identity() {
    let mut a = Registry::default();
    a.intern(vec![Op::Set {
        dst: 0,
        value: Value::Int(1),
    }]);
    let mut b = Registry::default();
    b.intern(vec![Op::Set {
        dst: 0,
        value: Value::Int(2),
    }]);
    let mut c = Registry::default();
    c.intern(vec![Op::Return]);
    let mut ab = a.clone();
    ab.merge(&b).unwrap();
    let mut ba = b.clone();
    ba.merge(&a).unwrap();
    assert_eq!(ab, ba);
    let old = ab.clone();
    ab.merge(&old).unwrap();
    assert_eq!(ab, old);
    ab.merge(&c).unwrap();
    let mut bc = b.clone();
    bc.merge(&c).unwrap();
    a.merge(&bc).unwrap();
    assert_eq!(a, ab);
    let mut bad = Registry::default();
    bad.0.insert("wrong".into(), vec![Op::Return]);
    assert!(a.merge(&bad).is_err());
}

#[test]
fn model_is_checksummed_roundtrippable_and_refuses_overwrite() {
    let dir = temporary();
    let file = dir.join("x.model");
    let mut m = Model::new(Limits::default(), 8);
    let doc = Document::new(vec![0, 255, 128, 0]);
    m.observe_document(&doc).unwrap();
    m.observe_document(&doc).unwrap();
    assert_eq!(m.root_evidence.len(), 1);
    assert_eq!(m.root_evidence[&doc.id].total(), 4.0);
    model::save(&file, &m).unwrap();
    assert_eq!(model::load(&file).unwrap(), m);
    assert!(model::save(&file, &m).is_err());
    let mut bytes = fs::read(&file).unwrap();
    *bytes.last_mut().unwrap() ^= 1;
    fs::write(dir.join("bad.model"), bytes).unwrap();
    assert!(model::load(&dir.join("bad.model")).is_err());
    fs::remove_dir_all(dir).unwrap();
}

#[test]
fn probability_mixture_remains_normalized_and_zero_budget_is_root_fallback() {
    let mut m = Model::new(Limits::default(), 8);
    let p = proposal::associative(b':', b';', b'?', &mut m.registry);
    m.experts.push(Expert {
        program: p,
        weight: 0.95,
        evidence: BTreeMap::new(),
    });
    m.validate().unwrap();
    let compiled = m.compile().unwrap();
    let mut s = compiled.session();
    for &b in b"a:R;a?" {
        s.observe(b);
    }
    let before = s.counters().work;
    let root = s.distribution(Some(0));
    assert_eq!(root, m.root().unwrap());
    assert_eq!(s.counters().work - before, 256);
    let p = s.distribution(None);
    assert!((p.iter().sum::<f64>() - 1.0).abs() < 1e-12);
    assert!(p.iter().all(|x| *x > 0.0));
    assert!(p[b'R' as usize] > 0.9);
}

#[test]
fn invalid_ir_and_histogram_overflow_are_errors() {
    let p = Program {
        observe: vec![Op::Input { dst: 16 }],
        predict: vec![],
    };
    assert!(p.validate(&Registry::default()).is_err());
    let p = Program {
        observe: vec![Op::Call {
            module: "missing".into(),
        }],
        predict: vec![],
    };
    assert!(p.validate(&Registry::default()).is_err());
    let mut h = Histogram::default();
    h.add(0, u64::MAX).unwrap();
    assert!(h.add(0, 1).is_err());
}

#[test]
fn binary_jsonl_and_explicit_file_caps() {
    let dir = temporary();
    let file = dir.join("bytes.jsonl");
    fs::write(&file, "{\"bytes\":[0,255,128]}\n").unwrap();
    let docs = afrn_v5::data::read(&file, 100).unwrap();
    assert_eq!(docs[0].bytes, vec![0, 255, 128]);
    assert!(afrn_v5::data::read(&file, 2).is_err());
    fs::remove_dir_all(dir).unwrap();
}

#[test]
fn duplicate_document_leakage_is_rejected() {
    let docs = vec![
        Document::new(b"a".to_vec()),
        Document::new(b"b".to_vec()),
        Document::new(b"c".to_vec()),
    ];
    assert!(afrn_v5::data::split(docs.clone(), Some(docs)).is_err());
}

#[test]
fn module_factoring_preserves_unbudgeted_program_behavior() {
    let p = proposal::context(4);
    let mut programs = vec![p.clone(), p.clone()];
    let mut registry = Registry::default();
    programs[0].observe.extend(vec![
        Op::Set {
            dst: 12,
            value: Value::Bytes(vec![7; 128])
        };
        3
    ]);
    programs[1] = programs[0].clone();
    let original = programs[0].clone();
    assert!(proposal::factor(&mut programs, &mut registry) > 0);
    let mut a = Runtime::default();
    let mut b = Runtime::default();
    observe(&mut a, &original, &Registry::default(), b"abcdefgh");
    observe(&mut b, &programs[0], &registry, b"abcdefgh");
    assert_eq!(
        a.predict(&original, &Registry::default(), &Limits::default(), 32768),
        b.predict(&programs[0], &registry, &Limits::default(), 32768)
    );
}
