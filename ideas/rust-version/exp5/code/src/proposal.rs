//! Search proposals, not semantic labels. These program skeletons are an explicit
//! inductive bias; none is active until selected using held-out predictive loss.
use crate::{
    canonical_id,
    data::Document,
    ir::{Arithmetic, Block, Op, Program, Registry, Value},
};
use std::collections::{BTreeMap, BTreeSet};

fn set(dst: u8, n: i64) -> Op {
    Op::Set {
        dst,
        value: Value::Int(n),
    }
}
fn clear(dst: u8) -> Op {
    Op::Set {
        dst,
        value: Value::Empty,
    }
}
fn test(byte: u8, yes: Block, no: Block) -> Block {
    vec![
        set(8, byte as i64),
        Op::IfEqual {
            a: 0,
            b: 8,
            yes,
            no,
        },
    ]
}
fn advance(registry: &mut Registry) -> Op {
    // Standard register convention: r6 candidate continuation, r7 cursor,
    // r0 observed byte. The equality test invalidates a failed continuation.
    let body = vec![
        Op::At {
            dst: 9,
            bytes: 6,
            index: 7,
        },
        Op::IfEqual {
            a: 9,
            b: 0,
            yes: vec![
                set(10, 1),
                Op::Arithmetic {
                    dst: 7,
                    a: 7,
                    b: 10,
                    kind: Arithmetic::Add,
                },
            ],
            no: vec![clear(6), set(7, 0)],
        },
    ];
    Op::Call {
        module: registry.intern(body),
    }
}
pub fn context(bytes: usize) -> Program {
    Program {
        observe: vec![
            Op::Input { dst: 0 },
            Op::Tail {
                reg: 1,
                bytes: bytes.saturating_sub(1),
            },
            Op::Append { dst: 1, src: 0 },
        ],
        predict: vec![Op::Feature { regs: vec![1] }],
    }
}
pub fn associative(separator: u8, terminator: u8, query: u8, registry: &mut Registry) -> Program {
    let ordinary = vec![Op::Append { dst: 1, src: 0 }];
    let lookup = vec![Op::Get { dst: 6, key: 1 }, set(7, 0), clear(1), set(3, 0)];
    let finish = vec![
        set(8, 1),
        Op::IfEqual {
            a: 3,
            b: 8,
            yes: vec![Op::Put { key: 2, value: 1 }],
            no: vec![],
        },
        clear(1),
        set(3, 0),
    ];
    let bind = vec![Op::Copy { dst: 2, src: 1 }, clear(1), set(3, 1)];
    let mut observe = vec![Op::Input { dst: 0 }, advance(registry)];
    observe.extend(test(
        separator,
        bind,
        test(terminator, finish, test(query, lookup, ordinary)),
    ));
    Program {
        observe,
        predict: vec![Op::Emit { bytes: 6, index: 7 }],
    }
}
pub fn arithmetic(
    separator: u8,
    query: u8,
    reset: u8,
    kind: Arithmetic,
    registry: &mut Registry,
) -> Program {
    let calculate = vec![
        Op::ParseDecimal { dst: 3, src: 2 },
        Op::ParseDecimal { dst: 4, src: 1 },
        Op::Arithmetic {
            dst: 5,
            a: 3,
            b: 4,
            kind,
        },
        Op::FormatDecimal { dst: 6, src: 5 },
        set(7, 0),
        clear(1),
    ];
    let mut observe = vec![Op::Input { dst: 0 }, advance(registry)];
    observe.extend(test(
        separator,
        vec![Op::Copy { dst: 2, src: 1 }, clear(1)],
        test(
            query,
            calculate,
            test(
                reset,
                vec![clear(1), clear(2)],
                vec![Op::Append { dst: 1, src: 0 }],
            ),
        ),
    ));
    Program {
        observe,
        predict: vec![Op::Emit { bytes: 6, index: 7 }],
    }
}

pub fn alphabet(docs: &[Document]) -> Vec<u8> {
    let mut counts = [0u64; 256];
    for doc in docs {
        for &b in &doc.bytes {
            counts[b as usize] += 1;
        }
    }
    let mut bytes: Vec<u8> = (0..=255).filter(|b| counts[*b as usize] > 0).collect();
    bytes.sort_by_key(|b| {
        (
            b.is_ascii_alphanumeric(),
            std::cmp::Reverse(counts[*b as usize]),
            *b,
        )
    });
    bytes.truncate(8);
    bytes
}

pub fn seeds(alphabet: &[u8], registry: &mut Registry) -> Vec<Program> {
    let mut result = vec![
        context(1),
        context(2),
        context(4),
        context(8),
        context(16),
        context(32),
        context(64),
    ];
    // Interleave procedural proposals so a small per-round quota eventually
    // explores each family rather than exhausting all context sizes first.
    for &a in alphabet {
        for &b in alphabet {
            for &c in alphabet {
                if a == b || b == c || a == c {
                    continue;
                }
                result.push(associative(a, b, c, registry));
                for kind in [Arithmetic::Add, Arithmetic::Subtract, Arithmetic::Multiply] {
                    result.push(arithmetic(a, b, c, kind, registry));
                }
            }
        }
    }
    result
}

pub struct Rng(pub u64);
impl Rng {
    pub fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9e3779b97f4a7c15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d049bb133111eb);
        z ^ (z >> 31)
    }
    pub fn index(&mut self, n: usize) -> usize {
        (self.next_u64() % n.max(1) as u64) as usize
    }
    pub fn uniform(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 / ((1u64 << 53) as f64)
    }
}

fn mutate_block(block: &mut Block, alphabet: &[u8], rng: &mut Rng, depth: usize) {
    if block.is_empty() {
        block.push(Op::Feature { regs: vec![1] });
        return;
    }
    let i = rng.index(block.len());
    match &mut block[i] {
        Op::IfEqual { yes, no, a, b } if depth < 8 => match rng.index(4) {
            0 => std::mem::swap(yes, no),
            1 => *a = rng.index(16) as u8,
            2 => *b = rng.index(16) as u8,
            _ => mutate_block(
                if rng.index(2) == 0 { yes } else { no },
                alphabet,
                rng,
                depth + 1,
            ),
        },
        Op::WhileNonzero { body, .. } if depth < 8 => mutate_block(body, alphabet, rng, depth + 1),
        Op::Set {
            value: Value::Int(n),
            ..
        } => {
            *n = if alphabet.is_empty() {
                rng.index(256) as i64
            } else {
                alphabet[rng.index(alphabet.len())] as i64
            }
        }
        Op::Tail { bytes, .. } => *bytes = (*bytes * 2 + 1).min(4095),
        Op::Arithmetic { kind, .. } => {
            *kind = [Arithmetic::Add, Arithmetic::Subtract, Arithmetic::Multiply][rng.index(3)]
        }
        _ => match rng.index(6) {
            0 => {
                block.remove(i);
            }
            1 if block.len() < 64 => {
                block.insert(i, block[i].clone());
            }
            2 => {
                let j = rng.index(block.len());
                block.swap(i, j);
            }
            3 if block.len() < 64 => block.insert(
                i,
                Op::Copy {
                    dst: rng.index(16) as u8,
                    src: rng.index(16) as u8,
                },
            ),
            4 if block.len() < 64 => block.insert(
                i,
                Op::Get {
                    dst: rng.index(16) as u8,
                    key: rng.index(16) as u8,
                },
            ),
            _ if block.len() < 64 => {
                // A finite loop is a proposal too; usefulness is evaluated, not assumed.
                block.insert(
                    i,
                    Op::WhileNonzero {
                        reg: 15,
                        body: vec![set(15, 0), block[i].clone()],
                    },
                );
            }
            _ => (),
        },
    }
}
pub fn mutate(parent: &Program, alphabet: &[u8], rng: &mut Rng) -> Program {
    let mut p = parent.clone();
    mutate_block(
        if rng.index(3) == 0 {
            &mut p.predict
        } else {
            &mut p.observe
        },
        alphabet,
        rng,
        0,
    );
    p
}

pub fn retain_modules(registry: &mut Registry, programs: &[Program]) {
    fn walk(block: &Block, registry: &Registry, used: &mut BTreeSet<String>) {
        for op in block {
            match op {
                Op::Call { module } if used.insert(module.clone()) => {
                    if let Some(b) = registry.0.get(module) {
                        walk(b, registry, used);
                    }
                }
                Op::IfEqual { yes, no, .. } => {
                    walk(yes, registry, used);
                    walk(no, registry, used);
                }
                Op::WhileNonzero { body, .. } => walk(body, registry, used),
                _ => (),
            }
        }
    }
    let mut used = BTreeSet::new();
    for p in programs {
        walk(&p.observe, registry, &mut used);
        walk(&p.predict, registry, &mut used);
    }
    registry.0.retain(|k, _| used.contains(k));
}

/// Factor repeated whole blocks into parameter-by-register shared routines.
/// Return-containing blocks are excluded because call boundaries change Return.
pub fn factor(programs: &mut [Program], registry: &mut Registry) -> usize {
    fn eligible(b: &Block) -> bool {
        b.iter().all(|op| match op {
            Op::Return => false,
            Op::IfEqual { yes, no, .. } => eligible(yes) && eligible(no),
            Op::WhileNonzero { body, .. } => eligible(body),
            _ => true,
        })
    }
    fn collect(b: &Block, counts: &mut BTreeMap<String, (Block, usize)>) {
        if b.len() > 1 && eligible(b) {
            counts.entry(canonical_id(b)).or_insert((b.clone(), 0)).1 += 1;
        }
        for op in b {
            match op {
                Op::IfEqual { yes, no, .. } => {
                    collect(yes, counts);
                    collect(no, counts);
                }
                Op::WhileNonzero { body, .. } => collect(body, counts),
                _ => (),
            }
        }
    }
    fn replace(b: &mut Block, selected: &BTreeSet<String>) {
        let id = canonical_id(b);
        if selected.contains(&id) {
            *b = vec![Op::Call { module: id }];
            return;
        }
        for op in b {
            match op {
                Op::IfEqual { yes, no, .. } => {
                    replace(yes, selected);
                    replace(no, selected);
                }
                Op::WhileNonzero { body, .. } => replace(body, selected),
                _ => (),
            }
        }
    }
    let mut counts = BTreeMap::new();
    for p in programs.iter() {
        collect(&p.observe, &mut counts);
        collect(&p.predict, &mut counts);
    }
    let mut selected = BTreeSet::new();
    for (id, (block, n)) in counts {
        if n > 1 && serde_json::to_vec(&block).unwrap().len() * (n - 1) > 100 * n {
            registry.intern(block);
            selected.insert(id);
        }
    }
    for p in programs {
        replace(&mut p.observe, &selected);
        replace(&mut p.predict, &selected);
    }
    selected.len()
}
