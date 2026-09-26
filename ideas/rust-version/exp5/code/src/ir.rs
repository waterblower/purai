//! Structured control-flow IR. Registers carry runtime values, never training labels.
use crate::{Result, canonical_id};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};

pub const REGISTERS: usize = 16;
pub type Reg = u8;
pub type Block = Vec<Op>;

#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub enum Value {
    #[default]
    Empty,
    Int(i64),
    Bytes(Vec<u8>),
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum Arithmetic {
    Add,
    Subtract,
    Multiply,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum Op {
    Set {
        dst: Reg,
        value: Value,
    },
    Copy {
        dst: Reg,
        src: Reg,
    },
    Input {
        dst: Reg,
    },
    Append {
        dst: Reg,
        src: Reg,
    },
    Tail {
        reg: Reg,
        bytes: usize,
    },
    At {
        dst: Reg,
        bytes: Reg,
        index: Reg,
    },
    ParseDecimal {
        dst: Reg,
        src: Reg,
    },
    FormatDecimal {
        dst: Reg,
        src: Reg,
    },
    Arithmetic {
        dst: Reg,
        a: Reg,
        b: Reg,
        kind: Arithmetic,
    },
    Get {
        dst: Reg,
        key: Reg,
    },
    Put {
        key: Reg,
        value: Reg,
    },
    IfEqual {
        a: Reg,
        b: Reg,
        yes: Block,
        no: Block,
    },
    WhileNonzero {
        reg: Reg,
        body: Block,
    },
    Call {
        module: String,
    },
    /// Propose a byte; the interpreter does not append it to the input.
    Emit {
        bytes: Reg,
        index: Reg,
    },
    /// Bind an arbitrary computed feature to learned next-byte statistics.
    Feature {
        regs: Vec<Reg>,
    },
    Return,
}

#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct Registry(pub BTreeMap<String, Block>);

impl Registry {
    pub fn intern(&mut self, block: Block) -> String {
        let id = canonical_id(&block);
        self.0.entry(id.clone()).or_insert(block);
        id
    }

    /// Content-addressed immutable definitions form an idempotent union registry.
    /// This is not a claim that fitted models or optimizer decisions are CRDTs.
    pub fn merge(&mut self, other: &Self) -> Result<()> {
        for (id, block) in &other.0 {
            if *id != canonical_id(block) {
                return Err("invalid module fingerprint".into());
            }
            if self.0.get(id).is_some_and(|existing| existing != block) {
                return Err("conflicting module definition".into());
            }
        }
        self.0.extend(other.0.clone());
        Ok(())
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Program {
    pub observe: Block,
    pub predict: Block,
}

impl Program {
    pub fn id(&self) -> String {
        canonical_id(self)
    }
    pub fn validate(&self, registry: &Registry) -> Result<()> {
        let mut visiting = BTreeSet::new();
        validate_block(&self.observe, registry, &mut visiting, 0)?;
        validate_block(&self.predict, registry, &mut visiting, 0)
    }
    pub fn description_bytes(&self) -> usize {
        serde_json::to_vec(self).expect("IR serialization").len()
    }
}

fn validate_block(
    block: &Block,
    registry: &Registry,
    visiting: &mut BTreeSet<String>,
    depth: usize,
) -> Result<()> {
    if depth > 24 || block.len() > 256 {
        return Err("IR nesting/block size limit exceeded".into());
    }
    for op in block {
        let regs = match op {
            Op::Set { dst, value } => {
                if matches!(value, Value::Bytes(b) if b.len() > 4096) {
                    return Err("literal too large".into());
                }
                vec![*dst]
            }
            Op::Copy { dst, src }
            | Op::Append { dst, src }
            | Op::ParseDecimal { dst, src }
            | Op::FormatDecimal { dst, src } => vec![*dst, *src],
            Op::Input { dst } => vec![*dst],
            Op::Tail { reg, bytes } => {
                if *bytes > 65536 {
                    return Err("tail too large".into());
                }
                vec![*reg]
            }
            Op::At { dst, bytes, index } => vec![*dst, *bytes, *index],
            Op::Arithmetic { dst, a, b, .. } => vec![*dst, *a, *b],
            Op::Get { dst, key } => vec![*dst, *key],
            Op::Put { key, value } => vec![*key, *value],
            Op::IfEqual { a, b, yes, no } => {
                validate_block(yes, registry, visiting, depth + 1)?;
                validate_block(no, registry, visiting, depth + 1)?;
                vec![*a, *b]
            }
            Op::WhileNonzero { reg, body } => {
                validate_block(body, registry, visiting, depth + 1)?;
                vec![*reg]
            }
            Op::Call { module } => {
                let body = registry.0.get(module).ok_or("missing module")?;
                if canonical_id(body) != *module {
                    return Err("invalid module hash".into());
                }
                if !visiting.insert(module.clone()) {
                    return Err("recursive calls are not allowed; use fuel-bounded loops".into());
                }
                validate_block(body, registry, visiting, depth + 1)?;
                visiting.remove(module);
                vec![]
            }
            Op::Emit { bytes, index } => vec![*bytes, *index],
            Op::Feature { regs } => {
                if regs.is_empty() || regs.len() > REGISTERS {
                    return Err("invalid feature registers".into());
                }
                regs.clone()
            }
            Op::Return => vec![],
        };
        if regs.iter().any(|r| *r as usize >= REGISTERS) {
            return Err("invalid register".into());
        }
    }
    Ok(())
}
