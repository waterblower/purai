//! Event-driven register machine. Observe consumes one real input byte. Predict
//! computes on scratch registers, with read-only document memory and no future input.
use crate::ir::{Arithmetic, Block, Op, Program, REGISTERS, Registry, Value};
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Limits {
    pub observe_fuel: u64,
    pub predict_fuel: u64,
    pub max_value_bytes: usize,
    pub memory_entries: usize,
    pub memory_bytes: usize,
}
impl Default for Limits {
    fn default() -> Self {
        Self {
            observe_fuel: 32768,
            predict_fuel: 32768,
            max_value_bytes: 4096,
            memory_entries: 2048,
            memory_bytes: 2 * 1024 * 1024,
        }
    }
}
impl Limits {
    pub fn validate(&self) -> crate::Result<()> {
        if self.observe_fuel == 0
            || self.predict_fuel == 0
            || self.observe_fuel > 10_000_000
            || self.predict_fuel > 10_000_000
            || !(1..=65536).contains(&self.max_value_bytes)
            || self.memory_entries > 100_000
            || self.memory_bytes > 256 * 1024 * 1024
        {
            return Err("invalid VM resource limits".into());
        }
        Ok(())
    }
}

#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct Counters {
    pub work: u64,
    pub instructions: u64,
    pub exhaustions: u64,
    pub evictions: u64,
    pub peak_accounted_bytes: usize,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Prediction {
    Byte(u8),
    Feature(String),
}

#[derive(Clone, Debug)]
pub struct Runtime {
    pub regs: [Value; REGISTERS],
    memory: BTreeMap<Vec<u8>, (Value, u64)>,
    memory_bytes: usize,
    clock: u64,
    pub counters: Counters,
    pub valid: bool,
    fuel: u64,
    input: Option<u8>,
    readonly: bool,
}

enum Flow {
    Continue,
    Return,
    Output(Prediction),
}
type VmResult<T> = std::result::Result<T, ()>;

fn size(v: &Value) -> usize {
    match v {
        Value::Bytes(b) => b.len(),
        _ => 8,
    }
}
fn key(v: &Value) -> Vec<u8> {
    serde_json::to_vec(v).expect("value serialization")
}

impl Default for Runtime {
    fn default() -> Self {
        Self {
            regs: std::array::from_fn(|_| Value::Empty),
            memory: BTreeMap::new(),
            memory_bytes: 0,
            clock: 0,
            counters: Counters::default(),
            valid: true,
            fuel: 0,
            input: None,
            readonly: false,
        }
    }
}

impl Runtime {
    fn charge(&mut self, amount: u64) -> VmResult<()> {
        let spent = amount.min(self.fuel);
        self.counters.work = self.counters.work.saturating_add(spent);
        self.fuel -= spent;
        if spent != amount {
            return Err(());
        }
        Ok(())
    }
    fn set(&mut self, r: u8, value: Value, limits: &Limits) -> VmResult<()> {
        if size(&value) > limits.max_value_bytes {
            return Err(());
        }
        self.regs[r as usize] = value;
        let bytes = self.memory_bytes
            + self.regs.iter().map(size).sum::<usize>()
            + std::mem::size_of::<Self>();
        self.counters.peak_accounted_bytes = self.counters.peak_accounted_bytes.max(bytes);
        Ok(())
    }
    fn fail(&mut self, observe: bool) {
        self.counters.exhaustions += 1;
        if observe {
            // Never continue with partially updated causal state after missing input.
            self.valid = false;
            self.memory.clear();
            self.memory_bytes = 0;
            self.regs.fill(Value::Empty);
        }
    }
    pub fn observe(&mut self, byte: u8, p: &Program, registry: &Registry, limits: &Limits) {
        if !self.valid {
            return;
        }
        self.fuel = limits.observe_fuel;
        self.input = Some(byte);
        self.readonly = false;
        if self.execute(&p.observe, registry, limits, 0).is_err() {
            self.fail(true);
        }
        self.input = None;
    }
    pub fn predict(
        &mut self,
        p: &Program,
        registry: &Registry,
        limits: &Limits,
        fuel: u64,
    ) -> Option<Prediction> {
        if !self.valid {
            return None;
        }
        self.fuel = fuel.min(limits.predict_fuel);
        self.readonly = true;
        self.input = None;
        if self
            .charge(self.regs.iter().map(size).sum::<usize>() as u64)
            .is_err()
        {
            self.fail(false);
            return None;
        }
        let register_bytes = self.regs.iter().map(size).sum::<usize>();
        self.counters.peak_accounted_bytes = self
            .counters
            .peak_accounted_bytes
            .max(self.memory_bytes + 2 * register_bytes + std::mem::size_of::<Self>());
        let saved = self.regs.clone();
        let result = self.execute(&p.predict, registry, limits, 0);
        self.regs = saved;
        match result {
            Ok(Flow::Output(p)) => Some(p),
            Ok(_) => None,
            Err(_) => {
                self.fail(false);
                None
            }
        }
    }
    fn execute(
        &mut self,
        block: &Block,
        registry: &Registry,
        limits: &Limits,
        depth: usize,
    ) -> VmResult<Flow> {
        if depth > 24 {
            return Err(());
        }
        for op in block {
            self.charge(1)?;
            self.counters.instructions += 1;
            match op {
                Op::Set { dst, value } => {
                    self.charge(size(value) as u64)?;
                    self.set(*dst, value.clone(), limits)?;
                }
                Op::Copy { dst, src } => {
                    self.charge(size(&self.regs[*src as usize]) as u64)?;
                    self.set(*dst, self.regs[*src as usize].clone(), limits)?;
                }
                Op::Input { dst } => {
                    let b = self.input.ok_or(())?;
                    self.set(*dst, Value::Int(b as i64), limits)?;
                }
                Op::Append { dst, src } => {
                    self.charge(
                        (size(&self.regs[*src as usize]) + size(&self.regs[*dst as usize])) as u64,
                    )?;
                    let mut out = match &self.regs[*dst as usize] {
                        Value::Bytes(b) => b.clone(),
                        _ => vec![],
                    };
                    match &self.regs[*src as usize] {
                        Value::Int(b) if (0..256).contains(b) => out.push(*b as u8),
                        Value::Bytes(b) => {
                            if out.len().saturating_add(b.len()) > limits.max_value_bytes {
                                return Err(());
                            }
                            out.extend_from_slice(b);
                        }
                        _ => (),
                    }
                    self.set(*dst, Value::Bytes(out), limits)?;
                }
                Op::Tail { reg, bytes } => {
                    self.charge(size(&self.regs[*reg as usize]) as u64)?;
                    if let Value::Bytes(b) = &mut self.regs[*reg as usize]
                        && b.len() > *bytes
                    {
                        b.drain(..b.len() - bytes);
                    }
                }
                Op::At { dst, bytes, index } => {
                    let result = match (&self.regs[*bytes as usize], &self.regs[*index as usize]) {
                        (Value::Bytes(b), Value::Int(i)) if *i >= 0 => {
                            b.get(*i as usize).map(|v| Value::Int(*v as i64))
                        }
                        _ => None,
                    };
                    self.set(*dst, result.unwrap_or_default(), limits)?;
                }
                Op::ParseDecimal { dst, src } => {
                    self.charge(size(&self.regs[*src as usize]) as u64)?;
                    let value = match &self.regs[*src as usize] {
                        Value::Bytes(b) => std::str::from_utf8(b)
                            .ok()
                            .and_then(|s| s.trim().parse::<i64>().ok())
                            .map(Value::Int),
                        _ => None,
                    };
                    self.set(*dst, value.unwrap_or_default(), limits)?;
                }
                Op::FormatDecimal { dst, src } => {
                    self.charge(32)?;
                    let result = match self.regs[*src as usize] {
                        Value::Int(i) => Value::Bytes(i.to_string().into_bytes()),
                        _ => Value::Empty,
                    };
                    self.set(*dst, result, limits)?;
                }
                Op::Arithmetic { dst, a, b, kind } => {
                    let out = match (&self.regs[*a as usize], &self.regs[*b as usize]) {
                        (Value::Int(a), Value::Int(b)) => match kind {
                            Arithmetic::Add => a.checked_add(*b),
                            Arithmetic::Subtract => a.checked_sub(*b),
                            Arithmetic::Multiply => a.checked_mul(*b),
                        },
                        _ => None,
                    };
                    self.set(*dst, out.map(Value::Int).unwrap_or_default(), limits)?;
                }
                Op::Get { dst, key: r } => {
                    self.charge(
                        (size(&self.regs[*r as usize]) * (2 + self.memory.len().ilog2_checked()))
                            as u64,
                    )?;
                    let k = key(&self.regs[*r as usize]);
                    let bytes = self.memory.get(&k).map_or(0, |(v, _)| size(v));
                    self.charge(bytes as u64)?;
                    let v = self
                        .memory
                        .get(&k)
                        .map(|(v, _)| v.clone())
                        .unwrap_or_default();
                    self.set(*dst, v, limits)?;
                }
                Op::Put { key: kr, value: vr } => {
                    if self.readonly {
                        return Err(());
                    }
                    self.charge(
                        (size(&self.regs[*kr as usize]) * (2 + self.memory.len().ilog2_checked())
                            + size(&self.regs[*vr as usize])) as u64,
                    )?;
                    let k = key(&self.regs[*kr as usize]);
                    let value = self.regs[*vr as usize].clone();
                    let needed = k.len() + size(&value) + 64;
                    if needed > limits.memory_bytes || limits.memory_entries == 0 {
                        return Err(());
                    }
                    if let Some((v, _)) = self.memory.remove(&k) {
                        self.memory_bytes -= k.len() + size(&v) + 64;
                    }
                    while self.memory.len() >= limits.memory_entries
                        || self.memory_bytes + needed > limits.memory_bytes
                    {
                        self.charge(self.memory.len() as u64)?;
                        let oldest = self
                            .memory
                            .iter()
                            .min_by_key(|(_, (_, time))| *time)
                            .map(|(k, _)| k.clone())
                            .ok_or(())?;
                        let (v, _) = self.memory.remove(&oldest).ok_or(())?;
                        self.memory_bytes -= oldest.len() + size(&v) + 64;
                        self.counters.evictions += 1;
                    }
                    self.clock = self.clock.checked_add(1).ok_or(())?;
                    self.memory.insert(k, (value, self.clock));
                    self.memory_bytes += needed;
                }
                Op::IfEqual { a, b, yes, no } => {
                    self.charge(
                        (size(&self.regs[*a as usize]) + size(&self.regs[*b as usize])) as u64,
                    )?;
                    let branch = if self.regs[*a as usize] == self.regs[*b as usize] {
                        yes
                    } else {
                        no
                    };
                    let flow = self.execute(branch, registry, limits, depth + 1)?;
                    if !matches!(flow, Flow::Continue) {
                        return Ok(flow);
                    }
                }
                Op::WhileNonzero { reg, body } => loop {
                    self.charge(1)?;
                    if !matches!(self.regs[*reg as usize], Value::Int(i) if i != 0) {
                        break;
                    }
                    let flow = self.execute(body, registry, limits, depth + 1)?;
                    if !matches!(flow, Flow::Continue) {
                        return Ok(flow);
                    }
                },
                Op::Call { module } => {
                    self.charge(module.len() as u64)?;
                    if let Flow::Output(p) = self.execute(
                        registry.0.get(module).ok_or(())?,
                        registry,
                        limits,
                        depth + 1,
                    )? {
                        return Ok(Flow::Output(p));
                    }
                }
                Op::Emit { bytes, index } => {
                    if let (Value::Bytes(b), Value::Int(i)) =
                        (&self.regs[*bytes as usize], &self.regs[*index as usize])
                        && *i >= 0
                        && let Some(b) = b.get(*i as usize)
                    {
                        return Ok(Flow::Output(Prediction::Byte(*b)));
                    }
                    return Ok(Flow::Return);
                }
                Op::Feature { regs } => {
                    let bytes = regs
                        .iter()
                        .map(|r| size(&self.regs[*r as usize]))
                        .sum::<usize>();
                    self.charge(bytes as u64)?;
                    if bytes > 1024 {
                        return Err(());
                    }
                    let values: Vec<_> = regs.iter().map(|r| &self.regs[*r as usize]).collect();
                    return Ok(Flow::Output(Prediction::Feature(
                        serde_json::to_string(&values).map_err(|_| ())?,
                    )));
                }
                Op::Return => return Ok(Flow::Return),
            }
            self.counters.peak_accounted_bytes = self.counters.peak_accounted_bytes.max(
                self.memory_bytes
                    + self.regs.iter().map(size).sum::<usize>()
                    + std::mem::size_of::<Self>(),
            );
        }
        Ok(Flow::Continue)
    }
}

trait LogSize {
    fn ilog2_checked(self) -> usize;
}
impl LogSize for usize {
    fn ilog2_checked(self) -> usize {
        self.checked_ilog2().unwrap_or(0) as usize
    }
}
