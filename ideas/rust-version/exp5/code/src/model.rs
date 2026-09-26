use crate::{
    Result,
    data::Document,
    fingerprint,
    ir::{Program, Registry},
    vm::{Counters, Limits, Prediction, Runtime},
};
use serde::{Deserialize, Serialize};
use std::{
    collections::BTreeMap,
    fs::{self, OpenOptions},
    io::{Read, Write},
    path::Path,
};

#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct Histogram {
    pub counts: BTreeMap<u8, u64>,
}
impl Histogram {
    pub fn add(&mut self, byte: u8, n: u64) -> Result<()> {
        let c = self.counts.entry(byte).or_default();
        *c = c.checked_add(n).ok_or("evidence count overflow")?;
        Ok(())
    }
    pub fn merge(&mut self, other: &Self) -> Result<()> {
        for (&b, &n) in &other.counts {
            self.add(b, n)?;
        }
        Ok(())
    }
    pub fn total(&self) -> f64 {
        self.counts.values().map(|n| *n as f64).sum()
    }
    pub fn distribution(&self, prior: &[f64; 256], strength: f64) -> [f64; 256] {
        let total = self.total() + strength;
        std::array::from_fn(|i| {
            (self.counts.get(&(i as u8)).copied().unwrap_or(0) as f64 + strength * prior[i]) / total
        })
    }
}

#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct Evidence {
    pub features: BTreeMap<String, Histogram>,
    pub direct_attempts: u64,
    pub direct_correct: u64,
}
impl Evidence {
    pub fn record(&mut self, prediction: Option<Prediction>, byte: u8, cap: usize) -> Result<()> {
        match prediction {
            Some(Prediction::Byte(b)) => {
                self.direct_attempts = self
                    .direct_attempts
                    .checked_add(1)
                    .ok_or("count overflow")?;
                if b == byte {
                    self.direct_correct =
                        self.direct_correct.checked_add(1).ok_or("count overflow")?;
                }
            }
            Some(Prediction::Feature(k)) if cap > 0 => {
                // Deterministic bounded dictionary; absent features use the root distribution.
                // Feature retention is by lexical key order, not a claimed frequency optimum.
                if !self.features.contains_key(&k)
                    && self.features.len() >= cap
                    && self
                        .features
                        .last_key_value()
                        .is_some_and(|(last, _)| k < *last)
                {
                    self.features.pop_last();
                }
                if self.features.contains_key(&k) || self.features.len() < cap {
                    self.features.entry(k).or_default().add(byte, 1)?;
                }
            }
            _ => (),
        }
        Ok(())
    }
    pub fn merge_counts(&mut self, other: &Self, cap: usize) -> Result<()> {
        self.direct_attempts = self
            .direct_attempts
            .checked_add(other.direct_attempts)
            .ok_or("count overflow")?;
        self.direct_correct = self
            .direct_correct
            .checked_add(other.direct_correct)
            .ok_or("count overflow")?;
        for (key, hist) in &other.features {
            self.features.entry(key.clone()).or_default().merge(hist)?;
        }
        while self.features.len() > cap {
            self.features.pop_last();
        }
        Ok(())
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Expert {
    pub program: Program,
    pub weight: f64,
    /// Immutable deterministic evidence per document. Re-observing a document
    /// does not count it twice. Distinct documents remain separate observations.
    pub evidence: BTreeMap<String, Evidence>,
}
impl Expert {
    pub fn aggregate(&self, cap: usize) -> Result<Evidence> {
        let mut result = Evidence::default();
        for stats in self.evidence.values() {
            result.merge_counts(stats, cap)?;
        }
        Ok(result)
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Model {
    pub version: u32,
    pub registry: Registry,
    /// Ordered local execution plan. This learned plan is not a CRDT.
    pub experts: Vec<Expert>,
    pub root_evidence: BTreeMap<String, Histogram>,
    pub replay: Vec<Document>,
    pub selection_documents: std::collections::BTreeSet<String>,
    pub limits: Limits,
    pub feature_cap: usize,
    pub stop_confidence: f64,
    pub prediction_budget: u64,
}

impl Model {
    pub fn new(limits: Limits, feature_cap: usize) -> Self {
        Self {
            version: 5,
            registry: Registry::default(),
            experts: vec![],
            root_evidence: BTreeMap::new(),
            replay: vec![],
            selection_documents: Default::default(),
            limits,
            feature_cap,
            stop_confidence: 1.0,
            prediction_budget: 100_000,
        }
    }
    pub fn observe_document(&mut self, doc: &Document) -> Result<()> {
        if self.root_evidence.contains_key(&doc.id) {
            return Ok(());
        }
        let mut h = Histogram::default();
        for &b in &doc.bytes {
            h.add(b, 1)?;
        }
        self.root_evidence.insert(doc.id.clone(), h);
        Ok(())
    }
    pub fn root(&self) -> Result<[f64; 256]> {
        let mut h = Histogram::default();
        for v in self.root_evidence.values() {
            h.merge(v)?;
        }
        Ok(h.distribution(&[1.0 / 256.0; 256], 128.0))
    }
    pub fn validate(&self) -> Result<()> {
        if self.version != 5
            || self.experts.len() > 32
            || self.feature_cap > 65536
            || !self.stop_confidence.is_finite()
            || !(0.5..=1.0).contains(&self.stop_confidence)
            || self.prediction_budget == 0
            || self.prediction_budget > 100_000_000
            || self.registry.0.len() > 4096
        {
            return Err("invalid model configuration".into());
        }
        self.limits.validate()?;
        let mut ids = std::collections::BTreeSet::new();
        for e in &self.experts {
            e.program.validate(&self.registry)?;
            if !e.weight.is_finite()
                || !(0.0..=1.0).contains(&e.weight)
                || !ids.insert(e.program.id())
            {
                return Err("invalid/duplicate expert".into());
            }
            for (doc, stats) in &e.evidence {
                if !self.root_evidence.contains_key(doc)
                    || stats.features.len() > self.feature_cap
                    || stats.direct_correct > stats.direct_attempts
                    || stats.features.keys().any(|k| k.len() > 8192)
                {
                    return Err("invalid evidence".into());
                }
            }
        }
        for d in &self.replay {
            if d.id != fingerprint(&d.bytes) || !self.root_evidence.contains_key(&d.id) {
                return Err("invalid replay document".into());
            }
        }
        self.root()?;
        Ok(())
    }
    pub fn encoded_bytes(&self) -> Result<usize> {
        Ok(serde_json::to_vec(self)?.len() + 48)
    }
    pub fn compile(&self) -> Result<Compiled<'_>> {
        Ok(Compiled {
            model: self,
            root: self.root()?,
            tables: self
                .experts
                .iter()
                .map(|e| e.aggregate(self.feature_cap))
                .collect::<Result<_>>()?,
        })
    }
}

pub struct Compiled<'a> {
    pub model: &'a Model,
    root: [f64; 256],
    tables: Vec<Evidence>,
}
pub struct Session<'a, 'b> {
    compiled: &'a Compiled<'b>,
    pub states: Vec<Runtime>,
    pub routing_work: u64,
    pub skipped_experts: u64,
}

impl<'b> Compiled<'b> {
    pub fn session(&self) -> Session<'_, 'b> {
        Session {
            compiled: self,
            states: vec![Runtime::default(); self.model.experts.len()],
            routing_work: 0,
            skipped_experts: 0,
        }
    }
}
impl Session<'_, '_> {
    pub fn distribution(&mut self, budget: Option<u64>) -> [f64; 256] {
        // Fixed baseline work is additional to the optional program-work budget.
        self.routing_work += 256;
        let mut out = self.compiled.root;
        let mut fuel = budget.unwrap_or(self.compiled.model.prediction_budget);
        for (i, e) in self.compiled.model.experts.iter().enumerate() {
            if fuel == 0 {
                self.skipped_experts += (self.states.len() - i) as u64;
                break;
            }
            // Charge dispatch and confidence reduction, rather than hiding routing work.
            let routing = 257.min(fuel);
            fuel -= routing;
            self.routing_work += routing;
            if routing < 257
                || out.iter().copied().fold(0.0, f64::max) >= self.compiled.model.stop_confidence
            {
                self.skipped_experts += (self.states.len() - i) as u64;
                break;
            }
            let before = self.states[i].counters.work;
            let prediction = self.states[i].predict(
                &e.program,
                &self.compiled.model.registry,
                &self.compiled.model.limits,
                fuel,
            );
            fuel = fuel.saturating_sub(self.states[i].counters.work - before);
            let query_work = match &prediction {
                Some(Prediction::Byte(_)) => 256,
                Some(Prediction::Feature(k)) => {
                    256 + k.len() as u64
                        * (2 + self.compiled.tables[i]
                            .features
                            .len()
                            .checked_ilog2()
                            .unwrap_or(0) as u64)
                }
                None => 0,
            };
            if query_work > fuel {
                self.skipped_experts += 1;
                break;
            }
            fuel -= query_work;
            self.routing_work += query_work;
            let candidate = match prediction {
                Some(Prediction::Byte(b)) => {
                    let mut p = self.compiled.root.map(|x| x * 0.001);
                    p[b as usize] += 0.999;
                    Some(p)
                }
                Some(Prediction::Feature(k)) => self.compiled.tables[i]
                    .features
                    .get(&k)
                    .map(|h| h.distribution(&self.compiled.root, 8.0)),
                None => None,
            };
            if let Some(p) = candidate {
                if fuel < 256 {
                    self.skipped_experts += 1;
                    break;
                }
                fuel -= 256;
                self.routing_work += 256;
                for b in 0..256 {
                    out[b] = (1.0 - e.weight) * out[b] + e.weight * p[b];
                }
            }
        }
        out
    }
    pub fn observe(&mut self, b: u8) {
        for (state, expert) in self.states.iter_mut().zip(&self.compiled.model.experts) {
            state.observe(
                b,
                &expert.program,
                &self.compiled.model.registry,
                &self.compiled.model.limits,
            );
        }
    }
    pub fn counters(&self) -> Counters {
        let mut c = Counters {
            work: self.routing_work,
            ..Counters::default()
        };
        for state in &self.states {
            c.work += state.counters.work;
            c.instructions += state.counters.instructions;
            c.exhaustions += state.counters.exhaustions;
            c.evictions += state.counters.evictions;
            // Sum of per-expert peaks is a conservative bound, not measured RSS.
            c.peak_accounted_bytes += state.counters.peak_accounted_bytes;
        }
        c
    }
}

const MAGIC: &[u8; 8] = b"AFRN5\r\n\0";
pub const FILE_LIMIT: usize = 256 * 1024 * 1024;
pub fn save(path: &Path, model: &Model) -> Result<usize> {
    model.validate()?;
    let data = serde_json::to_vec(model)?;
    if data.len() + 48 > FILE_LIMIT {
        return Err("model exceeds hard file limit".into());
    }
    let tmp = path.with_extension("model.tmp");
    let mut f = OpenOptions::new().create_new(true).write(true).open(&tmp)?;
    let result = (|| -> Result<()> {
        f.write_all(MAGIC)?;
        f.write_all(&(data.len() as u64).to_le_bytes())?;
        f.write_all(&crate::sha256::digest(&data))?;
        f.write_all(&data)?;
        f.sync_all()?;
        // create_new prevents overwriting an existing final model, including races.
        fs::hard_link(&tmp, path)?;
        Ok(())
    })();
    drop(f);
    let _ = fs::remove_file(&tmp);
    result?;
    Ok(data.len() + 48)
}
pub fn load(path: &Path) -> Result<Model> {
    let mut f = fs::File::open(path)?;
    if f.metadata()?.len() > FILE_LIMIT as u64 {
        return Err("model file too large".into());
    }
    let mut header = [0u8; 48];
    f.read_exact(&mut header)?;
    if &header[..8] != MAGIC {
        return Err("unknown model format".into());
    }
    let len = u64::from_le_bytes(header[8..16].try_into()?) as usize;
    if len > FILE_LIMIT - 48 || len + 48 != f.metadata()?.len() as usize {
        return Err("invalid model length".into());
    }
    let mut data = vec![0; len];
    f.read_exact(&mut data)?;
    if crate::sha256::digest(&data) != header[16..48] {
        return Err("model checksum mismatch".into());
    }
    let m: Model = serde_json::from_slice(&data)?;
    m.validate()?;
    Ok(m)
}
