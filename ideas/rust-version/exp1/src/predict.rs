use crate::{
    Result,
    graph::{Calibration, Compiled, Model},
    math::Rng,
};

const HASH_BITS: usize = 18;
const MATCH_MIN: usize = 6;

/// Transient, document-local copy state. Epochs avoid clearing a large table for
/// every short story; no information from another document is ever consulted.
pub struct MatchState {
    positions: Vec<usize>,
    epochs: Vec<u32>,
    epoch: u32,
    pub pointer: usize,
    pub length: usize,
}
impl Default for MatchState {
    fn default() -> Self {
        Self {
            positions: vec![0; 1 << HASH_BITS],
            epochs: vec![0; 1 << HASH_BITS],
            epoch: 1,
            pointer: 0,
            length: 0,
        }
    }
}
impl MatchState {
    pub fn reset(&mut self) {
        self.pointer = 0;
        self.length = 0;
        self.epoch = self.epoch.wrapping_add(1);
        if self.epoch == 0 {
            self.epochs.fill(0);
            self.epoch = 1;
        }
    }
    pub fn predicted(&self, text: &[u8]) -> Option<u8> {
        (self.length > 0).then(|| text[self.pointer])
    }
    pub fn observe(&mut self, text: &[u8], i: usize) {
        if self.length > 0 {
            if text[self.pointer] == text[i] {
                self.pointer += 1;
                self.length += 1;
            } else {
                self.length = 0;
            }
        }
        if i + 1 < MATCH_MIN {
            return;
        }
        let mut h = 0u32;
        for &b in &text[i + 1 - MATCH_MIN..=i] {
            h = (h ^ b as u32).wrapping_mul(0x9e37_79b1);
        }
        let h = (h >> (32 - HASH_BITS)) as usize;
        if self.length == 0 && self.epochs[h] == self.epoch {
            let candidate = self.positions[h];
            let mut length = 0;
            while length < 64
                && length < candidate
                && text[candidate - 1 - length] == text[i - length]
            {
                length += 1;
            }
            if length >= MATCH_MIN {
                self.pointer = candidate;
                self.length = length;
            }
        }
        self.positions[h] = i + 1;
        self.epochs[h] = self.epoch;
    }
}

pub fn bucket(length: usize, q: f64) -> usize {
    let lb = (2.0 * (length.max(1) as f64).log2())
        .floor()
        .clamp(0.0, 15.0) as usize;
    let qb = (-2.0 * q.max(1e-12).log2()).floor().clamp(0.0, 15.0) as usize;
    lb * 16 + qb
}
fn hit_probability(t: &Calibration, bucket: usize, q: f64) -> f64 {
    let mut p = q;
    for l in 0..=bucket / 16 {
        let b = l * 16 + bucket % 16;
        p = (t.hits[b] as f64 + 2.0 * p) / (t.totals[b] as f64 + 2.0);
    }
    p.clamp(1e-12, 1.0 - 1e-12)
}

/// Fit only copy reliability, on training-side documents withheld from structure
/// learning. The independent validation set never changes graph or calibration.
pub fn calibrate(model: &mut Model, docs: &[Vec<u8>]) {
    model.calibration = Calibration::default();
    model.calibration_bytes = docs.iter().map(|d| d.len() as u64).sum();
    let compiled = Compiled::new(model);
    let mut mm = MatchState::default();
    for text in docs {
        mm.reset();
        for i in 0..text.len() {
            if let Some(b) = mm.predicted(text) {
                let (node, _, _) = compiled.context(text, i, usize::MAX);
                let q = compiled.distributions[node][b as usize];
                let k = bucket(mm.length, q);
                model.calibration.totals[k] += 1;
                if text[i] == b {
                    model.calibration.hits[k] += 1;
                }
            }
            mm.observe(text, i);
        }
    }
}

pub struct Predictor<'a> {
    pub model: &'a Model,
    pub compiled: Compiled,
}
pub struct Step {
    pub node: usize,
    pub steps: usize,
    pub match_length: usize,
    pub match_byte: Option<u8>,
}

impl<'a> Predictor<'a> {
    pub fn new(model: &'a Model) -> Self {
        Self {
            model,
            compiled: Compiled::new(model),
        }
    }
    pub fn distribution(
        &self,
        text: &[u8],
        i: usize,
        mm: &MatchState,
        use_match: bool,
        budget: usize,
    ) -> ([f64; 256], Step) {
        let (node, _, steps) = self.compiled.context(text, i, budget.max(1));
        let mut out = self.compiled.distributions[node];
        let guess = if use_match { mm.predicted(text) } else { None };
        if let Some(b) = guess {
            let q = out[b as usize];
            if q < 1.0 - 1e-12 {
                let p = hit_probability(&self.model.calibration, bucket(mm.length, q), q);
                let scale = (1.0 - p) / (1.0 - q);
                for x in &mut out {
                    *x *= scale;
                }
                out[b as usize] = p;
            }
        }
        (
            out,
            Step {
                node,
                steps,
                match_length: if guess.is_some() { mm.length } else { 0 },
                match_byte: guess,
            },
        )
    }
    /// Scalar evaluation avoids mixing/scaling 256 probabilities for every byte.
    pub fn evaluate(&self, docs: &[Vec<u8>], use_match: bool, budget: usize) -> Evaluation {
        let mut result = Evaluation::default();
        let mut mm = MatchState::default();
        for text in docs {
            mm.reset();
            for i in 0..text.len() {
                let (node, _, steps) = self.compiled.context(text, i, budget.max(1));
                let p = &self.compiled.distributions[node];
                let mut qx = p[text[i] as usize];
                if use_match {
                    if let Some(b) = mm.predicted(text) {
                        let q = p[b as usize];
                        if q < 1.0 - 1e-12 {
                            let ph =
                                hit_probability(&self.model.calibration, bucket(mm.length, q), q);
                            qx = if b == text[i] {
                                ph
                            } else {
                                qx * (1.0 - ph) / (1.0 - q)
                            };
                        }
                        result.match_predictions += 1;
                    }
                    mm.observe(text, i);
                }
                result.bits -= qx.max(1e-300).log2();
                result.bytes += 1;
                result.steps += steps as u64;
                result.min_steps = Some(result.min_steps.map_or(steps, |v| v.min(steps)));
                result.max_steps = result.max_steps.max(steps);
            }
        }
        result
    }

    pub fn generate(
        &self,
        prompt: &[u8],
        opts: &GenerateOptions,
        mut on_step: impl FnMut(u8, &Step, &[f64; 256]),
    ) -> Result<Vec<u8>> {
        if !opts.temperature.is_finite()
            || opts.temperature < 0.0
            || !(0.0 < opts.top_p && opts.top_p <= 1.0)
            || opts.budget == 0
        {
            return Err("require temperature >= 0, 0 < top-p <= 1, and max-steps >= 1".into());
        }
        let mut text = prompt.to_vec();
        text.try_reserve(opts.bytes)?;
        let mut rng = Rng(opts.seed);
        let mut mm = MatchState::default();
        if opts.use_match {
            for i in 0..text.len() {
                mm.observe(&text, i);
            }
        }
        for _ in 0..opts.bytes {
            let i = text.len();
            let (p, step) = self.distribution(&text, i, &mm, opts.use_match, opts.budget);
            let b = sample(&p, opts.temperature, opts.top_p, &mut rng);
            text.push(b);
            if opts.use_match {
                mm.observe(&text, i);
            }
            on_step(b, &step, &p);
        }
        Ok(text.split_off(prompt.len()))
    }
}

#[derive(Clone, Debug)]
pub struct GenerateOptions {
    pub bytes: usize,
    pub temperature: f64,
    pub top_p: f64,
    pub seed: u64,
    pub use_match: bool,
    pub budget: usize,
}
impl Default for GenerateOptions {
    fn default() -> Self {
        Self {
            bytes: 400,
            temperature: 0.85,
            top_p: 1.0,
            seed: 42,
            use_match: true,
            budget: usize::MAX,
        }
    }
}

fn sample(p: &[f64; 256], temperature: f64, top_p: f64, rng: &mut Rng) -> u8 {
    let best = (0..256)
        .max_by(|&a, &b| p[a].total_cmp(&p[b]).then(b.cmp(&a)))
        .unwrap();
    if temperature == 0.0 {
        return best as u8;
    }
    // Log-space rescaling avoids underflow for small positive temperatures.
    let top = p[best].ln();
    let mut q = [0.0; 256];
    for x in 0..256 {
        q[x] = ((p[x].ln() - top) / temperature).exp();
    }
    if top_p < 1.0 {
        let mut ids: Vec<usize> = (0..256).collect();
        ids.sort_unstable_by(|&a, &b| q[b].total_cmp(&q[a]).then(a.cmp(&b)));
        let target = q.iter().sum::<f64>() * top_p;
        let mut acc = 0.0;
        for x in ids {
            if acc >= target {
                q[x] = 0.0;
            } else {
                acc += q[x];
            }
        }
    }
    let mut r = rng.uniform() * q.iter().sum::<f64>();
    for (x, v) in q.iter().enumerate() {
        if *v > 0.0 {
            if r < *v {
                return x as u8;
            }
            r -= v;
        }
    }
    best as u8
}

#[derive(Clone, Debug, Default)]
pub struct Evaluation {
    pub bits: f64,
    pub bytes: u64,
    pub steps: u64,
    pub min_steps: Option<usize>,
    pub max_steps: usize,
    pub match_predictions: u64,
}
impl Evaluation {
    pub fn bpb(&self) -> f64 {
        if self.bytes == 0 {
            0.0
        } else {
            self.bits / self.bytes as f64
        }
    }
    pub fn steps_per_byte(&self) -> f64 {
        if self.bytes == 0 {
            0.0
        } else {
            self.steps as f64 / self.bytes as f64
        }
    }
}
