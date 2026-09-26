use crate::{
    Result,
    data::{self, Document, Split},
    ir::Program,
    model::{Evidence, Expert, Model},
    proposal::{self, Rng},
    vm::Runtime,
};
use serde::{Deserialize, Serialize};
use std::{collections::BTreeSet, time::Instant};

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Options {
    pub rounds: usize,
    pub candidates_per_round: usize,
    pub beam: usize,
    pub max_experts: usize,
    pub max_model_bytes: usize,
    pub mdl_weight: f64,
    pub work_weight: f64,
    pub max_work: u64,
    pub max_seconds: f64,
    pub seed: u64,
    pub replay_bytes: usize,
    pub stop_on_bpb_increase: bool,
}
impl Default for Options {
    fn default() -> Self {
        Self {
            rounds: 24,
            candidates_per_round: 16,
            beam: 4,
            max_experts: 8,
            max_model_bytes: 64 * 1024 * 1024,
            mdl_weight: 0.02,
            work_weight: 0.000001,
            max_work: 500_000_000,
            max_seconds: 600.0,
            seed: 42,
            replay_bytes: 65536,
            stop_on_bpb_increase: false,
        }
    }
}
impl Options {
    pub fn validate(&self) -> Result<()> {
        if self.rounds > 10000
            || !(1..=256).contains(&self.candidates_per_round)
            || !(1..=32).contains(&self.beam)
            || !(1..=32).contains(&self.max_experts)
            || self.max_model_bytes > crate::model::FILE_LIMIT
            || self.max_model_bytes < 1024
            || !self.mdl_weight.is_finite()
            || self.mdl_weight < 0.0
            || !self.work_weight.is_finite()
            || self.work_weight < 0.0
            || self.max_work == 0
            || !self.max_seconds.is_finite()
            || self.max_seconds <= 0.0
            || self.replay_bytes > 1024 * 1024
        {
            return Err("invalid training budgets/options".into());
        }
        Ok(())
    }
}

#[derive(Debug)]
pub struct BudgetExceeded;
impl std::fmt::Display for BudgetExceeded {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "training work/time budget exhausted")
    }
}
impl std::error::Error for BudgetExceeded {}
pub struct Budget {
    start: Instant,
    pub work: u64,
    max_work: u64,
    seconds: f64,
}
impl Budget {
    pub fn new(work: u64, seconds: f64) -> Self {
        Self {
            start: Instant::now(),
            work: 0,
            max_work: work,
            seconds,
        }
    }
    pub fn spend(&mut self, work: u64) -> Result<()> {
        self.work = self.work.saturating_add(work);
        self.check()
    }
    pub fn check(&self) -> Result<()> {
        if self.work > self.max_work || self.start.elapsed().as_secs_f64() > self.seconds {
            return Err(Box::new(BudgetExceeded));
        }
        Ok(())
    }
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct Evaluation {
    pub bytes: u64,
    pub bits: f64,
    pub work: u64,
    pub instructions: u64,
    pub exhaustions: u64,
    pub evictions: u64,
    pub skipped_experts: u64,
    pub peak_accounted_state_bytes: usize,
}
impl Evaluation {
    pub fn bpb(&self) -> f64 {
        if self.bytes == 0 {
            0.0
        } else {
            self.bits / self.bytes as f64
        }
    }
    fn join(&mut self, e: Self) {
        self.bytes += e.bytes;
        self.bits += e.bits;
        self.work += e.work;
        self.instructions += e.instructions;
        self.exhaustions += e.exhaustions;
        self.evictions += e.evictions;
        self.skipped_experts += e.skipped_experts;
        self.peak_accounted_state_bytes = self
            .peak_accounted_state_bytes
            .max(e.peak_accounted_state_bytes);
    }
}

pub fn evaluate(
    model: &Model,
    docs: &[Document],
    budget: &mut Budget,
    prediction_budget: Option<u64>,
) -> Result<Evaluation> {
    let compiled = model.compile()?;
    let mut result = Evaluation::default();
    for doc in docs {
        budget.check()?;
        let mut session = compiled.session();
        let mut charged = 0;
        for (i, &byte) in doc.bytes.iter().enumerate() {
            // No byte reaches the VM until after its probability has been scored.
            let p = session.distribution(prediction_budget);
            result.bits -= p[byte as usize].log2();
            result.bytes += 1;
            session.observe(byte);
            if i % 64 == 0 {
                let work = session.counters().work;
                budget.spend(work - charged)?;
                charged = work;
            }
        }
        let c = session.counters();
        budget.spend(c.work - charged)?;
        result.work += c.work;
        result.instructions += c.instructions;
        result.exhaustions += c.exhaustions;
        result.evictions += c.evictions;
        result.skipped_experts += session.skipped_experts;
        result.peak_accounted_state_bytes = result
            .peak_accounted_state_bytes
            .max(c.peak_accounted_bytes);
    }
    Ok(result)
}

pub fn fit(
    program: &Program,
    model: &Model,
    docs: &[Document],
    budget: &mut Budget,
) -> Result<Expert> {
    program.validate(&model.registry)?;
    let mut expert = Expert {
        program: program.clone(),
        weight: 0.5,
        evidence: Default::default(),
    };
    for doc in docs {
        budget.check()?;
        let mut vm = Runtime::default();
        let mut evidence = Evidence::default();
        let mut charged = 0;
        for (i, &b) in doc.bytes.iter().enumerate() {
            let prediction = vm.predict(
                program,
                &model.registry,
                &model.limits,
                model.limits.predict_fuel,
            );
            evidence.record(prediction, b, model.feature_cap)?;
            vm.observe(b, program, &model.registry, &model.limits);
            if i % 64 == 0 {
                budget.spend(vm.counters.work - charged)?;
                charged = vm.counters.work;
            }
        }
        budget.spend(vm.counters.work - charged)?;
        expert.evidence.insert(doc.id.clone(), evidence);
    }
    Ok(expert)
}

fn compact(model: &mut Model) {
    proposal::retain_modules(
        &mut model.registry,
        &model
            .experts
            .iter()
            .map(|e| e.program.clone())
            .collect::<Vec<_>>(),
    );
}
fn cost(
    model: &Model,
    evaluation: &Evaluation,
    opts: &Options,
    training_bytes: usize,
) -> Result<f64> {
    let bytes = model.encoded_bytes()?;
    if bytes > opts.max_model_bytes {
        return Ok(f64::INFINITY);
    }
    Ok(evaluation.bpb()
        + opts.mdl_weight * bytes as f64 * 8.0 / training_bytes.max(1) as f64
        + opts.work_weight * evaluation.work as f64 / evaluation.bytes.max(1) as f64)
}
fn assess(
    model: &Model,
    dev: &[Document],
    replay: &[Document],
    budget: &mut Budget,
) -> Result<(Evaluation, Evaluation)> {
    let development = evaluate(model, dev, budget, None)?;
    let mut selection = development.clone();
    if !replay.is_empty() {
        selection.join(evaluate(model, replay, budget, None)?);
    }
    Ok((development, selection))
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Round {
    pub round: usize,
    pub candidates: usize,
    pub accepted: String,
    pub experts: usize,
    pub modules: usize,
    pub model_bytes: usize,
    pub development_bpb: f64,
    pub objective: f64,
    pub elapsed_seconds: f64,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Report {
    pub options: Options,
    pub rounds: Vec<Round>,
    pub stop_reason: String,
    pub selected_round: usize,
    pub training_seconds: f64,
    pub search_work: u64,
    pub train_bytes: usize,
    pub calibration_bytes: usize,
    pub development_bytes: usize,
    pub final_development: Evaluation,
    pub protocol: String,
}

/// Discrete program search: fit evidence on training documents, tune mixture
/// strength on calibration, select topology/cost/stopping on development.
pub fn train(
    mut model: Model,
    split: Split,
    opts: Options,
    mut on_round: impl FnMut(&Round),
) -> Result<(Model, Report)> {
    opts.validate()?;
    model.validate()?;
    let start = Instant::now();
    let mut budget = Budget::new(opts.max_work, opts.max_seconds);
    for d in &split.train {
        if model.selection_documents.contains(&d.id) {
            return Err("resume input contains a previous selection document".into());
        }
    }
    for d in split.calibration.iter().chain(&split.development) {
        if model.root_evidence.contains_key(&d.id) {
            return Err("selection corpus overlaps prior training evidence".into());
        }
        model.selection_documents.insert(d.id.clone());
    }
    let protected_replay = model.replay.clone();
    let mut training = split.train.clone();
    let mut seen: BTreeSet<String> = training.iter().map(|d| d.id.clone()).collect();
    for d in &protected_replay {
        if seen.insert(d.id.clone()) {
            training.push(d.clone());
        }
    }
    for d in &training {
        model.observe_document(d)?;
    }
    let train_bytes = data::byte_count(&training);
    // Update retained experts using new evidence; exact duplicate observations
    // are checked and deduplicated rather than counted as new samples.
    for i in 0..model.experts.len() {
        let fresh = fit(&model.experts[i].program, &model, &training, &mut budget)?;
        for (id, evidence) in fresh.evidence {
            if model.experts[i]
                .evidence
                .get(&id)
                .is_some_and(|old| *old != evidence)
            {
                return Err("conflicting deterministic evidence on resume".into());
            }
            model.experts[i].evidence.insert(id, evidence);
        }
    }
    // Bounded replay contains only whole training documents, never dev/test bytes.
    let mut replay = training.clone();
    replay.sort_by(|a, b| a.id.cmp(&b.id));
    let mut kept = 0;
    replay.retain(|d| {
        if kept + d.bytes.len() <= opts.replay_bytes {
            kept += d.bytes.len();
            true
        } else {
            false
        }
    });
    model.replay = replay;
    if model.encoded_bytes()? > opts.max_model_bytes {
        return Err("root/continued model already exceeds --max-model-bytes".into());
    }
    let (mut dev, initial_selection) =
        assess(&model, &split.development, &protected_replay, &mut budget)?;
    let mut objective = cost(&model, &initial_selection, &opts, train_bytes)?;
    let alphabet = proposal::alphabet(&training);
    let mut registry = model.registry.clone();
    let mut seeds = proposal::seeds(&alphabet, &mut registry);
    let mut rng = Rng(opts.seed);
    // Keep the first unigram-context proposal, shuffle remaining skeletons.
    for i in (2..seeds.len()).rev() {
        let j = 1 + rng.index(i);
        seeds.swap(i, j);
    }
    let mut cursor = 0;
    let mut parents: Vec<Program> = model.experts.iter().map(|e| e.program.clone()).collect();
    let mut tried = BTreeSet::new();
    let mut report = Report { options: opts.clone(), rounds: vec![], stop_reason: "round_limit".into(), selected_round: 0,
        training_seconds: 0.0, search_work: 0, train_bytes, calibration_bytes: data::byte_count(&split.calibration),
        development_bytes: data::byte_count(&split.development), final_development: dev.clone(),
        protocol: "Byte input; document-local state reset; score then observe. Evidence uses training only. Calibration tunes mixture weights. Development plus bounded prior-training replay selects topology, routing and stopping; neither is an independent final test. Resume stores exact-document receipts, not multiplicity-weighted repeated copies. Discrete template-and-mutation search is bounded, not universal program discovery. No full-model CRDT guarantee.".into() };
    'rounds: for round in 1..=opts.rounds {
        let previous_bpb = dev.bpb();
        let mut winner: Option<(Model, Evaluation, f64, String)> = None;
        let mut best = objective;
        let mut ranked = Vec::<(f64, Program)>::new();
        let mut count = 0;
        // Every candidate is evaluated from the last committed model.
        let attempt = (|| -> Result<()> {
            // Prune whole experts, or change execution order and stopping policy.
            let mut policy_variants = Vec::new();
            for i in 0..model.experts.len() {
                let mut m = model.clone();
                m.experts.remove(i);
                compact(&mut m);
                policy_variants.push((m, format!("prune:{i}")));
                if i + 1 < model.experts.len() {
                    let mut m = model.clone();
                    m.experts.swap(i, i + 1);
                    policy_variants.push((m, format!("reorder:{i}")));
                }
            }
            for threshold in [0.75, 0.9, 0.99, 1.0] {
                if model.stop_confidence != threshold {
                    let mut m = model.clone();
                    m.stop_confidence = threshold;
                    policy_variants.push((m, format!("stop-confidence:{threshold}")));
                }
            }
            for (m, name) in policy_variants {
                let (d, e) = assess(&m, &split.development, &protected_replay, &mut budget)?;
                let c = cost(&m, &e, &opts, train_bytes)?;
                if c + 1e-10 < best {
                    best = c;
                    winner = Some((m, d, c, name));
                }
            }
            for n in 0..opts.candidates_per_round {
                budget.check()?;
                let p = if !parents.is_empty() && n % 2 == 1 {
                    proposal::mutate(&parents[rng.index(parents.len())], &alphabet, &mut rng)
                } else if cursor < seeds.len() {
                    let p = seeds[cursor].clone();
                    cursor += 1;
                    p
                } else {
                    proposal::mutate(&proposal::context(1 + rng.index(64)), &alphabet, &mut rng)
                };
                if !tried.insert(p.id())
                    || model.experts.iter().any(|e| e.program == p)
                    || p.validate(&registry).is_err()
                {
                    continue;
                }
                count += 1;
                let mut m = model.clone();
                m.registry.merge(&registry)?;
                let expert = fit(&p, &m, &training, &mut budget)?;
                let replace = if m.experts.len() >= opts.max_experts {
                    Some((round + n) % m.experts.len())
                } else {
                    None
                };
                let index = replace.unwrap_or(m.experts.len());
                if let Some(i) = replace {
                    m.experts[i] = expert;
                } else {
                    m.experts.push(expert);
                }
                compact(&mut m);
                if m.encoded_bytes()? > opts.max_model_bytes {
                    continue;
                }
                let mut weight_cost = f64::INFINITY;
                let mut weight = 0.0;
                for w in [0.0, 0.05, 0.25, 0.5, 0.75, 0.95, 0.999] {
                    m.experts[index].weight = w;
                    let e = evaluate(&m, &split.calibration, &mut budget, None)?;
                    let c = e.bpb() + opts.work_weight * e.work as f64 / e.bytes.max(1) as f64;
                    if c < weight_cost {
                        weight_cost = c;
                        weight = w;
                    }
                }
                m.experts[index].weight = weight;
                let (d, e) = assess(&m, &split.development, &protected_replay, &mut budget)?;
                let c = cost(&m, &e, &opts, train_bytes)?;
                ranked.push((c, p));
                if weight > 0.0 && c + 1e-10 < best {
                    best = c;
                    winner = Some((
                        m,
                        d,
                        c,
                        if replace.is_some() {
                            "replace-program"
                        } else {
                            "add-program"
                        }
                        .into(),
                    ));
                }
            }
            // Propose common-block extraction, then refit because added Call work
            // can affect fuel-limited executions. Never reuse invalid evidence.
            if model.experts.len() > 1 {
                let mut m = model.clone();
                let mut programs: Vec<_> = m.experts.iter().map(|e| e.program.clone()).collect();
                if proposal::factor(&mut programs, &mut m.registry) > 0 {
                    for (i, p) in programs.iter().enumerate() {
                        let mut e = fit(p, &m, &training, &mut budget)?;
                        e.weight = m.experts[i].weight;
                        m.experts[i] = e;
                    }
                    compact(&mut m);
                    if m.validate().is_ok() {
                        let (d, e) =
                            assess(&m, &split.development, &protected_replay, &mut budget)?;
                        let c = cost(&m, &e, &opts, train_bytes)?;
                        if c + 1e-10 < best {
                            winner = Some((m, d, c, "factor-shared-blocks".into()));
                        }
                    }
                }
            }
            Ok(())
        })();
        if let Err(e) = attempt {
            if e.downcast_ref::<BudgetExceeded>().is_some() {
                report.stop_reason = "work_or_time_budget; incomplete_round_discarded".into();
                break 'rounds;
            }
            return Err(e);
        }
        ranked.sort_by(|a, b| a.0.total_cmp(&b.0).then(a.1.id().cmp(&b.1.id())));
        parents = ranked.into_iter().take(opts.beam).map(|(_, p)| p).collect();
        parents.extend(model.experts.iter().map(|e| e.program.clone()));
        let mut accepted = "unchanged".to_string();
        let mut rejected_increase = false;
        let mut round_bpb = dev.bpb();
        if let Some((m, d, c, name)) = winner {
            round_bpb = d.bpb();
            if opts.stop_on_bpb_increase && d.bpb() > previous_bpb {
                rejected_increase = true;
                accepted = "rejected:first-bpb-increase".into();
            } else {
                model = m;
                dev = d;
                objective = c;
                accepted = name;
                report.selected_round = round;
                registry.merge(&model.registry)?;
            }
        }
        let r = Round {
            round,
            candidates: count,
            accepted,
            experts: model.experts.len(),
            modules: model.registry.0.len(),
            model_bytes: model.encoded_bytes()?,
            development_bpb: round_bpb,
            objective,
            elapsed_seconds: start.elapsed().as_secs_f64(),
        };
        on_round(&r);
        report.rounds.push(r);
        if rejected_increase {
            report.stop_reason = "first_development_bpb_increase; prior_model_retained".into();
            break;
        }
    }
    model.validate()?;
    report.final_development = dev;
    report.training_seconds = start.elapsed().as_secs_f64();
    report.search_work = budget.work;
    Ok((model, report))
}
