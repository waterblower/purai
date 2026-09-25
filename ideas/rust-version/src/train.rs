use crate::{
    Result,
    graph::{ByteSet, Compiled, Histogram, Model, Params},
    math,
};
use std::collections::{HashMap, HashSet};
use std::time::Instant;

#[derive(Clone, Debug)]
pub struct Options {
    pub rounds: usize,
    pub max_depth: usize,
    pub max_nodes: usize,
    pub max_candidates: usize,
    pub forks_per_round: usize,
    pub merges_per_round: usize,
    pub min_count: u64,
    pub structure_weight: f64,
    pub merge_top: usize,
}
impl Default for Options {
    fn default() -> Self {
        Self {
            rounds: 12,
            max_depth: 12,
            max_nodes: 60_000,
            max_candidates: 100_000,
            forks_per_round: 10_000,
            merges_per_round: 64,
            min_count: 4,
            structure_weight: 0.25,
            merge_top: 16,
        }
    }
}
impl Options {
    pub fn validate(&self) -> Result<()> {
        if self.max_depth == 0
            || self.max_depth > 128
            || self.max_nodes == 0
            || self.max_nodes > 250_000
            || self.max_candidates == 0
            || self.max_candidates > 1_000_000
            || self.min_count < 2
            || !self.structure_weight.is_finite()
            || self.structure_weight < 0.0
            || self.merge_top > 256
        {
            return Err("invalid training limits (depth 1..128, nodes 1..250000, candidates 1..1000000, min-count >= 2)".into());
        }
        Ok(())
    }
}

#[derive(Clone, Debug)]
pub struct RoundReport {
    pub round: usize,
    pub nodes: usize,
    pub classes: usize,
    pub depth: usize,
    pub candidates: usize,
    pub added: usize,
    pub merged: usize,
    pub pruned: usize,
    pub estimated_gain_bits: f64,
    pub seconds: f64,
}

fn node_cost(label_len: usize, nodes: usize) -> f64 {
    16.0 + 8.0 * label_len as f64 + ((nodes + 256) as f64).log2()
}

/// Leave-child-out prior prevents a candidate from supplying its own evidence.
fn fork_delta(
    model: &Model,
    compiled: &Compiled,
    parent: usize,
    counts: &Histogram,
    label_len: usize,
    weight: f64,
) -> f64 {
    let p = &model.nodes[parent];
    let base = p
        .parent
        .map_or(&math::UNIFORM, |id| &compiled.distributions[id]);
    let rest = p.counts.subtract(counts);
    let params = &model.params;
    let rest_p = math::smooth(&rest, base, params.discount, params.theta);
    let before = math::py_bits(&p.counts, base, params.discount, params.theta)
        - math::py_bits(&rest, base, params.discount, params.theta);
    math::py_bits(counts, &rest_p, params.discount, params.theta) - before
        + weight * node_cost(label_len, model.nodes.len())
}

struct Candidate {
    key: usize,
    counts: Histogram,
    delta: f64,
}

fn grow(model: &mut Model, docs: &[Vec<u8>], opts: &Options) -> (usize, usize, f64) {
    if opts.forks_per_round == 0 || model.nodes.len() >= opts.max_nodes {
        return (0, 0, 0.0);
    }
    let compiled = Compiled::new(model);
    // Bounded by max_nodes, not by the number of distinct substrings in the data.
    let mut totals = vec![0u64; model.nodes.len() * 256];
    for text in docs {
        for i in 1..text.len() {
            let (parent, start, _) = compiled.context(text, i, usize::MAX);
            if start > 0 && model.nodes[parent].depth < opts.max_depth {
                totals[parent * 256 + text[start - 1] as usize] += 1;
            }
        }
    }
    let mut keys: Vec<usize> = totals
        .iter()
        .enumerate()
        .filter_map(|(k, &n)| (n >= opts.min_count).then_some(k))
        .collect();
    keys.sort_unstable_by(|&a, &b| totals[b].cmp(&totals[a]).then(a.cmp(&b)));
    keys.truncate(opts.max_candidates);
    drop(totals);
    let index: HashMap<usize, usize> = keys.iter().enumerate().map(|(i, &key)| (key, i)).collect();
    let mut candidates: Vec<Candidate> = keys
        .into_iter()
        .map(|key| Candidate {
            key,
            counts: Histogram::default(),
            delta: 0.0,
        })
        .collect();
    if candidates.is_empty() {
        return (0, 0, 0.0);
    }
    for text in docs {
        for i in 1..text.len() {
            let (parent, start, _) = compiled.context(text, i, usize::MAX);
            if start > 0 && model.nodes[parent].depth < opts.max_depth {
                let key = parent * 256 + text[start - 1] as usize;
                if let Some(&id) = index.get(&key) {
                    candidates[id].counts.add(text[i], 1);
                }
            }
        }
    }
    let considered = candidates.len();
    for c in &mut candidates {
        c.delta = fork_delta(
            model,
            &compiled,
            c.key / 256,
            &c.counts,
            1,
            opts.structure_weight,
        );
    }
    candidates.retain(|c| c.delta < -1e-6);
    candidates.sort_unstable_by(|a, b| a.delta.total_cmp(&b.delta).then(a.key.cmp(&b.key)));
    candidates.truncate(opts.forks_per_round.min(opts.max_nodes - model.nodes.len()));
    let accepted = candidates.len();
    let gain = candidates.iter().map(|c| -c.delta).sum();
    for c in candidates {
        model.add_child(
            c.key / 256,
            ByteSet::singleton((c.key % 256) as u8),
            c.counts,
        );
    }
    (considered, accepted, gain)
}

/// Merge disjoint sibling leaves into a shared conditional distribution. Future
/// growth may specialize this generalized state using additional history.
pub fn merge_leaves(model: &mut Model, opts: &Options) -> (usize, f64) {
    if opts.merges_per_round == 0 {
        return (0, 0.0);
    }
    let comp = Compiled::new(model);
    let mut proposals = Vec::new();
    for (parent, n) in model.nodes.iter().enumerate() {
        let mut leaves: Vec<usize> = n
            .children
            .iter()
            .copied()
            .filter(|&c| {
                model.nodes[c].children.is_empty()
                    && model.nodes[c].counts.total() >= opts.min_count
            })
            .collect();
        leaves.sort_unstable_by(|&a, &b| {
            model.nodes[b]
                .counts
                .total()
                .cmp(&model.nodes[a].counts.total())
                .then(a.cmp(&b))
        });
        leaves.truncate(opts.merge_top);
        for (ai, &a) in leaves.iter().enumerate() {
            for &b in &leaves[ai + 1..] {
                let na = &model.nodes[a];
                let nb = &model.nodes[b];
                // A cheap proposal filter; every surviving pair is scored exactly.
                if na.counts.top() != nb.counts.top() {
                    continue;
                }
                let base = &comp.distributions[parent];
                let sum = na.counts.sum(&nb.counts);
                let d = model.params.discount;
                let t = model.params.theta;
                let delta = math::py_bits(&sum, base, d, t)
                    - math::py_bits(&na.counts, base, d, t)
                    - math::py_bits(&nb.counts, base, d, t)
                    + opts.structure_weight
                        * (node_cost(na.label.len() + nb.label.len(), model.nodes.len())
                            - node_cost(na.label.len(), model.nodes.len())
                            - node_cost(nb.label.len(), model.nodes.len()));
                if delta < -1e-6 {
                    proposals.push((delta, a.min(b), a.max(b), parent));
                }
            }
        }
    }
    proposals.sort_unstable_by(|a, b| a.0.total_cmp(&b.0).then(a.1.cmp(&b.1)).then(a.2.cmp(&b.2)));
    let mut used = HashSet::new();
    let mut merged = 0;
    let mut gain = 0.0;
    for (delta, a, b, parent) in proposals {
        if merged >= opts.merges_per_round {
            break;
        }
        if used.contains(&a) || used.contains(&b) {
            continue;
        }
        used.insert(a);
        used.insert(b);
        model.nodes[a].counts = model.nodes[a].counts.sum(&model.nodes[b].counts);
        model.nodes[a].label = model.nodes[a].label.union(model.nodes[b].label);
        model.nodes[parent].children.retain(|&c| c != b);
        merged += 1;
        gain -= delta;
    }
    if merged > 0 {
        model.compact();
    }
    (merged, gain)
}

/// Discard leaf refinements whose evidence no longer justifies their complexity.
pub fn prune(model: &mut Model, weight: f64) -> usize {
    let mut removed = 0;
    loop {
        let comp = Compiled::new(model);
        let mut drop_ids = Vec::new();
        for i in 1..model.nodes.len() {
            let n = &model.nodes[i];
            if n.children.is_empty()
                && fork_delta(
                    model,
                    &comp,
                    n.parent.unwrap(),
                    &n.counts,
                    n.label.len(),
                    weight,
                ) >= 0.0
            {
                drop_ids.push(i);
            }
        }
        if drop_ids.is_empty() {
            break;
        }
        for &i in &drop_ids {
            let p = model.nodes[i].parent.unwrap();
            model.nodes[p].children.retain(|&c| c != i);
        }
        removed += drop_ids.len();
        model.compact();
    }
    removed
}

pub fn train(
    docs: &[Vec<u8>],
    params: Params,
    opts: &Options,
    mut on_round: impl FnMut(&Model, &RoundReport),
) -> Result<Model> {
    params.validate()?;
    opts.validate()?;
    if docs.iter().all(Vec::is_empty) {
        return Err("training data is empty".into());
    }
    let mut model = Model::root(params, docs);
    for round in 1..=opts.rounds {
        let start = Instant::now();
        let (candidates, added, grow_gain) = grow(&mut model, docs, opts);
        let (merged, merge_gain) = merge_leaves(&mut model, opts);
        // Merges can change the evidence represented by a leaf, so rescore it.
        let pruned = if merged > 0 || round == opts.rounds {
            prune(&mut model, opts.structure_weight)
        } else {
            0
        };
        let report = RoundReport {
            round,
            nodes: model.nodes.len(),
            classes: model.classes(),
            depth: model.max_depth(),
            candidates,
            added,
            merged,
            pruned,
            estimated_gain_bits: grow_gain + merge_gain,
            seconds: start.elapsed().as_secs_f64(),
        };
        on_round(&model, &report);
        if added == 0 && merged == 0 && pruned == 0 {
            break;
        }
    }
    model.validate()?;
    Ok(model)
}

/// Independent recount, used to verify structural edits preserve exact evidence.
pub fn recount(model: &Model, docs: &[Vec<u8>]) -> Vec<Histogram> {
    let c = Compiled::new(model);
    let mut counts = vec![Histogram::default(); model.nodes.len()];
    for text in docs {
        for i in 0..text.len() {
            let mut node = 0;
            let mut j = i;
            counts[0].add(text[i], 1);
            while j > 0 {
                let child = c.next[node][text[j - 1] as usize];
                if child == 0 {
                    break;
                }
                node = child as usize - 1;
                j -= 1;
                counts[node].add(text[i], 1);
            }
        }
    }
    counts
}
