use crate::{Result, math};

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Histogram(pub Vec<(u8, u64)>);

impl Histogram {
    pub fn add(&mut self, byte: u8, n: u64) {
        if n == 0 {
            return;
        }
        match self.0.binary_search_by_key(&byte, |&(b, _)| b) {
            Ok(i) => self.0[i].1 = self.0[i].1.checked_add(n).expect("count overflow"),
            Err(i) => self.0.insert(i, (byte, n)),
        }
    }
    pub fn total(&self) -> u64 {
        self.0.iter().map(|&(_, c)| c).sum()
    }
    pub fn get(&self, b: u8) -> u64 {
        self.0
            .binary_search_by_key(&b, |&(x, _)| x)
            .map_or(0, |i| self.0[i].1)
    }
    pub fn sum(&self, other: &Self) -> Self {
        let mut result = self.clone();
        for &(b, n) in &other.0 {
            result.add(b, n);
        }
        result
    }
    pub fn subtract(&self, other: &Self) -> Self {
        Self(
            self.0
                .iter()
                .filter_map(|&(b, n)| {
                    let c = n
                        .checked_sub(other.get(b))
                        .expect("child count exceeds parent");
                    (c != 0).then_some((b, c))
                })
                .collect(),
        )
    }
    pub fn top(&self) -> u8 {
        self.0
            .iter()
            .max_by(|(a, ac), (b, bc)| ac.cmp(bc).then_with(|| b.cmp(a)))
            .map_or(0, |&(b, _)| b)
    }
}

/// A learned equivalence class on the preceding byte. Siblings are disjoint.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct ByteSet(pub [u64; 4]);
impl ByteSet {
    pub fn singleton(b: u8) -> Self {
        let mut s = Self::default();
        s.insert(b);
        s
    }
    pub fn insert(&mut self, b: u8) {
        self.0[b as usize / 64] |= 1u64 << (b % 64);
    }
    pub fn contains(&self, b: u8) -> bool {
        self.0[b as usize / 64] & (1u64 << (b % 64)) != 0
    }
    pub fn union(self, other: Self) -> Self {
        Self(std::array::from_fn(|i| self.0[i] | other.0[i]))
    }
    pub fn len(&self) -> usize {
        self.0.iter().map(|v| v.count_ones() as usize).sum()
    }
    pub fn is_empty(&self) -> bool {
        self.0 == [0; 4]
    }
    pub fn bytes(&self) -> impl Iterator<Item = u8> + '_ {
        (0..=255u8).filter(|&b| self.contains(b))
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Node {
    pub parent: Option<usize>,
    pub label: ByteSet,
    pub counts: Histogram,
    pub children: Vec<usize>,
    pub depth: usize,
}

#[derive(Clone, Debug, PartialEq)]
pub struct Params {
    pub discount: f64,
    pub theta: f64,
}
impl Default for Params {
    fn default() -> Self {
        Self {
            discount: 0.75,
            theta: 2.0,
        }
    }
}
impl Params {
    pub fn validate(&self) -> Result<()> {
        if !(0.0..1.0).contains(&self.discount) || !self.theta.is_finite() || self.theta <= 0.0 {
            return Err("require 0 <= discount < 1 and finite theta > 0".into());
        }
        Ok(())
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Calibration {
    pub hits: [u64; 256],
    pub totals: [u64; 256],
}
impl Default for Calibration {
    fn default() -> Self {
        Self {
            hits: [0; 256],
            totals: [0; 256],
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct Model {
    pub params: Params,
    pub nodes: Vec<Node>,
    pub calibration: Calibration,
    pub training_bytes: u64,
    pub calibration_bytes: u64,
}

impl Model {
    pub fn root(params: Params, docs: &[Vec<u8>]) -> Self {
        let mut counts = Histogram::default();
        for doc in docs {
            for &b in doc {
                counts.add(b, 1);
            }
        }
        let training_bytes = counts.total();
        Self {
            params,
            nodes: vec![Node {
                parent: None,
                label: ByteSet::default(),
                counts,
                children: Vec::new(),
                depth: 0,
            }],
            calibration: Calibration::default(),
            training_bytes,
            calibration_bytes: 0,
        }
    }
    pub fn add_child(&mut self, parent: usize, label: ByteSet, counts: Histogram) -> usize {
        let id = self.nodes.len();
        let depth = self.nodes[parent].depth + 1;
        self.nodes.push(Node {
            parent: Some(parent),
            label,
            counts,
            children: Vec::new(),
            depth,
        });
        self.nodes[parent].children.push(id);
        id
    }
    pub fn max_depth(&self) -> usize {
        self.nodes.iter().map(|n| n.depth).max().unwrap_or(0)
    }
    pub fn classes(&self) -> usize {
        self.nodes.iter().filter(|n| n.label.len() > 1).count()
    }

    /// Keep the authoritative counts, remove unreachable nodes, renumber topologically.
    pub fn compact(&mut self) {
        let mut live = vec![false; self.nodes.len()];
        live[0] = true;
        for i in 0..self.nodes.len() {
            if live[i] {
                for &c in &self.nodes[i].children {
                    live[c] = true;
                }
            }
        }
        let mut remap = vec![usize::MAX; live.len()];
        let mut nodes = Vec::new();
        for (i, n) in self.nodes.iter().enumerate() {
            if live[i] {
                remap[i] = nodes.len();
                nodes.push(n.clone());
            }
        }
        for n in &mut nodes {
            n.parent = n.parent.map(|p| remap[p]);
            for c in &mut n.children {
                *c = remap[*c];
            }
            n.children.sort_unstable();
        }
        self.nodes = nodes;
    }

    pub fn validate(&self) -> Result<()> {
        self.params.validate()?;
        if self.nodes.is_empty()
            || self.nodes[0].parent.is_some()
            || !self.nodes[0].label.is_empty()
            || self.nodes[0].depth != 0
        {
            return Err("invalid graph root".into());
        }
        let mut incoming = vec![0usize; self.nodes.len()];
        for (i, n) in self.nodes.iter().enumerate() {
            let mut last = None;
            let mut total = 0u64;
            for &(b, count) in &n.counts.0 {
                if count == 0 || last.is_some_and(|p| b <= p) {
                    return Err("invalid sparse counts".into());
                }
                total = total.checked_add(count).ok_or("count overflow")?;
                last = Some(b);
            }
            if total > self.training_bytes {
                return Err("node count exceeds training size".into());
            }
            if i == 0 && total != self.training_bytes {
                return Err("root/training byte count mismatch".into());
            }
            if i != 0 {
                let p = n.parent.ok_or("non-root has no parent")?;
                if p >= i
                    || n.depth != self.nodes[p].depth + 1
                    || n.depth > 128
                    || n.label.is_empty()
                {
                    return Err("graph is not a valid topologically ordered context tree".into());
                }
            }
            let mut labels = ByteSet::default();
            let mut child_counts = [0u64; 256];
            for &c in &n.children {
                if c <= i || c >= self.nodes.len() || self.nodes[c].parent != Some(i) {
                    return Err("invalid child reference".into());
                }
                incoming[c] += 1;
                for b in self.nodes[c].label.bytes() {
                    if labels.contains(b) {
                        return Err("overlapping child labels".into());
                    }
                    labels.insert(b);
                }
                for &(b, count) in &self.nodes[c].counts.0 {
                    child_counts[b as usize] = child_counts[b as usize]
                        .checked_add(count)
                        .ok_or("count overflow")?;
                }
            }
            for (b, c) in child_counts.iter().enumerate() {
                if *c > n.counts.get(b as u8) {
                    return Err("children exceed parent evidence".into());
                }
            }
        }
        if incoming.iter().skip(1).any(|&n| n != 1) {
            return Err("unreachable or multiply owned node".into());
        }
        for i in 0..256 {
            if self.calibration.hits[i] > self.calibration.totals[i]
                || self.calibration.totals[i] > self.calibration_bytes
            {
                return Err("invalid calibration table".into());
            }
        }
        Ok(())
    }
}

/// Derived tables. Topology and sparse integer evidence remain authoritative.
pub struct Compiled {
    pub next: Vec<[u32; 256]>,
    pub distributions: Vec<[f64; 256]>,
}
impl Compiled {
    pub fn new(model: &Model) -> Self {
        let mut next = vec![[0u32; 256]; model.nodes.len()];
        let mut distributions: Vec<[f64; 256]> = Vec::with_capacity(model.nodes.len());
        for (i, n) in model.nodes.iter().enumerate() {
            for &c in &n.children {
                for b in model.nodes[c].label.bytes() {
                    next[i][b as usize] = (c + 1) as u32;
                }
            }
            let base = n.parent.map_or(&math::UNIFORM, |p| &distributions[p]);
            distributions.push(math::smooth(
                &n.counts,
                base,
                model.params.discount,
                model.params.theta,
            ));
        }
        Self {
            next,
            distributions,
        }
    }
    /// Traverse history backwards. Never reads text[end] or any future byte.
    pub fn context(&self, text: &[u8], end: usize, budget: usize) -> (usize, usize, usize) {
        let mut node = 0;
        let mut start = end;
        let mut steps = 1; // root distribution access
        while start > 0 && steps < budget {
            steps += 1;
            let child = self.next[node][text[start - 1] as usize];
            if child == 0 {
                break;
            }
            node = child as usize - 1;
            start -= 1;
        }
        (node, start, steps)
    }
}
