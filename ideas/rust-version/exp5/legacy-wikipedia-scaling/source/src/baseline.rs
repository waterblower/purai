use crate::graph::{Histogram, Params};
use std::collections::HashMap;

/// A fixed-order byte n-gram baseline. No topology selection, class merging or
/// match copier. Smoothing and document boundaries match the learned model.
pub struct Ngram {
    tables: Vec<HashMap<Vec<u8>, Histogram>>,
    params: Params,
}
impl Ngram {
    pub fn fit(docs: &[Vec<u8>], order: usize, params: Params) -> Self {
        let mut tables: Vec<HashMap<Vec<u8>, Histogram>> =
            (0..=order).map(|_| HashMap::new()).collect();
        for text in docs {
            for i in 0..text.len() {
                for (k, table) in tables.iter_mut().enumerate().take(i.min(order) + 1) {
                    let key = &text[i - k..i];
                    if let Some(counts) = table.get_mut(key) {
                        counts.add(text[i], 1);
                    } else {
                        let mut counts = Histogram::default();
                        counts.add(text[i], 1);
                        table.insert(key.to_vec(), counts);
                    }
                }
            }
        }
        Self { tables, params }
    }
    pub fn contexts(&self) -> usize {
        self.tables.iter().map(HashMap::len).sum()
    }
    pub fn evaluate(&self, docs: &[Vec<u8>]) -> f64 {
        let mut bits = 0.0;
        for text in docs {
            for i in 0..text.len() {
                let mut p = 1.0 / 256.0;
                for (k, table) in self
                    .tables
                    .iter()
                    .enumerate()
                    .take(i.min(self.tables.len() - 1) + 1)
                {
                    let Some(c) = table.get(&text[i - k..i]) else {
                        break;
                    };
                    let d = self.params.discount;
                    let theta = self.params.theta;
                    p = ((c.get(text[i]) as f64 - d).max(0.0) + (theta + d * c.0.len() as f64) * p)
                        / (c.total() as f64 + theta);
                }
                bits -= p.log2();
            }
        }
        bits
    }
}
