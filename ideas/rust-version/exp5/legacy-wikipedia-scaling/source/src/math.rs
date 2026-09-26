use crate::graph::Histogram;

pub const UNIFORM: [f64; 256] = [1.0 / 256.0; 256];

const COEFFICIENTS: [f64; 9] = [
    0.999_999_999_999_809_9,
    676.520_368_121_885_1,
    -1_259.139_216_722_402_8,
    771.323_428_777_653_1,
    -176.615_029_162_140_6,
    12.507_343_278_686_905,
    -0.138_571_095_265_720_12,
    9.984_369_578_019_572e-6,
    1.505_632_735_149_311_6e-7,
];

pub fn lgamma(x: f64) -> f64 {
    if x < 0.5 {
        return (std::f64::consts::PI / (std::f64::consts::PI * x).sin()).ln() - lgamma(1.0 - x);
    }
    let x = x - 1.0;
    let t = x + 7.5;
    let a = COEFFICIENTS
        .iter()
        .enumerate()
        .skip(1)
        .fold(COEFFICIENTS[0], |s, (i, c)| s + c / (x + i as f64));
    0.5 * (2.0 * std::f64::consts::PI).ln() + (x + 0.5) * t.ln() - t + a.ln()
}

/// Pitman-Yor marginal code length, one occupied table per observed byte.
pub fn py_bits(counts: &Histogram, base: &[f64; 256], discount: f64, theta: f64) -> f64 {
    if counts.total() == 0 {
        return 0.0;
    }
    let g = lgamma(1.0 - discount);
    let mut s = 0.0;
    for (u, &(x, c)) in counts.0.iter().enumerate() {
        s += base[x as usize].ln()
            + (theta + discount * u as f64).ln()
            + lgamma(c as f64 - discount)
            - g;
    }
    -(s + lgamma(theta) - lgamma(theta + counts.total() as f64)) / std::f64::consts::LN_2
}

pub fn smooth(counts: &Histogram, base: &[f64; 256], discount: f64, theta: f64) -> [f64; 256] {
    let n = counts.total();
    if n == 0 {
        return *base;
    }
    let den = n as f64 + theta;
    let g = (theta + discount * counts.0.len() as f64) / den;
    let mut p = base.map(|v| v * g);
    for &(b, c) in &counts.0 {
        p[b as usize] += (c as f64 - discount).max(0.0) / den;
    }
    p
}

/// SplitMix64 with a specified mapping to [0, 1); independent of std RNG versions.
pub struct Rng(pub u64);
impl Rng {
    pub fn uniform(&mut self) -> f64 {
        self.0 = self.0.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        z ^= z >> 31;
        (z >> 11) as f64 / (1u64 << 53) as f64
    }
}
