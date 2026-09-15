/// Semantic Partition Module (SPM) for Flux v3.
/// Generic over Float type.

use crate::float::Float;
use crate::rng::Rng;

pub const K: usize = 4;
pub const STRIDE: usize = 4;

pub fn spm_global_count(d: usize, nl: usize) -> usize {
    K * d + nl * (K + d)
}

pub struct SpmOffsets {
    pub w: usize,
    pub delta: usize,
    pub gate: usize,
}

pub fn spm_offsets(
    base: usize, d: usize, layer: usize,
) -> SpmOffsets {
    SpmOffsets {
        w: base,
        delta: base + K * d + layer * (K + d),
        gate: base + K * d + layer * (K + d) + K,
    }
}

pub fn spm_init<F: Float>(
    params: &mut [F], base: usize,
    d: usize, nl: usize, rng: &mut Rng,
) {
    let xavier = (2.0 / (K + d) as f64).sqrt();
    for i in 0..K * d {
        params[base + i] = F::from_f64(rng.gaussian_scaled(xavier));
    }
    for li in 0..nl {
        let off = spm_offsets(base, d, li);
        for i in 0..K {
            let frac = i as f64 / (K.max(2) - 1) as f64;
            params[off.delta + i] = F::from_f64(-5.0 + frac * 2.0);
        }
        for j in 0..d {
            params[off.gate + j] = F::from_f64(-5.0);
        }
    }
}
