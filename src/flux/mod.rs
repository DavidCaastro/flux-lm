/// Flux v3 model: WHT-D with content-dependent gating + dual-timescale memory.
/// Generic over Float type for f32/f64 support.

pub mod wht;
pub mod spm;
pub mod forward;
pub mod backward;
#[cfg(test)]
mod grad_check;

use crate::float::Float;
use crate::rng::Rng;

pub struct FluxModel<F: Float> {
    pub d: usize,
    pub n_layers: usize,
    pub params: Vec<F>,
}

/// Per-layer params (v2/v3): 525*d
pub fn flux_layer_param_count(d: usize) -> usize {
    525 * d
}

pub fn flux_total_param_count(
    d: usize, n_layers: usize,
) -> usize {
    256 * d + n_layers * flux_layer_param_count(d)
        + 256 * d + 256
        + spm::spm_global_count(d, n_layers)
}

pub fn flux_pre_spm_param_count(
    d: usize, n_layers: usize,
) -> usize {
    256 * d + n_layers * flux_layer_param_count(d)
        + 256 * d + 256
}

pub fn flux_layer_offset(d: usize, layer: usize) -> usize {
    256 * d + layer * flux_layer_param_count(d)
}

impl<F: Float> FluxModel<F> {
    pub fn new(
        d: usize, n_layers: usize, rng: &mut Rng,
    ) -> Self {
        assert!(d.is_power_of_two(), "d must be power of 2 for WHT");
        let np = flux_total_param_count(d, n_layers);
        let mut params = vec![F::ZERO; np];
        init_flux(&mut params, d, n_layers, rng);
        Self { d, n_layers, params }
    }

    pub fn n_params(&self) -> usize {
        self.params.len()
    }
}

fn init_flux<F: Float>(
    p: &mut [F], d: usize, nl: usize, rng: &mut Rng,
) {
    for i in 0..256 * d {
        p[i] = F::from_f64(rng.gaussian_scaled(0.1));
    }
    for li in 0..nl {
        let lo = flux_layer_offset(d, li);
        init_flux_layer(p, lo, d, rng);
    }
    let out_off = flux_layer_offset(d, nl);
    let s = (2.0 / (d + 256) as f64).sqrt();
    for i in 0..256 * d {
        p[out_off + i] = F::from_f64(rng.gaussian_scaled(s));
    }

    let spm_base = flux_pre_spm_param_count(d, nl);
    spm::spm_init(p, spm_base, d, nl, rng);
}

fn init_flux_layer<F: Float>(
    p: &mut [F], lo: usize, d: usize, rng: &mut Rng,
) {
    let mut o = lo;

    for k in 0..d { p[o + k] = F::ONE; }
    o += d;

    for i in 0..256 * d {
        p[o + i] = F::from_f64(rng.gaussian_scaled(0.1));
    }
    o += 256 * d;

    for byte in 0..256usize {
        for k in 0..d {
            p[o + byte * d + k] =
                if byte == k % 256 { F::from_f64(0.3) } else { F::ZERO };
        }
    }
    o += 256 * d;

    for k in 0..d {
        p[o + k] = F::from_f64(rng.gaussian_scaled(0.1));
    }
    o += d;

    for k in 0..d { p[o + k] = F::ONE; }
    o += d;
    o += d; // b1 zeros

    for k in 0..d { p[o + k] = F::ONE; }
    o += d;
    o += d; // b2 zeros

    for k in 0..d {
        let frac = k as f64 / d.max(2) as f64;
        p[o + k] = F::from_f64(0.1 + frac * 0.4);
    }
    o += d;

    for k in 0..d { p[o + k] = F::ONE; }
    o += d;

    let sc = 0.3 / (d as f64).sqrt();
    for k in 0..d {
        p[o + k] = F::from_f64(rng.gaussian_scaled(sc));
    }
    o += d;

    for k in 0..d { p[o + k] = F::from_f64(0.5); }
    o += d;

    for k in 0..d {
        let frac = k as f64 / d.max(2) as f64;
        p[o + k] = F::from_f64(-2.0 + frac * 2.5);
    }
    o += d;

    for k in 0..d { p[o + k] = F::ONE; }
    o += d;

    for k in 0..d {
        p[o + k] = F::from_f64(rng.gaussian_scaled(sc));
    }
    let _ = o;
}
