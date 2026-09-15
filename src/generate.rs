/// Text generation for Flux v3 — generic over Float.

use crate::float::Float;
use crate::math::softmax_inplace;
use crate::rng::Rng;
use crate::flux::FluxModel;
use crate::flux::forward::flux_forward_inference;

pub fn generate_flux<F: Float>(
    model: &FluxModel<F>, seed_text: &[u8], length: usize,
    temperature: f64, rng: &mut Rng,
) -> Vec<u8> {
    let mut seq: Vec<u8> = seed_text.to_vec();
    let max_ctx = 256;

    for _ in 0..length {
        let start = if seq.len() > max_ctx {
            seq.len() - max_ctx
        } else {
            0
        };
        let ctx = &seq[start..];

        let mut probs = flux_forward_inference(
            ctx, &model.params, model.d, model.n_layers,
        );

        if (temperature - 1.0).abs() > 1e-6 {
            apply_temperature(&mut probs, temperature);
        }

        let chosen = sample(&probs, rng);
        seq.push(chosen);
    }

    seq[seed_text.len()..].to_vec()
}

fn apply_temperature<F: Float>(probs: &mut [F], temperature: f64) {
    let t = F::from_f64(temperature);
    let min_p = F::from_f64(1e-15);
    for p in probs.iter_mut() {
        *p = (*p).max(min_p).ln() / t;
    }
    softmax_inplace(probs);
}

fn sample<F: Float>(probs: &[F], rng: &mut Rng) -> u8 {
    let r = rng.random();
    let mut cum = 0.0;
    for (i, &p) in probs.iter().enumerate() {
        cum += p.to_f64();
        if r < cum {
            return i as u8;
        }
    }
    255
}
