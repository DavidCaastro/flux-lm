/// Flux v3 forward pass: generic + activation checkpointing.

use crate::float::Float;
use crate::math::softmax_inplace;
use super::wht::wht_inplace;
use super::flux_layer_offset;
use super::flux_pre_spm_param_count;
use super::spm::{self, K, STRIDE};

fn rmsnorm<F: Float>(x: &mut [F], gamma: &[F], d: usize) {
    let eps = F::from_f64(1e-8);
    let mut ss = F::ZERO;
    for k in 0..d { ss = ss + x[k] * x[k]; }
    let rms = (ss / F::from_usize(d) + eps).sqrt();
    let inv = F::ONE / rms;
    for k in 0..d { x[k] = x[k] * inv * gamma[k]; }
}

fn sigmoid<F: Float>(x: F) -> F {
    let fifteen = F::from_f64(15.0);
    if x > fifteen { F::ONE }
    else if x < -fifteen { F::ZERO }
    else { F::ONE / (F::ONE + (-x).exp()) }
}

fn softplus<F: Float>(x: F) -> F {
    let twenty = F::from_f64(20.0);
    if x > twenty { x } else { (F::ONE + x.exp()).ln() }
}

pub struct FluxStepData<F: Float> {
    pub input: Vec<F>,
    pub logits: Vec<F>,
    pub probs: Vec<F>,
    pub target: usize,
}

pub struct FluxLayerOffsets {
    pub rn: usize,
    pub g_gate: usize,
    pub a_bias: usize,
    pub w_gate_h: usize,
    pub s1: usize,
    pub b1: usize,
    pub s2: usize,
    pub b2: usize,
    pub delta_fast: usize,
    pub b_in_fast: usize,
    pub c_out_fast: usize,
    pub skip: usize,
    pub delta_slow: usize,
    pub b_in_slow: usize,
    pub c_out_slow: usize,
}

pub fn flux_layer_offsets(
    lo: usize, d: usize,
) -> FluxLayerOffsets {
    let mut o = lo;
    let rn = o; o += d;
    let g_gate = o; o += 256 * d;
    let a_bias = o; o += 256 * d;
    let w_gate_h = o; o += d;
    let s1 = o; o += d;
    let b1 = o; o += d;
    let s2 = o; o += d;
    let b2 = o; o += d;
    let delta_fast = o; o += d;
    let b_in_fast = o; o += d;
    let c_out_fast = o; o += d;
    let skip = o; o += d;
    let delta_slow = o; o += d;
    let b_in_slow = o; o += d;
    let c_out_slow = o;
    let _ = o;
    FluxLayerOffsets {
        rn, g_gate, a_bias, w_gate_h,
        s1, b1, s2, b2,
        delta_fast, b_in_fast, c_out_fast, skip,
        delta_slow, b_in_slow, c_out_slow,
    }
}

/// Full forward pass for training with activation checkpointing.
/// layer_checkpoints stores layer_in snapshot before each layer mutates it.
pub fn flux_forward_pass<F: Float>(
    corpus: &[u8], start: usize, seq_len: usize,
    params: &[F], d: usize, n_layers: usize,
    intermediates: &mut Vec<FluxStepData<F>>,
    layer_checkpoints: &mut Vec<Vec<Vec<F>>>,
) -> F {
    let embed = &params[0..256 * d];
    let out_base = flux_layer_offset(d, n_layers);
    let w_out = &params[out_base..out_base + 256 * d];
    let b_out =
        &params[out_base + 256 * d..out_base + 256 * d + 256];

    let bytes: Vec<u8> =
        (0..seq_len).map(|t| corpus[start + t]).collect();

    let mut layer_in: Vec<Vec<F>> = bytes.iter().map(|&b| {
        embed[b as usize * d..(b as usize + 1) * d].to_vec()
    }).collect();

    intermediates.clear();
    layer_checkpoints.clear();

    let mut state = vec![F::ZERO; d];
    let mut out = vec![F::ZERO; d];
    let spm_base = flux_pre_spm_param_count(d, n_layers);

    for li in 0..n_layers {
        // Activation checkpoint: snapshot layer_in before this layer
        layer_checkpoints.push(layer_in.iter().map(|v| v.clone()).collect());

        let lo = flux_layer_offset(d, li);
        let lf = flux_layer_offsets(lo, d);
        let spm_off = spm::spm_offsets(spm_base, d, li);

        let lambdas_fast: Vec<F> = (0..d).map(|k| {
            (-softplus(params[lf.delta_fast + k])).exp()
        }).collect();
        let lambdas_slow: Vec<F> = (0..d).map(|k| {
            (-softplus(params[lf.delta_slow + k])).exp()
        }).collect();
        let lambdas_sem: [F; K] = {
            let mut l = [F::ZERO; K];
            for i in 0..K {
                l[i] = (-softplus(params[spm_off.delta + i])).exp();
            }
            l
        };
        let gate_sig: Vec<F> = (0..d).map(|j| {
            sigmoid(params[spm_off.gate + j])
        }).collect();

        let mut h_fast = vec![F::ZERO; d];
        let mut h_slow = vec![F::ZERO; d];
        let mut h_sem = [F::ZERO; K];
        let mut cond = vec![F::ZERO; d];
        let res_scale = F::ONE / F::from_usize(li + 2).ln();

        for t in 0..seq_len {
            let bi = bytes[t] as usize;
            state.copy_from_slice(&layer_in[t]);

            rmsnorm(
                &mut state, &params[lf.rn..lf.rn + d], d,
            );

            for k in 0..d {
                let gate_in = params[lf.g_gate + bi * d + k]
                    + params[lf.w_gate_h + k] * h_fast[k];
                let gate = sigmoid(gate_in);
                state[k] = gate * state[k]
                    + params[lf.a_bias + bi * d + k];
            }

            wht_inplace(&mut state);
            for k in 0..d {
                state[k] = (params[lf.s1 + k] * state[k]
                    + params[lf.b1 + k]).tanh();
            }

            wht_inplace(&mut state);
            for k in 0..d {
                state[k] = (params[lf.s2 + k] * state[k]
                    + params[lf.b2 + k]).tanh();
            }

            for k in 0..d {
                h_fast[k] = lambdas_fast[k] * h_fast[k]
                    + params[lf.b_in_fast + k] * state[k];
                h_slow[k] = lambdas_slow[k] * h_slow[k]
                    + params[lf.b_in_slow + k] * state[k];
            }

            if t & (STRIDE - 1) == 0 {
                for i in 0..K {
                    let row = &params[spm_off.w + i * d
                        ..spm_off.w + (i + 1) * d];
                    let mut s = F::ZERO;
                    for j in 0..d { s = s + row[j] * h_slow[j]; }
                    h_sem[i] = lambdas_sem[i] * h_sem[i]
                        + (F::ONE - lambdas_sem[i]) * s;
                }
                cond.iter_mut().for_each(|v| *v = F::ZERO);
                for i in 0..K {
                    let scale = h_sem[i];
                    let row = &params[spm_off.w + i * d
                        ..spm_off.w + (i + 1) * d];
                    for j in 0..d {
                        cond[j] = cond[j] + row[j] * scale;
                    }
                }
                for j in 0..d { cond[j] = cond[j] * gate_sig[j]; }
            }

            for k in 0..d {
                out[k] = params[lf.c_out_fast + k] * h_fast[k]
                    + params[lf.c_out_slow + k] * h_slow[k]
                    + params[lf.skip + k] * state[k]
                    + res_scale * layer_in[t][k]
                    + cond[k];
            }
            layer_in[t].copy_from_slice(&out);
        }
    }

    compute_loss(
        corpus, start, seq_len, w_out, b_out, d,
        &layer_in, intermediates,
    )
}

fn compute_loss<F: Float>(
    corpus: &[u8], start: usize, seq_len: usize,
    w_out: &[F], b_out: &[F], d: usize,
    layer_in: &[Vec<F>],
    intermediates: &mut Vec<FluxStepData<F>>,
) -> F {
    let mut total_loss = F::ZERO;
    let mut logits = vec![F::ZERO; 256];
    let mut probs = vec![F::ZERO; 256];
    let min_p = F::from_f64(1e-15);
    for t in 0..seq_len {
        let target = corpus[start + t + 1] as usize;
        let x = &layer_in[t];

        for i in 0..256 {
            let mut s = b_out[i];
            let base = i * d;
            for j in 0..d {
                s = s + w_out[base + j] * x[j];
            }
            logits[i] = s;
        }

        probs.copy_from_slice(&logits);
        softmax_inplace(&mut probs);

        let p = probs[target].max(min_p);
        total_loss = total_loss + (-p.ln());

        intermediates.push(FluxStepData {
            input: layer_in[t].clone(),
            logits: logits.clone(),
            probs: probs.clone(),
            target,
        });
    }
    total_loss / F::from_usize(seq_len)
}

/// Inference: process sequence, return P(next byte).
pub fn flux_forward_inference<F: Float>(
    seq: &[u8], params: &[F], d: usize, n_layers: usize,
) -> Vec<F> {
    let embed = &params[0..256 * d];
    let out_base = flux_layer_offset(d, n_layers);
    let w_out = &params[out_base..out_base + 256 * d];
    let b_out =
        &params[out_base + 256 * d..out_base + 256 * d + 256];

    let mut layer_in: Vec<Vec<F>> = seq.iter().map(|&b| {
        embed[b as usize * d..(b as usize + 1) * d].to_vec()
    }).collect();

    let mut state = vec![F::ZERO; d];
    let mut out = vec![F::ZERO; d];
    let spm_base = flux_pre_spm_param_count(d, n_layers);

    for li in 0..n_layers {
        let lo = flux_layer_offset(d, li);
        let lf = flux_layer_offsets(lo, d);
        let spm_off = spm::spm_offsets(spm_base, d, li);

        let lambdas_fast: Vec<F> = (0..d).map(|k| {
            (-softplus(params[lf.delta_fast + k])).exp()
        }).collect();
        let lambdas_slow: Vec<F> = (0..d).map(|k| {
            (-softplus(params[lf.delta_slow + k])).exp()
        }).collect();
        let lambdas_sem: [F; K] = {
            let mut l = [F::ZERO; K];
            for i in 0..K {
                l[i] = (-softplus(params[spm_off.delta + i])).exp();
            }
            l
        };
        let gate_sig: Vec<F> = (0..d).map(|j| {
            sigmoid(params[spm_off.gate + j])
        }).collect();

        let mut h_fast = vec![F::ZERO; d];
        let mut h_slow = vec![F::ZERO; d];
        let mut h_sem = [F::ZERO; K];
        let mut cond = vec![F::ZERO; d];
        let res_scale = F::ONE / F::from_usize(li + 2).ln();

        for t in 0..seq.len() {
            let bi = seq[t] as usize;
            state.copy_from_slice(&layer_in[t]);

            rmsnorm(
                &mut state, &params[lf.rn..lf.rn + d], d,
            );

            for k in 0..d {
                let gate_in = params[lf.g_gate + bi * d + k]
                    + params[lf.w_gate_h + k] * h_fast[k];
                let gate = sigmoid(gate_in);
                state[k] = gate * state[k]
                    + params[lf.a_bias + bi * d + k];
            }

            wht_inplace(&mut state);
            for k in 0..d {
                state[k] = (params[lf.s1 + k] * state[k]
                    + params[lf.b1 + k]).tanh();
            }

            wht_inplace(&mut state);
            for k in 0..d {
                state[k] = (params[lf.s2 + k] * state[k]
                    + params[lf.b2 + k]).tanh();
            }

            for k in 0..d {
                h_fast[k] = lambdas_fast[k] * h_fast[k]
                    + params[lf.b_in_fast + k] * state[k];
                h_slow[k] = lambdas_slow[k] * h_slow[k]
                    + params[lf.b_in_slow + k] * state[k];
            }

            if t & (STRIDE - 1) == 0 {
                for i in 0..K {
                    let row = &params[spm_off.w + i * d
                        ..spm_off.w + (i + 1) * d];
                    let mut s = F::ZERO;
                    for j in 0..d { s = s + row[j] * h_slow[j]; }
                    h_sem[i] = lambdas_sem[i] * h_sem[i]
                        + (F::ONE - lambdas_sem[i]) * s;
                }
                cond.iter_mut().for_each(|v| *v = F::ZERO);
                for i in 0..K {
                    let scale = h_sem[i];
                    let row = &params[spm_off.w + i * d
                        ..spm_off.w + (i + 1) * d];
                    for j in 0..d {
                        cond[j] = cond[j] + row[j] * scale;
                    }
                }
                for j in 0..d { cond[j] = cond[j] * gate_sig[j]; }
            }

            for k in 0..d {
                out[k] = params[lf.c_out_fast + k] * h_fast[k]
                    + params[lf.c_out_slow + k] * h_slow[k]
                    + params[lf.skip + k] * state[k]
                    + res_scale * layer_in[t][k]
                    + cond[k];
            }
            layer_in[t].copy_from_slice(&out);
        }
    }

    let last = layer_in.last().unwrap();
    let mut logits = vec![F::ZERO; 256];
    for i in 0..256 {
        let mut s = b_out[i];
        let base = i * d;
        for j in 0..d { s = s + w_out[base + j] * last[j]; }
        logits[i] = s;
    }
    softmax_inplace(&mut logits);
    logits
}
