/// Flux v3 backward pass: generic + O(L) via activation checkpoints.
/// Eliminated: recompute_layer_inputs, run_layer_forward, infer_n_layers.

use crate::float::Float;
use super::forward::{FluxStepData, FluxLayerOffsets, flux_layer_offsets};
use super::wht::wht_inplace;
use super::flux_layer_offset;
use super::flux_pre_spm_param_count;
use super::spm::{self, K, STRIDE};

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

pub fn flux_backward_pass<F: Float>(
    corpus: &[u8], start: usize, seq_len: usize,
    params: &[F], d: usize, n_layers: usize,
    intermediates: &[FluxStepData<F>],
    layer_checkpoints: &[Vec<Vec<F>>],
    grads: &mut [F],
) {
    let scale = F::ONE / F::from_usize(seq_len);
    for g in grads.iter_mut() { *g = F::ZERO; }

    let out_base = flux_layer_offset(d, n_layers);
    let w_out = &params[out_base..out_base + 256 * d];

    let mut dl_dinput = vec![vec![F::ZERO; d]; seq_len];

    grad_output_proj(
        intermediates, w_out, grads, out_base,
        d, seq_len, scale, &mut dl_dinput,
    );

    for li in (0..n_layers).rev() {
        backward_flux_layer(
            corpus, start, seq_len, params, grads,
            d, li, n_layers, &mut dl_dinput,
            &layer_checkpoints[li],
        );
    }

    grad_embedding(corpus, start, seq_len, d, grads, &dl_dinput);
}

fn grad_output_proj<F: Float>(
    intermediates: &[FluxStepData<F>], w_out: &[F],
    grads: &mut [F], out_base: usize,
    d: usize, seq_len: usize, scale: F,
    dl_dinput: &mut [Vec<F>],
) {
    let mut dl_dlogits = vec![F::ZERO; 256];
    for t in 0..seq_len {
        let step = &intermediates[t];
        for i in 0..256 {
            let tgt = if i == step.target { F::ONE } else { F::ZERO };
            dl_dlogits[i] = (step.probs[i] - tgt) * scale;
        }

        for i in 0..256 {
            let base = out_base + i * d;
            for j in 0..d {
                grads[base + j] = grads[base + j]
                    + dl_dlogits[i] * step.input[j];
            }
            grads[out_base + 256 * d + i] =
                grads[out_base + 256 * d + i] + dl_dlogits[i];
        }

        for j in 0..d {
            let mut s = F::ZERO;
            for i in 0..256 {
                s = s + w_out[i * d + j] * dl_dlogits[i];
            }
            dl_dinput[t][j] = s;
        }
    }
}

fn grad_embedding<F: Float>(
    corpus: &[u8], start: usize, seq_len: usize,
    d: usize, grads: &mut [F], dl_dinput: &[Vec<F>],
) {
    for t in 0..seq_len {
        let bi = corpus[start + t] as usize;
        for k in 0..d {
            grads[bi * d + k] = grads[bi * d + k] + dl_dinput[t][k];
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn backward_flux_layer<F: Float>(
    corpus: &[u8], start: usize, seq_len: usize,
    params: &[F], grads: &mut [F],
    d: usize, li: usize, n_layers: usize,
    dl_dout: &mut Vec<Vec<F>>,
    layer_inputs: &[Vec<F>],
) {
    let lo = flux_layer_offset(d, li);
    let lf = flux_layer_offsets(lo, d);
    let res_scale = F::ONE / F::from_usize(li + 2).ln();

    let spm_base = flux_pre_spm_param_count(d, n_layers);
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

    let bytes: Vec<u8> =
        (0..seq_len).map(|t| corpus[start + t]).collect();

    // Use layer_inputs from checkpoint (O(1) access, no recompute)
    let (states_after_rn,
         states_after_wht1, states_after_tanh1,
         states_after_wht2, states_after_tanh2,
         h_fast_hist, h_slow_hist) = recompute_layer_internals(
        &bytes, params, d, &lf,
        &lambdas_fast, &lambdas_slow, seq_len,
        layer_inputs,
    );

    let (h_sem_hist, z_hist) = recompute_spm_state(
        params, d, &spm_off, &lambdas_sem, &h_slow_hist, seq_len,
    );

    let mut dh_fast = vec![F::ZERO; d];
    let mut dh_slow = vec![F::ZERO; d];
    let mut dl_dstate = vec![F::ZERO; d];
    let mut dl_dwht2 = vec![F::ZERO; d];
    let mut dl_dwht1 = vec![F::ZERO; d];
    let mut dl_pre_rn = vec![F::ZERO; d];
    let mut dl_layer_in = vec![F::ZERO; d];

    let mut dl_dcond_acc = vec![F::ZERO; d];
    let mut dh_sem_carry = [F::ZERO; K];

    for t in (0..seq_len).rev() {
        let bi = bytes[t] as usize;

        for k in 0..d {
            dl_dcond_acc[k] = dl_dcond_acc[k] + dl_dout[t][k];
        }

        if t & (STRIDE - 1) == 0 {
            let si = t / STRIDE;
            let mut dl_dh_slow_extra = vec![F::ZERO; d];

            for j in 0..d {
                let sig = gate_sig[j];
                let mut expand_j = F::ZERO;
                for i in 0..K {
                    expand_j = expand_j
                        + params[spm_off.w + i * d + j]
                        * h_sem_hist[si][i];
                }
                grads[spm_off.gate + j] = grads[spm_off.gate + j]
                    + dl_dcond_acc[j]
                    * expand_j * sig * (F::ONE - sig);
            }

            let dl_gated: Vec<F> = (0..d).map(|j| {
                dl_dcond_acc[j] * gate_sig[j]
            }).collect();

            for i in 0..K {
                let mut dl_hsem = F::ZERO;
                let row = &params[spm_off.w + i * d
                    ..spm_off.w + (i + 1) * d];
                for j in 0..d {
                    dl_hsem = dl_hsem + dl_gated[j] * row[j];
                    grads[spm_off.w + i * d + j] =
                        grads[spm_off.w + i * d + j]
                        + dl_gated[j] * h_sem_hist[si][i];
                }
                dl_hsem = dl_hsem + dh_sem_carry[i];

                let dl_dz = dl_hsem * (F::ONE - lambdas_sem[i]);
                let h_sem_prev = if si > 0 {
                    h_sem_hist[si - 1][i]
                } else {
                    F::ZERO
                };
                let sp_sig = sigmoid(params[spm_off.delta + i]);
                grads[spm_off.delta + i] =
                    grads[spm_off.delta + i]
                    + dl_hsem
                    * (h_sem_prev - z_hist[si][i])
                    * (-lambdas_sem[i]) * sp_sig;
                dh_sem_carry[i] = dl_hsem * lambdas_sem[i];

                for j in 0..d {
                    grads[spm_off.w + i * d + j] =
                        grads[spm_off.w + i * d + j]
                        + dl_dz * h_slow_hist[t][j];
                    dl_dh_slow_extra[j] = dl_dh_slow_extra[j]
                        + dl_dz * params[spm_off.w + i * d + j];
                }
            }

            for j in 0..d {
                dh_slow[j] = dh_slow[j] + dl_dh_slow_extra[j];
            }

            dl_dcond_acc.iter_mut().for_each(|v| *v = F::ZERO);
        }

        for k in 0..d {
            let dl_k = dl_dout[t][k];

            grads[lf.c_out_fast + k] = grads[lf.c_out_fast + k]
                + dl_k * h_fast_hist[t][k];
            grads[lf.c_out_slow + k] = grads[lf.c_out_slow + k]
                + dl_k * h_slow_hist[t][k];
            grads[lf.skip + k] = grads[lf.skip + k]
                + dl_k * states_after_tanh2[t][k];

            let dl_dh_fast_cur =
                dl_k * params[lf.c_out_fast + k] + dh_fast[k];
            let dl_dh_slow_cur =
                dl_k * params[lf.c_out_slow + k] + dh_slow[k];

            dl_dstate[k] = dl_k * params[lf.skip + k];

            grads[lf.b_in_fast + k] = grads[lf.b_in_fast + k]
                + dl_dh_fast_cur * states_after_tanh2[t][k];
            dl_dstate[k] = dl_dstate[k]
                + dl_dh_fast_cur * params[lf.b_in_fast + k];

            let h_fast_prev =
                if t > 0 { h_fast_hist[t - 1][k] } else { F::ZERO };
            let sp_sig_fast =
                sigmoid(params[lf.delta_fast + k]);
            grads[lf.delta_fast + k] = grads[lf.delta_fast + k]
                + dl_dh_fast_cur
                * h_fast_prev * (-lambdas_fast[k]) * sp_sig_fast;

            grads[lf.b_in_slow + k] = grads[lf.b_in_slow + k]
                + dl_dh_slow_cur * states_after_tanh2[t][k];
            dl_dstate[k] = dl_dstate[k]
                + dl_dh_slow_cur * params[lf.b_in_slow + k];

            let h_slow_prev =
                if t > 0 { h_slow_hist[t - 1][k] } else { F::ZERO };
            let sp_sig_slow =
                sigmoid(params[lf.delta_slow + k]);
            grads[lf.delta_slow + k] = grads[lf.delta_slow + k]
                + dl_dh_slow_cur
                * h_slow_prev * (-lambdas_slow[k]) * sp_sig_slow;

            dh_fast[k] = dl_dh_fast_cur * lambdas_fast[k];
            dh_slow[k] = dl_dh_slow_cur * lambdas_slow[k];
        }

        for k in 0..d {
            let y = states_after_tanh2[t][k];
            let dtanh = F::ONE - y * y;
            let dl_dpre = dl_dstate[k] * dtanh;
            grads[lf.s2 + k] = grads[lf.s2 + k]
                + dl_dpre * states_after_wht2[t][k];
            grads[lf.b2 + k] = grads[lf.b2 + k] + dl_dpre;
            dl_dwht2[k] = dl_dpre * params[lf.s2 + k];
        }

        wht_inplace(&mut dl_dwht2);

        for k in 0..d {
            let y = states_after_tanh1[t][k];
            let dtanh = F::ONE - y * y;
            let dl_dpre = dl_dwht2[k] * dtanh;
            grads[lf.s1 + k] = grads[lf.s1 + k]
                + dl_dpre * states_after_wht1[t][k];
            grads[lf.b1 + k] = grads[lf.b1 + k] + dl_dpre;
            dl_dwht1[k] = dl_dpre * params[lf.s1 + k];
        }

        wht_inplace(&mut dl_dwht1);

        let h_fast_for_gate =
            if t > 0 { &h_fast_hist[t - 1] }
            else { &vec![F::ZERO; d] };

        for k in 0..d {
            let gate_in = params[lf.g_gate + bi * d + k]
                + params[lf.w_gate_h + k] * h_fast_for_gate[k];
            let sig = sigmoid(gate_in);
            let dl_dgate_in = dl_dwht1[k]
                * states_after_rn[t][k] * sig * (F::ONE - sig);

            grads[lf.g_gate + bi * d + k] =
                grads[lf.g_gate + bi * d + k] + dl_dgate_in;
            grads[lf.a_bias + bi * d + k] =
                grads[lf.a_bias + bi * d + k] + dl_dwht1[k];
            grads[lf.w_gate_h + k] = grads[lf.w_gate_h + k]
                + dl_dgate_in * h_fast_for_gate[k];

            dh_fast[k] = dh_fast[k]
                + dl_dgate_in * params[lf.w_gate_h + k];

            dl_pre_rn[k] = dl_dwht1[k] * sig;
        }

        backward_rmsnorm(
            &layer_inputs[t],
            &params[lf.rn..lf.rn + d],
            &dl_pre_rn, grads, lf.rn, d,
            &mut dl_layer_in,
        );

        for k in 0..d {
            dl_dout[t][k] = dl_layer_in[k]
                + res_scale * dl_dout[t][k];
        }
    }
}

fn recompute_spm_state<F: Float>(
    params: &[F], d: usize,
    spm_off: &spm::SpmOffsets,
    lambdas_sem: &[F; K],
    h_slow_hist: &[Vec<F>],
    seq_len: usize,
) -> (Vec<[F; K]>, Vec<[F; K]>) {
    let n_steps = (seq_len + STRIDE - 1) / STRIDE;
    let mut h_sem_hist = Vec::with_capacity(n_steps);
    let mut z_hist = Vec::with_capacity(n_steps);
    let mut h_sem = [F::ZERO; K];

    for t in 0..seq_len {
        if t & (STRIDE - 1) == 0 {
            let mut z = [F::ZERO; K];
            for i in 0..K {
                let row = &params[spm_off.w + i * d
                    ..spm_off.w + (i + 1) * d];
                let mut s = F::ZERO;
                for j in 0..d { s = s + row[j] * h_slow_hist[t][j]; }
                z[i] = s;
                h_sem[i] = lambdas_sem[i] * h_sem[i]
                    + (F::ONE - lambdas_sem[i]) * s;
            }
            z_hist.push(z);
            h_sem_hist.push(h_sem);
        }
    }
    (h_sem_hist, z_hist)
}

fn backward_rmsnorm<F: Float>(
    x: &[F], gamma: &[F], dl: &[F],
    grads: &mut [F], gamma_off: usize, d: usize,
    dx: &mut [F],
) {
    let eps = F::from_f64(1e-8);
    let mut ss = F::ZERO;
    for k in 0..d { ss = ss + x[k] * x[k]; }
    let rms = (ss / F::from_usize(d) + eps).sqrt();
    let inv = F::ONE / rms;

    for k in 0..d {
        grads[gamma_off + k] = grads[gamma_off + k]
            + dl[k] * x[k] * inv;
    }

    let mut dot = F::ZERO;
    for k in 0..d {
        dot = dot + dl[k] * gamma[k] * x[k];
    }
    dot = dot * inv * inv * inv / F::from_usize(d);

    for k in 0..d {
        dx[k] = dl[k] * gamma[k] * inv - dot * x[k];
    }
}

fn rmsnorm_apply<F: Float>(x: &mut [F], gamma: &[F], d: usize) {
    let eps = F::from_f64(1e-8);
    let mut ss = F::ZERO;
    for k in 0..d { ss = ss + x[k] * x[k]; }
    let rms = (ss / F::from_usize(d) + eps).sqrt();
    let inv = F::ONE / rms;
    for k in 0..d { x[k] = x[k] * inv * gamma[k]; }
}

#[allow(clippy::type_complexity)]
fn recompute_layer_internals<F: Float>(
    bytes: &[u8], params: &[F], d: usize,
    lf: &FluxLayerOffsets,
    lambdas_fast: &[F], lambdas_slow: &[F],
    seq_len: usize, inputs: &[Vec<F>],
) -> (Vec<Vec<F>>, Vec<Vec<F>>, Vec<Vec<F>>,
      Vec<Vec<F>>, Vec<Vec<F>>,
      Vec<Vec<F>>, Vec<Vec<F>>) {
    let mut after_rn = Vec::with_capacity(seq_len);
    let mut after_wht1 = Vec::with_capacity(seq_len);
    let mut after_tanh1 = Vec::with_capacity(seq_len);
    let mut after_wht2 = Vec::with_capacity(seq_len);
    let mut after_tanh2 = Vec::with_capacity(seq_len);
    let mut h_fast_hist = Vec::with_capacity(seq_len);
    let mut h_slow_hist = Vec::with_capacity(seq_len);

    let mut h_fast = vec![F::ZERO; d];
    let mut h_slow = vec![F::ZERO; d];

    for t in 0..seq_len {
        let bi = bytes[t] as usize;

        let mut state = inputs[t].clone();
        rmsnorm_apply(
            &mut state, &params[lf.rn..lf.rn + d], d,
        );
        after_rn.push(state.clone());

        for k in 0..d {
            let gate_in = params[lf.g_gate + bi * d + k]
                + params[lf.w_gate_h + k] * h_fast[k];
            let gate = sigmoid(gate_in);
            state[k] = gate * state[k]
                + params[lf.a_bias + bi * d + k];
        }

        wht_inplace(&mut state);
        after_wht1.push(state.clone());

        for k in 0..d {
            state[k] = (params[lf.s1 + k] * state[k]
                + params[lf.b1 + k]).tanh();
        }
        after_tanh1.push(state.clone());

        wht_inplace(&mut state);
        after_wht2.push(state.clone());

        for k in 0..d {
            state[k] = (params[lf.s2 + k] * state[k]
                + params[lf.b2 + k]).tanh();
        }
        after_tanh2.push(state.clone());

        for k in 0..d {
            h_fast[k] = lambdas_fast[k] * h_fast[k]
                + params[lf.b_in_fast + k] * state[k];
            h_slow[k] = lambdas_slow[k] * h_slow[k]
                + params[lf.b_in_slow + k] * state[k];
        }
        h_fast_hist.push(h_fast.clone());
        h_slow_hist.push(h_slow.clone());
    }

    (after_rn, after_wht1, after_tanh1,
     after_wht2, after_tanh2,
     h_fast_hist, h_slow_hist)
}
