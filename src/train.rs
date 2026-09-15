/// Training loop for Flux v3 — generic over Float.

use std::thread;
use std::time::Instant;

use crate::float::Float;
use crate::checkpoint;
use crate::config::Config;
use crate::hw;
use crate::logger::Logger;
use crate::flux::{FluxModel, flux_layer_offset, flux_pre_spm_param_count};
use crate::flux::forward::{flux_forward_pass, flux_layer_offsets};
use crate::flux::spm::{self, K};
use crate::flux::backward::flux_backward_pass;
use crate::optim::EntropicAdam;
use crate::hw::check_stop;

const LN2: f64 = 0.6931471805599453;
const PI: f64 = std::f64::consts::PI;

pub fn train_flux<F: Float>(
    model: &mut FluxModel<F>, optim: &mut EntropicAdam<F>,
    corpus: &[u8], cfg: &Config, logger: &Logger,
    start_epoch: u32,
) {
    let d = model.d;
    let nl = model.n_layers;
    let np = model.n_params();
    let sl = cfg.seq_len;
    let total_epochs = cfg.epochs as u32;
    let wd = F::from_f64(1e-5);
    let one_m_wd = F::ONE - wd;

    let split = (corpus.len() as f64 * 0.9) as usize;
    let split = split.max(sl + 1).min(corpus.len() - sl - 1);
    let train_corpus = &corpus[..split];
    let test_corpus = &corpus[split..];
    let has_test = test_corpus.len() > sl + 1;

    let cmd_line: String = std::env::args()
        .collect::<Vec<_>>().join(" ");
    logger.log(&format!("cmd: {}", cmd_line));
    let prec = if F::BYTE_SIZE == 4 { "f32" } else { "f64" };
    logger.log(&format!(
        "Flux v3 [{}]: d={}, layers={}, params={}", prec, d, nl, np,
    ));
    let n_threads = hw::core_budget(cfg.core_fraction);
    logger.log(&format!(
        "Corpus: {} bytes (train={}, test={}), seq_len={}",
        corpus.len(), train_corpus.len(),
        test_corpus.len(), sl,
    ));
    logger.log(&format!(
        "Threads: {} (core_fraction={:.2})", n_threads, cfg.core_fraction,
    ));

    let t_0: u32 = (total_epochs / 5).max(50);
    let t_train_start = Instant::now();

    for epoch in (start_epoch + 1)..=total_epochs {
        if check_stop() {
            logger.log("Stop file detected.");
            return;
        }

        let t_epoch = Instant::now();
        let (loss, grads) = train_epoch_flux(
            train_corpus, &model.params, d, nl, sl, np,
            cfg.core_fraction,
        );
        let epoch_ms = t_epoch.elapsed().as_millis();
        let bpb = loss.to_f64() / LN2;

        let (grads, grad_norm) = clip_gradients(grads, F::from_f64(5.0));

        let warmup_frac = 0.05;
        let warmup_factor = (epoch as f64
            / (warmup_frac * total_epochs as f64)).min(1.0);
        let cosine_factor = warm_restart_cosine(
            epoch, start_epoch, t_0,
        );
        let lr_scale = warmup_factor * cosine_factor;
        let original_lr = optim.lr;
        optim.lr = F::from_f64(original_lr.to_f64() * lr_scale);

        optim.step(&mut model.params, &grads, epoch as usize);

        for i in 0..np {
            model.params[i] = model.params[i] * one_m_wd;
        }

        optim.lr = original_lr;

        let do_print = epoch % cfg.print_every as u32 == 0
            || epoch == start_epoch + 1;

        if do_print {
            let elapsed = t_train_start.elapsed().as_secs();
            let h = elapsed / 3600;
            let m = (elapsed % 3600) / 60;
            let s = elapsed % 60;
            if has_test {
                let test_loss = eval_loss(
                    test_corpus, &model.params, d, nl, sl,
                );
                let test_bpb = test_loss.to_f64() / LN2;
                logger.log(&format!(
                    "epoch {:4}  train_bpb={:.3}  test_bpb={:.3}  \
                     lr_s={:.4}  grad={:.1}  {:.0}ms  [{:02}:{:02}:{:02}]",
                    epoch, bpb, test_bpb, lr_scale,
                    grad_norm.to_f64(),
                    epoch_ms, h, m, s,
                ));
            } else {
                logger.log(&format!(
                    "epoch {:4}  bpb={:.3}  lr_s={:.4}  \
                     grad={:.1}  {:.0}ms  [{:02}:{:02}:{:02}]",
                    epoch, bpb, lr_scale,
                    grad_norm.to_f64(),
                    epoch_ms, h, m, s,
                ));
            }
        }

        let diag_every = (cfg.print_every as u32) * 10;
        if diag_every > 0 && epoch % diag_every == 0 {
            log_diagnostics(
                &model.params, d, nl, epoch,
                grad_norm.to_f64(), logger,
            );
        }

        save_if_needed(
            cfg, model, optim, epoch, loss.to_f64(), logger,
        );
    }
}

fn warm_restart_cosine(
    epoch: u32, start: u32, t_0: u32,
) -> f64 {
    let e = epoch.saturating_sub(start);
    let mut t_cur = t_0;
    let mut consumed = 0u32;
    loop {
        if consumed + t_cur >= e {
            let progress = (e - consumed) as f64 / t_cur as f64;
            return 0.5 * (1.0 + (PI * progress).cos());
        }
        consumed += t_cur;
        t_cur = t_cur.saturating_mul(2);
        if t_cur == 0 { return 0.5; }
    }
}

fn eval_loss<F: Float>(
    corpus: &[u8], params: &[F],
    d: usize, nl: usize, sl: usize,
) -> F {
    let n_chunks = (corpus.len() - 1) / sl;
    if n_chunks == 0 { return F::ZERO; }
    let mut total_loss = F::ZERO;
    let mut count = 0usize;
    let mut inter = Vec::new();
    let mut ckpts = Vec::new();
    for ci in 0..n_chunks {
        let start = ci * sl;
        if start + sl >= corpus.len() { break; }
        inter.clear();
        ckpts.clear();
        let loss = flux_forward_pass(
            corpus, start, sl, params, d, nl, &mut inter, &mut ckpts,
        );
        total_loss = total_loss + loss;
        count += 1;
    }
    if count > 0 {
        total_loss / F::from_usize(count)
    } else {
        F::ZERO
    }
}

fn train_epoch_flux<F: Float>(
    corpus: &[u8], params: &[F],
    d: usize, nl: usize, sl: usize, np: usize,
    core_fraction: f64,
) -> (F, Vec<F>) {
    let n_chunks = (corpus.len() - 1) / sl;
    if n_chunks == 0 {
        return (F::ZERO, vec![F::ZERO; np]);
    }

    let n_threads = hw::core_budget(core_fraction);
    let chunks_per_thread =
        (n_chunks + n_threads - 1) / n_threads;

    let mut total_g = vec![F::ZERO; np];
    let mut total_loss = F::ZERO;
    let mut total_count = 0usize;

    thread::scope(|s| {
        let mut handles = Vec::with_capacity(n_threads);
        for tid in 0..n_threads {
            let c_start = tid * chunks_per_thread;
            let c_end =
                ((tid + 1) * chunks_per_thread).min(n_chunks);
            if c_start >= c_end { break; }

            handles.push(s.spawn(move || {
                let mut local_g = vec![F::ZERO; np];
                let mut chunk_g = vec![F::ZERO; np];
                let mut inter = Vec::new();
                let mut ckpts = Vec::new();
                let mut local_loss = F::ZERO;
                let mut count = 0usize;
                for ci in c_start..c_end {
                    let start = ci * sl;
                    if start + sl >= corpus.len() { break; }
                    inter.clear();
                    ckpts.clear();
                    let loss = flux_forward_pass(
                        corpus, start, sl, params,
                        d, nl, &mut inter, &mut ckpts,
                    );
                    flux_backward_pass(
                        corpus, start, sl, params,
                        d, nl, &inter, &ckpts, &mut chunk_g,
                    );
                    local_loss = local_loss + loss;
                    count += 1;
                    for i in 0..np {
                        local_g[i] = local_g[i] + chunk_g[i];
                    }
                }
                (local_g, local_loss, count)
            }));
        }

        for h in handles {
            let (lg, ll, c) = h.join().unwrap();
            total_loss = total_loss + ll;
            total_count += c;
            for i in 0..np { total_g[i] = total_g[i] + lg[i]; }
        }
    });

    if total_count > 0 {
        let inv = F::ONE / F::from_usize(total_count);
        for g in total_g.iter_mut() { *g = *g * inv; }
        total_loss = total_loss * inv;
    }
    (total_loss, total_g)
}

fn save_if_needed<F: Float>(
    cfg: &Config, model: &FluxModel<F>,
    optim: &EntropicAdam<F>, epoch: u32, loss: f64,
    logger: &Logger,
) {
    let path = match &cfg.ckpt_path {
        Some(p) => p,
        None => return,
    };
    let total = cfg.epochs as u32;
    let every = cfg.ckpt_every as u32;
    if every == 0 { return; }
    if epoch % every != 0 && epoch != total { return; }

    match checkpoint::save_flux(
        model.d, model.n_layers,
        &model.params, optim,
        epoch, loss, path,
    ) {
        Ok(()) => {
            logger.log(&format!(
                "  ckpt saved: {} (epoch {})", path, epoch,
            ));
        }
        Err(e) => {
            logger.log(&format!(
                "  ckpt save error: {}", e,
            ));
        }
    }
}

fn log_diagnostics<F: Float>(
    params: &[F], d: usize, nl: usize,
    epoch: u32, grad_norm: f64, logger: &Logger,
) {
    let clip_note = if grad_norm > 5.0 { "->5.0" } else { "" };
    let mut msg = format!(
        "[diag] epoch {}  grad={:.1}{}", epoch, grad_norm, clip_note,
    );
    for li in 0..nl {
        let lo = flux_layer_offset(d, li);
        let lf = flux_layer_offsets(lo, d);
        let stats = |base: usize| -> (f64, f64) {
            let vals: Vec<f64> =
                (0..d).map(|k| params[base + k].to_f64()).collect();
            let mean = vals.iter().sum::<f64>() / d as f64;
            let var = vals.iter()
                .map(|v| (v - mean) * (v - mean))
                .sum::<f64>() / d as f64;
            (mean, var.sqrt())
        };
        let (sm, ss) = stats(lf.skip);
        let (cfm, cfs) = stats(lf.c_out_fast);
        let (csm, css) = stats(lf.c_out_slow);
        let (dfm, dfs) = stats(lf.delta_fast);
        let (dsm, dss) = stats(lf.delta_slow);
        let (wm, ws) = stats(lf.w_gate_h);
        let spm_base = flux_pre_spm_param_count(d, nl);
        let spm_off = spm::spm_offsets(spm_base, d, li);
        let gate_mean: f64 = (0..d).map(|j| {
            let x = params[spm_off.gate + j].to_f64();
            1.0 / (1.0 + (-x).exp())
        }).sum::<f64>() / d as f64;
        let delta_sem: Vec<f64> = (0..K).map(|i| {
            params[spm_off.delta + i].to_f64()
        }).collect();

        msg.push_str(&format!(
            "\n  L{}  skip={:.2}+/-{:.2}  c_f={:.3}+/-{:.3}  \
             c_s={:.3}+/-{:.3}  d_f={:.2}+/-{:.2}  \
             d_s={:.2}+/-{:.2}  wgh={:.2}+/-{:.2}  \
             spm_g={:.4}  d_sem=[{:.2},{:.2},{:.2},{:.2}]",
            li, sm, ss, cfm, cfs, csm, css,
            dfm, dfs, dsm, dss, wm, ws,
            gate_mean,
            delta_sem[0], delta_sem[1], delta_sem[2], delta_sem[3],
        ));
    }
    logger.log(&msg);
}

fn clip_gradients<F: Float>(
    mut grads: Vec<F>, max_norm: F,
) -> (Vec<F>, F) {
    let mut norm_sq = F::ZERO;
    for &g in &grads { norm_sq = norm_sq + g * g; }
    let norm = norm_sq.sqrt();
    if norm > max_norm {
        let scale = max_norm / norm;
        for g in grads.iter_mut() { *g = *g * scale; }
    }
    (grads, norm)
}
