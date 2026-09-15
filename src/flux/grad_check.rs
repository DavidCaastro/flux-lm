/// Gradient checking for Flux v3: tests both f64 and f32.

#[cfg(test)]
mod tests {
    use crate::rng::Rng;
    use crate::float::Float;
    use crate::flux::FluxModel;
    use crate::flux::forward::flux_forward_pass;
    use crate::flux::backward::flux_backward_pass;

    fn numerical_grad<F: Float>(
        corpus: &[u8], start: usize, sl: usize,
        params: &mut [F], d: usize, nl: usize, idx: usize,
        eps: F,
    ) -> F {
        let orig = params[idx];

        params[idx] = orig + eps;
        let mut inter = Vec::new();
        let mut ckpts = Vec::new();
        let loss_p = flux_forward_pass(
            corpus, start, sl, params, d, nl, &mut inter, &mut ckpts,
        );

        params[idx] = orig - eps;
        inter.clear();
        ckpts.clear();
        let loss_m = flux_forward_pass(
            corpus, start, sl, params, d, nl, &mut inter, &mut ckpts,
        );

        params[idx] = orig;
        (loss_p - loss_m) / (eps + eps)
    }

    fn check_grad_subset<F: Float>(
        corpus: &[u8], start: usize, sl: usize,
        params: &mut [F], d: usize, nl: usize,
        analytic: &[F], indices: &[usize], label: &str,
        eps: F, rel_tol: f64, abs_tol: f64,
    ) {
        let mut max_rel = 0.0_f64;
        let mut worst_idx = 0;
        let mut n_checked = 0;
        let mut n_bad = 0;

        for &idx in indices {
            let ng = numerical_grad(
                corpus, start, sl, params, d, nl, idx, eps,
            );
            let ag = analytic[idx];
            let diff = (ag - ng).abs().to_f64();
            let denom = ag.abs().to_f64()
                .max(ng.abs().to_f64())
                .max(1e-8);
            let rel = diff / denom;

            if rel > max_rel {
                max_rel = rel;
                worst_idx = idx;
            }
            n_checked += 1;
            if rel > rel_tol && diff > abs_tol {
                n_bad += 1;
                if n_bad <= 5 {
                    eprintln!(
                        "  FAIL [{}/{}] idx={} a={:.6e} n={:.6e} \
                         rel={:.4e}",
                        label, n_bad, idx, ag.to_f64(), ng.to_f64(), rel,
                    );
                }
            }
        }

        eprintln!(
            "  {} checked={} bad={} max_rel={:.4e} (idx={})",
            label, n_checked, n_bad, max_rel, worst_idx,
        );
        assert!(
            n_bad == 0,
            "{}: {}/{} params failed gradient check",
            label, n_bad, n_checked,
        );
    }

    fn param_range(start: usize, len: usize, stride: usize)
        -> Vec<usize>
    {
        (start..start + len).step_by(stride.max(1)).collect()
    }

    fn run_full_check<F: Float>(
        d: usize, nl: usize, sl: usize, seed: u64,
        eps: F, rel_tol: f64, abs_tol: f64, label: &str,
    ) {
        let mut rng = Rng::new(seed);
        let model = FluxModel::<F>::new(d, nl, &mut rng);
        let mut params = model.params.clone();
        let np = params.len();

        let corpus: Vec<u8> =
            (0..sl + 1).map(|i| (i * 37 + 11) as u8).collect();

        let mut inter = Vec::new();
        let mut ckpts = Vec::new();
        let _loss = flux_forward_pass(
            &corpus, 0, sl, &params, d, nl, &mut inter, &mut ckpts,
        );
        let mut grads = vec![F::ZERO; np];
        flux_backward_pass(
            &corpus, 0, sl, &params, d, nl, &inter, &ckpts,
            &mut grads,
        );

        let all: Vec<usize> = (0..np).collect();
        check_grad_subset(
            &corpus, 0, sl, &mut params, d, nl,
            &grads, &all, &format!("{}_all_params", label),
            eps, rel_tol, abs_tol,
        );
    }

    fn run_component_check<F: Float>(
        d: usize, nl: usize, sl: usize, seed: u64,
        eps: F, rel_tol: f64, abs_tol: f64, label: &str,
    ) {
        let mut rng = Rng::new(seed);
        let model = FluxModel::<F>::new(d, nl, &mut rng);
        let mut params = model.params.clone();
        let np = params.len();

        let corpus: Vec<u8> =
            (0..sl + 1).map(|i| ((i * 53 + 7) % 256) as u8).collect();

        let mut inter = Vec::new();
        let mut ckpts = Vec::new();
        let _loss = flux_forward_pass(
            &corpus, 0, sl, &params, d, nl, &mut inter, &mut ckpts,
        );
        let mut grads = vec![F::ZERO; np];
        flux_backward_pass(
            &corpus, 0, sl, &params, d, nl, &inter, &ckpts,
            &mut grads,
        );

        let emb = param_range(0, 256 * d, 31);
        check_grad_subset(
            &corpus, 0, sl, &mut params, d, nl,
            &grads, &emb, &format!("{}_embedding", label),
            eps, rel_tol, abs_tol,
        );

        for li in 0..nl {
            let lo = crate::flux::flux_layer_offset(d, li);
            let mut o = lo;
            let rn = param_range(o, d, 1); o += d;
            check_grad_subset(
                &corpus, 0, sl, &mut params, d, nl,
                &grads, &rn, &format!("{}_L{}_rmsnorm", label, li),
                eps, rel_tol, abs_tol,
            );

            let g = param_range(o, 256 * d, 47); o += 256 * d;
            check_grad_subset(
                &corpus, 0, sl, &mut params, d, nl,
                &grads, &g, &format!("{}_L{}_gate", label, li),
                eps, rel_tol, abs_tol,
            );

            let a = param_range(o, 256 * d, 47); o += 256 * d;
            check_grad_subset(
                &corpus, 0, sl, &mut params, d, nl,
                &grads, &a, &format!("{}_L{}_abias", label, li),
                eps, rel_tol, abs_tol,
            );

            let wgh = param_range(o, d, 1); o += d;
            check_grad_subset(
                &corpus, 0, sl, &mut params, d, nl,
                &grads, &wgh, &format!("{}_L{}_w_gate_h", label, li),
                eps, rel_tol, abs_tol,
            );

            let s1 = param_range(o, d, 1); o += d;
            check_grad_subset(
                &corpus, 0, sl, &mut params, d, nl,
                &grads, &s1, &format!("{}_L{}_s1", label, li),
                eps, rel_tol, abs_tol,
            );

            let b1 = param_range(o, d, 1); o += d;
            check_grad_subset(
                &corpus, 0, sl, &mut params, d, nl,
                &grads, &b1, &format!("{}_L{}_b1", label, li),
                eps, rel_tol, abs_tol,
            );

            let s2 = param_range(o, d, 1); o += d;
            check_grad_subset(
                &corpus, 0, sl, &mut params, d, nl,
                &grads, &s2, &format!("{}_L{}_s2", label, li),
                eps, rel_tol, abs_tol,
            );

            let b2 = param_range(o, d, 1); o += d;
            check_grad_subset(
                &corpus, 0, sl, &mut params, d, nl,
                &grads, &b2, &format!("{}_L{}_b2", label, li),
                eps, rel_tol, abs_tol,
            );

            let df = param_range(o, d, 1); o += d;
            check_grad_subset(
                &corpus, 0, sl, &mut params, d, nl,
                &grads, &df, &format!("{}_L{}_delta_fast", label, li),
                eps, rel_tol, abs_tol,
            );

            let bf = param_range(o, d, 1); o += d;
            check_grad_subset(
                &corpus, 0, sl, &mut params, d, nl,
                &grads, &bf, &format!("{}_L{}_b_in_fast", label, li),
                eps, rel_tol, abs_tol,
            );

            let cf = param_range(o, d, 1); o += d;
            check_grad_subset(
                &corpus, 0, sl, &mut params, d, nl,
                &grads, &cf, &format!("{}_L{}_c_out_fast", label, li),
                eps, rel_tol, abs_tol,
            );

            let skip = param_range(o, d, 1); o += d;
            check_grad_subset(
                &corpus, 0, sl, &mut params, d, nl,
                &grads, &skip, &format!("{}_L{}_skip", label, li),
                eps, rel_tol, abs_tol,
            );

            let ds = param_range(o, d, 1); o += d;
            check_grad_subset(
                &corpus, 0, sl, &mut params, d, nl,
                &grads, &ds, &format!("{}_L{}_delta_slow", label, li),
                eps, rel_tol, abs_tol,
            );

            let bs = param_range(o, d, 1); o += d;
            check_grad_subset(
                &corpus, 0, sl, &mut params, d, nl,
                &grads, &bs, &format!("{}_L{}_b_in_slow", label, li),
                eps, rel_tol, abs_tol,
            );

            let cs = param_range(o, d, 1);
            check_grad_subset(
                &corpus, 0, sl, &mut params, d, nl,
                &grads, &cs, &format!("{}_L{}_c_out_slow", label, li),
                eps, rel_tol, abs_tol,
            );
        }

        let out_base = crate::flux::flux_layer_offset(d, nl);
        let wout = param_range(out_base, 256 * d, 31);
        check_grad_subset(
            &corpus, 0, sl, &mut params, d, nl,
            &grads, &wout, &format!("{}_w_out", label),
            eps, rel_tol, abs_tol,
        );

        let bout = param_range(out_base + 256 * d, 256, 7);
        check_grad_subset(
            &corpus, 0, sl, &mut params, d, nl,
            &grads, &bout, &format!("{}_b_out", label),
            eps, rel_tol, abs_tol,
        );

        use crate::flux::spm;
        let spm_base =
            crate::flux::flux_pre_spm_param_count(d, nl);
        let wc = param_range(spm_base, spm::K * d, 1);
        check_grad_subset(
            &corpus, 0, sl, &mut params, d, nl,
            &grads, &wc, &format!("{}_spm_w_compress", label),
            eps, rel_tol, abs_tol,
        );

        for li in 0..nl {
            let soff = spm::spm_offsets(spm_base, d, li);
            let ds: Vec<usize> =
                (soff.delta..soff.delta + spm::K).collect();
            check_grad_subset(
                &corpus, 0, sl, &mut params, d, nl,
                &grads, &ds, &format!("{}_L{}_spm_delta", label, li),
                eps, rel_tol, abs_tol,
            );
            let gs = param_range(soff.gate, d, 1);
            check_grad_subset(
                &corpus, 0, sl, &mut params, d, nl,
                &grads, &gs, &format!("{}_L{}_spm_gate", label, li),
                eps, rel_tol, abs_tol,
            );
        }
    }

    fn run_temporal_check<F: Float>(
        d: usize, nl: usize, sl: usize, seed: u64,
        eps: F, rel_tol: f64, abs_tol: f64, label: &str,
    ) {
        let mut rng = Rng::new(seed);
        let model = FluxModel::<F>::new(d, nl, &mut rng);
        let mut params = model.params.clone();
        let np = params.len();

        let corpus: Vec<u8> =
            (0..sl + 1).map(|i| ((i * 17 + 3) % 256) as u8).collect();

        let mut inter = Vec::new();
        let mut ckpts = Vec::new();
        let _loss = flux_forward_pass(
            &corpus, 0, sl, &params, d, nl, &mut inter, &mut ckpts,
        );
        let mut grads = vec![F::ZERO; np];
        flux_backward_pass(
            &corpus, 0, sl, &params, d, nl, &inter, &ckpts,
            &mut grads,
        );

        let lo = crate::flux::flux_layer_offset(d, 0);
        let wgh_off = lo + d + 2 * 256 * d;
        let wgh: Vec<usize> = (wgh_off..wgh_off + d).collect();
        check_grad_subset(
            &corpus, 0, sl, &mut params, d, nl,
            &grads, &wgh, &format!("{}_gate_h_bptt", label),
            eps, rel_tol, abs_tol,
        );

        let mem_off = lo + d + 2 * 256 * d + d + 4 * d;
        let mem_params: Vec<usize> =
            (mem_off..mem_off + 7 * d).collect();
        check_grad_subset(
            &corpus, 0, sl, &mut params, d, nl,
            &grads, &mem_params, &format!("{}_memory_bptt_long", label),
            eps, rel_tol, abs_tol,
        );

        use crate::flux::spm;
        let spm_base =
            crate::flux::flux_pre_spm_param_count(d, nl);
        let soff = spm::spm_offsets(spm_base, d, 0);
        let spm_delta: Vec<usize> =
            (soff.delta..soff.delta + spm::K).collect();
        check_grad_subset(
            &corpus, 0, sl, &mut params, d, nl,
            &grads, &spm_delta, &format!("{}_spm_delta_bptt", label),
            eps, rel_tol, abs_tol,
        );
        let spm_w: Vec<usize> =
            (spm_base..spm_base + spm::K * d).collect();
        check_grad_subset(
            &corpus, 0, sl, &mut params, d, nl,
            &grads, &spm_w, &format!("{}_spm_w_bptt", label),
            eps, rel_tol, abs_tol,
        );
    }

    // ============ f64 tests ============

    #[test]
    fn flux_gradient_check_full_f64() {
        run_full_check::<f64>(
            4, 1, 4, 12345,
            1e-5, 1e-3, 1e-6, "f64",
        );
    }

    #[test]
    fn flux_gradient_check_by_component_f64() {
        run_component_check::<f64>(
            8, 2, 6, 99887,
            1e-5, 1e-3, 1e-6, "f64",
        );
    }

    #[test]
    fn flux_gradient_check_temporal_f64() {
        run_temporal_check::<f64>(
            4, 1, 12, 55555,
            1e-5, 1e-3, 1e-6, "f64",
        );
    }

    // ============ f32 tests ============

    #[test]
    fn flux_gradient_check_full_f32() {
        run_full_check::<f32>(
            4, 1, 4, 12345,
            1e-3, 5e-2, 1e-3, "f32",
        );
    }

    #[test]
    fn flux_gradient_check_by_component_f32() {
        run_component_check::<f32>(
            8, 2, 6, 99887,
            1e-3, 5e-2, 1e-3, "f32",
        );
    }

    #[test]
    fn flux_gradient_check_temporal_f32() {
        run_temporal_check::<f32>(
            4, 1, 12, 55555,
            1e-3, 5e-2, 1e-3, "f32",
        );
    }
}
