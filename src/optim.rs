/// Entropic Adam optimizer — generic over Float.

use crate::float::Float;

pub struct EntropicAdam<F: Float> {
    pub m: Vec<F>,
    pub v: Vec<F>,
    pub sign_history: Vec<u32>,
    pub t: usize,
    pub lr: F,
    pub beta1: F,
    pub beta2: F,
    pub eps: F,
    pub t_initial: f64,
    pub t_final: f64,
    pub total_epochs: usize,
    pub group_size: usize,
}

impl<F: Float> EntropicAdam<F> {
    pub fn new(n_params: usize, lr: f64, total_epochs: usize) -> Self {
        let gs = 64;
        let n_groups = (n_params + gs - 1) / gs;
        Self {
            m: vec![F::ZERO; n_params],
            v: vec![F::ZERO; n_params],
            sign_history: vec![0u32; n_groups],
            t: 0,
            lr: F::from_f64(lr),
            beta1: F::from_f64(0.9),
            beta2: F::from_f64(0.999),
            eps: F::from_f64(1e-8),
            t_initial: 2.0,
            t_final: 0.5,
            total_epochs: total_epochs.max(1),
            group_size: gs,
        }
    }

    pub fn step(
        &mut self, params: &mut [F], grads: &[F], epoch: usize,
    ) {
        self.t += 1;
        let t = self.t as f64;
        let b1 = self.beta1.to_f64();
        let b2 = self.beta2.to_f64();
        let bc1 = 1.0 - b1.powf(t);
        let bc2 = 1.0 - b2.powf(t);
        let lr_base = self.lr.to_f64() * bc2.sqrt() / bc1;

        let frac = epoch as f64 / self.total_epochs as f64;
        let t_epoch = self.t_initial
            * (self.t_final / self.t_initial).powf(frac);

        let n = params.len();
        let gs = self.group_size;
        let n_groups = (n + gs - 1) / gs;

        let mut group_lr = vec![lr_base; n_groups];
        for g in 0..n_groups {
            let start = g * gs;
            let end = (start + gs).min(n);

            let mut pos = 0u32;
            for i in start..end {
                if grads[i].to_f64() > 0.0 { pos += 1; }
            }
            let majority = if pos * 2 >= (end - start) as u32 { 1u32 }
                else { 0u32 };

            self.sign_history[g] =
                (self.sign_history[g] << 1) | majority;

            let bits = self.sign_history[g].count_ones();
            let p = bits as f64 / 32.0;
            let h = if p < 1e-10 || p > 1.0 - 1e-10 {
                0.0
            } else {
                -(p * p.ln() + (1.0 - p) * (1.0 - p).ln())
            };

            group_lr[g] = lr_base * (-h / t_epoch).exp();
        }

        let one_m_b1 = F::ONE - self.beta1;
        let one_m_b2 = F::ONE - self.beta2;
        for i in 0..n {
            self.m[i] = self.beta1 * self.m[i]
                + one_m_b1 * grads[i];
            self.v[i] = self.beta2 * self.v[i]
                + one_m_b2 * grads[i] * grads[i];
            let g = i / gs;
            let glr = F::from_f64(group_lr[g]);
            params[i] = params[i] - glr * self.m[i]
                / (self.v[i].sqrt() + self.eps);
        }
    }
}
