use crate::float::Float;

pub struct Rng {
    state: u64,
}

impl Rng {
    pub fn new(seed: u64) -> Self {
        let s = if seed == 0 { 0xDEAD_BEEF_CAFE_1234 } else { seed };
        Self { state: s }
    }

    pub fn next_u64(&mut self) -> u64 {
        let mut x = self.state;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.state = x;
        x
    }

    pub fn next_f64(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 / ((1u64 << 53) as f64)
    }

    pub fn gaussian(&mut self) -> f64 {
        loop {
            let u1 = self.next_f64();
            let u2 = self.next_f64();
            if u1 < 1e-30 {
                continue;
            }
            let r = (-2.0 * u1.ln()).sqrt();
            let theta = 2.0 * std::f64::consts::PI * u2;
            return r * theta.cos();
        }
    }

    pub fn gaussian_scaled(&mut self, std: f64) -> f64 {
        self.gaussian() * std
    }

    pub fn gaussian_float<F: Float>(&mut self) -> F {
        F::from_f64(self.gaussian())
    }

    pub fn gaussian_scaled_float<F: Float>(&mut self, std: F) -> F {
        F::from_f64(self.gaussian()) * std
    }

    pub fn random(&mut self) -> f64 {
        self.next_f64()
    }
}
