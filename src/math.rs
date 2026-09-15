/// Shared math utilities.

use crate::float::Float;

pub fn softmax_inplace<F: Float>(v: &mut [F]) {
    let n = v.len();
    let mut mx = F::NEG_INFINITY;
    for i in 0..n {
        if v[i] > mx { mx = v[i]; }
    }
    let mut z = F::ZERO;
    for i in 0..n {
        let e = (v[i] - mx).exp();
        v[i] = e;
        z = z + e;
    }
    let inv = if z.to_f64() < 1e-30 {
        F::ONE / F::from_usize(n)
    } else {
        F::ONE / z
    };
    for i in 0..n {
        v[i] = v[i] * inv;
    }
}
