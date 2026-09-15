/// Walsh-Hadamard Transform (WHT) — normalized, in-place, generic.
/// O(d*log2(d)) with zero parameters. d must be a power of 2.

use crate::float::Float;

pub fn wht_inplace<F: Float>(x: &mut [F]) {
    let d = x.len();
    debug_assert!(d > 0 && d.is_power_of_two());
    let mut half = 1;
    while half < d {
        let mut i = 0;
        while i < d {
            for j in i..i + half {
                let a = x[j];
                let b = x[j + half];
                x[j] = a + b;
                x[j + half] = a - b;
            }
            i += half * 2;
        }
        half *= 2;
    }
    let inv = F::ONE / F::from_usize(d).sqrt();
    for v in x.iter_mut() {
        *v = *v * inv;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn wht_involution_f64() {
        let original: Vec<f64> = vec![1.0, -2.0, 3.0, 0.5, -1.5, 2.5, 0.0, 4.0];
        let mut x = original.clone();
        wht_inplace(&mut x);
        assert!((x[0] - original[0]).abs() > 1e-10);
        wht_inplace(&mut x);
        for (a, b) in x.iter().zip(original.iter()) {
            assert!(
                (a - b).abs() < 1e-12,
                "WHT f64 involution failed: {} != {}", a, b,
            );
        }
    }

    #[test]
    fn wht_involution_f32() {
        let original: Vec<f32> = vec![1.0, -2.0, 3.0, 0.5, -1.5, 2.5, 0.0, 4.0];
        let mut x = original.clone();
        wht_inplace(&mut x);
        assert!((x[0] - original[0]).abs() > 1e-5);
        wht_inplace(&mut x);
        for (a, b) in x.iter().zip(original.iter()) {
            assert!(
                (a - b).abs() < 1e-4,
                "WHT f32 involution failed: {} != {}", a, b,
            );
        }
    }

    #[test]
    fn wht_size_2_f64() {
        let mut x: Vec<f64> = vec![3.0, 1.0];
        wht_inplace(&mut x);
        let s = (2.0_f64).sqrt();
        assert!((x[0] - 4.0 / s).abs() < 1e-12);
        assert!((x[1] - 2.0 / s).abs() < 1e-12);
    }

    #[test]
    fn wht_size_2_f32() {
        let mut x: Vec<f32> = vec![3.0, 1.0];
        wht_inplace(&mut x);
        let s = (2.0_f32).sqrt();
        assert!((x[0] - 4.0 / s).abs() < 1e-5);
        assert!((x[1] - 2.0 / s).abs() < 1e-5);
    }
}
