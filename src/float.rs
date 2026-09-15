/// Trait Float — zero-cost generic over f32/f64 via monomorphization.

use std::iter::Sum;
use std::ops::{Add, Sub, Mul, Div, Neg, AddAssign, SubAssign, MulAssign, DivAssign};

pub trait Float:
    Copy + Clone + Send + Sync + 'static + Default
    + std::fmt::Debug + std::fmt::Display
    + PartialEq + PartialOrd
    + Add<Output = Self> + Sub<Output = Self>
    + Mul<Output = Self> + Div<Output = Self>
    + Neg<Output = Self>
    + AddAssign + SubAssign + MulAssign + DivAssign
    + Sum
{
    const ZERO: Self;
    const ONE: Self;
    const NEG_INFINITY: Self;
    const BYTE_SIZE: usize;

    fn from_f64(v: f64) -> Self;
    fn to_f64(self) -> f64;
    fn from_usize(v: usize) -> Self;

    fn exp(self) -> Self;
    fn ln(self) -> Self;
    fn sqrt(self) -> Self;
    fn tanh(self) -> Self;
    fn abs(self) -> Self;
    fn cos(self) -> Self;
    fn powf(self, n: Self) -> Self;
    fn max(self, other: Self) -> Self;
    fn min(self, other: Self) -> Self;

    fn write_le<W: std::io::Write>(self, w: &mut W) -> std::io::Result<()>;
    fn read_le<R: std::io::Read>(r: &mut R) -> std::io::Result<Self>;
}

impl Float for f64 {
    const ZERO: f64 = 0.0;
    const ONE: f64 = 1.0;
    const NEG_INFINITY: f64 = f64::NEG_INFINITY;
    const BYTE_SIZE: usize = 8;

    #[inline] fn from_f64(v: f64) -> Self { v }
    #[inline] fn to_f64(self) -> f64 { self }
    #[inline] fn from_usize(v: usize) -> Self { v as f64 }

    #[inline] fn exp(self) -> Self { f64::exp(self) }
    #[inline] fn ln(self) -> Self { f64::ln(self) }
    #[inline] fn sqrt(self) -> Self { f64::sqrt(self) }
    #[inline] fn tanh(self) -> Self { f64::tanh(self) }
    #[inline] fn abs(self) -> Self { f64::abs(self) }
    #[inline] fn cos(self) -> Self { f64::cos(self) }
    #[inline] fn powf(self, n: Self) -> Self { f64::powf(self, n) }
    #[inline] fn max(self, other: Self) -> Self { f64::max(self, other) }
    #[inline] fn min(self, other: Self) -> Self { f64::min(self, other) }

    #[inline]
    fn write_le<W: std::io::Write>(self, w: &mut W) -> std::io::Result<()> {
        w.write_all(&self.to_le_bytes())
    }
    #[inline]
    fn read_le<R: std::io::Read>(r: &mut R) -> std::io::Result<Self> {
        let mut buf = [0u8; 8];
        r.read_exact(&mut buf)?;
        Ok(f64::from_le_bytes(buf))
    }
}

impl Float for f32 {
    const ZERO: f32 = 0.0;
    const ONE: f32 = 1.0;
    const NEG_INFINITY: f32 = f32::NEG_INFINITY;
    const BYTE_SIZE: usize = 4;

    #[inline] fn from_f64(v: f64) -> Self { v as f32 }
    #[inline] fn to_f64(self) -> f64 { self as f64 }
    #[inline] fn from_usize(v: usize) -> Self { v as f32 }

    #[inline] fn exp(self) -> Self { f32::exp(self) }
    #[inline] fn ln(self) -> Self { f32::ln(self) }
    #[inline] fn sqrt(self) -> Self { f32::sqrt(self) }
    #[inline] fn tanh(self) -> Self { f32::tanh(self) }
    #[inline] fn abs(self) -> Self { f32::abs(self) }
    #[inline] fn cos(self) -> Self { f32::cos(self) }
    #[inline] fn powf(self, n: Self) -> Self { f32::powf(self, n) }
    #[inline] fn max(self, other: Self) -> Self { f32::max(self, other) }
    #[inline] fn min(self, other: Self) -> Self { f32::min(self, other) }

    #[inline]
    fn write_le<W: std::io::Write>(self, w: &mut W) -> std::io::Result<()> {
        w.write_all(&self.to_le_bytes())
    }
    #[inline]
    fn read_le<R: std::io::Read>(r: &mut R) -> std::io::Result<Self> {
        let mut buf = [0u8; 4];
        r.read_exact(&mut buf)?;
        Ok(f32::from_le_bytes(buf))
    }
}
