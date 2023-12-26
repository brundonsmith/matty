#![cfg_attr(feature = "simd", feature(portable_simd))]

mod matrix;
mod vector;

pub use matrix::*;
pub use vector::*;

#[cfg(feature = "simd")]
pub mod simd;
