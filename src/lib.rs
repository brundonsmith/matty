#![cfg_attr(feature = "simd", feature(portable_simd))]

mod matrix;
mod utils;
mod vector;

#[cfg(feature = "simd")]
mod vector_simd;

pub use matrix::*;
pub use utils::Element;
pub use vector::*;

#[cfg(feature = "simd")]
pub use vector_simd::*;
