#![cfg_attr(feature = "simd", feature(portable_simd))]

mod matrix;
mod utils;
mod vector;

pub use matrix::*;
pub use utils::Element;
pub use vector::*;

#[cfg(feature = "simd")]
pub mod simd;
