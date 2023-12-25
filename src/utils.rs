use std::{fmt::Display, iter::Sum};

use num_traits::{FromPrimitive, Num, NumAssign};

#[cfg(not(feature = "simd"))]
pub trait Element:
    Copy + Display + Default + PartialEq + Num + FromPrimitive + Sum + NumAssign
{
}

#[cfg(not(feature = "simd"))]
impl<T: Copy + Display + Default + PartialEq + Num + FromPrimitive + Sum + NumAssign> Element
    for T
{
}

#[cfg(feature = "simd")]
pub trait Element:
    Copy
    + Display
    + Default
    + PartialEq
    + Num
    + FromPrimitive
    + Sum
    + NumAssign
    + std::simd::SimdElement
{
}

#[cfg(feature = "simd")]
impl<
        T: Copy
            + Display
            + Default
            + PartialEq
            + Num
            + FromPrimitive
            + Sum
            + NumAssign
            + std::simd::SimdElement,
    > Element for T
{
}
