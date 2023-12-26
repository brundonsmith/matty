use std::{
    iter::Sum,
    simd::{LaneCount, Simd, SimdElement, SupportedLaneCount},
};

use num_traits::real::Real;

use crate::{RealVector, Vector, Vector2, Vector3, VectorAdd, VectorDiv, VectorMul, VectorSub};

impl<
        T: Copy
            + Default
            + std::ops::Mul<Output = T>
            + std::ops::Sub<Output = T>
            + Sum<T>
            + SimdElement,
        const N: usize,
    > Vector<T, N> for Simd<T, N>
where
    LaneCount<N>: SupportedLaneCount,
    Simd<T, N>: std::ops::Mul<Output = Simd<T, N>>,
{
    #[inline]
    fn dot(self, other: Self) -> T {
        self.mul(other).to_array().into_iter().sum()
    }

    #[inline]
    fn cross(self, other: Self) -> Self {
        let mut res = [T::default(); N];

        for i in 0..N {
            res[i] =
                self[(i + 1) % N] * other[(i + 2) % N] - self[(i + 2) % N] * other[(i + 1) % N];
        }

        res.into()
    }
}

impl<T: SimdElement> Vector2<T> for Simd<T, 2> {
    #[inline]
    fn new(x: T, y: T) -> Self {
        Simd::from([x, y])
    }

    #[inline]
    fn x(&self) -> T {
        self[0]
    }

    #[inline]
    fn y(&self) -> T {
        self[1]
    }
}

impl<T: Default + SimdElement> Vector3<T> for Simd<T, 4> {
    #[inline]
    fn new(x: T, y: T, z: T) -> Self {
        Simd::from([x, y, z, T::default()])
    }

    #[inline]
    fn x(&self) -> T {
        self[0]
    }

    #[inline]
    fn y(&self) -> T {
        self[1]
    }

    #[inline]
    fn z(&self) -> T {
        self[2]
    }
}

impl<
        T: Copy
            + Default
            + std::ops::Mul<Output = T>
            + std::ops::Sub<Output = T>
            + Sum<T>
            + SimdElement
            + Real,
        const N: usize,
    > RealVector<T, N> for Simd<T, N>
where
    LaneCount<N>: SupportedLaneCount,
    Simd<T, N>: std::ops::Mul<Output = Simd<T, N>>,
    Simd<T, N>: std::ops::Div<Output = Simd<T, N>>,
{
    #[inline]
    fn magnitude(self) -> T {
        self.to_array().into_iter().map(|n| n * n).sum::<T>().sqrt()
    }

    #[inline]
    fn normalized(self) -> Self {
        self.div_scalar(self.magnitude())
    }

    #[inline]
    fn angle_to(self, other: Self) -> T {
        self.normalized().dot(other.normalized()).acos()
    }
}

macro_rules! simd_op {
    ($vectortrait:ident, $optrait:ident, $fullmethod:ident, $scalarmethod:ident) => {
        #[cfg(feature = "simd")]
        impl<T: Copy + std::ops::$optrait<Output = T> + SimdElement, const N: usize>
            $vectortrait<T, N> for Simd<T, N>
        where
            LaneCount<N>: SupportedLaneCount,
            Simd<T, N>: std::ops::$optrait<Output = Simd<T, N>>,
        {
            #[inline]
            fn $fullmethod(self, other: Self) -> Self {
                std::ops::$optrait::$fullmethod(self, other)
            }

            #[inline]
            fn $scalarmethod(self, scalar: T) -> Self {
                std::ops::$optrait::$fullmethod(self, Simd::from([scalar; N]))
            }
        }
    };
}

simd_op!(VectorAdd, Add, add, add_scalar);
simd_op!(VectorSub, Sub, sub, sub_scalar);
simd_op!(VectorMul, Mul, mul, mul_scalar);
simd_op!(VectorDiv, Div, div, div_scalar);

#[test]
fn addition() {
    assert_eq!([4, 5].add([12, 2]), [16, 7])
}

#[test]
fn subtraction() {
    assert_eq!([4, 5].sub([12, 2]), [-8, 3])
}

#[test]
fn dot() {
    assert_eq!([2, 3].dot([-1, -2]), -8)
}
