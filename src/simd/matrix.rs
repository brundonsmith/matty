use std::{
    iter::Sum,
    ops::{AddAssign, MulAssign},
    simd::{LaneCount, Simd, SimdElement, SupportedLaneCount},
};

use num_traits::{One, Zero};

use crate::{ArrayMatrix, Matrix, MatrixMul, MatrixVectorMul, Vector};

use super::SimdVector;

pub type SimdMatrix<T, const R: usize, const C: usize> = [Simd<T, C>; R];

impl<
        T: Copy + Zero + One + AddAssign + MulAssign + SimdElement,
        const R: usize,
        const C: usize,
    > Matrix<T, R, C> for SimdMatrix<T, R, C>
where
    LaneCount<C>: SupportedLaneCount,
{
    #[inline]
    fn identity() -> Self {
        std::array::from_fn(|r| {
            std::array::from_fn(|c| if r == c { T::one() } else { T::zero() }).into()
        })
    }

    #[inline]
    fn translate<V: Vector<T, R>>(mut self, vec: V) -> Self {
        for r in 0..R {
            self[r][C - 1] += vec[r];
        }

        self
    }

    #[inline]
    fn scale<V: Vector<T, R>>(mut self, vec: V) -> Self {
        for r in 0..R {
            self[r][r] *= vec[r];
        }

        self
    }
}

impl<
        T: Copy
            + Default
            + Zero
            + One
            + AddAssign
            + MulAssign
            + std::ops::Sub<T, Output = T>
            + Sum<T>
            + SimdElement,
        const R1: usize,
        const C1: usize,
        const C2: usize,
    > MatrixMul<T, R1, C1, C2, SimdMatrix<T, C1, C2>, SimdMatrix<T, R1, C2>>
    for SimdMatrix<T, R1, C1>
where
    LaneCount<C1>: SupportedLaneCount,
    LaneCount<C2>: SupportedLaneCount,
    Simd<T, C1>: std::ops::Mul<Output = Simd<T, C1>>,
{
    #[inline]
    fn mul(self, other: SimdMatrix<T, C1, C2>) -> SimdMatrix<T, R1, C2> {
        std::array::from_fn(|index| {
            let row = Simd::from(self[index]);

            Simd::from(std::array::from_fn(|c2| {
                let col = Simd::from(std::array::from_fn(|i| other[i][c2]));
                (row * col).to_array().into_iter().sum()
            }))
        })
    }
}

impl<
        T: Copy
            + Default
            + std::ops::Mul<Output = T>
            + std::ops::Sub<Output = T>
            + Sum<T>
            + SimdElement,
        const R: usize,
        const C: usize,
    > MatrixVectorMul<T, R, C, SimdVector<T, C>, SimdVector<T, R>> for SimdMatrix<T, R, C>
where
    LaneCount<R>: SupportedLaneCount,
    LaneCount<C>: SupportedLaneCount,
    Simd<T, R>: std::ops::Mul<Output = Simd<T, R>>,
    Simd<T, C>: std::ops::Mul<Output = Simd<T, C>>,
{
    #[inline]
    fn mul_vec(self, vec: SimdVector<T, C>) -> SimdVector<T, R> {
        Simd::from(std::array::from_fn(|index| self[index].dot(vec)))
    }
}

trait MatrixSimdify<T: SimdElement, const R: usize, const C: usize>
where
    LaneCount<C>: SupportedLaneCount,
{
    fn simd(self) -> SimdMatrix<T, R, C>;
}

impl<T: SimdElement, const R: usize, const C: usize> MatrixSimdify<T, R, C> for ArrayMatrix<T, R, C>
where
    LaneCount<C>: SupportedLaneCount,
{
    fn simd(self) -> SimdMatrix<T, R, C> {
        std::array::from_fn(|i| Simd::from(self[i]))
    }
}

#[test]
fn mul() {
    assert_eq!(
        [[1, 7], [2, 4]].simd().mul([[3, 3], [5, 2]].simd()),
        [[38, 17], [26, 14]].simd()
    )
}

#[test]
fn mul_2() {
    assert_eq!(
        [[1, 2, 3, 4], [5, 6, 7, 8]]
            .simd()
            .mul([[9, 10], [11, 12], [13, 14], [15, 16]].simd()),
        [[130, 140], [322, 348]].simd()
    )
}
