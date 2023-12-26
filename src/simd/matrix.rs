use std::{
    iter::Sum,
    ops::{AddAssign, MulAssign},
    simd::{LaneCount, Simd, SimdElement, SupportedLaneCount},
};

use num_traits::{One, Zero};

use crate::{Matrix, MatrixMul, Vector};

impl<
        T: Copy + Zero + One + AddAssign + MulAssign + SimdElement,
        const R: usize,
        const C: usize,
    > Matrix<T, R, C> for [Simd<T, C>; R]
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
        T: Copy + Zero + One + AddAssign + MulAssign + Sum<T> + SimdElement,
        const R1: usize,
        const C1: usize,
        const C2: usize,
    > MatrixMul<T, R1, C1, C2, [Simd<T, C2>; C1], [Simd<T, C2>; R1]> for [Simd<T, C1>; R1]
where
    LaneCount<C1>: SupportedLaneCount,
    LaneCount<C2>: SupportedLaneCount,
    Simd<T, C1>: std::ops::Mul<Output = Simd<T, C1>>,
{
    #[inline]
    fn mul(self, other: [Simd<T, C2>; C1]) -> [Simd<T, C2>; R1] {
        std::array::from_fn(|index| {
            let row = Simd::from(self[index]);

            std::array::from_fn(|c2| {
                let col = Simd::from(std::array::from_fn(|i| other[i][c2]));
                (row * col).to_array().into_iter().sum()
            })
            .into()
        })
    }
}

trait MatrixSimdify<T, const R: usize, const C: usize, M: Matrix<T, R, C>> {
    fn simdify(self) -> M;
}

impl<
        T: Copy + Zero + One + AddAssign + MulAssign + SimdElement,
        const R: usize,
        const C: usize,
    > MatrixSimdify<T, R, C, [Simd<T, C>; R]> for [[T; C]; R]
where
    LaneCount<C>: SupportedLaneCount,
{
    fn simdify(self) -> [Simd<T, C>; R] {
        std::array::from_fn(|i| Simd::from(self[i]))
    }
}

#[test]
fn mul() {
    assert_eq!(
        [[1, 7], [2, 4]].simdify().mul([[3, 3], [5, 2]].simdify()),
        [[38, 17], [26, 14]].simdify()
    )
}
