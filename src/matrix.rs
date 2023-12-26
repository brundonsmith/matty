use std::{
    iter::Sum,
    ops::{AddAssign, MulAssign},
};

use num_traits::{One, Zero};

use crate::{Vector, VectorMul};

pub trait Matrix<T, const R: usize, const C: usize> {
    fn identity() -> Self;

    fn translate<V: Vector<T, R>>(self, vec: V) -> Self;

    fn scale<V: Vector<T, R>>(self, vec: V) -> Self;
}

pub trait MatrixMul<
    T,
    const R1: usize,
    const C1: usize,
    const C2: usize,
    MOther: Matrix<T, C1, C2>,
    MResult: Matrix<T, R1, C2>,
>
{
    fn mul(self, other: MOther) -> MResult;
}

impl<T: Copy + Zero + One + AddAssign + MulAssign, const R: usize, const C: usize> Matrix<T, R, C>
    for [[T; C]; R]
{
    #[inline]
    fn identity() -> Self {
        std::array::from_fn(|r| std::array::from_fn(|c| if r == c { T::one() } else { T::zero() }))
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
        T: Copy + Zero + One + AddAssign + MulAssign + Sum<T>,
        const R1: usize,
        const C1: usize,
        const C2: usize,
    > MatrixMul<T, R1, C1, C2, [[T; C2]; C1], [[T; C2]; R1]> for [[T; C1]; R1]
{
    #[inline]
    fn mul(self, other: [[T; C2]; C1]) -> [[T; C2]; R1] {
        std::array::from_fn(|index| {
            let row = self[index];

            std::array::from_fn(|c2| {
                let col = std::array::from_fn(|i| other[i][c2]);
                row.mul(col).into_iter().sum()
            })
        })
    }
}

#[test]
fn mul() {
    assert_eq!([[1, 7], [2, 4]].mul([[3, 3], [5, 2]]), [[38, 17], [26, 14]])
}
