use std::{
    iter::Sum,
    ops::{AddAssign, Index, MulAssign},
};

use num_traits::{One, Zero};

use crate::{ArrayVector, Vector, VectorMul};

pub type ArrayMatrix<T, const R: usize, const C: usize> = [[T; C]; R];

pub trait Matrix<T, const R: usize, const C: usize>: Index<usize> {
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

pub trait MatrixVectorMul<T, const R: usize, const C: usize, V1: Vector<T, C>, V2: Vector<T, R>> {
    fn mul_vec(self, vec: V1) -> V2;
}

impl<T: Copy + Zero + One + AddAssign + MulAssign, const R: usize, const C: usize> Matrix<T, R, C>
    for ArrayMatrix<T, R, C>
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
        T: Copy + Default + Zero + One + AddAssign + MulAssign + std::ops::Sub<T, Output = T> + Sum<T>,
        const R1: usize,
        const C1: usize,
        const C2: usize,
    > MatrixMul<T, R1, C1, C2, ArrayMatrix<T, C1, C2>, ArrayMatrix<T, R1, C2>>
    for ArrayMatrix<T, R1, C1>
{
    #[inline]
    fn mul(self, other: ArrayMatrix<T, C1, C2>) -> ArrayMatrix<T, R1, C2> {
        std::array::from_fn(|index| {
            let row = self[index];

            std::array::from_fn(|c2| {
                let col = std::array::from_fn(|i| other[i][c2]);
                row.mul(col).into_iter().sum()
            })
        })
    }
}

impl<
        T: Copy + Default + std::ops::Mul<Output = T> + std::ops::Sub<Output = T> + Sum<T>,
        const R: usize,
        const C: usize,
    > MatrixVectorMul<T, R, C, ArrayVector<T, C>, ArrayVector<T, R>> for ArrayMatrix<T, R, C>
{
    fn mul_vec(self, vec: ArrayVector<T, C>) -> ArrayVector<T, R> {
        std::array::from_fn(|index| self[index].dot(vec))
    }
}

#[test]
fn mul() {
    assert_eq!([[1, 7], [2, 4]].mul([[3, 3], [5, 2]]), [[38, 17], [26, 14]])
}

#[test]
fn mul_2() {
    assert_eq!(
        [[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]].mul([
            [13, 14, 15],
            [16, 17, 18],
            [19, 20, 21],
            [22, 23, 24]
        ]),
        [[190, 200, 210], [470, 496, 522], [750, 792, 834]]
    )
}
