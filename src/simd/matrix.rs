use std::{
    ops::{Add, Div, Index, Mul, Sub},
    simd::{LaneCount, Simd, SupportedLaneCount},
};

use crate::{simd::Vector, Element};

#[derive(Debug, Clone, Copy, PartialEq, PartialOrd)]
pub struct Matrix<T: Element, const R: usize, const C: usize>
where
    LaneCount<R>: SupportedLaneCount,
    LaneCount<C>: SupportedLaneCount,
    Simd<T, R>: Add<Output = Simd<T, R>>
        + Sub<Output = Simd<T, R>>
        + Mul<Simd<T, R>, Output = Simd<T, R>>
        + Div<Simd<T, R>, Output = Simd<T, R>>,
    Simd<T, C>: Add<Output = Simd<T, C>>
        + Sub<Output = Simd<T, C>>
        + Mul<Simd<T, C>, Output = Simd<T, C>>
        + Div<Simd<T, C>, Output = Simd<T, C>>,
{
    data: [[T; C]; R],
}

pub type TransformationMatrix<T> = Matrix<T, 4, 4>;

impl<T: Element, const R: usize, const C: usize> Matrix<T, R, C>
where
    LaneCount<R>: SupportedLaneCount,
    LaneCount<C>: SupportedLaneCount,
    Simd<T, 1>: Add<Output = Simd<T, 1>>
        + Sub<Output = Simd<T, 1>>
        + Mul<Simd<T, 1>, Output = Simd<T, 1>>
        + Div<Simd<T, 1>, Output = Simd<T, 1>>,
    Simd<T, R>: Add<Output = Simd<T, R>>
        + Sub<Output = Simd<T, R>>
        + Mul<Simd<T, R>, Output = Simd<T, R>>
        + Div<Simd<T, R>, Output = Simd<T, R>>,
    Simd<T, C>: Add<Output = Simd<T, C>>
        + Sub<Output = Simd<T, C>>
        + Mul<Simd<T, C>, Output = Simd<T, C>>
        + Div<Simd<T, C>, Output = Simd<T, C>>,
{
    pub fn identity() -> Self {
        Self {
            data: std::array::from_fn(|r| {
                std::array::from_fn(|c| if r == c { T::one() } else { T::zero() })
            }),
        }
    }

    pub fn translate(mut self, vec: Vector<T, R>) -> Self {
        let last_col = Simd::from(std::array::from_fn(|i| self.data[i][C - 1]));
        let translated = last_col + vec.to_simd();

        for r in 0..R {
            self.data[r][C - 1] = translated[r];
        }

        self
    }

    pub fn scale(mut self, vec: Vector<T, R>) -> Self {
        let diagonal = Simd::from(std::array::from_fn(|i| self.data[i][i]));
        let scaled = diagonal * vec.to_simd();

        for r in 0..R {
            self.data[r][r] = scaled[r];
        }

        self
    }
}

impl<T: Element, const R: usize, const C: usize> Index<usize> for Matrix<T, R, C>
where
    LaneCount<R>: SupportedLaneCount,
    LaneCount<C>: SupportedLaneCount,
    Simd<T, R>: Add<Output = Simd<T, R>>
        + Sub<Output = Simd<T, R>>
        + Mul<Simd<T, R>, Output = Simd<T, R>>
        + Div<Simd<T, R>, Output = Simd<T, R>>,
    Simd<T, C>: Add<Output = Simd<T, C>>
        + Sub<Output = Simd<T, C>>
        + Mul<Simd<T, C>, Output = Simd<T, C>>
        + Div<Simd<T, C>, Output = Simd<T, C>>,
{
    type Output = [T; C];

    #[inline]
    fn index(&self, index: usize) -> &Self::Output {
        &self.data[index]
    }
}

impl<T: Element, const R1: usize, const C1: usize, const C2: usize> Mul<Matrix<T, C1, C2>>
    for Matrix<T, R1, C1>
where
    LaneCount<R1>: SupportedLaneCount,
    LaneCount<C1>: SupportedLaneCount,
    LaneCount<C2>: SupportedLaneCount,
    Simd<T, R1>: Add<Output = Simd<T, R1>>
        + Sub<Output = Simd<T, R1>>
        + Mul<Simd<T, R1>, Output = Simd<T, R1>>
        + Div<Simd<T, R1>, Output = Simd<T, R1>>,
    Simd<T, C1>: Add<Output = Simd<T, C1>>
        + Sub<Output = Simd<T, C1>>
        + Mul<Simd<T, C1>, Output = Simd<T, C1>>
        + Div<Simd<T, C1>, Output = Simd<T, C1>>,
    Simd<T, C2>: Add<Output = Simd<T, C2>>
        + Sub<Output = Simd<T, C2>>
        + Mul<Simd<T, C2>, Output = Simd<T, C2>>
        + Div<Simd<T, C2>, Output = Simd<T, C2>>,
{
    type Output = Matrix<T, R1, C2>;

    fn mul(self, rhs: Matrix<T, C1, C2>) -> Self::Output {
        Matrix::from(std::array::from_fn(|index| {
            let row = Simd::from(self.data[index]);

            std::array::from_fn(|c2| {
                let col = Simd::from(std::array::from_fn(|i| rhs.data[i][c2]));
                (row * col).to_array().into_iter().sum()
            })
        }))
    }
}

impl<T: Element, const R1: usize, const C1: usize> Mul<Vector<T, C1>> for Matrix<T, R1, C1>
where
    LaneCount<R1>: SupportedLaneCount,
    LaneCount<C1>: SupportedLaneCount,
    Simd<T, 1>: Add<Output = Simd<T, 1>>
        + Sub<Output = Simd<T, 1>>
        + Mul<Simd<T, 1>, Output = Simd<T, 1>>
        + Div<Simd<T, 1>, Output = Simd<T, 1>>,
    Simd<T, R1>: Add<Output = Simd<T, R1>>
        + Sub<Output = Simd<T, R1>>
        + Mul<Simd<T, R1>, Output = Simd<T, R1>>
        + Div<Simd<T, R1>, Output = Simd<T, R1>>,
    Simd<T, C1>: Add<Output = Simd<T, C1>>
        + Sub<Output = Simd<T, C1>>
        + Mul<Simd<T, C1>, Output = Simd<T, C1>>
        + Div<Simd<T, C1>, Output = Simd<T, C1>>,
{
    type Output = Vector<T, R1>;

    #[inline]
    fn mul(self, rhs: Vector<T, C1>) -> Self::Output {
        let mat = self * rhs.column_matrix();
        std::array::from_fn(|i| mat[i][0]).into()
    }
}

impl<T: Element, const R: usize, const C: usize> From<[[T; C]; R]> for Matrix<T, R, C>
where
    LaneCount<R>: SupportedLaneCount,
    LaneCount<C>: SupportedLaneCount,
    Simd<T, R>: Add<Output = Simd<T, R>>
        + Sub<Output = Simd<T, R>>
        + Mul<Simd<T, R>, Output = Simd<T, R>>
        + Div<Simd<T, R>, Output = Simd<T, R>>,
    Simd<T, C>: Add<Output = Simd<T, C>>
        + Sub<Output = Simd<T, C>>
        + Mul<Simd<T, C>, Output = Simd<T, C>>
        + Div<Simd<T, C>, Output = Simd<T, C>>,
{
    #[inline]
    fn from(value: [[T; C]; R]) -> Self {
        Self { data: value }
    }
}

#[test]
fn mul() {
    assert_eq!(
        Matrix::from([[1, 7], [2, 4]]) * Matrix::from([[3, 3], [5, 2]]),
        Matrix::from([[38, 17], [26, 14]])
    )
}
