use std::ops::{Index, Mul};

use crate::{Element, Vector};

#[derive(Debug, Clone, Copy, PartialEq, PartialOrd)]
pub struct Matrix<T: Element, const R: usize, const C: usize> {
    data: [[T; C]; R],
}

// pub type TransformationMatrix<T> = Matrix<T, 4, 4>;

// impl<T: Element, const R: usize, const C: usize> Matrix<T, R, C> {
//     pub fn identity() -> Self {
//         let mut res = [[T::zero(); C]; R];
//         let mut i = 0;

//         while i < R {
//             res[i][i] = T::one();
//             i += 1;
//         }

//         Self { data: res }
//     }

//     pub fn translate(mut self, vec: Vector<T, R>) -> Self {
//         for r in 0..R {
//             self.data[r][C - 1] += vec[r];
//         }

//         self
//     }

//     pub fn scale(mut self, vec: Vector<T, R>) -> Self {
//         for r in 0..R {
//             self.data[r][r] *= vec[r];
//         }

//         self
//     }
// }

impl<T: Element, const R: usize, const C: usize> Index<usize> for Matrix<T, R, C> {
    type Output = [T; C];

    #[inline]
    fn index(&self, index: usize) -> &Self::Output {
        &self.data[index]
    }
}

// impl<T: Element, const R1: usize, const C1: usize, const C2: usize> Mul<Matrix<T, C1, C2>>
//     for Matrix<T, R1, C1>
// {
//     type Output = Matrix<T, R1, C2>;

//     fn mul(self, rhs: Matrix<T, C1, C2>) -> Self::Output {
//         let mut res = [[T::zero(); C2]; R1];

//         for (index, row) in self.data.into_iter().enumerate() {
//             for c2 in 0..C2 {
//                 res[index][c2] = row
//                     .into_iter()
//                     .zip(rhs.data.into_iter().map(|r| r[c2]))
//                     .map(|(l, r)| l * r)
//                     .sum();
//             }
//         }

//         res.into()
//     }
// }

// impl<T: Element, const R1: usize, const C1: usize> Mul<Vector<T, C1>> for Matrix<T, R1, C1> {
//     type Output = Vector<T, R1>;

//     #[inline]
//     fn mul(self, rhs: Vector<T, C1>) -> Self::Output {
//         (self * rhs.column_matrix()).into()
//     }
// }

impl<T: Element, const R: usize, const C: usize> From<[[T; C]; R]> for Matrix<T, R, C> {
    #[inline]
    fn from(value: [[T; C]; R]) -> Self {
        Self { data: value }
    }
}
