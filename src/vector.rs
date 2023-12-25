use std::{
    fmt::{Display, Write},
    iter::Sum,
    ops::{Add, Div, Index, Mul, Sub},
};

use num_traits::real::Real;

use crate::{utils::collect_arr, Element, Matrix};

pub trait IVector<T: Element, const N: usize> {
    fn new(data: [T; N]) -> Self;

    fn column_matrix(self) -> Matrix<T, N, 1>;

    fn row_matrix(self) -> Matrix<T, 1, N>;

    fn dot(self, other: Self) -> T;

    fn cross(self, other: Self) -> Self;
}

pub trait IRealVector<T: Element + Real, const N: usize> {
    fn magnitude(self) -> T;

    fn normalized(self) -> Self;

    fn angle_to(self, other: Self) -> T;
}

#[derive(Debug, Clone, Copy, PartialEq, PartialOrd)]
pub struct Vector<T: Element, const N: usize> {
    data: [T; N],
    normalized: bool,
}

impl<T: Element, const N: usize> Vector<T, N> {
    #[inline]
    pub fn resize<const N2: usize>(self) -> Vector<T, N2> {
        collect_arr(
            (0..(self.data.len().max(N2)))
                .map(|i| self.data.get(i).map(|x| *x).unwrap_or(T::default())),
        )
        .into()
    }
}

impl<T: Element, const N: usize> IVector<T, N> for Vector<T, N> {
    #[inline]
    fn new(data: [T; N]) -> Self {
        Self {
            data,
            normalized: false,
        }
    }

    #[inline]
    fn column_matrix(self) -> Matrix<T, N, 1> {
        collect_arr(self.data.into_iter().map(|n| [n])).into()
    }

    #[inline]
    fn row_matrix(self) -> Matrix<T, 1, N> {
        [self.data].into()
    }

    #[inline]
    fn dot(self, other: Vector<T, N>) -> T {
        self.data
            .into_iter()
            .zip(other.data.into_iter())
            .map(|(s, o)| s * o)
            .sum()
    }

    #[inline]
    fn cross(self, other: Vector<T, N>) -> Vector<T, N> {
        let mut res = [T::zero(); N];

        for i in 0..N {
            res[i] = self.data[(i + 1) % N] * other.data[(i + 2) % N]
                - self.data[(i + 2) % N] * other.data[(i + 1) % N];
        }

        res.into()
    }
}

impl<T: Element + Real, const N: usize> IRealVector<T, N> for Vector<T, N> {
    #[inline]
    fn magnitude(self) -> T {
        self.data.into_iter().map(|n| n * n).sum::<T>().sqrt()
    }

    #[inline]
    fn normalized(self) -> Vector<T, N> {
        if self.normalized {
            self
        } else {
            let mut res = self / self.magnitude();
            res.normalized = true;
            res
        }
    }

    #[inline]
    fn angle_to(self, other: Vector<T, N>) -> T {
        self.normalized().dot(other.normalized()).acos()
    }
}

impl<T: Element, const N: usize> Display for Vector<T, N> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_char('(')?;
        for n in 0..N {
            if n > 0 {
                f.write_str(", ")?;
            }

            f.write_fmt(format_args!("{}", self.data[n]))?;
        }
        f.write_char(')')
    }
}

impl<T: Element, const N: usize> Index<usize> for Vector<T, N> {
    type Output = T;

    #[inline]
    fn index(&self, index: usize) -> &Self::Output {
        &self.data[index]
    }
}

impl<T: Element, const N: usize> Default for Vector<T, N> {
    #[inline]
    fn default() -> Self {
        Self {
            data: [T::default(); N].into(),
            normalized: Default::default(),
        }
    }
}

/// From traits

impl<T: Element, const N: usize, A> From<A> for Vector<T, N>
where
    A: Into<[T; N]>,
{
    #[inline]
    fn from(value: A) -> Self {
        Vector::new(value.into())
    }
}

impl<T: Element, const N: usize> From<Matrix<T, N, 1>> for Vector<T, N> {
    #[inline]
    fn from(value: Matrix<T, N, 1>) -> Self {
        collect_arr((0..N).map(|i| value[i][0])).into()
    }
}

/// Math traits

impl<T: Element, const N: usize> Add<Vector<T, N>> for Vector<T, N> {
    type Output = Self;

    #[inline]
    fn add(self, rhs: Vector<T, N>) -> Self::Output {
        collect_arr(
            self.data
                .into_iter()
                .zip(rhs.data.into_iter())
                .map(|(l, r)| l + r),
        )
        .into()
    }
}

impl<T: Element, const N: usize> Sub<Vector<T, N>> for Vector<T, N> {
    type Output = Vector<T, N>;

    #[inline]
    fn sub(self, rhs: Vector<T, N>) -> Self::Output {
        collect_arr(
            self.data
                .into_iter()
                .zip(rhs.data.into_iter())
                .map(|(l, r)| l - r),
        )
        .into()
    }
}

impl<T: Element, const N: usize> Mul<T> for Vector<T, N> {
    type Output = Vector<T, N>;

    #[inline]
    fn mul(self, rhs: T) -> Self::Output {
        collect_arr(self.data.into_iter().map(|n| n * rhs)).into()
    }
}

impl<T: Element, const N: usize> Div<T> for Vector<T, N> {
    type Output = Vector<T, N>;

    #[inline]
    fn div(self, rhs: T) -> Self::Output {
        self * (T::one() / rhs)
    }
}

impl<T: Element, const N: usize> Sum for Vector<T, N>
where
    Vector<T, N>: Add<Output = Vector<T, N>>,
{
    #[inline]
    fn sum<I: Iterator<Item = Vector<T, N>>>(iter: I) -> Self {
        iter.fold(Self::default(), Add::add)
    }
}

/// Convenience consts

macro_rules! impl_vector2_consts {
    ($type:ident) => {
        impl Vector<$type, 2> {
            pub const RIGHT: Self = Vector {
                data: [1.0, 0.0],
                normalized: true,
            };
            pub const UP: Self = Vector {
                data: [0.0, 1.0],
                normalized: true,
            };
            pub const LEFT: Self = Vector {
                data: [-1.0, 0.0],
                normalized: true,
            };
            pub const DOWN: Self = Vector {
                data: [0.0, -1.0],
                normalized: true,
            };

            #[inline]
            pub fn x(self) -> $type {
                self.data[0]
            }

            #[inline]
            pub fn y(self) -> $type {
                self.data[1]
            }
        }
    };
}

impl_vector2_consts!(f32);
impl_vector2_consts!(f64);

macro_rules! impl_vector3_consts {
    ($type:ident) => {
        impl Vector<$type, 3> {
            pub const RIGHT: Self = Vector {
                data: [1.0, 0.0, 0.0],
                normalized: true,
            };
            pub const UP: Self = Vector {
                data: [0.0, 1.0, 0.0],
                normalized: true,
            };
            pub const FORWARD: Self = Vector {
                data: [0.0, 0.0, 1.0],
                normalized: true,
            };
            pub const LEFT: Self = Vector {
                data: [-1.0, 0.0, 0.0],
                normalized: true,
            };
            pub const DOWN: Self = Vector {
                data: [0.0, -1.0, 0.0],
                normalized: true,
            };
            pub const BACK: Self = Vector {
                data: [0.0, 0.0, -1.0],
                normalized: true,
            };

            #[inline]
            pub fn x(self) -> $type {
                self.data[0]
            }

            #[inline]
            pub fn y(self) -> $type {
                self.data[1]
            }

            #[inline]
            pub fn z(self) -> $type {
                self.data[2]
            }
        }
    };
}

impl_vector3_consts! {f32}
impl_vector3_consts! {f64}

#[test]
fn addition() {
    assert_eq!(
        Vector::new([4, 5]) + Vector::new([12, 2]),
        Vector::new([16, 7])
    )
}

#[test]
fn subtraction() {
    assert_eq!(
        Vector::new([4, 5]) - Vector::new([12, 2]),
        Vector::new([-8, 3])
    )
}

#[test]
fn dot() {
    assert_eq!(Vector::new([2, 7, 1]).dot(Vector::new([8, 2, 8])), 38)
}
