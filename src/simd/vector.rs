use std::{
    fmt::{Display, Write},
    iter::Sum,
    ops::{Add, Div, Index, Mul, Sub},
    simd::{LaneCount, Simd, SupportedLaneCount},
};

use num_traits::real::Real;

use crate::{simd::Matrix, Element, IRealVector, IVector};

#[derive(Debug, Clone, Copy, PartialEq, PartialOrd)]
pub struct Vector<T: Element, const N: usize>
where
    LaneCount<N>: SupportedLaneCount,
    Simd<T, N>: Add<Output = Simd<T, N>>
        + Sub<Output = Simd<T, N>>
        + Mul<Simd<T, N>, Output = Simd<T, N>>
        + Div<Simd<T, N>, Output = Simd<T, N>>,
{
    data: Simd<T, N>,
    #[cfg(feature = "remember_normalization")]
    normalized: bool,
}

impl<T: Element, const N: usize> Vector<T, N>
where
    LaneCount<N>: SupportedLaneCount,
    Simd<T, 1>: Add<Output = Simd<T, 1>>
        + Sub<Output = Simd<T, 1>>
        + Mul<Simd<T, 1>, Output = Simd<T, 1>>
        + Div<Simd<T, 1>, Output = Simd<T, 1>>,
    Simd<T, N>: Add<Output = Simd<T, N>>
        + Sub<Output = Simd<T, N>>
        + Mul<Simd<T, N>, Output = Simd<T, N>>
        + Div<Simd<T, N>, Output = Simd<T, N>>,
{
    #[inline]
    pub fn resize<const N2: usize>(self) -> Vector<T, N2>
    where
        LaneCount<N2>: SupportedLaneCount,
        Simd<T, N2>: Add<Output = Simd<T, N2>>
            + Sub<Output = Simd<T, N2>>
            + Mul<Simd<T, N2>, Output = Simd<T, N2>>
            + Div<Simd<T, N2>, Output = Simd<T, N2>>,
    {
        std::array::from_fn(|i| {
            self.data
                .to_array()
                .get(i)
                .map(|x| *x)
                .unwrap_or(T::default())
        })
        .into()
    }

    #[inline]
    pub fn to_simd(self) -> Simd<T, N> {
        self.data
    }

    #[inline]
    pub fn column_matrix(self) -> Matrix<T, N, 1> {
        std::array::from_fn(|i| [self.data[i]]).into()
    }

    #[inline]
    pub fn row_matrix(self) -> Matrix<T, 1, N> {
        [self.data.to_array()].into()
    }
}

impl<T: Element, const N: usize> IVector<T, N> for Vector<T, N>
where
    LaneCount<N>: SupportedLaneCount,
    Simd<T, N>: Add<Output = Simd<T, N>>
        + Sub<Output = Simd<T, N>>
        + Mul<Simd<T, N>, Output = Simd<T, N>>
        + Div<Simd<T, N>, Output = Simd<T, N>>,
{
    #[inline]
    fn new(data: [T; N]) -> Self {
        Self {
            data: data.into(),
            #[cfg(feature = "remember_normalization")]
            normalized: false,
        }
    }

    #[inline]
    fn to_array(self) -> [T; N] {
        self.data.to_array()
    }

    #[inline]
    fn dot(self, other: Self) -> T {
        (self.data * other.data).to_array().into_iter().sum()
    }

    #[inline]
    fn cross(self, other: Self) -> Self {
        let mut res = [T::zero(); N];

        for i in 0..N {
            res[i] = self.data[(i + 1) % N] * other.data[(i + 2) % N]
                - self.data[(i + 2) % N] * other.data[(i + 1) % N];
        }

        res.into()
    }
}

impl<T: Element + Real, const N: usize> IRealVector<T, N> for Vector<T, N>
where
    LaneCount<N>: SupportedLaneCount,
    Simd<T, N>: Add<Output = Simd<T, N>>
        + Sub<Output = Simd<T, N>>
        + Mul<Simd<T, N>, Output = Simd<T, N>>
        + Div<Simd<T, N>, Output = Simd<T, N>>,
{
    #[inline]
    fn magnitude(self) -> T {
        self.data
            .to_array()
            .into_iter()
            .map(|n| n * n)
            .sum::<T>()
            .sqrt()
    }

    #[inline]
    fn normalized(self) -> Self {
        #[cfg(feature = "remember_normalization")]
        if self.normalized {
            return self;
        }

        let mut res = self / self.magnitude();

        #[cfg(feature = "remember_normalization")]
        {
            res.normalized = true;
        }

        res
    }

    #[inline]
    fn angle_to(self, other: Self) -> T {
        self.normalized().dot(other.normalized()).acos()
    }
}

impl<T: Element, const N: usize> Display for Vector<T, N>
where
    LaneCount<N>: SupportedLaneCount,
    Simd<T, N>: Add<Output = Simd<T, N>>
        + Sub<Output = Simd<T, N>>
        + Mul<Simd<T, N>, Output = Simd<T, N>>
        + Div<Simd<T, N>, Output = Simd<T, N>>,
{
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

impl<T: Element, const N: usize> Index<usize> for Vector<T, N>
where
    LaneCount<N>: SupportedLaneCount,
    Simd<T, N>: Add<Output = Simd<T, N>>
        + Sub<Output = Simd<T, N>>
        + Mul<Simd<T, N>, Output = Simd<T, N>>
        + Div<Simd<T, N>, Output = Simd<T, N>>,
{
    type Output = T;

    #[inline]
    fn index(&self, index: usize) -> &Self::Output {
        &self.data[index]
    }
}

impl<T: Element, const N: usize> Default for Vector<T, N>
where
    LaneCount<N>: SupportedLaneCount,
    Simd<T, N>: Add<Output = Simd<T, N>>
        + Sub<Output = Simd<T, N>>
        + Mul<Simd<T, N>, Output = Simd<T, N>>
        + Div<Simd<T, N>, Output = Simd<T, N>>,
{
    #[inline]
    fn default() -> Self {
        Self {
            data: [T::default(); N].into(),
            #[cfg(feature = "remember_normalization")]
            normalized: Default::default(),
        }
    }
}

/// From traits

impl<T: Element, const N: usize> From<[T; N]> for Vector<T, N>
where
    LaneCount<N>: SupportedLaneCount,
    Simd<T, N>: Add<Output = Simd<T, N>>
        + Sub<Output = Simd<T, N>>
        + Mul<Simd<T, N>, Output = Simd<T, N>>
        + Div<Simd<T, N>, Output = Simd<T, N>>,
{
    #[inline]
    fn from(value: [T; N]) -> Self {
        Self {
            data: Simd::from(value),
            #[cfg(feature = "remember_normalization")]
            normalized: false,
        }
    }
}

impl<T: Element, const N: usize> From<Vector<T, N>> for [T; N]
where
    LaneCount<N>: SupportedLaneCount,
    Simd<T, N>: Add<Output = Simd<T, N>>
        + Sub<Output = Simd<T, N>>
        + Mul<Simd<T, N>, Output = Simd<T, N>>
        + Div<Simd<T, N>, Output = Simd<T, N>>,
{
    #[inline]
    fn from(value: Vector<T, N>) -> Self {
        value.to_array()
    }
}

impl<T: Element, const N: usize> From<Simd<T, N>> for Vector<T, N>
where
    LaneCount<N>: SupportedLaneCount,
    Simd<T, N>: Add<Output = Simd<T, N>>
        + Sub<Output = Simd<T, N>>
        + Mul<Simd<T, N>, Output = Simd<T, N>>
        + Div<Simd<T, N>, Output = Simd<T, N>>,
{
    #[inline]
    fn from(value: Simd<T, N>) -> Self {
        Self {
            data: value,
            #[cfg(feature = "remember_normalization")]
            normalized: false,
        }
    }
}

impl<T: Element, const N: usize> From<Vector<T, N>> for crate::Vector<T, N>
where
    LaneCount<N>: SupportedLaneCount,
    Simd<T, N>: Add<Output = Simd<T, N>>
        + Sub<Output = Simd<T, N>>
        + Mul<Simd<T, N>, Output = Simd<T, N>>
        + Div<Simd<T, N>, Output = Simd<T, N>>,
{
    #[inline]
    fn from(value: Vector<T, N>) -> Self {
        Self::new(value.to_array())
    }
}

impl<T: Element, const N: usize> From<crate::Vector<T, N>> for Vector<T, N>
where
    LaneCount<N>: SupportedLaneCount,
    Simd<T, N>: Add<Output = Simd<T, N>>
        + Sub<Output = Simd<T, N>>
        + Mul<Simd<T, N>, Output = Simd<T, N>>
        + Div<Simd<T, N>, Output = Simd<T, N>>,
{
    #[inline]
    fn from(value: crate::Vector<T, N>) -> Self {
        Self::new(value.to_array())
    }
}

/// Math traits

impl<T: Element, const N: usize> Add<Vector<T, N>> for Vector<T, N>
where
    LaneCount<N>: SupportedLaneCount,
    Simd<T, N>: Add<Output = Simd<T, N>>
        + Sub<Output = Simd<T, N>>
        + Mul<Simd<T, N>, Output = Simd<T, N>>
        + Div<Simd<T, N>, Output = Simd<T, N>>,
{
    type Output = Self;

    #[inline]
    fn add(self, rhs: Vector<T, N>) -> Self::Output {
        Self {
            data: (self.data + rhs.data).into(),
            #[cfg(feature = "remember_normalization")]
            normalized: false,
        }
    }
}

impl<T: Element, const N: usize> Sub<Vector<T, N>> for Vector<T, N>
where
    LaneCount<N>: SupportedLaneCount,
    Simd<T, N>: Add<Output = Simd<T, N>>
        + Sub<Output = Simd<T, N>>
        + Mul<Simd<T, N>, Output = Simd<T, N>>
        + Div<Simd<T, N>, Output = Simd<T, N>>,
{
    type Output = Vector<T, N>;

    #[inline]
    fn sub(self, rhs: Vector<T, N>) -> Self::Output {
        Self {
            data: (self.data - rhs.data).into(),
            #[cfg(feature = "remember_normalization")]
            normalized: false,
        }
    }
}

impl<T: Element, const N: usize> Mul<T> for Vector<T, N>
where
    LaneCount<N>: SupportedLaneCount,
    Simd<T, N>: Add<Output = Simd<T, N>>
        + Sub<Output = Simd<T, N>>
        + Mul<Simd<T, N>, Output = Simd<T, N>>
        + Div<Simd<T, N>, Output = Simd<T, N>>,
{
    type Output = Vector<T, N>;

    #[inline]
    fn mul(self, rhs: T) -> Self::Output {
        (self.data * Simd::from([rhs; N])).into()
    }
}

impl<T: Element, const N: usize> Div<T> for Vector<T, N>
where
    LaneCount<N>: SupportedLaneCount,
    Simd<T, N>: Add<Output = Simd<T, N>>
        + Sub<Output = Simd<T, N>>
        + Mul<Simd<T, N>, Output = Simd<T, N>>
        + Div<Simd<T, N>, Output = Simd<T, N>>,
{
    type Output = Vector<T, N>;

    #[inline]
    fn div(self, rhs: T) -> Self::Output {
        (self.data / Simd::from([rhs; N])).into()
    }
}

impl<T: Element, const N: usize> Sum for Vector<T, N>
where
    LaneCount<N>: SupportedLaneCount,
    Simd<T, N>: Add<Output = Simd<T, N>>
        + Sub<Output = Simd<T, N>>
        + Mul<Simd<T, N>, Output = Simd<T, N>>
        + Div<Simd<T, N>, Output = Simd<T, N>>,
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
                data: Simd::from_array([1.0, 0.0]),
                #[cfg(feature = "remember_normalization")]
                normalized: true,
            };
            pub const UP: Self = Vector {
                data: Simd::from_array([0.0, 1.0]),
                #[cfg(feature = "remember_normalization")]
                normalized: true,
            };
            pub const LEFT: Self = Vector {
                data: Simd::from_array([-1.0, 0.0]),
                #[cfg(feature = "remember_normalization")]
                normalized: true,
            };
            pub const DOWN: Self = Vector {
                data: Simd::from_array([0.0, -1.0]),
                #[cfg(feature = "remember_normalization")]
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
    assert_eq!(Vector::new([2, 3]).dot(Vector::new([-1, -2])), -8)
}
