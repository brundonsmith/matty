use std::{
    fmt::{Display, Write},
    iter::Sum,
    ops::{Add, Div, Index, Mul, Sub},
    simd::{LaneCount, Simd, SupportedLaneCount},
};

use num_traits::real::Real;

use crate::{utils::collect_arr, Element, IRealVector, IVector, Matrix};

#[derive(Debug, Clone, Copy, PartialEq, PartialOrd)]
pub struct SimdVector<T: Element, const N: usize>
where
    LaneCount<N>: SupportedLaneCount,
    Simd<T, N>: Add<Output = Simd<T, N>>
        + Sub<Output = Simd<T, N>>
        + Mul<Simd<T, N>, Output = Simd<T, N>>
        + Div<Simd<T, N>, Output = Simd<T, N>>,
{
    data: Simd<T, N>,
    normalized: bool,
}
impl<T: Element, const N: usize> SimdVector<T, N>
where
    LaneCount<N>: SupportedLaneCount,
    Simd<T, N>: Add<Output = Simd<T, N>>
        + Sub<Output = Simd<T, N>>
        + Mul<Simd<T, N>, Output = Simd<T, N>>
        + Div<Simd<T, N>, Output = Simd<T, N>>,
{
    #[inline]
    pub fn resize<const N2: usize>(self) -> SimdVector<T, N2>
    where
        LaneCount<N2>: SupportedLaneCount,
        Simd<T, N2>: Add<Output = Simd<T, N2>>
            + Sub<Output = Simd<T, N2>>
            + Mul<Simd<T, N2>, Output = Simd<T, N2>>
            + Div<Simd<T, N2>, Output = Simd<T, N2>>,
    {
        collect_arr((0..(self.data.lanes().max(N2))).map(|i| {
            self.data
                .to_array()
                .get(i)
                .map(|x| *x)
                .unwrap_or(T::default())
        }))
        .into()
    }
}

impl<T: Element, const N: usize> IVector<T, N> for SimdVector<T, N>
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
            normalized: false,
        }
    }

    #[inline]
    fn column_matrix(self) -> Matrix<T, N, 1> {
        collect_arr(self.data.to_array().into_iter().map(|n| [n])).into()
    }

    #[inline]
    fn row_matrix(self) -> Matrix<T, 1, N> {
        [self.data.to_array()].into()
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

impl<T: Element + Real, const N: usize> IRealVector<T, N> for SimdVector<T, N>
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
        if self.normalized {
            self
        } else {
            let mut res = self / self.magnitude();
            res.normalized = true;
            res
        }
    }

    #[inline]
    fn angle_to(self, other: Self) -> T {
        self.normalized().dot(other.normalized()).acos()
    }
}

impl<T: Element, const N: usize> Display for SimdVector<T, N>
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

impl<T: Element, const N: usize> Index<usize> for SimdVector<T, N>
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

impl<T: Element, const N: usize> Default for SimdVector<T, N>
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
            normalized: Default::default(),
        }
    }
}

/// From traits

impl<T: Element, const N: usize> From<[T; N]> for SimdVector<T, N>
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
            normalized: false,
        }
    }
}

impl<T: Element, const N: usize> From<Simd<T, N>> for SimdVector<T, N>
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
            normalized: false,
        }
    }
}

/// Math traits

impl<T: Element, const N: usize> Add<SimdVector<T, N>> for SimdVector<T, N>
where
    LaneCount<N>: SupportedLaneCount,
    Simd<T, N>: Add<Output = Simd<T, N>>
        + Sub<Output = Simd<T, N>>
        + Mul<Simd<T, N>, Output = Simd<T, N>>
        + Div<Simd<T, N>, Output = Simd<T, N>>,
{
    type Output = Self;

    #[inline]
    fn add(self, rhs: SimdVector<T, N>) -> Self::Output {
        Self {
            data: (self.data + rhs.data).into(),
            normalized: false,
        }
    }
}

impl<T: Element, const N: usize> Sub<SimdVector<T, N>> for SimdVector<T, N>
where
    LaneCount<N>: SupportedLaneCount,
    Simd<T, N>: Add<Output = Simd<T, N>>
        + Sub<Output = Simd<T, N>>
        + Mul<Simd<T, N>, Output = Simd<T, N>>
        + Div<Simd<T, N>, Output = Simd<T, N>>,
{
    type Output = SimdVector<T, N>;

    #[inline]
    fn sub(self, rhs: SimdVector<T, N>) -> Self::Output {
        Self {
            data: (self.data - rhs.data).into(),
            normalized: false,
        }
    }
}

impl<T: Element, const N: usize> Mul<T> for SimdVector<T, N>
where
    LaneCount<N>: SupportedLaneCount,
    Simd<T, N>: Add<Output = Simd<T, N>>
        + Sub<Output = Simd<T, N>>
        + Mul<Simd<T, N>, Output = Simd<T, N>>
        + Div<Simd<T, N>, Output = Simd<T, N>>,
{
    type Output = SimdVector<T, N>;

    #[inline]
    fn mul(self, rhs: T) -> Self::Output {
        (self.data * Simd::from([rhs; N])).into()
    }
}

impl<T: Element, const N: usize> Div<T> for SimdVector<T, N>
where
    LaneCount<N>: SupportedLaneCount,
    Simd<T, N>: Add<Output = Simd<T, N>>
        + Sub<Output = Simd<T, N>>
        + Mul<Simd<T, N>, Output = Simd<T, N>>
        + Div<Simd<T, N>, Output = Simd<T, N>>,
{
    type Output = SimdVector<T, N>;

    #[inline]
    fn div(self, rhs: T) -> Self::Output {
        (self.data / Simd::from([rhs; N])).into()
    }
}

impl<T: Element, const N: usize> Sum for SimdVector<T, N>
where
    LaneCount<N>: SupportedLaneCount,
    Simd<T, N>: Add<Output = Simd<T, N>>
        + Sub<Output = Simd<T, N>>
        + Mul<Simd<T, N>, Output = Simd<T, N>>
        + Div<Simd<T, N>, Output = Simd<T, N>>,
{
    #[inline]
    fn sum<I: Iterator<Item = SimdVector<T, N>>>(iter: I) -> Self {
        iter.fold(Self::default(), Add::add)
    }
}

/// Convenience consts

macro_rules! impl_vector2_consts {
    ($type:ident) => {
        impl SimdVector<$type, 2> {
            pub const RIGHT: Self = SimdVector {
                data: Simd::from_array([1.0, 0.0]),
                normalized: true,
            };
            pub const UP: Self = SimdVector {
                data: Simd::from_array([0.0, 1.0]),
                normalized: true,
            };
            pub const LEFT: Self = SimdVector {
                data: Simd::from_array([-1.0, 0.0]),
                normalized: true,
            };
            pub const DOWN: Self = SimdVector {
                data: Simd::from_array([0.0, -1.0]),
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
        SimdVector::new([4, 5]) + SimdVector::new([12, 2]),
        SimdVector::new([16, 7])
    )
}

#[test]
fn subtraction() {
    assert_eq!(
        SimdVector::new([4, 5]) - SimdVector::new([12, 2]),
        SimdVector::new([-8, 3])
    )
}

#[test]
fn dot() {
    assert_eq!(SimdVector::new([2, 3]).dot(SimdVector::new([-1, -2])), -8)
}
