use num_traits::real::Real;
use std::{iter::Sum, ops::Index};

pub trait Vector<T, const N: usize>: Index<usize, Output = T> {
    fn dot(self, other: Self) -> T;

    fn cross(self, other: Self) -> Self;
}

pub trait Vector2<T>: Index<usize> {
    fn new(x: T, y: T) -> Self;

    fn x(&self) -> T;

    fn y(&self) -> T;
}

pub trait Vector3<T>: Index<usize> {
    fn new(x: T, y: T, z: T) -> Self;

    fn x(&self) -> T;

    fn y(&self) -> T;

    fn z(&self) -> T;
}

pub trait Vector2Swizzle<T>: Vector2<T> {
    fn xy(&self) -> Self;

    fn yx(&self) -> Self;
}

pub trait Vector3Swizzle<T, V2: Vector2<T>>: Vector3<T> {
    fn xy(&self) -> V2;

    fn yx(&self) -> V2;

    fn xz(&self) -> V2;

    fn zx(&self) -> V2;

    fn yz(&self) -> V2;

    fn zy(&self) -> V2;

    fn xyz(&self) -> Self;

    fn xzy(&self) -> Self;

    fn yzx(&self) -> Self;

    fn yxz(&self) -> Self;

    fn zxy(&self) -> Self;

    fn zyx(&self) -> Self;
}

pub trait RealVector<T: Real, const N: usize> {
    fn magnitude(self) -> T;

    fn normalized(self) -> Self;

    fn angle_to(self, other: Self) -> T;
}

pub trait VectorAdd<T, const N: usize> {
    fn add(self, other: Self) -> Self;

    fn add_scalar(self, scalar: T) -> Self;
}

pub trait VectorSub<T, const N: usize> {
    fn sub(self, other: Self) -> Self;

    fn sub_scalar(self, scalar: T) -> Self;
}

pub trait VectorMul<T, const N: usize> {
    fn mul(self, other: Self) -> Self;

    fn mul_scalar(self, scalar: T) -> Self;
}

pub trait VectorDiv<T, const N: usize> {
    fn div(self, other: Self) -> Self;

    fn div_scalar(self, scalar: T) -> Self;
}

/// Implementations for array

impl<
        T: Copy + Default + std::ops::Mul<Output = T> + std::ops::Sub<Output = T> + Sum<T>,
        const N: usize,
    > Vector<T, N> for [T; N]
{
    #[inline]
    fn dot(self, other: Self) -> T {
        self.mul(other).into_iter().sum()
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

impl<T: Copy> Vector2<T> for [T; 2] {
    #[inline]
    fn new(x: T, y: T) -> Self {
        [x, y]
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

impl<T: Copy> Vector3<T> for [T; 3] {
    #[inline]
    fn new(x: T, y: T, z: T) -> Self {
        [x, y, z]
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

impl<T: Copy, V: Vector2<T>> Vector2Swizzle<T> for V {
    #[inline]
    fn xy(&self) -> Self {
        V::new(self.x(), self.y())
    }

    #[inline]
    fn yx(&self) -> Self {
        V::new(self.y(), self.x())
    }
}

impl<T: Copy, V3: Vector3<T>, V2: Vector2<T>> Vector3Swizzle<T, V2> for V3 {
    #[inline]
    fn xy(&self) -> V2 {
        V2::new(self.x(), self.y())
    }

    #[inline]
    fn yx(&self) -> V2 {
        V2::new(self.y(), self.x())
    }

    #[inline]
    fn xz(&self) -> V2 {
        V2::new(self.x(), self.z())
    }

    #[inline]
    fn zx(&self) -> V2 {
        V2::new(self.z(), self.x())
    }

    #[inline]
    fn yz(&self) -> V2 {
        V2::new(self.y(), self.z())
    }

    #[inline]
    fn zy(&self) -> V2 {
        V2::new(self.z(), self.y())
    }

    #[inline]
    fn xyz(&self) -> Self {
        Self::new(self.x(), self.y(), self.z())
    }

    #[inline]
    fn xzy(&self) -> Self {
        Self::new(self.x(), self.z(), self.y())
    }

    #[inline]
    fn yzx(&self) -> Self {
        Self::new(self.y(), self.z(), self.x())
    }

    #[inline]
    fn yxz(&self) -> Self {
        Self::new(self.y(), self.x(), self.z())
    }

    #[inline]
    fn zxy(&self) -> Self {
        Self::new(self.z(), self.x(), self.y())
    }

    #[inline]
    fn zyx(&self) -> Self {
        Self::new(self.z(), self.y(), self.x())
    }
}

impl<
        T: Copy + Default + std::ops::Mul<Output = T> + std::ops::Sub<Output = T> + Sum<T> + Real,
        const N: usize,
    > RealVector<T, N> for [T; N]
{
    #[inline]
    fn magnitude(self) -> T {
        self.into_iter().map(|n| n * n).sum::<T>().sqrt()
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

macro_rules! array_op {
    ($vectortrait:ident, $optrait:ident, $fullmethod:ident, $scalarmethod:ident) => {
        impl<T: Copy + std::ops::$optrait<Output = T>, const N: usize> $vectortrait<T, N>
            for [T; N]
        {
            #[inline]
            fn $fullmethod(self, other: Self) -> Self {
                std::array::from_fn(|i| std::ops::$optrait::$fullmethod(self[i], other[i]))
            }

            #[inline]
            fn $scalarmethod(self, scalar: T) -> Self {
                std::array::from_fn(|i| std::ops::$optrait::$fullmethod(self[i], scalar))
            }
        }
    };
}

array_op!(VectorAdd, Add, add, add_scalar);
array_op!(VectorSub, Sub, sub, sub_scalar);
array_op!(VectorMul, Mul, mul, mul_scalar);
array_op!(VectorDiv, Div, div, div_scalar);

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
    assert_eq!([2, 7, 1].dot([8, 2, 8]), 38)
}

#[test]
fn cross() {
    assert_eq!([2, 3, 4].cross([5, 6, 7]), [-3, 6, -3])
}
