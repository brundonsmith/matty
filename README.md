
# Matty

Provides traits that allow regular 1D and 2D arrays to be used as Vectors and
Matrices with common operations. Optionally supports SIMD for accelerating
operations.

## Usage

```rust
use matty::*;

fn main() {
    let vec1 = [4.0, 5.0];
    let vec2 = [6.0, 7.0];
    let vec3 = vec1.add(vec2).normalized().dot([1.0, 0.0]);

    let vec = [1.0, 1.0, 1.0];
    let transformation = ArrayMatrix::<f32, 3, 3>::identity()
        .translate([1.0, 0.0, 0.0])
        .scale([2.0, 2.0, 2.0]);
    let transformed_vec = transformation.mul_vec(vec);
}
```

## Why not structs?

I started with Vector and Matrix structs, but quickly found myself deep in the
weeds of complex trait restrictions and lots of "passthrough" trait
implementations that just called out to the inner array's implementation. There
was twice as much code, and lots of unexposed array functionality.

With traits and raw arrays, you can do anything you would do with a regular
array, or even implement these traits for your own data structures. It also
makes polymorphism easier across Simd and non-Simd cases.

The only downside is that the standard math operators can't be overridden,
because you can only implement a trait for a type if either the trait or the
type is defined in your own crate. However, you could trivially wrap up the
arrays in a struct yourself and implement those traits if you wanted to :)

## Feature flags

`simd` - Uses Rust's `std::simd`, which is an unstable API, so the nightly
compiler is required. If enabled, a `simd::` module will be included which has
SIMD-accelerated versions of Vector and Matrix. These currently only work for
constructs whose sizes are valid numbers of SIMD lanes (1, 2, 4, 8, etc).