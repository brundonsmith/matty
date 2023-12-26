
# Matty

Provides traits that turn regular 1D and 2D arrays into Vectors and Matrices

## Feature flags

`simd` - Uses Rust's `std::simd`, which is an unstable API, so the nightly
compiler is required. If enabled, a `simd::` module will be exposed which has
SIMD-accelerated versions of Vector and Matrix. These currently only work for
constructs whose sizes are valid numbers of SIMD lanes (1, 2, 4, 8, etc).