
# Matty

Provides basic Vector and Matrix structs, which are generic over both the
numeric type in each cell, and the size (length/width).

## Feature flags

`remember_normalization` - If enabled, Vector structs will include a private
boolean that tracks whether they've already been normalized, to avoid repeating
work. This makes vectors slightly larger though, so the performance trade-off
has been exposed to library users.

`simd` - Uses Rust's `std::simd`, which is an unstable API, so the nightly
compiler is required. If enabled, a `simd::` module will be exposed which has
SIMD-accelerated versions of Vector and Matrix. These structs currently only
work for valid numbers of SIMD lanes (1, 2, 4, 8, etc).