// Exact constant-divisor reductions of DESIGN §3.5 and §3.6, usable from host C++ and
// CUDA device code (the CPU exhaustive test runs this exact code; cert_barrett and
// cert_crt_y_reduction in spec/params.py prove it).
#pragma once

#include <cstdint>

#if defined(__CUDACC__)
#  define IMD_HD __host__ __device__ __forceinline__
#else
#  define IMD_HD inline
#endif

namespace imd::device {

// floor(a * b / 2^64).
IMD_HD uint64_t umulhi64(uint64_t a, uint64_t b) {
#if defined(__CUDA_ARCH__)
  return __umul64hi(a, b);
#else
  __extension__ typedef unsigned __int128 u128;
  return static_cast<uint64_t>((static_cast<u128>(a) * b) >> 64);
#endif
}

// §3.5 epilogue: the non-negative residue v mod p for v in the certified reachable range
// [epi_vmin, epi_vmax]. p = 256: two's-complement mask; otherwise n = v + bias in [0, 2^32),
// q = umulhi64(magic, n) = n div p, v mod p = n - p q (all u32 / u64 words).
IMD_HD uint32_t mod_p_epilogue(int32_t v, uint32_t p, uint32_t bias, uint64_t magic) {
  if (p == 256u) return static_cast<uint32_t>(v) & 255u;
  const uint32_t n = static_cast<uint32_t>(v) + bias;
  const auto q = static_cast<uint32_t>(umulhi64(magic, n));
  return n - p * q;
}

// §3.6: n mod p for n in [0, 2^16) (n = q_l W_l), same magic, bias 0.
IMD_HD uint32_t mod_p_small(uint32_t n, uint32_t p, uint64_t magic) {
  if (p == 256u) return n & 255u;
  const auto q = static_cast<uint32_t>(umulhi64(magic, n));
  return n - p * q;
}

}  // namespace imd::device
