// Device copy of the generated constant tables (include/immerdink/detail/constants.hpp).
//
// Statically initialized __constant__ memory: no cudaMemcpyToSymbol, no workspace. Without
// relocatable device code every CUDA translation unit that includes this header gets its
// own (identical) copy, which is what the kernels of one TU read.
#pragma once

#include "immerdink/detail/constants.hpp"

namespace imd::device {

static __constant__ imd::constants::Tables c_tables = imd::constants::kTables;

// §3.5 exact non-negative v mod p_l for v in [epi_vmin[l], epi_vmax[l]] (cert_barrett):
// p = 256 by mask, otherwise bias + umulhi64.
__device__ __forceinline__ uint32_t mod_p(int32_t v, int l) {
  const uint32_t p = c_tables.moduli[l];
  if (p == 256u) return static_cast<uint32_t>(v) & 255u;
  const uint32_t n = static_cast<uint32_t>(v) + c_tables.epi_bias[l];  // in [0, 2^32)
  const auto q = static_cast<uint32_t>(__umul64hi(c_tables.epi_magic[l], n));
  return n - p * q;
}

}  // namespace imd::device
