// Device copy of the generated constant tables (include/immerdink/detail/constants.hpp).
//
// Statically initialized __constant__ memory: no cudaMemcpyToSymbol, no workspace. Without
// relocatable device code every CUDA translation unit that includes this header gets its
// own (identical) copy, which is what the kernels of one TU read.
#pragma once

#include "device/modp.hpp"
#include "device/residue.hpp"
#include "immerdink/detail/constants.hpp"

namespace imd::device {

static __constant__ imd::constants::Tables c_tables = imd::constants::kTables;
static __constant__ ByteWeights c_byte_weights = kByteWeights;  // derived from pow2_mod
static __constant__ Magic32 c_magic32 = kMagic32;                // derived from epi_magic

// §3.5 exact non-negative v mod p_l for v in [epi_vmin[l], epi_vmax[l]] (cert_barrett):
// p = 256 by mask, otherwise bias + umulhi64.
__device__ __forceinline__ uint32_t mod_p(int32_t v, int l) {
  return mod_p_epilogue(v, c_tables.moduli[l], c_tables.epi_bias[l], c_tables.epi_magic[l]);
}

}  // namespace imd::device
