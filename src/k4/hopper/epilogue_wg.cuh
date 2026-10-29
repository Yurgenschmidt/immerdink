// The K4 register epilogue for wgmma accumulators (issue 17, docs/NOTES-17.md; design candidate for issue 22).
//
// The mod-p -> u8 epilogue of src/kernels/k4_gemm.cuh (epilogue_rows), with its accumulator indexing
// changed from the mma.sync order to the wgmma order and nothing else: the same byte packing, the same 4 x 4
// quad transpose, the same Barrett reduction, one 128-bit store per thread per row. A warpgroup runs
// wgmma m64nNk32, so the register v of thread (warp w, lane l = 4 g + q) of an N-column tile holds
//     row 16 w + g + 8 ((v / 2) % 2),  column 8 (v / 4) + 2 q + (v % 2)
// (tests/test_wgmma_layout.py checks this against CUTLASS's CuTe tables for every atom, and that this
// epilogue's data movement stores the right bytes). Per group of 64 columns the registers are
// acc[32 c + 4 nj + 2 h + e], c the group, nj the 8-column block in it, h the row half, e the element:
// the same (nj, h, e) order as the mma.sync fragment, so the epilogue body is unchanged.
//
// Not yet used by the library (no wgmma kernel exists before issue 22). tests/test_hopper_epilogue.cu runs it
// on any sm_80+ GPU with accumulators placed in wgmma order, and compares with K4Gemm's own epilogue.
#pragma once

#include <cstdint>

#include "device/constants.cuh"

namespace imd::kernels {

struct HopperEpilogue {
  // In-quad 4 x 4 transpose of 32-bit words (k4_gemm.cuh quad_transpose).
  __device__ static void quad_transpose(uint32_t (&w)[4], int t) {
#pragma unroll
    for (int s = 1; s <= 2; s <<= 1) {
      const bool hi = (t & s) != 0;
#pragma unroll
      for (int k = 0; k < 4; ++k) {
        if (k & s) continue;
        const uint32_t send = hi ? w[k] : w[k ^ s];
        const uint32_t recv = __shfl_xor_sync(0xffffffffu, send, s);
        if (hi)
          w[k] = recv;
        else
          w[k ^ s] = recv;
      }
    }
  }

  // Store-mod (kAdd = false) or add-mod (kAdd = true) of this thread's share of one 64 x kN wgmma tile.
  //   acc      the kN / 2 accumulator registers of this thread, wgmma order
  //   warp     warp index inside the warpgroup (0..3), lane  lane index
  //   row0     first row of the warpgroup's 64 rows, col0  first column of the tile
  //   plane    the modulus' u8 plane (row stride ldw), m / n  the output extent (rows beyond m and
  //            columns beyond n are not stored; n is a multiple of 16, as in K4)
  //   mod / neg_p / bias / magic  the §3.5 constants of the modulus (neg_p, bias, magic unused when p = 256)
  template <int kN, bool kAdd>
  __device__ static void run(const int32_t (&acc)[kN / 2], int warp, int lane, int row0, int col0,
                             uint8_t* plane, int64_t ldw, int m, int n, uint32_t mod, uint32_t bias,
                             uint64_t magic) {
    static_assert(kN % 64 == 0 && kN <= 256, "the epilogue works on 64-column groups");
    if (mod == 256u)
      body<kN, kAdd, true>(acc, warp, lane, row0, col0, plane, ldw, m, n, 0u, 0u, 0u);
    else
      body<kN, kAdd, false>(acc, warp, lane, row0, col0, plane, ldw, m, n, 0u - mod, bias, magic);
  }

 private:
  template <int kN, bool kAdd, bool k256>
  __device__ static void body(const int32_t (&acc)[kN / 2], int warp, int lane, int row0, int col0,
                              uint8_t* plane, int64_t ldw, int m, int n, uint32_t neg_p, uint32_t bias,
                              uint64_t magic) {
    const int g = lane >> 2, t = lane & 3;
#pragma unroll
    for (int c = 0; c < kN / 64; ++c) {
      const int col = col0 + 64 * c + 16 * t;  // this thread's 16 output bytes of the group
      const bool col_ok = col < n;
      uint4 old[2];
      if constexpr (kAdd) {
#pragma unroll
        for (int h = 0; h < 2; ++h) {
          const int row = row0 + 16 * warp + 8 * h + g;
          old[h] = make_uint4(0, 0, 0, 0);
          if (col_ok && row < m) old[h] = *reinterpret_cast<const uint4*>(plane + row * ldw + col);
        }
      }
#pragma unroll
      for (int h = 0; h < 2; ++h) {
        const int row = row0 + 16 * warp + 8 * h + g;
        int32_t v[8][2];
#pragma unroll
        for (int nj = 0; nj < 8; ++nj) {
          v[nj][0] = acc[32 * c + 4 * nj + 2 * h];
          v[nj][1] = acc[32 * c + 4 * nj + 2 * h + 1];
        }
        if constexpr (kAdd) {
          uint32_t w[4] = {__byte_perm(old[h].x, old[h].z, 0x5410), __byte_perm(old[h].x, old[h].z, 0x7632),
                           __byte_perm(old[h].y, old[h].w, 0x5410), __byte_perm(old[h].y, old[h].w, 0x7632)};
          quad_transpose(w, t);
#pragma unroll
          for (int nj = 0; nj < 8; ++nj) {
            const uint32_t word = w[nj >> 1] >> (16 * (nj & 1));
            v[nj][0] += static_cast<int32_t>(word & 255u);
            v[nj][1] += static_cast<int32_t>((word >> 8) & 255u);
          }
        }
        uint32_t w[4];
#pragma unroll
        for (int k = 0; k < 4; ++k) {
          uint32_t q[4];
#pragma unroll
          for (int e = 0; e < 4; ++e) {
            const int32_t x = v[2 * k + (e >> 1)][e & 1];
            if constexpr (k256)
              q[e] = static_cast<uint32_t>(x) & 255u;
            else
              q[e] = imd::device::mod_p_barrett_neg(x, neg_p, bias, magic);
          }
          w[k] = __byte_perm(__byte_perm(q[0], q[1], 0x0040), __byte_perm(q[2], q[3], 0x0040), 0x5410);
        }
        quad_transpose(w, t);
        if (col_ok && row < m) {
          const uint4 out = make_uint4(__byte_perm(w[0], w[1], 0x5410), __byte_perm(w[2], w[3], 0x5410),
                                       __byte_perm(w[0], w[1], 0x7632), __byte_perm(w[2], w[3], 0x7632));
          *reinterpret_cast<uint4*>(plane + row * ldw + col) = out;
        }
      }
    }
  }
};

}  // namespace imd::kernels
