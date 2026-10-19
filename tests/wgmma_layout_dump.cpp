// Machine-readable dump of the layouts the Hopper K4 design depends on (issue 17, docs/NOTES-17.md), read
// from CUTLASS's own CuTe tables (third_party/cutlass v4.8.0), so that a CUTLASS update that changes them
// is caught. Host code only; no GPU. Output (text, one record per line):
//   acc N t v idx        wgmma m64nNk32 s32 <- s8 x s8 accumulator: thread t (0..127), register v
//                        (0..N/2-1) holds element row = idx % 64, column = idx / 64 of the 64 x N tile
//   swz KIND r b off     shared-memory layout of an int8 K-major tile atom (KIND SW128: 8 rows x 128
//                        bytes, SW64: 8 rows x 64 bytes, SW32: 8 rows x 32 bytes): byte b of row r is
//                        at byte offset off of the atom
// CuTe defines the int8 atoms for N = 8, 16, 32, 64, 96, 128, 192, 256 (the PTX shape allows every multiple
// of 8; the layout formula is checked for all of them in tests/test_wgmma_layout.py).
// Compiled and read by tests/test_wgmma_layout.py.
#include <cstdio>

#include <cute/tensor.hpp>
#include <cute/atom/mma_atom.hpp>
#include <cute/arch/mma_sm90_gmma.hpp>
#include <cute/atom/mma_traits_sm90_gmma.hpp>

using namespace cute;

template <class Atom>
void dump(int n) {
  using T = MMA_Traits<Atom>;
  auto cl = typename T::CLayout{};
  const int nthr = size<0>(cl), nval = size<1>(cl);
  if (nthr != 128 || nval != n / 2) {
    std::printf("error acc %d threads %d values %d\n", n, nthr, nval);
    return;
  }
  for (int t = 0; t < nthr; ++t)
    for (int v = 0; v < nval; ++v) std::printf("acc %d %d %d %d\n", n, t, v, int(cl(t, v)));
}

template <class AtomLayout>
void dump_swz(const char* kind, int rows, int bytes) {
  auto l = AtomLayout{};
  // an (rows x bytes) K-major atom of int8: coordinate (row, k), k contiguous
  for (int r = 0; r < rows; ++r)
    for (int b = 0; b < bytes; ++b) std::printf("swz %s %d %d %d\n", kind, r, b, int(l(make_coord(r, b))));
}

int main() {
  dump<SM90_64x8x32_S32S8S8_SS_TN>(8);
  dump<SM90_64x16x32_S32S8S8_SS_TN>(16);
  dump<SM90_64x32x32_S32S8S8_SS_TN>(32);
  dump<SM90_64x64x32_S32S8S8_SS_TN>(64);
  dump<SM90_64x96x32_S32S8S8_SS_TN>(96);
  dump<SM90_64x128x32_S32S8S8_SS_TN>(128);
  dump<SM90_64x192x32_S32S8S8_SS_TN>(192);
  dump<SM90_64x256x32_S32S8S8_SS_TN>(256);
  dump_swz<GMMA::Layout_K_SW128_Atom<int8_t>>("SW128", 8, 128);
  dump_swz<GMMA::Layout_K_SW64_Atom<int8_t>>("SW64", 8, 64);
  dump_swz<GMMA::Layout_K_SW32_Atom<int8_t>>("SW32", 8, 32);
  return 0;
}
