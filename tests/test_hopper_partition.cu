// Host-side check of the register order the Hopper K4 fast epilogue assumes, for the REAL TiledMma of every variant
// (issue 17, docs/NOTES-17.md; reviewer item). No GPU is used: CuTe's partition_C is evaluated on the host.
//
// For each variant of src/k4/hopper/k4_hopper_variants.def and every thread t of its TiledMma, the accumulator register i must hold
// the element K4ModEpilogue / HopperEpilogue assume:
//   rep = i / (N / 2), v = i % (N / 2), wg = t / 128, warp = (t % 128) / 32, lane = t % 32
//   row = 64 (wg + kWG rep) + 16 warp + lane / 4 + 8 ((v / 2) % 2),   column = 8 (v / 4) + 2 (lane % 4) + v % 2
// with kWG = (rows of the TiledMma) / 64 and kRep = registers / (N / 2). Every element of the CTA tile must be held exactly once.
#include <cstdio>
#include <set>
#include <utility>

#include "k4/hopper/k4_hopper.cuh"

namespace {
using namespace imd::hopper;
using namespace cute;
namespace cg = cutlass::gemm;
using Coop = cg::KernelTmaWarpSpecializedCooperative;
using Ping = cg::KernelTmaWarpSpecializedPingpong;
using _1 = Int<1>;
using _2 = Int<2>;
int failures = 0, checked = 0;

template <class Tile, class Cluster, class Sched>
void check(const char* name) {
  using V = Variant<Tile, Cluster, Sched, false, false>;
  using Mma = typename V::Main::TiledMma;
  constexpr int TM = int(size<0>(Tile{})), TN = int(size<1>(Tile{}));
  const int threads = int(size(Mma{}));
  const int kWG = int(tile_size<0>(Mma{})) / 64;
  Tensor cD = make_identity_tensor(make_shape(Int<TM>{}, Int<TN>{}));
  std::set<std::pair<int, int>> seen;
  for (int t = 0; t < threads; ++t) {
    auto thr = Mma{}.get_thread_slice(t);
    Tensor tC = thr.partition_C(cD);
    const int regs = int(size(tC));
    const int kRep = regs / (TN / 2);
    if (kRep * (TN / 2) != regs || 64 * kWG * kRep != TM) {
      std::printf("FAIL %s: %d registers, kRep %d, kWG %d do not tile %d x %d\n", name, regs, kRep, kWG, TM, TN);
      ++failures;
      return;
    }
    const int wg = t / 128, warp = (t % 128) / 32, lane = t % 32;
    for (int i = 0; i < regs; ++i) {
      const int rep = i / (TN / 2), v = i % (TN / 2);
      const int row = 64 * (wg + kWG * rep) + 16 * warp + lane / 4 + 8 * ((v / 2) % 2);
      const int col = 8 * (v / 4) + 2 * (lane % 4) + (v % 2);
      const int gr = int(get<0>(tC(i))), gc = int(get<1>(tC(i)));
      ++checked;
      if (gr != row || gc != col) {
        std::printf("FAIL %s: thread %d register %d is (%d, %d), the epilogue assumes (%d, %d)\n", name, t, i, gr, gc, row, col);
        ++failures;
        return;
      }
      seen.insert({row, col});
    }
  }
  if (int(seen.size()) != TM * TN) {
    std::printf("FAIL %s: %zu distinct elements for a %d x %d tile\n", name, seen.size(), TM, TN);
    ++failures;
  }
}
}  // namespace

int main() {
#define K4H_VARIANT(NAME, TILE_M, TILE_N, TILE_K, CLM, CLN, SCHED, ADD, FAST) \
  if (FAST) check<Shape<Int<TILE_M>, Int<TILE_N>, Int<TILE_K>>, Shape<Int<CLM>, Int<CLN>, _1>, SCHED>(#NAME);
#include "k4/hopper/k4_hopper_variants.def"
#undef K4H_VARIANT
  if (failures) {
    std::printf("%d variant(s) differ from the assumed register order\n", failures);
    return 1;
  }
  std::printf("PASS  %d registers of the fast-epilogue variants sit where the epilogue assumes\n", checked);
  return 0;
}
