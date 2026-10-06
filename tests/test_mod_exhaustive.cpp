// PLAN M1: the modular reductions, verified EXHAUSTIVELY on the CPU against the oracle's
// definition (spec/oracle.py: the non-negative residue, Python's `%`), running the very
// code the kernels run (src/device/modp.hpp is host/device):
//   §3.5 epilogue  mod_p_epilogue(v, p) for EVERY reachable int32 accumulator value
//                  v in [epi_vmin, epi_vmax] of every modulus (6.4e10 values);
//   §3.6 CRT step  mod_p_small(n, p) for every n in [0, 2^16) and every modulus.
// The reference is incremental (r = v mod p is advanced with v), so no division is
// involved. Work is split over all hardware threads.
#include <algorithm>
#include <atomic>
#include <cstdint>
#include <cstdio>
#include <thread>
#include <vector>

#include "device/modp.hpp"
#include "device/residue.hpp"
#include "immerdink/detail/constants.hpp"

namespace {

namespace c = imd::constants;
using imd::device::mod_p_epilogue;
using imd::device::mod_p_small;

std::atomic<uint64_t> g_bad{0}, g_done{0};

// v in [lo, hi] for modulus l; returns mismatches.
uint64_t check_range(int l, int64_t lo, int64_t hi) {
  const uint32_t p = c::kTables.moduli[l], bias = c::kTables.epi_bias[l];
  const uint64_t magic = c::kTables.epi_magic[l];
  int64_t r = lo % static_cast<int64_t>(p);  // the oracle's non-negative residue of lo
  if (r < 0) r += p;
  auto ref = static_cast<uint32_t>(r);
  uint64_t bad = 0;
  for (int64_t v = lo; v <= hi; ++v) {
    bad += mod_p_epilogue(static_cast<int32_t>(v), p, bias, magic) != ref;
    if (++ref == p) ref = 0;
  }
  return bad;
}

}  // namespace

int main() {
  // §3.6: n = q_l W_l < 2^16 (exhaustive, every modulus).
  uint64_t small_bad = 0;
  for (int l = 0; l < c::kNMax; ++l) {
    const uint32_t p = c::kTables.moduli[l];
    for (uint32_t n = 0; n < (1u << 16); ++n)
      small_bad += mod_p_small(n, p, c::kTables.epi_magic[l]) != n % p;
  }
  for (int l = 0; l < c::kNMax; ++l) {  // the 32-bit-magic variant used by K5 (M4)
    const uint32_t p = c::kTables.moduli[l];
    for (uint32_t n = 0; n < (1u << 16); ++n)
      small_bad += imd::device::mod_p_small32(n, p, imd::device::kMagic32.m[l]) != n % p;
  }
  std::printf("CRT step (q W) mod p: 2 x 20 x 65536 values, %llu mismatches\n",
              static_cast<unsigned long long>(small_bad));

  // §3.5: every reachable accumulator of every modulus, split into work items.
  struct Item {
    int l;
    int64_t lo, hi;
  };
  std::vector<Item> items;
  constexpr int64_t kSlice = int64_t{1} << 26;
  uint64_t total = 0;
  for (int l = 0; l < c::kNMax; ++l) {
    const int64_t lo = c::kTables.epi_vmin[l], hi = c::kTables.epi_vmax[l];
    total += static_cast<uint64_t>(hi - lo + 1);
    for (int64_t a = lo; a <= hi; a += kSlice) items.push_back({l, a, std::min(hi, a + kSlice - 1)});
  }
  std::atomic<size_t> next{0};
  const unsigned nthreads = std::max(1u, std::thread::hardware_concurrency());
  std::vector<std::thread> pool;
  for (unsigned t = 0; t < nthreads; ++t)
    pool.emplace_back([&] {
      for (size_t i; (i = next.fetch_add(1)) < items.size();) {
        g_bad += check_range(items[i].l, items[i].lo, items[i].hi);
        g_done += static_cast<uint64_t>(items[i].hi - items[i].lo + 1);
      }
    });
  for (auto& th : pool) th.join();
  std::printf("epilogue v mod p: %llu reachable values of 20 moduli (%u threads), %llu mismatches\n",
              static_cast<unsigned long long>(g_done.load()), nthreads,
              static_cast<unsigned long long>(g_bad.load()));
  // §3.4 pack residue by byte weights (mod_p_bytes, M4): (1) every byte weight equals
  // 2^(8i) mod p (computed here by repeated doubling); (2) the reduction of the weighted
  // byte sum, exhaustively over its whole range [0, 7 * 255^2]; (3) end to end against
  // mu % p on edge values and 2^20 pseudo-random mu < 2^54 per modulus; (v1.14 §3.4.1) the
  // pack weights and mod_p_words32 on X < 2^78 (cert_pack_words).
  uint64_t bytes_bad = 0;
  {
    const imd::device::ByteWeights& bw = imd::device::kByteWeights;
    for (int l = 0; l < c::kNMax; ++l) {
      const uint32_t p = c::kTables.moduli[l];
      const uint64_t magic = c::kTables.epi_magic[l];
      uint32_t pw = 1 % p;
      for (int i = 0; i < 8; ++i) {
        const uint32_t w = i < 4 ? (bw.lo[l] >> (8 * i)) & 255u
                                 : (i < 7 ? (bw.hi[l] >> (8 * (i - 4))) & 255u : 0u);
        bytes_bad += i < 7 ? w != pw : w != 0;
        for (int d = 0; d < 8; ++d) pw = (2 * pw) % p;
      }
      if (p != 256u)
        for (uint32_t s = 0; s <= 7u * 255u * 255u; ++s)
          bytes_bad += s - p * static_cast<uint32_t>(imd::device::umulhi64(magic, s)) != s % p;
      // A5: the 32-bit magic of mod_p_bytes32, every p (2^24 for 256); v1.14 §3.4.1: over
      // [0, 12 * 255^2], the range of mod_p_words32's sum (W <= 3 dp4a).
      const uint32_t m32 = imd::device::kMagic32.m[l];
      for (uint32_t s = 0; s <= 12u * 255u * 255u; ++s)
        bytes_bad += s - p * static_cast<uint32_t>((uint64_t{s} * m32) >> 32) != s % p;
      // v1.14: the pack weights, bytes 0..11 (2^(8i) mod p for i < kPackWeightBytes, else 0).
      {
        uint32_t pw8 = 1 % p;
        for (int i = 0; i < 12; ++i) {
          const uint32_t w = (imd::device::kPackWeights.w[i / 4][l] >> (8 * (i % 4))) & 255u;
          bytes_bad += i < c::kPackWeightBytes ? w != pw8 : w != 0;
          for (int d = 0; d < 8; ++d) pw8 = (2 * pw8) % p;
        }
      }
      // v1.14: mod_p_words32<2> on X < 2^kPackXBits2 and <3> on X < 2^kPackXBits3 against
      // X mod p (u128), edge values and 2^20 pseudo-random X of every bit length.
      {
        __extension__ typedef unsigned __int128 u128;
        const uint32_t(&pwt)[3][c::kNMax] = imd::device::kPackWeights.w;
        const uint32_t w2[2] = {pwt[0][l], pwt[1][l]}, w3[3] = {pwt[0][l], pwt[1][l], pwt[2][l]};
        uint64_t y = 0xD1B54A32D192ED03ull ^ static_cast<uint64_t>(l);
        const u128 one = 1;
        const u128 edges[] = {0, 1, 255, 256, (one << 32) - 1, one << 32, (one << 53) - 1,
                              one << 53, (one << 53) + 1, (one << 63) - 1, one << 63, one << 64,
                              (one << 64) + 1, (one << 77), (one << 78) - 1};
        for (int t = 0; t < (1 << 20) + 15; ++t) {
          u128 X;
          if (t < 15) {
            X = edges[t];
          } else {
            y ^= y << 13, y ^= y >> 7, y ^= y << 17;
            const uint64_t z = y * 0x9E3779B97F4A7C15ull;
            X = ((static_cast<u128>(z) << 64) | y) >> (128 - 1 - (y % c::kPackXBits3));  // < 2^78
          }
          const uint32_t x3[3] = {static_cast<uint32_t>(X), static_cast<uint32_t>(X >> 32),
                                  static_cast<uint32_t>(X >> 64)};
          bytes_bad += imd::device::mod_p_words32<3>(x3, w3, p, m32) != static_cast<uint32_t>(X % p);
          if (X >> c::kPackXBits2 == 0) {
            const uint32_t x2[2] = {x3[0], x3[1]};
            bytes_bad += imd::device::mod_p_words32<2>(x2, w2, p, m32) != static_cast<uint32_t>(X % p);
          }
        }
      }
      uint64_t x = 0x9E3779B97F4A7C15ull ^ static_cast<uint64_t>(l);
      const uint64_t edges[] = {0, 1, 255, 256, 0xFFFFFFFFull, 1ull << 32, (1ull << 53) - 1,
                                1ull << 53, (1ull << 54) - 1};
      for (int t = 0; t < (1 << 20) + 9; ++t) {
        uint64_t mu;
        if (t < 9) {
          mu = edges[t];
        } else {
          x ^= x << 13, x ^= x >> 7, x ^= x << 17;
          mu = x >> (10 + (x & 31));  // < 2^54, spread over bit lengths
        }
        const uint32_t r = imd::device::mod_p_bytes(static_cast<uint32_t>(mu),
                                                    static_cast<uint32_t>(mu >> 32), bw.lo[l],
                                                    bw.hi[l], p, magic);
        bytes_bad += r != mu % p;
        bytes_bad += imd::device::mod_p_bytes32(static_cast<uint32_t>(mu),
                                                static_cast<uint32_t>(mu >> 32), bw.lo[l],
                                                bw.hi[l], p, m32) != mu % p;
      }
    }
  }
  std::printf("pack residue by byte weights: weights (A5 and v1.14), reduction over [0, 12*255^2], "
              "2^20 mu and 2^20 X (mod_p_words32<2>, <3>) per modulus: %llu mismatches\n",
              static_cast<unsigned long long>(bytes_bad));
  const bool ok = small_bad == 0 && g_bad == 0 && g_done == total && bytes_bad == 0;
  std::printf("%s\n", ok ? "PASS" : "FAIL");
  return ok ? 0 : 1;
}
