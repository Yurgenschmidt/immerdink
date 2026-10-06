// The performance model (DESIGN v1.8 / v1.10 §6c): features of every phase from the plan's
// lowering and the shared launch rules, the calibration file (formats 1 and 2),
// imd_perf_predict, imd_perf_predict_pack, imd_perf_rank. Pure: no allocation visible to the
// caller, thread-safe, and no global or static mutable state but one, test-only: the
// K-alignment hook (k_hat.cpp imd_test_set_k_align, DESIGN v1.13 §4b.5), read once per API
// call (one KAlignSel snapshot, passed to every analysis of the call). The shipped
// calibration is parsed once into an immutable function-local static.
#include "perf.hpp"

#include <algorithm>
#include <charconv>
#include <cstddef>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <initializer_list>
#include <limits>
#include <new>
#include <string>
#include <string_view>
#include <vector>

#include "accuracy.hpp"
#include "analysis.hpp"
#include "immerdink/detail/constants.hpp"
#include "immerdink/immerdink_host.h"

namespace imd::detail {
namespace {

namespace c = imd::constants;

// The shipped calibration (src/host/calib/rtx4090_sm89.calib), embedded by CMake as
// kPerfCalibDefault (a raw string literal): the file is the one source.
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Woverlength-strings"
#include "perf_calib_default.inc"
#pragma GCC diagnostic pop

constexpr int kP = IMD_PERF_MAX_PARAMS;
constexpr int kK4ByteSlot = 4;  // K4 classes: t = max(sum_{j != 4} th_j x_j, th_4 x_4)
constexpr int kK4TailSlot = 5;  // K4: the MACs of chunks with a partial 64-wide K tile
constexpr int64_t kK4TileK = 64;    // format 1's partial-K-tile criterion
constexpr int64_t kK4RowAlign = 32; // format 2's: residue rows of k_hat bytes, 32-byte sectors

const char* const kClassNames[IMD_PERF_CLASS_COUNT] = {
    "k0", "kf", "kl", "pack", "k4_128x256", "k4_128x128", "k4_64x128", "k4_64x64",
    "k5", "direct", "beta", "call"};
// Feature names per class (nullptr: the class does not use that slot).
const char* const kParamNames[IMD_PERF_CLASS_COUNT][kP] = {
    {"launch", "byte", "chain", nullptr, nullptr, nullptr},          // k0
    {"launch", "byte", "op", "row", "op_eps", "chain"},              // kf
    {"launch", "byte", "op", "l2byte", "op_eps", "wave"},            // kl
    {"launch", "byte", "op", "op_eps", "chain", "wave"},             // pack
    {"round", "mac1", nullptr, nullptr, "byte", "mac_ktail"},        // k4_128x256 (1 CTA per SM)
    {"round", "mac1", "mac2", nullptr, "byte", "mac_ktail"},         // k4_128x128 (2)
    {"round", "mac1", "mac2", nullptr, "byte", "mac_ktail"},         // k4_64x128  (2)
    {"round", "mac1", "mac2", "mac3", "byte", "mac_ktail"},          // k4_64x64   (3)
    {"launch", "byte", "op", "chain", "dram", nullptr},              // k5
    {"launch", "byte", "flop", "chain", nullptr, nullptr},           // direct
    {"launch", "byte", "chain", nullptr, nullptr, nullptr},          // beta
    {"call", "launch", nullptr, nullptr, nullptr, nullptr},          // call (signed: a correction)
};
// v1.10: the parameters new in format 2 (a format-1 file has none of them; they are 0 there).
constexpr bool kFormat2Param[IMD_PERF_CLASS_COUNT][kP] = {
    {false, false, true, false, false, false},   // k0.chain
    {},                                          // kf (kf.chain is format 1)
    {false, false, false, false, false, true},   // kl.wave
    {false, false, false, false, true, true},    // pack.chain, pack.wave
    {}, {}, {}, {},
    {false, false, false, true, true, false},    // k5.chain, k5.dram
    {false, false, false, true, false, false},   // direct.chain
    {false, false, true, false, false, false},   // beta.chain
    {}};
// v1.10 (§6c.1 "Clock regimes"): the slots whose time scales with the SM clock (ops, chains,
// L2 traffic, K4's round model); DRAM bytes, launches and the call term do not.
constexpr bool kClockBound[IMD_PERF_CLASS_COUNT][kP] = {
    {false, false, true, false, false, false},   // k0: chain
    {false, false, true, true, true, true},      // kf: op, row, op_eps, chain
    {false, false, true, true, true, false},     // kl: op, l2byte, op_eps (not wave: memory)
    {false, false, true, true, true, false},     // pack: op, op_eps, chain (not wave: memory)
    {true, true, true, true, false, true},       // k4: the round model, not the DRAM term
    {true, true, true, true, false, true},
    {true, true, true, true, false, true},
    {true, true, true, true, false, true},
    {false, false, true, true, false, false},    // k5: op, chain
    {false, false, true, true, false, false},    // direct: flop, chain
    {false, false, true, false, false, false},   // beta: chain
    {false, false, false, false, false, false}}; // call
const char* const kPhaseNames[IMD_PERF_PHASE_COUNT] = {
    "K0 vec", "KF fused", "KL A", "KL B", "K2 pack A", "K3 pack B", "K4 gemm", "K5 crt",
    "DIRECT", "K5 beta-only", "call", "pack prep", "pack KL"};

static_assert(IMD_PERF_CLASS_COUNT * kP <= IMD_PERF_PARAM_CAPACITY, "param slots");
static_assert(IMD_PERF_CLASS_COUNT <= IMD_PERF_CLASS_CAPACITY, "band rows");
static_assert(IMD_PERF_KERNEL_COUNT <= IMD_PERF_KERNEL_CAPACITY, "kernel rows");
static_assert(IMD_PERF_KERNEL_K5 + 4 * (c::kNMax - c::kNMin + 1) == IMD_PERF_KERNEL_COUNT,
              "K5 kernels: 4 variants per N");
static_assert(IMD_PERF_CLASS_K4_128x256 + IMD_K4_64x64 == IMD_PERF_CLASS_K4_64x64,
              "K4 classes follow the IMD_K4_* order");

// Version-6 struct sizes (the v7 fields are appended; DESIGN §6c.3).
constexpr uint32_t kCalibV1Size = offsetof(imd_perf_calib_t, kernel);
constexpr uint32_t kCalibV7Size = offsetof(imd_perf_calib_t, energy_byte_f_pj);  // v7..v10
constexpr uint32_t kCalibV11Size = sizeof(imd_perf_calib_t);
constexpr uint32_t kRequestV1Size = offsetof(imd_perf_request_t, regime);
constexpr uint32_t kPerfV1Size = offsetof(imd_perf_t, regime);
constexpr uint32_t kRankRequestV1Size = offsetof(imd_perf_rank_request_t, regime);
constexpr uint32_t kRankRequestV7Size = offsetof(imd_perf_rank_request_t, accuracy);
constexpr uint32_t kRankRequestV8Size = sizeof(imd_perf_rank_request_t);
constexpr uint32_t kRequestV7Size = offsetof(imd_perf_request_t, accuracy);
constexpr uint32_t kRequestV8Size = sizeof(imd_perf_request_t);
constexpr uint32_t kCandidateV1Size = offsetof(imd_perf_candidate_t, time_lo_s);
constexpr uint32_t kCandidateV7Size = offsetof(imd_perf_candidate_t, err_bound);
constexpr uint32_t kPerfV7Size = offsetof(imd_perf_t, err_bound);
static_assert(sizeof(imd_perf_phase_t) == 96, "imd_perf_phase_t v1 layout changed");
static_assert(kCalibV1Size == 1232, "imd_perf_calib_t v1 layout changed");
static_assert(kPerfV1Size == 1648, "imd_perf_t v1 layout changed");
static_assert(kCandidateV1Size == 128, "imd_perf_candidate_t v1 layout changed");
static_assert(kRankRequestV1Size == 64, "imd_perf_rank_request_t v1 layout changed");
static_assert(kRequestV1Size == 24, "imd_perf_request_t v1 layout changed");
static_assert(kCalibV7Size == 1232 + 16 * 128 + 8 * 10 + 8 * 6 + 8 * 32 + 8, "v7 calib");
static_assert(kCalibV11Size == kCalibV7Size + 8, "v11 calib");
static_assert(kRequestV7Size == 40, "v7 request");
static_assert(kPerfV7Size == 1648 + 8 + 8 * 5 + 16 * 16, "v7 prediction");
static_assert(kRankRequestV7Size == 88, "v7 rank request");
static_assert(kCandidateV7Size == 168, "v7 candidate");
// v8 (DESIGN §6f): appended accuracy pointer / error fields only.
static_assert(kRequestV8Size == 48 && kRankRequestV8Size == 96, "v8 requests");
static_assert(sizeof(imd_perf_t) == kPerfV7Size + 24, "v8 prediction");
static_assert(sizeof(imd_perf_candidate_t) == 192, "v8 candidate");
static_assert(sizeof(imd_err_request_t) == 64 && sizeof(imd_err_t) == 72, "v8 error structs");

template <class F>
imd_status_t guarded(F&& f) {
  try {
    return f();
  } catch (const std::bad_alloc&) {
    return IMD_STATUS_ALLOC_FAILED;
  } catch (...) {
    return IMD_STATUS_INVALID_VALUE;
  }
}

// ---- kernels (v1.10 occupancy table) --------------------------------------------------------

// The kernel names (imd_perf_kernel_name, the calibration's kernel.<name> keys), built once
// into a constant table at compile time is awkward for the K5 rows, so they are formatted on
// demand from the index; this table holds the fixed ones.
const char* const kKernelFixed[IMD_PERF_KERNEL_K5] = {
    "k0", "kf16", "kf4", "kl8", "kl4", "pack", "zero_rows", "direct", "beta", "pack_prep",
    "k4_128x256", "k4_128x128", "k4_64x128", "k4_64x64"};

// "k5_n<N>_e<E>" / "k5b_n<N>" for N = 2..20: static storage, one string per kernel.
struct K5Names {
  char s[IMD_PERF_KERNEL_COUNT - IMD_PERF_KERNEL_K5][16];
  constexpr K5Names() : s{} {
    for (int i = 0; i < IMD_PERF_KERNEL_COUNT - IMD_PERF_KERNEL_K5; ++i) {
      const int N = c::kNMin + i / 4, v = i % 4;
      char* p = s[i];
      int q = 0;
      const char* pre = v == 3 ? "k5b_n" : "k5_n";
      for (int t = 0; pre[t] != '\0'; ++t) p[q++] = pre[t];
      if (N >= 10) p[q++] = static_cast<char>('0' + N / 10);
      p[q++] = static_cast<char>('0' + N % 10);
      if (v != 3) {
        p[q++] = '_', p[q++] = 'e';
        p[q++] = static_cast<char>('0' + (v == 0 ? 1 : v == 1 ? 2 : 4));
      }
      p[q] = '\0';
    }
  }
};
constexpr K5Names kK5Names{};

const char* kernel_name(int32_t k) {
  if (k < 0 || k >= IMD_PERF_KERNEL_COUNT) return nullptr;
  return k < IMD_PERF_KERNEL_K5 ? kKernelFixed[k] : kK5Names.s[k - IMD_PERF_KERNEL_K5];
}

int32_t k5_kernel(int N, int variant) { return IMD_PERF_KERNEL_K5 + 4 * (N - c::kNMin) + variant; }

// The built-in occupancy (a format-1 calibration, DESIGN §6c.2): KF as fitted in v1.8 (E = 16:
// 2, E = 4: 4), KL by its launch bounds, the 256-thread tile kernels by their ~34 KB of shared
// memory (2), the plain grid-stride kernels 6, K4 the configuration table's.
imd_perf_kernel_t builtin_kernel(int32_t k) {
  imd_perf_kernel_t r{2, 0, 0, kLaunchThreads};
  switch (k) {
    case IMD_PERF_KERNEL_KF16: r.resident = 2; break;
    case IMD_PERF_KERNEL_KF4: r.resident = 4; break;
    case IMD_PERF_KERNEL_KL8: r.resident = 2, r.threads = 512; break;
    case IMD_PERF_KERNEL_KL4: r.resident = 1, r.threads = 1024; break;
    case IMD_PERF_KERNEL_PACK: r.resident = 2; break;
    case IMD_PERF_KERNEL_K0: case IMD_PERF_KERNEL_ZERO_ROWS: case IMD_PERF_KERNEL_DIRECT:
    case IMD_PERF_KERNEL_BETA: case IMD_PERF_KERNEL_PACK_PREP: r.resident = 6; break;
    default:
      if (k >= IMD_PERF_KERNEL_K4 && k < IMD_PERF_KERNEL_K5) {
        const K4ConfigDesc& d = kK4Configs[k - IMD_PERF_KERNEL_K4];
        r.resident = d.ctas_per_sm, r.threads = k4_threads(d);
      } else {
        r.resident = 6;  // K5: 256 threads, small shared memory
      }
  }
  return r;
}

// The v7 fields of a format-1 calibration (DESIGN §6c.2 migration): built-in occupancy, no
// clock or power model, no cache term, bands 1 (zero width). Predictions equal host API 6's.
void migrate_v7_fields(imd_perf_calib_t* cal) {
  for (int k = 0; k < IMD_PERF_KERNEL_CAPACITY; ++k)
    cal->kernel[k] = k < IMD_PERF_KERNEL_COUNT ? builtin_kernel(k) : imd_perf_kernel_t{0, 0, 0, 0};
  cal->clock_eff_mhz = cal->clock_min_mhz = cal->power_cap_w = cal->power_idle_w = 0;
  cal->energy_tc_pj = cal->energy_alu_pj = cal->energy_byte_pj = cal->energy_byte_f_pj = 0;
  cal->energy_gamma = 2;
  cal->warm_l2_bytes = cal->warm_l2byte_ns = 0;
  cal->tie_ratio = 1;
  for (auto& b : cal->band_call) b[0] = b[1] = 1;
  for (auto& b : cal->band_class) b[0] = b[1] = 1;
}

// A true format-1 calibration (DESIGN §6c.2): format 1, every format-2 parameter 0 (format 1
// has none: a format-2 parameter kept with format-1 features would be an unvalidated hybrid).
// The v7 fields (if the struct has them) are migrated by the caller.
void to_format1_params(imd_perf_calib_t* cal) {
  cal->format = 1;
  for (int cl = 0; cl < IMD_PERF_CLASS_COUNT; ++cl)
    for (int j = 0; j < kP; ++j)
      if (kFormat2Param[cl][j]) cal->param[cl * kP + j] = 0;
}

// ---- features -----------------------------------------------------------------------------

PerfPhase& add_phase(PerfPhases* P, int32_t phase, int32_t cls, int64_t launches,
                     std::initializer_list<double> x, double bytes, double ops = 0, int peak = 0) {
  PerfPhase& p = P->ph[P->n++];
  p.phase = phase, p.cls = cls, p.launches = launches;
  int j = 0;
  for (const double v : x) p.x[j++] = v;
  p.bytes = bytes, p.ops = ops, p.peak = peak;
  return p;
}

// The eps-path part of feature `slot`: x[slot] += f * ops (f = the call's eps fraction).
PerfPhase& set_eps(PerfPhase& p, int slot, double ops, double f) {
  p.eps_x[slot] = ops;
  p.x[slot] += f * ops;
  return p;
}

double dbl(int64_t v) { return static_cast<double>(v); }

// v1.10 (DESIGN §6c.1 "Waves and occupancy"): the busiest SM's per-thread chain of one launch of
// grid_x x grid_y CTAs (resident r per SM) over `units` work units distributed grid-stride over
// grid_x, u serial steps per unit: ceil(B / (S r)) ceil(units / grid_x) u.
double chain_of(int64_t grid_x, int64_t grid_y, int64_t units, double u, int32_t resident,
                int32_t sms) {
  if (units <= 0 || grid_x <= 0) return 0;
  const int64_t per_sm = int64_t{sms} * std::max<int32_t>(resident, 1);
  const int64_t B = grid_x * grid_y;
  const int64_t waves = (B + per_sm - 1) / per_sm;
  const int64_t per_block = (units + grid_x - 1) / grid_x;
  return dbl(waves) * dbl(per_block) * u;
}

int32_t resident_of(const imd_perf_calib_t& cal, int32_t k) { return cal.kernel[k].resident; }

// A plain grid-stride kernel of 256-thread blocks over `outputs` items, u steps per item.
double grid_chain(const imd_perf_calib_t& cal, int32_t kernel, int64_t outputs, double u) {
  const int64_t units = (outputs + kLaunchThreads - 1) / kLaunchThreads;
  return chain_of(launch_blocks(outputs), 1, units, u, resident_of(cal, kernel), cal.sm_count);
}

// KF's per-thread chain on the busiest SM (DESIGN §6c.1): its launch (kf_* rules, shared with
// contract.cu) has B = tiles x parts blocks; each thread holds E values of a tile and emits the
// residues of ceil(g / parts) moduli for each of them in sequence. The eps factor (1 + f) is
// applied by the caller (set_eps).
double kf_chain(const ExecPlan& e, const imd_perf_calib_t& cal) {
  const int32_t sms = cal.sm_count;
  const int lg_tk = kf_lg_tk(e.k_hat);
  const int gs = e.g;
  const auto tiles_for = [&](int lg_te) {
    int64_t t = 0;
    for (int x = 0; x < 2; ++x)
      if (!(e.packed & (1 << x)))
        t += kf_operand_tiles(lg_te, lg_tk, e.b, x == 0 ? e.m : e.n, e.pack_radix[x]);
    return t;
  };
  int64_t tiles = tiles_for(12);
  const bool small = kf_small(tiles, gs, sms);
  if (small) tiles = tiles_for(10);
  if (tiles <= 0) return 0;
  const int64_t parts = pack_moduli_parts(tiles, gs, sms);
  const int32_t r = resident_of(cal, small ? IMD_PERF_KERNEL_KF4 : IMD_PERF_KERNEL_KF16);
  return chain_of(launch_blocks(tiles, 1), parts, tiles, (small ? 4.0 : 16.0) * dbl((gs + parts - 1) / parts),
                  r, sms);
}

double u128_to_double(u128 v) { return static_cast<double>(v); }

// KL of one operand (rows, padded rows, g moduli of group 1, KL tile): launch, DRAM bytes
// (read the operand once, write group 1's residues, sigma and the flag), residue ops, the
// L2 re-read of rows longer than two tiles (passes 2 and 3 re-read the middle chunks), the
// residue ops in the eps > 0 regime, and (v1.10) the chain of its launch (kl_launch: rows per
// batch rows_b, radix; lg_rows >= 0 for imd_pack's tile).
void add_kl(PerfPhases* P, const imd_perf_calib_t& cal, int32_t phase, double rows, double pad,
            double k, double k_hat, double g, bool along_rows, int64_t k_hat_i, double f,
            int32_t lg_rows, int64_t b, int64_t rows_b, int64_t radix) {
  const double tk = static_cast<double>(int64_t{1} << kl_tile_lg_k(along_rows, k_hat_i));
  const double bytes = 8 * rows * k + g * pad * k_hat + 3 * rows;
  const double l2 = 16 * rows * std::max(0.0, k_hat - 2 * tk);
  const KlLaunch L = kl_launch(along_rows, lg_rows, k_hat_i, b, rows_b, radix);
  // v1.10: the busiest SM's tile-chunk loads in sequence (memory latency of a small grid; a
  // compute chain nc x values x g fitted to 0 on the v1.8 data and was replaced by it).
  const double wave =
      chain_of(L.grid, 1, L.tiles, dbl(L.nc),
               resident_of(cal, L.values == 4 ? IMD_PERF_KERNEL_KL4 : IMD_PERF_KERNEL_KL8),
               cal.sm_count);
  PerfPhase& p = add_phase(P, phase, IMD_PERF_CLASS_KL, 1, {1, bytes, g * rows * k, l2, 0, wave},
                           bytes);
  set_eps(p, 4, g * rows * k, f);
  p.in_bytes = 8 * rows * k;
}

}  // namespace

double perf_log2_T(int N) {
  double t = 0;
  for (int w = c::kMaxLimbs - 1; w >= 0; --w) t = t * 4294967296.0 + c::kTables.T[N][w];
  return std::log2(t);
}

double perf_eps_fraction(int N, double k, double req) {
  const double lt = perf_log2_T(N);
  if (lt < 53 || k < 1) return 0;  // |a'| <= T < 2^53: eps = 0 everywhere
  if (req >= 0) return req;
  const double D = lt - 0.5 * std::log2(k) - 53;
  return std::min(1.0, std::max(0.0, (D + 2.5) / 1.5));
}

void perf_contract_phases(const Analysis& an, const ExecPlan& e, int32_t flags, int32_t k4,
                          const imd_perf_calib_t& cal, double eps, PerfPhases* out) {
  PerfPhases& P = *out;
  P = PerfPhases{};
  if (e.path == Path::kNoop) return;
  const int32_t sms = cal.sm_count;
  const double l2 = cal.l2_bytes;
  const bool resident = an.options.vec_mode == IMD_VEC_RESIDENT;
  const int64_t v = resident ? 0 : 1;
  const double V = static_cast<double>(e.vec_count);
  const double b = static_cast<double>(e.b), m = static_cast<double>(e.m),
               n = static_cast<double>(e.n), k = static_cast<double>(e.k),
               nh = static_cast<double>(e.n_hat), kh = static_cast<double>(e.k_hat);
  const bool beta_read = (flags & IMD_PERF_BETA_NONZERO) != 0;
  const double out_bytes = 8 * b * m * n * (beta_read ? 2 : 1);
  const double c_in = beta_read ? 8 * b * m * n : 0;  // C is an input when beta != 0
  const int64_t bmn = e.b * e.m * e.n;
  const auto k0 = [&] {
    if (!resident)
      add_phase(&P, IMD_PERF_PHASE_K0, IMD_PERF_CLASS_K0, 1,
                {1, 8 * V, grid_chain(cal, IMD_PERF_KERNEL_K0, e.vec_count, 1)}, 8 * V);
  };
  P.footprint = static_cast<double>(an.layout.total) + 8 * b * m * n;
  if ((flags & IMD_PERF_ALPHA_ZERO) || e.k == 0) {  // §3.7: K0 and the beta kernel
    k0();
    add_phase(&P, IMD_PERF_PHASE_BETA, IMD_PERF_CLASS_BETA, 1,
              {1, out_bytes, grid_chain(cal, IMD_PERF_KERNEL_BETA, bmn, 1)}, out_bytes)
        .in_bytes = c_in;
    P.inputs = c_in;
    P.launches = v + 1;
  } else if (!emulates(e)) {  // DIRECT dispatch (mode DIRECT or K-empty)
    k0();
    const double bytes = 8 * b * (m * k + n * k) + out_bytes, flops = 2 * b * m * n * k;
    add_phase(&P, IMD_PERF_PHASE_DIRECT, IMD_PERF_CLASS_DIRECT, 1,
              {1, bytes, flops, grid_chain(cal, IMD_PERF_KERNEL_DIRECT, bmn, k)}, bytes, flops, 2)
        .in_bytes = 8 * b * (m * k + n * k) + c_in;
    P.inputs = 8 * b * (m * k + n * k) + c_in;
    P.launches = v + 1;
  } else {
    const bool fused = fused_path(e);
    const int32_t fl = fused_large(e);
    const double N = e.N, g = e.g;  // g in [1, N]
    const double rows[2] = {b * m, b * n}, pad[2] = {b * m, b * nh};
    const bool inl[2] = {!(e.packed & 1), !(e.packed & 2)};
    const double f = perf_eps_fraction(e.N, k, eps);
    P.eps_fraction = f;
    P.eps_possible = perf_log2_T(e.N) >= 53;
    for (int x = 0; x < 2; ++x)
      if (inl[x]) P.inputs += 8 * rows[x] * k;
    P.inputs += c_in;
    for (int x = 0; x < 2; ++x)  // a packed operand's buffer is part of the footprint
      if (!inl[x]) P.footprint += static_cast<double>(an.packed.layout[x].total);
    if (!fused) k0();
    if (fused) {  // KF: K0 + K1 + group 1's packs of the in-line operands
      double bytes = resident ? 0 : 8 * V, ops = 0, r = 0, in = 0;
      for (int x = 0; x < 2; ++x) {
        if (!inl[x]) continue;
        bytes += 8 * rows[x] * k + g * pad[x] * kh + 3 * rows[x];
        ops += g * rows[x] * k;
        r += rows[x];
        in += 8 * rows[x] * k;
      }
      const double chain = kf_chain(e, cal);
      set_eps(set_eps(add_phase(&P, IMD_PERF_PHASE_KF, IMD_PERF_CLASS_KF, 1,
                                {1, bytes, ops, r, 0, chain}, bytes),
                      4, ops, f),
              5, chain, f)
          .in_bytes = in;
    } else {  // KL per in-line operand (v1.6: every in-line operand without KF)
      for (int x = 0; x < 2; ++x)
        if (inl[x])
          add_kl(&P, cal, x == 0 ? IMD_PERF_PHASE_KL_A : IMD_PERF_PHASE_KL_B, rows[x], pad[x], k,
                 kh, g, e.pack_along_rows[x], e.k_hat, f, -1, e.b, x == 0 ? e.m : e.n,
                 e.pack_radix[x]);
    }
    // K2 / K3: the groups of an in-line operand that KF / KL did not pack (groups 2..G):
    // per group one read of the operand and the group's residues (+ the zero-rows launch of B).
    const int64_t G = (e.N + e.g - 1) / e.g;
    for (int x = 0; x < 2; ++x) {
      if (!inl[x]) continue;
      const bool first_done = fused || (fl & (1 << x));
      const int64_t groups = G - (first_done ? 1 : 0);
      if (groups <= 0) continue;
      const double moduli = first_done ? N - g : N;  // moduli of the packed groups
      const double bytes = groups * 8 * rows[x] * k + moduli * pad[x] * kh;
      const int64_t per = 1 + ((x == 1 && e.n_hat > e.n) ? 1 : 0);
      // v1.10: the chain of each group's k23_pack launch (pack_launch, shared with contract.cu).
      double chain = 0, wave = 0;
      for (int64_t gi = first_done ? 1 : 0; gi < G; ++gi) {
        const int gsize = static_cast<int>(std::min<int64_t>(e.g, e.N - gi * e.g));
        const PackLaunch L = pack_launch(e.pack_along_rows[x], e.b, x == 0 ? e.m : e.n,
                                         e.pack_radix[x], e.k_hat, gsize, sms);
        chain += chain_of(L.grid_x, L.parts, L.tiles, 16.0 * dbl((gsize + L.parts - 1) / L.parts),
                          resident_of(cal, IMD_PERF_KERNEL_PACK), sms);
        // the tile loads of the busiest SM, in sequence (a memory-latency term: a grid of
        // a few blocks per SM cannot hide DRAM latency)
        wave += chain_of(L.grid_x, L.parts, L.tiles, 1.0, resident_of(cal, IMD_PERF_KERNEL_PACK), sms);
      }
      PerfPhase& p = add_phase(&P, x == 0 ? IMD_PERF_PHASE_PACK_A : IMD_PERF_PHASE_PACK_B,
                               IMD_PERF_CLASS_PACK, groups * per,
                               {static_cast<double>(groups * per), bytes, moduli * rows[x] * k, 0,
                                chain, wave},
                               bytes);
      set_eps(set_eps(p, 3, moduli * rows[x] * k, f), 4, chain, f);
    }
    // K4: the A3 round model of the configuration, over the groups and the K-chunks.
    const int32_t cfg = k4 >= 0 ? k4 : k4_select(e, sms);
    P.k4_config = cfg;
    const K4Counts kc = plan_k4_counts(e, production_launch_limits());
    const int64_t chunk = c::kKcPrime;
    const int64_t gsz[2] = {e.g, e.N % e.g}, gcnt[2] = {e.N / e.g, e.N % e.g ? 1 : 0};
    const int64_t last = e.k_hat - (kc.k_chunks - 1) * chunk;
    // The K-chunk kinds: full chunks with aligned / misaligned rows, then the last chunk.
    // The slow K-chunks (mac_ktail): format 1 (v1.8) took a chunk whose length is not a
    // multiple of the 64-wide K tile (CUTLASS's predicated residue tile); format 2 (v1.10,
    // NOTES-B2 §6) one whose rows are misaligned, by the rule "k_hat and the chunk offset
    // c * K_c' are multiples of 32". The measured mechanism (DESIGN v1.13 §4b, NOTES-K4ALIGN):
    // CUTLASS loads a chunk's residue tile (kc mod 64) first, so row r's main-loop loads sit at
    // phase (r k_hat + chunk end) mod 64. Since v1.13 K_c' = 131008 = 0 mod 64: every chunk
    // end is 0 mod 64 or k_hat, the c * K_c' term is always 0 mod 32, and the format-2 rule
    // reduces to "slow iff k_hat mod 32 != 0", which is exact for the phase classes 16 / 48
    // (class 32, +3 % to +19 % measured, is not a separate term). v1.13 stage 2 (§4b.6): an
    // A = 64 plan (AUTO: k > 96) has k_hat = 0 mod 64 and is never slow; only A = 16 plans
    // (k <= 96, or forced by the test hook) can be. Calibration format 2 is unchanged; theta was
    // refit after stage 2 (2026-10-06, NOTES-B2 §8): the A = 16-forced calibration cases at
    // k > 96 (--k-align 16) keep the slow term determined.
    const int64_t nfull = kc.k_chunks - 1;
    bool last_slow;
    int64_t full_slow;
    if (cal.format == 1) {
      full_slow = chunk % kK4TileK != 0 ? nfull : 0;
      last_slow = last % kK4TileK != 0;
    } else if (e.k_hat % kK4RowAlign != 0) {
      full_slow = nfull, last_slow = true;
    } else {
      const auto misaligned = [&](int64_t c) { return (c * chunk) % kK4RowAlign != 0; };
      full_slow = 0;
      for (int64_t c = 0; c < nfull; ++c) full_slow += misaligned(c) ? 1 : 0;
      last_slow = misaligned(nfull);
    }
    // v1.13 §4b.6: the MMA work of a chunk is ceil(kc)_64 (CUTLASS runs its residue tile as a
    // full zero-filled 64-wide tile), whatever A: the same for A = 16 and A = 64 plans. Only the
    // byte terms use the padded k_hat.
    const auto mma_k = [](int64_t len) { return (len + kK4TileK - 1) / kK4TileK * kK4TileK; };
    const int64_t ksz[3] = {mma_k(chunk), mma_k(chunk), mma_k(last)},
                  kcnt[3] = {nfull - full_slow, full_slow, 1};
    const bool kslow[3] = {false, true, last_slow};
    double x[kP] = {0, 0, 0, 0, 0, 0};
    const int occ = kK4Configs[cfg].ctas_per_sm;
    for (int i = 0; i < 2; ++i)
      for (int j = 0; j < 3; ++j) {
        if (gcnt[i] == 0 || kcnt[j] == 0) continue;
        const K4Rounds r = k4_rounds(cfg, e.m, e.n_hat, ksz[j], gsz[i] * e.b, sms);
        const double cnt = static_cast<double>(gcnt[i]) * static_cast<double>(kcnt[j]);
        const double full = u128_to_double(r.full), macs = u128_to_double(r.tile_macs);
        x[0] += cnt * (full + (r.rest > 0 ? 1 : 0));
        x[occ] += cnt * full * occ * macs;
        if (r.rest > 0) x[r.rest] += cnt * r.rest * macs;
        // A slow chunk's MACs again, at their extra cost (mac_ktail).
        if (kslow[j]) x[kK4TailSlot] += cnt * (full * occ + r.rest) * macs;
      }
    const double k4_bytes = N * b * (m + nh) * kh + N * b * m * nh * (2.0 * kc.k_chunks - 1);
    // The DRAM term: per group, its operand residues (just written by the packs) and its W
    // planes stay in L2 while they fit; the bytes beyond L2 stream from DRAM.
    double dram = 0;
    for (int i = 0; i < 2; ++i) {
      const double gi = static_cast<double>(gsz[i]);
      const double group = gi * b * (m + nh) * kh + gi * b * m * nh * (2.0 * kc.k_chunks - 1);
      dram += static_cast<double>(gcnt[i]) * std::max(0.0, group - l2);
    }
    const double k4_ops = 2 * N * b * m * nh * static_cast<double>(nfull * ksz[0] + ksz[2]);
    add_phase(&P, IMD_PERF_PHASE_K4, IMD_PERF_CLASS_K4_128x256 + cfg, kc.launches,
              {x[0], x[1], x[2], x[3], dram, x[kK4TailSlot]}, k4_bytes, k4_ops, 1);
    // K5: the N planes, sigma / tau and the flags read, C written (and read if beta != 0);
    // v1.10: the chain of its launch (k5_launch, shared with contract.cu).
    const double k5_bytes = N * b * m * nh + 3 * b * (m + n) + out_bytes;
    const double limbs = N * c::kTables.L32[e.N];
    const K5Launch L5 = k5_launch(e.k5_batch_fast, e.b, e.m, e.n, sms);
    const int32_t k5k = k5_kernel(e.N, L5.variant);
    const double k5_chain =
        L5.variant == 3
            ? chain_of(L5.grid, 1, L5.work, 4 * limbs, resident_of(cal, k5k), sms)
            : chain_of(L5.grid, 1, (L5.work + kLaunchThreads - 1) / kLaunchThreads,
                       L5.elems * limbs, resident_of(cal, k5k), sms);
    // v1.10: the W plane bytes beyond the (effective) L2 that K4 left them in come from DRAM
    // (K5 at 8192^2, N = 8: 1.07 GB at 0.93 TB/s; the byte term alone, fitted on mostly
    // L2-resident planes, was 24 % low there).
    const double k5_dram = std::max(0.0, N * b * m * nh - l2);
    add_phase(&P, IMD_PERF_PHASE_K5, IMD_PERF_CLASS_K5, 1,
              {1, k5_bytes, limbs * b * m * n, k5_chain, k5_dram}, k5_bytes)
        .in_bytes = c_in;
    P.launches = plan_launches(e, fused, fl, resident, kc.launches);
  }
  P.footprint += P.inputs;  // F = I + workspace (+ packed buffers) + the output
  add_phase(&P, IMD_PERF_PHASE_CALL, IMD_PERF_CLASS_CALL, 0,
            {1, static_cast<double>(P.launches)}, 0);
}

void perf_pack_phases(const PackLowered& pl, const imd_perf_calib_t& cal, double eps,
                      PerfPhases* out) {
  PerfPhases& P = *out;
  P = PerfPhases{};
  if (pl.layout.total == 0) return;
  ExecPlan e;
  pack_exec(pl, &e);
  const double b = static_cast<double>(pl.b), r = static_cast<double>(pl.r),
               k = static_cast<double>(pl.k), kh = static_cast<double>(pl.k_hat),
               N = static_cast<double>(pl.N);
  // The prep launch: X's scatter vectors and every byte KL does not write (gaps, tail).
  const double zeros = static_cast<double>(pl.layout.total) - 3 * b * r - N * b * r * kh;
  const double prep = 8 * (r + k + b) + zeros;
  // kp_prep's grid (contract.cu launch_kp_prep): max(vec entries, the tail's 16-byte words).
  const uint64_t planes_end = pl.layout.offset[kPackRes] +
                              static_cast<uint64_t>(pl.N) * static_cast<uint64_t>(pl.b * pl.r * pl.k_hat);
  const int64_t work = std::max<int64_t>(e.vec_count, static_cast<int64_t>((pl.layout.total - planes_end) / 16));
  add_phase(&P, IMD_PERF_PHASE_PACK_PREP, IMD_PERF_CLASS_K0, 1,
            {1, prep, grid_chain(cal, IMD_PERF_KERNEL_PACK_PREP, work, 1)}, prep);
  P.eps_fraction = perf_eps_fraction(pl.N, k, eps);
  P.eps_possible = perf_log2_T(pl.N) >= 53;
  add_kl(&P, cal, IMD_PERF_PHASE_PACK_KL, b * r, b * r, k, kh, N, e.pack_along_rows[0], pl.k_hat,
         P.eps_fraction, pack_kl_lg_rows_for(e.pack_along_rows[0], pl.k_hat), pl.b, pl.r,
         e.pack_radix[0]);
  P.launches = 2;
  P.inputs = 8 * b * r * k;
  P.footprint = P.inputs + static_cast<double>(pl.layout.total) + static_cast<double>(pl.ws_bytes);
  add_phase(&P, IMD_PERF_PHASE_CALL, IMD_PERF_CLASS_CALL, 0, {1, 2}, 0);
}

double perf_phase_seconds(const imd_perf_calib_t& cal, const PerfPhase& p, double scale) {
  const double* th = cal.param + p.cls * kP;
  const bool* cb = kClockBound[p.cls];
  if (p.cls >= IMD_PERF_CLASS_K4_128x256 && p.cls <= IMD_PERF_CLASS_K4_64x64) {
    // K4: a roofline, max(the A3 round model, the DRAM term).
    double rounds = 0;
    for (int j = 0; j < kP; ++j)
      if (j != kK4ByteSlot) rounds += th[j] * p.x[j];
    return std::max(rounds * scale, th[kK4ByteSlot] * p.x[kK4ByteSlot]) * 1e-9;
  }
  double ns = 0, nsc = 0;
  for (int j = 0; j < kP; ++j) (cb[j] ? nsc : ns) += th[j] * p.x[j];
  return (ns + nsc * scale) * 1e-9;
}

namespace {

// ---- calibration text -----------------------------------------------------------------------

std::string_view trim(std::string_view s) {
  while (!s.empty() && (s.front() == ' ' || s.front() == '\t')) s.remove_prefix(1);
  while (!s.empty() && (s.back() == ' ' || s.back() == '\t' || s.back() == '\r')) s.remove_suffix(1);
  return s;
}

// A number, optionally followed by whitespace and a '#' comment. With `rest` non-null, the text
// after the number (trimmed) is returned there instead of being required empty.
bool parse_number(std::string_view v, double* out, std::string_view* rest_out = nullptr) {
  const char* first = v.data();
  const char* last = v.data() + v.size();
  if (first != last && *first == '+') ++first;
  const auto res = std::from_chars(first, last, *out, std::chars_format::general);
  if (res.ec != std::errc() || !std::isfinite(*out)) return false;
  const std::string_view rest = trim(std::string_view(res.ptr, static_cast<size_t>(last - res.ptr)));
  if (rest_out != nullptr) {
    // a number must end at a blank (or the end)
    if (!rest.empty() && res.ptr != last && *res.ptr != ' ' && *res.ptr != '\t') return false;
    *rest_out = rest;
    return true;
  }
  return rest.empty() || rest.front() == '#';
}

// `n` numbers separated by blanks (then the end or a comment).
bool parse_numbers(std::string_view v, double* out, int n) {
  for (int i = 0; i < n; ++i) {
    std::string_view rest;
    if (!parse_number(v, &out[i], &rest)) return false;
    v = rest;
  }
  return v.empty() || v.front() == '#';
}

bool parse_int(std::string_view v, int64_t* out) {
  double d = 0;
  if (!parse_number(v, &d) || d != std::floor(d) || std::fabs(d) > 1e15) return false;
  *out = static_cast<int64_t>(d);
  return true;
}

struct StrField {
  const char* key;
  char* dst;
  size_t cap;
};

bool finite_nonneg(double v) { return std::isfinite(v) && v >= 0; }

bool calib_valid(const imd_perf_calib_t& c) {
  if ((c.format != 1 && c.format != 2) || c.sm_count < 1) return false;
  for (const double pk : {c.peak_dram_bytes_per_s, c.peak_int8_ops_per_s, c.peak_fp64_flops_per_s,
                          c.l2_bytes})
    if (!(pk > 0) || !std::isfinite(pk)) return false;
  for (int cl = 0; cl < IMD_PERF_CLASS_COUNT; ++cl)
    for (int j = 0; j < kP; ++j) {
      const double v = c.param[cl * kP + j];
      if (!std::isfinite(v) || (cl != IMD_PERF_CLASS_CALL && v < 0)) return false;
    }
  if (c.format == 1) return true;  // the v7 fields are migrated (resolve_calib)
  for (int k = 0; k < IMD_PERF_KERNEL_COUNT; ++k) {
    const imd_perf_kernel_t& q = c.kernel[k];
    if (q.resident < 1 || q.regs < 0 || q.smem_bytes < 0 || q.threads < 0) return false;
  }
  for (const double v : {c.clock_eff_mhz, c.clock_min_mhz, c.power_cap_w, c.power_idle_w,
                         c.energy_tc_pj, c.energy_alu_pj, c.energy_byte_pj, c.energy_byte_f_pj,
                         c.energy_gamma,
                         c.warm_l2_bytes, c.warm_l2byte_ns, c.tie_ratio})
    if (!finite_nonneg(v)) return false;
  if (c.power_cap_w > 0 && !(c.clock_eff_mhz > 0 && c.clock_min_mhz > 0 &&
                             c.clock_min_mhz <= c.clock_eff_mhz))
    return false;
  if (!(c.tie_ratio >= 1 && c.tie_ratio <= 1.25)) return false;
  const auto band_ok = [](const double* b) {
    return std::isfinite(b[0]) && std::isfinite(b[1]) && b[0] > 0 && b[0] <= b[1];
  };
  for (const auto& b : c.band_call)
    if (!band_ok(b)) return false;
  for (int cl = 0; cl < IMD_PERF_CLASS_COUNT; ++cl)
    if (!band_ok(c.band_class[cl])) return false;
  return true;
}

// v1.10 format-2 scalar keys (DESIGN §6c.2).
struct ScalarKey {
  const char* key;
  double imd_perf_calib_t::*field;
};
constexpr ScalarKey kScalarKeys[] = {
    {"power.clock_eff_mhz", &imd_perf_calib_t::clock_eff_mhz},
    {"power.clock_min_mhz", &imd_perf_calib_t::clock_min_mhz},
    {"power.cap_w", &imd_perf_calib_t::power_cap_w},
    {"power.idle_w", &imd_perf_calib_t::power_idle_w},
    {"power.tc_pj", &imd_perf_calib_t::energy_tc_pj},
    {"power.alu_pj", &imd_perf_calib_t::energy_alu_pj},
    {"power.byte_pj", &imd_perf_calib_t::energy_byte_pj},
    {"power.gamma", &imd_perf_calib_t::energy_gamma},
    {"warm.l2_bytes", &imd_perf_calib_t::warm_l2_bytes},
    {"warm.l2byte", &imd_perf_calib_t::warm_l2byte_ns},
    {"rank.tie_ratio", &imd_perf_calib_t::tie_ratio}};
constexpr int kScalarKeyCount = sizeof kScalarKeys / sizeof kScalarKeys[0];
const char* const kBandCallKeys[3] = {"band.call_small", "band.call_mid", "band.call_large"};
// v1.15 (host API 11, DESIGN §6c.7): the one optional format-2 key (absent = 0).
constexpr const char* kByteFKey = "power.byte_f_pj";

// A v7..v10-sized caller has no energy_byte_f_pj: the folded view (DESIGN §6c.2), the
// unscaled form with the total byte energy (an API-10 library's prediction).
void fold_byte_f(imd_perf_calib_t* cal, uint32_t struct_size) {
  if (struct_size >= kCalibV7Size && struct_size < kCalibV11Size) {
    cal->energy_byte_pj += cal->energy_byte_f_pj;
    cal->energy_byte_f_pj = 0;
  }
}

// A rejected calibration (host API 10, DESIGN §6c.3): the line (0: none), the reason
// (IMD_PERF_CALIB_ERR_*), the key and a message. reason 0 = no error.
struct CalibErr {
  int32_t line = 0;
  int32_t reason = IMD_PERF_CALIB_OK;
  std::string key;
  std::string message;
};

// Text as it may be shown in a message: at most `cap` bytes, non-printable bytes as '?'.
std::string printable(std::string_view k, size_t cap = 63) {
  std::string s(k.substr(0, cap));
  for (char& ch : s)
    if (static_cast<unsigned char>(ch) < 0x20 || static_cast<unsigned char>(ch) > 0x7e) ch = '?';
  return s;
}

imd_status_t calib_fail(CalibErr* e, int32_t line, int32_t reason, std::string_view key,
                        const std::string& detail) {
  if (e != nullptr) {
    e->line = line;
    e->reason = reason;
    e->key = printable(key);
    std::string m;
    if (line > 0) m = "line " + std::to_string(line) + ": ";
    if (!e->key.empty()) m += e->key + ": ";
    e->message = m + detail;
  }
  return IMD_STATUS_INVALID_VALUE;
}

imd_status_t parse_calib(std::string_view text, imd_perf_calib_t* out, CalibErr* err) {
  imd_perf_calib_t r;
  std::memset(&r, 0, sizeof r);
  r.struct_size = sizeof r;
  // Seen keys: 0 format, 1 gpu, 2 sm_count, 3..5 peaks, 6..9 strings, 10 l2_bytes, then the
  // params, then (format 2) the kernels, the scalar keys and the bands.
  constexpr int kFixed = 11;
  constexpr int kKernelSlot = kFixed + IMD_PERF_PARAM_CAPACITY;
  constexpr int kScalarSlot = kKernelSlot + IMD_PERF_KERNEL_COUNT;
  constexpr int kBandSlot = kScalarSlot + kScalarKeyCount;  // 3 call bands, then the classes
  constexpr int kOptSlot = kBandSlot + 3 + IMD_PERF_CLASS_COUNT;  // power.byte_f_pj (optional)
  constexpr int kSeen = kOptSlot + 1;
  int32_t seen[kSeen] = {};   // the line of each key seen (0: not seen)
  int32_t first_v2_line = 0;  // the first line of a format-2-only key
  std::string first_v2_key;
  StrField strs[5] = {{"gpu", r.gpu, sizeof r.gpu},
                      {"date", r.date, sizeof r.date},
                      {"version", r.version, sizeof r.version},
                      {"clocks", r.clocks, sizeof r.clocks},
                      {"notes", r.notes, sizeof r.notes}};
  const int str_seen[5] = {1, 6, 7, 8, 9};
  // The canonical key of a slot (MISSING_KEY names it).
  const auto slot_key = [&](int slot) -> std::string {
    static const char* const kFixedKeys[kFixed] = {
        "format", "gpu", "sm_count", "peak_dram_bytes_per_s", "peak_int8_ops_per_s",
        "peak_fp64_flops_per_s", "date", "version", "clocks", "notes", "l2_bytes"};
    if (slot < kFixed) return kFixedKeys[slot];
    if (slot < kKernelSlot) {
      const int cl = (slot - kFixed) / kP, j = (slot - kFixed) % kP;
      return std::string(kClassNames[cl]) + "." + kParamNames[cl][j];
    }
    if (slot < kScalarSlot) return std::string("kernel.") + kernel_name(slot - kKernelSlot);
    if (slot < kBandSlot) return kScalarKeys[slot - kScalarSlot].key;
    if (slot == kOptSlot) return kByteFKey;
    const int b = slot - kBandSlot;
    return b < 3 ? std::string(kBandCallKeys[b]) : std::string("band.") + kClassNames[b - 3];
  };
  int32_t line = 0;
  size_t pos = 0;
  while (pos <= text.size()) {
    size_t eol = text.find('\n', pos);
    if (eol == std::string_view::npos) eol = text.size();
    const std::string_view raw = text.substr(pos, eol - pos);
    pos = eol + 1;
    ++line;
    const std::string_view s = trim(raw);
    if (s.empty() || s.front() == '#') {
      if (eol == text.size()) break;
      continue;
    }
    const size_t eq = s.find('=');
    if (eq == std::string_view::npos)
      return calib_fail(err, line, IMD_PERF_CALIB_ERR_SYNTAX, "",
                        "'" + printable(s, 40) + "' is not 'key = value', a '#' comment or blank");
    const std::string_view key = trim(s.substr(0, eq)), val = trim(s.substr(eq + 1));
    const auto bad = [&](const char* want) {
      return calib_fail(err, line, IMD_PERF_CALIB_ERR_VALUE, key,
                        "invalid value '" + printable(val, 40) + "' (want " + want + ")");
    };
    int slot = -1;
    bool v2 = false;
    if (key == "format") {
      int64_t f = 0;
      if (!parse_int(val, &f) || (f != 1 && f != 2)) return bad("1 or 2");
      r.format = static_cast<int32_t>(f), slot = 0;
    } else if (key == "sm_count") {
      int64_t s2 = 0;
      if (!parse_int(val, &s2) || s2 < 1 || s2 > 1000000) return bad("an integer in [1, 1000000]");
      r.sm_count = static_cast<int32_t>(s2), slot = 2;
    } else if (key == "peak_dram_bytes_per_s" || key == "peak_int8_ops_per_s" ||
               key == "peak_fp64_flops_per_s" || key == "l2_bytes") {
      double d = 0;
      if (!parse_number(val, &d) || !(d > 0)) return bad("a finite number > 0");
      if (key == "peak_dram_bytes_per_s") r.peak_dram_bytes_per_s = d, slot = 3;
      else if (key == "peak_int8_ops_per_s") r.peak_int8_ops_per_s = d, slot = 4;
      else if (key == "peak_fp64_flops_per_s") r.peak_fp64_flops_per_s = d, slot = 5;
      else r.l2_bytes = d, slot = 10;
    } else if (key.substr(0, 7) == "kernel.") {
      const std::string_view nm = key.substr(7);
      for (int k = 0; k < IMD_PERF_KERNEL_COUNT && slot < 0; ++k) {
        if (nm != kernel_name(k)) continue;
        double d[4];
        bool ok = parse_numbers(val, d, 4) && d[0] >= 1;
        for (int i = 0; ok && i < 4; ++i)
          ok = d[i] == std::floor(d[i]) && d[i] >= 0 && d[i] <= 1e9;
        if (!ok) return bad("4 integers 'resident regs smem threads' <= 1e9, resident >= 1");
        r.kernel[k] = {static_cast<int32_t>(d[0]), static_cast<int32_t>(d[1]),
                       static_cast<int32_t>(d[2]), static_cast<int32_t>(d[3])};
        slot = kKernelSlot + k, v2 = true;
      }
    } else if (key.substr(0, 5) == "band.") {
      double d[2];
      int idx = -1;
      for (int i = 0; i < 3 && idx < 0; ++i)
        if (key == kBandCallKeys[i]) idx = i;
      for (int cl = 0; cl < IMD_PERF_CLASS_COUNT && idx < 0; ++cl)
        if (key.substr(5) == kClassNames[cl]) idx = 3 + cl;
      if (idx >= 0) {
        if (!parse_numbers(val, d, 2) || !(d[0] > 0) || !(d[0] <= d[1]))
          return bad("two numbers 'lo hi', 0 < lo <= hi");
        double* dst = idx < 3 ? r.band_call[idx] : r.band_class[idx - 3];
        dst[0] = d[0], dst[1] = d[1];
        slot = kBandSlot + idx, v2 = true;
      }
    } else if (key == kByteFKey) {
      double d = 0;
      if (!parse_number(val, &d) || d < 0) return bad("a finite number >= 0");
      r.energy_byte_f_pj = d;
      slot = kOptSlot, v2 = true;
    } else {
      for (int i = 0; i < kScalarKeyCount && slot < 0; ++i)
        if (key == kScalarKeys[i].key) {
          double d = 0;
          if (!parse_number(val, &d) || d < 0) return bad("a finite number >= 0");
          r.*(kScalarKeys[i].field) = d;
          slot = kScalarSlot + i, v2 = true;
        }
      for (int i = 0; i < 5 && slot < 0; ++i)
        if (key == strs[i].key) {
          if (val.size() >= strs[i].cap)
            return calib_fail(err, line, IMD_PERF_CALIB_ERR_TOO_LONG, key,
                              "value is " + std::to_string(val.size()) +
                                  " bytes, the field holds " + std::to_string(strs[i].cap - 1));
          std::memcpy(strs[i].dst, val.data(), val.size());
          strs[i].dst[val.size()] = '\0';
          slot = str_seen[i];
        }
      const size_t dot = key.find('.');
      if (slot < 0 && dot != std::string_view::npos) {
        const std::string_view cls = key.substr(0, dot), feat = key.substr(dot + 1);
        for (int cl = 0; cl < IMD_PERF_CLASS_COUNT && slot < 0; ++cl) {
          if (cls != kClassNames[cl]) continue;
          for (int j = 0; j < kP; ++j)
            if (kParamNames[cl][j] != nullptr && feat == kParamNames[cl][j]) {
              double d = 0;
              if (!parse_number(val, &d) || (cl != IMD_PERF_CLASS_CALL && d < 0))
                return bad(cl == IMD_PERF_CLASS_CALL ? "a finite number" : "a finite number >= 0");
              r.param[cl * kP + j] = d;
              slot = kFixed + cl * kP + j;
              v2 = kFormat2Param[cl][j];
            }
        }
      }
    }
    if (slot < 0) return calib_fail(err, line, IMD_PERF_CALIB_ERR_UNKNOWN_KEY, key, "unknown key");
    if (seen[slot] != 0)
      return calib_fail(err, line, IMD_PERF_CALIB_ERR_REPEATED_KEY, key,
                        "repeated key (first on line " + std::to_string(seen[slot]) + ")");
    seen[slot] = line;
    if (v2 && first_v2_line == 0) first_v2_line = line, first_v2_key = std::string(key);
    if (eol == text.size()) break;
  }
  // Required: format, gpu, sm_count, the peaks and every parameter of the format; format 2
  // also every kernel, scalar and band key. A format-2 key in a format-1 file is rejected.
  const auto missing = [&](int slot) {
    return calib_fail(err, 0, IMD_PERF_CALIB_ERR_MISSING_KEY, slot_key(slot), "missing key");
  };
  for (const int i : {0, 1, 2, 3, 4, 5, 10})
    if (seen[i] == 0) return missing(i);
  const bool f2 = r.format == 2;
  if (!f2 && first_v2_line != 0)
    return calib_fail(err, first_v2_line, IMD_PERF_CALIB_ERR_FORMAT2_KEY, first_v2_key,
                      "a format-2 key in a format-1 calibration");
  for (int cl = 0; cl < IMD_PERF_CLASS_COUNT; ++cl)
    for (int j = 0; j < kP; ++j)
      if (kParamNames[cl][j] != nullptr && (f2 || !kFormat2Param[cl][j]) &&
          seen[kFixed + cl * kP + j] == 0)
        return missing(kFixed + cl * kP + j);
  if (f2) {
    for (int i = kKernelSlot; i < kOptSlot; ++i)  // power.byte_f_pj is optional
      if (seen[i] == 0) return missing(i);
    if (r.power_cap_w > 0 && !(r.clock_eff_mhz > 0 && r.clock_min_mhz > 0 &&
                               r.clock_min_mhz <= r.clock_eff_mhz))
      return calib_fail(err, 0, IMD_PERF_CALIB_ERR_INCONSISTENT, "power.clock_min_mhz",
                        "power.cap_w > 0 needs 0 < power.clock_min_mhz <= power.clock_eff_mhz");
    if (!(r.tie_ratio >= 1 && r.tie_ratio <= 1.25))
      return calib_fail(err, 0, IMD_PERF_CALIB_ERR_INCONSISTENT, "rank.tie_ratio",
                        "must be in [1, 1.25]");
    for (int k = IMD_PERF_KERNEL_COUNT; k < IMD_PERF_KERNEL_CAPACITY; ++k) r.kernel[k] = {};
    for (int cl = IMD_PERF_CLASS_COUNT; cl < IMD_PERF_CLASS_CAPACITY; ++cl)
      r.band_class[cl][0] = r.band_class[cl][1] = 1;
  } else {
    migrate_v7_fields(&r);
  }
  // A caller compiled against version 6 has no room for the v7 fields: it gets a true
  // format-1 view (the format-2 parameters zeroed): a degraded model (DESIGN §6c.3).
  if (out->struct_size < kCalibV7Size) to_format1_params(&r);
  fold_byte_f(&r, out->struct_size);  // v7..v10 caller (API 11)
  r.struct_size = out->struct_size;
  std::memcpy(out, &r, std::min<size_t>(out->struct_size, sizeof r));
  if (err != nullptr) *err = CalibErr{};
  return IMD_STATUS_SUCCESS;
}

void append_number(std::string* s, double v) {
  char buf[64];
  const auto res = std::to_chars(buf, buf + sizeof buf, v);  // shortest round-trip form
  s->append(buf, res.ptr);
}

std::string format_calib(const imd_perf_calib_t& c) {
  const bool f2 = c.format == 2;
  std::string s;
  s += f2 ? "# immerdink performance-model calibration (docs/DESIGN.md §6c.2), format 2.\n"
          : "# immerdink performance-model calibration (docs/DESIGN.md §6c.2), format 1.\n";
  s += "# Parameters: <class>.<feature> = ns per feature unit (§6c.1).\n";
  s += f2 ? "format = 2\n" : "format = 1\n";
  const auto str = [&](const char* key, const char* v, size_t cap) {
    s += key;
    s += " = ";
    s.append(v, strnlen(v, cap));
    s += '\n';
  };
  str("gpu", c.gpu, sizeof c.gpu);
  s += "sm_count = " + std::to_string(c.sm_count) + "\n";
  str("date", c.date, sizeof c.date);
  str("version", c.version, sizeof c.version);
  str("clocks", c.clocks, sizeof c.clocks);
  str("notes", c.notes, sizeof c.notes);
  const std::pair<const char*, double> peaks[4] = {
      {"peak_dram_bytes_per_s", c.peak_dram_bytes_per_s},
      {"peak_int8_ops_per_s", c.peak_int8_ops_per_s},
      {"peak_fp64_flops_per_s", c.peak_fp64_flops_per_s},
      {"l2_bytes", c.l2_bytes}};
  for (const auto& [k, v] : peaks) {
    s += k;
    s += " = ";
    append_number(&s, v);
    s += '\n';
  }
  for (int cl = 0; cl < IMD_PERF_CLASS_COUNT; ++cl)
    for (int j = 0; j < kP; ++j) {
      if (kParamNames[cl][j] == nullptr || (!f2 && kFormat2Param[cl][j])) continue;
      s += kClassNames[cl];
      s += '.';
      s += kParamNames[cl][j];
      s += " = ";
      append_number(&s, c.param[cl * kP + j]);
      s += '\n';
    }
  if (!f2) return s;
  s += "# Occupancy (§6c.1): kernel.<name> = CTAs per SM, registers per thread, shared memory\n"
       "# bytes per CTA, threads per CTA (measured: imd_bench --methods occupancy).\n";
  for (int k = 0; k < IMD_PERF_KERNEL_COUNT; ++k) {
    const imd_perf_kernel_t& q = c.kernel[k];
    s += "kernel.";
    s += kernel_name(k);
    s += " = " + std::to_string(q.resident) + " " + std::to_string(q.regs) + " " +
         std::to_string(q.smem_bytes) + " " + std::to_string(q.threads) + "\n";
  }
  s += "# Clock / power model and cache state (§6c.1).\n";
  for (const ScalarKey& sk : kScalarKeys) {
    s += sk.key;
    s += " = ";
    append_number(&s, c.*(sk.field));
    s += '\n';
    if (sk.field == &imd_perf_calib_t::energy_byte_pj && c.energy_byte_f_pj != 0) {
      // v1.15: written only when non-zero (the text of a calibration without it is unchanged)
      s += kByteFKey;
      s += " = ";
      append_number(&s, c.energy_byte_f_pj);
      s += '\n';
    }
  }
  s += "# Prediction bands (§6c.1): p10 p90 of measured / predicted on the hold-out set.\n";
  const auto band = [&](const std::string& key, const double* b) {
    s += key + " = ";
    append_number(&s, b[0]);
    s += ' ';
    append_number(&s, b[1]);
    s += '\n';
  };
  for (int i = 0; i < 3; ++i) band(kBandCallKeys[i], c.band_call[i]);
  for (int cl = 0; cl < IMD_PERF_CLASS_COUNT; ++cl)
    band(std::string("band.") + kClassNames[cl], c.band_class[cl]);
  return s;
}

// The shipped calibration, parsed once (C++11 thread-safe initialization of a function-local
// static; immutable afterwards, so the model keeps no mutable state), with its parse error
// (host API 10: kept, so imd_perf_calib_default_ex can say why; CMake and host_c_api / host_perf
// make a failure unreachable in a correct build). A NULL calibration pointer anywhere means
// this one, at its full (v7) size: the recommended way to use it.
struct Shipped {
  imd_perf_calib_t cal;
  CalibErr err;  // reason OK, or why the embedded text did not parse (cal.format = 0)
};
const Shipped& shipped() {
  static const Shipped kShipped = [] {
    Shipped s;
    std::memset(&s.cal, 0, sizeof s.cal);
    s.cal.struct_size = sizeof s.cal;
    if (parse_calib(std::string_view(kPerfCalibDefault, sizeof kPerfCalibDefault - 1), &s.cal,
                    &s.err) != IMD_STATUS_SUCCESS) {
      s.cal.format = 0;
      s.err.message = "shipped calibration: " + s.err.message;
    }
    return s;
  }();
  return kShipped;
}
const imd_perf_calib_t* shipped_calib() { return &shipped().cal; }

// The calibration a function uses: the shipped one (NULL; no copy), or the caller's,
// validated and copied into *scratch with the v7 fields of a format-1 or version-6-sized
// calibration migrated (built-in occupancy, no power model, no cache term, no bands).
imd_status_t resolve_calib(const imd_perf_calib_t* in, imd_perf_calib_t* scratch,
                           const imd_perf_calib_t** use) {
  if (in == nullptr) {
    *use = shipped_calib();
    return (*use)->format == 0 ? IMD_STATUS_INVALID_VALUE : IMD_STATUS_SUCCESS;
  }
  imd_perf_calib_t* cal = scratch;
  *use = cal;
  std::memset(cal, 0, sizeof *cal);
  if (in->struct_size < kCalibV1Size) return IMD_STATUS_INVALID_VALUE;
  std::memcpy(cal, in, std::min<size_t>(in->struct_size, sizeof *cal));
  if (in->struct_size < kCalibV7Size) {
    if (cal->format != 1 && cal->format != 2) return IMD_STATUS_INVALID_VALUE;
    cal->format = 1;
  }
  if (!calib_valid(*cal)) return IMD_STATUS_INVALID_VALUE;
  if (cal->format == 1) {
    to_format1_params(cal);
    migrate_v7_fields(cal);
  }
  cal->struct_size = sizeof *cal;
  return IMD_STATUS_SUCCESS;
}

// ---- the calibration functions (host API 10: with the reason) -----------------------------

static_assert(sizeof(imd_perf_calib_error_t) == 272, "imd_perf_calib_error_t v1 layout");
constexpr uint32_t kCalibErrMinSize = 16;  // struct_size, line, reason, reserved

imd_status_t calib_default(imd_perf_calib_t* out, CalibErr* e) {
  if (out == nullptr || out->struct_size < kCalibV1Size)
    return calib_fail(e, 0, IMD_PERF_CALIB_ERR_ARGUMENT, "", "out is NULL or undersized");
  // The parsed-once shipped calibration, the caller's struct_size bytes of it (a version-6
  // caller gets its format-1 view: the v7 fields do not fit; NULL is the full one).
  const Shipped& sh = shipped();
  if (sh.cal.format == 0) {
    if (e != nullptr) *e = sh.err;
    return IMD_STATUS_INVALID_VALUE;
  }
  imd_perf_calib_t c = sh.cal;
  if (out->struct_size < kCalibV7Size) to_format1_params(&c);
  fold_byte_f(&c, out->struct_size);  // v7..v10 caller (API 11)
  c.struct_size = out->struct_size;
  std::memcpy(out, &c, std::min<size_t>(out->struct_size, sizeof c));
  return IMD_STATUS_SUCCESS;
}

imd_status_t calib_parse(const char* text, size_t length, imd_perf_calib_t* out, CalibErr* e) {
  if ((text == nullptr && length > 0) || out == nullptr || out->struct_size < kCalibV1Size)
    return calib_fail(e, 0, IMD_PERF_CALIB_ERR_ARGUMENT, "",
                      "text is NULL with length > 0, or out is NULL or undersized");
  return parse_calib(std::string_view(text == nullptr ? "" : text, length), out, e);
}

imd_status_t calib_load(const char* path, imd_perf_calib_t* out, CalibErr* e) {
  if (path == nullptr || out == nullptr || out->struct_size < kCalibV1Size)
    return calib_fail(e, 0, IMD_PERF_CALIB_ERR_ARGUMENT, "",
                      "path or out is NULL, or out is undersized");
  std::FILE* f = std::fopen(path, "rb");
  if (f == nullptr)
    return calib_fail(e, 0, IMD_PERF_CALIB_ERR_FILE, "",
                      "cannot open '" + printable(path, 120) + "'");
  std::string text;
  char buf[4096];
  size_t got = 0;
  while ((got = std::fread(buf, 1, sizeof buf, f)) > 0 && text.size() < (size_t{1} << 20))
    text.append(buf, got);
  const bool rd_err = std::ferror(f) != 0;
  std::fclose(f);
  if (rd_err)
    return calib_fail(e, 0, IMD_PERF_CALIB_ERR_FILE, "",
                      "read error on '" + printable(path, 120) + "'");
  if (text.size() >= (size_t{1} << 20))
    return calib_fail(e, 0, IMD_PERF_CALIB_ERR_FILE, "",
                      "'" + printable(path, 120) + "' is 1 MiB or larger");
  return parse_calib(text, out, e);
}

// The functions without _ex: the status and error line of f's CalibErr (unchanged results).
template <class F>
imd_status_t with_line(int32_t* error_line, F&& f) {
  if (error_line != nullptr) *error_line = 0;
  return guarded([&] {
    CalibErr e;
    const imd_status_t st = f(&e);
    if (error_line != nullptr) *error_line = st == IMD_STATUS_SUCCESS ? 0 : e.line;
    return st;
  });
}

// The _ex functions: the same status and *err filled (its struct_size bytes; < 16:
// INVALID_VALUE, nothing written). An exception (allocation) leaves reason OK.
template <class F>
imd_status_t with_error(imd_perf_calib_error_t* err, F&& f) {
  if (err != nullptr && err->struct_size < kCalibErrMinSize) return IMD_STATUS_INVALID_VALUE;
  CalibErr e;
  const imd_status_t st = guarded([&] { return f(&e); });
  if (err != nullptr) {
    imd_perf_calib_error_t r;
    std::memset(&r, 0, sizeof r);
    r.struct_size = err->struct_size;
    if (st != IMD_STATUS_SUCCESS) {
      r.line = e.line;
      r.reason = e.reason;
      std::snprintf(r.key, sizeof r.key, "%s", e.key.c_str());
      std::snprintf(r.message, sizeof r.message, "%s", e.message.c_str());
    }
    std::memcpy(err, &r, std::min<size_t>(err->struct_size, sizeof r));
  }
  return st;
}

// ---- prediction ---------------------------------------------------------------------------

// What the request asks beyond the options (v7 fields: DEFAULT, COLD, no clock for a v6 one).
struct Req {
  int32_t flags = 0, k4 = -1, regime = IMD_PERF_REGIME_DEFAULT, cache = IMD_PERF_CACHE_COLD;
  double eps = -1, clock = 0;
  ErrReq err;  // v8 (§6f): the accuracy request (default: NORMWISE, no target)
};

// The time of every phase at clock scale s = f_eff / f (no cache term), and its sum. The power
// model and the SUSTAINED clock use this COLD time in every cache state (documented
// approximation: the REPEAT saving is up to 16 % only on calls that fit the L2, which are far
// below the power cap, so the solved clock is the same).
double call_seconds(const imd_perf_calib_t& cal, const PerfPhases& P, double scale) {
  double t = 0;
  for (int i = 0; i < P.n; ++i) t += perf_phase_seconds(cal, P.ph[i], scale);
  return t;
}

// The call's average power at clock f (DESIGN §6c.1 "Clock regimes", v1.15 §6c.7): idle +
// ((tensor-core, ALU and clock-scaled byte energy) (f / f_ref)^gamma + DRAM energy) / time; 0
// without a power model. With energy_byte_f_pj = 0 the added term is +0: bit-identical to v1.10.
double call_power(const imd_perf_calib_t& cal, const PerfPhases& P, double f, double t) {
  if (!(cal.power_cap_w > 0) || !(cal.clock_eff_mhz > 0)) return 0;
  double tc = 0, alu = 0, bytes = 0;
  for (int i = 0; i < P.n; ++i) {
    const PerfPhase& p = P.ph[i];
    bytes += p.bytes;
    if (p.peak == 1) tc += p.ops;
    else if (p.cls == IMD_PERF_CLASS_KF || p.cls == IMD_PERF_CLASS_KL ||
             p.cls == IMD_PERF_CLASS_PACK || p.cls == IMD_PERF_CLASS_K5 ||
             p.cls == IMD_PERF_CLASS_DIRECT)
      alu += p.x[2];
  }
  const double dyn = std::pow(f / cal.clock_eff_mhz, cal.energy_gamma);
  const double joules = ((cal.energy_tc_pj * tc + cal.energy_alu_pj * alu +
                          cal.energy_byte_f_pj * bytes) * dyn +
                         cal.energy_byte_pj * bytes) * 1e-12;
  return cal.power_idle_w + (t > 0 ? joules / t : 0);
}

// The clock of the request (DESIGN §6c.1): an explicit clock, else BURST = f_ref, SUSTAINED =
// f_ref below the cap, else the f in [f_min, f_ref] where the call's power meets the cap
// (bisection; f_min if even that is above). 0: no clock model (f_ref = 0).
double resolve_clock(const imd_perf_calib_t& cal, const PerfPhases& P, const Req& q,
                     int32_t* regime_used) {
  const bool has_power = cal.power_cap_w > 0 && cal.clock_eff_mhz > 0;
  int32_t reg = q.regime;
  if (reg == IMD_PERF_REGIME_DEFAULT) reg = has_power ? IMD_PERF_REGIME_SUSTAINED : IMD_PERF_REGIME_BURST;
  *regime_used = reg;
  const double fr = cal.clock_eff_mhz;
  if (!(fr > 0)) return 0;
  if (q.clock > 0) return std::min(q.clock, fr);  // at or above the effective clock: BURST
  if (reg == IMD_PERF_REGIME_BURST || !has_power) return fr;
  const auto power_at = [&](double f) { return call_power(cal, P, f, call_seconds(cal, P, fr / f)); };
  if (power_at(fr) <= cal.power_cap_w) return fr;
  double lo = cal.clock_min_mhz, hi = fr;
  if (power_at(lo) >= cal.power_cap_w) return lo;
  for (int it = 0; it < 40 && hi - lo > 0.05; ++it) {  // P increases in f; 0.05 MHz
    const double mid = 0.5 * (lo + hi);
    (power_at(mid) > cal.power_cap_w ? hi : lo) = mid;
  }
  return 0.5 * (lo + hi);
}

int band_class_of(double t) { return t < 100e-6 ? IMD_PERF_BAND_SMALL : t < 1e-3 ? IMD_PERF_BAND_MID : IMD_PERF_BAND_LARGE; }

void fill_prediction(const imd_perf_calib_t& cal, const PerfPhases& P, const Req& q, imd_perf_t* r) {
  r->num_phases = P.n;
  r->k4_config = P.k4_config;
  r->sm_count = cal.sm_count;
  r->kernel_launches = P.launches;
  int32_t regime = IMD_PERF_REGIME_BURST;
  const double f = resolve_clock(cal, P, q, &regime);
  const double scale = f > 0 ? cal.clock_eff_mhz / f : 1.0;
  // COLD / WARM (DESIGN §6c.1 "Cache state"): the parameters are COLD costs (the fitter adds the
  // calibration records' resident-input saving back); a REPEAT / WARM call saves warm.l2byte per
  // resident input byte in the phases that read them first.
  const double Lw = cal.warm_l2_bytes, I = P.inputs;
  const double res_repeat = std::clamp(Lw - (P.footprint - I), 0.0, I);
  // WARM is modeled as REPEAT (measured on the 4090, NOTES-B2 §6: a call whose A was the
  // previous call's C runs within -7 ... +3 % of the same call repeated; COLD, an evicted L2,
  // is the real effect: up to +16 % for calls whose footprint fits the L2).
  const double res = q.cache == IMD_PERF_CACHE_COLD ? 0.0 : res_repeat;
  const double cache_ns = -res * cal.warm_l2byte_ns;
  double t = 0, bound = 0, eps_s = 0, lo = 0, hi = 0;
  for (int i = 0; i < P.n; ++i) {
    const PerfPhase& p = P.ph[i];
    if (P.eps_possible)
      for (int j = 0; j < kP; ++j)
        eps_s += cal.param[p.cls * kP + j] * p.eps_x[j] * (kClockBound[p.cls][j] ? scale : 1.0) * 1e-9;
    imd_perf_phase_t& o = r->phase[i];
    o.phase = p.phase, o.cls = p.cls, o.launches = p.launches;
    o.time_s = perf_phase_seconds(cal, p, scale);
    if (I > 0 && p.in_bytes > 0) o.time_s = std::max(0.0, o.time_s + cache_ns * 1e-9 * p.in_bytes / I);
    o.bytes = p.bytes, o.ops = p.ops;
    const double peak = p.peak == 1 ? cal.peak_int8_ops_per_s
                        : p.peak == 2 ? cal.peak_fp64_flops_per_s
                                      : 0;
    o.bound_s = std::max(p.bytes / cal.peak_dram_bytes_per_s, peak > 0 ? p.ops / peak : 0.0);
    for (int j = 0; j < kP; ++j) o.x[j] = p.x[j];
    const bool call = p.cls == IMD_PERF_CLASS_CALL;
    const double* b = cal.band_class[p.cls];
    r->phase_lo_s[i] = call ? o.time_s : o.time_s * b[0];
    r->phase_hi_s[i] = call ? o.time_s : o.time_s * b[1];
    t += o.time_s;
    bound += o.bound_s;
  }
  r->time_s = t;
  r->bound_s = bound;
  r->eps_s = eps_s;
  r->eps_fraction = P.eps_fraction;
  r->tflops_eq = t > 0 ? r->flops_eq / t * 1e-12 : 0;
  const double* bc = cal.band_call[band_class_of(t)];
  lo = t * bc[0], hi = t * bc[1];
  r->time_lo_s = lo, r->time_hi_s = hi;
  r->regime = regime;
  r->cache = q.cache;
  r->clock_mhz = f;
  r->power_w = f > 0 && P.n > 0 ? call_power(cal, P, f, call_seconds(cal, P, scale)) : 0;
  r->warm_bytes = res;
}

int32_t dispatch_of(const ExecPlan& e) {
  return e.path == Path::kNoop ? IMD_DISPATCH_NOOP
         : e.k == 0            ? IMD_DISPATCH_SCALE
         : emulates(e)         ? IMD_DISPATCH_EMULATE
                               : IMD_DISPATCH_DIRECT;
}

// The contract prediction of a lowered packing: status, and on SUCCESS *r filled.
imd_status_t predict_contract(const imd_tensor_t* A, const imd_tensor_t* B, const imd_tensor_t* C,
                              const imd_options_t* options, const imd_packing_t* packing,
                              const KAlignSel& sel, const Req& q, const imd_perf_calib_t& cal,
                              imd_perf_t* r) {
  Analysis an;
  if (const imd_status_t st = analyze_with(A, B, C, options, sel, packing, &an);
      st != IMD_STATUS_SUCCESS)
    return st;
  ExecPlan e;
  if (const imd_status_t st = plan_data(an, &e); st != IMD_STATUS_SUCCESS) return st;
  PerfPhases P;
  perf_contract_phases(an, e, q.flags, emulates(e) ? q.k4 : -1, cal, q.eps, &P);
  r->dispatch = dispatch_of(e);
  r->workspace_bytes = an.layout.total;
  r->plan_device_bytes = e.device_bytes;
  r->flops_eq = 2.0 * static_cast<double>(e.b) * static_cast<double>(e.m) *
                static_cast<double>(e.n) * static_cast<double>(e.k);
  fill_prediction(cal, P, q, r);
  // v8 (§6f): the error estimates of this configuration (the same function as
  // imd_err_estimate); written only into a v8-sized imd_perf_t (finish copies struct_size).
  const ErrOut eo = err_compute(e.N, options->rounding, r->dispatch, e.k, e.b * e.m * e.n, q.err);
  r->err_bound = eo.bound, r->err_typical = eo.typical;
  r->err_status = eo.err_status, r->err_metric = q.err.metric;
  return IMD_STATUS_SUCCESS;
}

imd_status_t predict_pack(const imd_pack_desc_t& d, const KAlignSel& sel, const Req& q,
                          const imd_perf_calib_t& cal, imd_perf_t* r) {
  PackLowered pl;
  if (const imd_status_t st = pack_lower(d, sel, &pl); st != IMD_STATUS_SUCCESS) return st;
  PerfPhases P;
  perf_pack_phases(pl, cal, q.eps, &P);
  r->dispatch = IMD_DISPATCH_EMULATE;
  r->pack_bytes = pl.layout.total;
  r->pack_ws_bytes = pl.ws_bytes;
  fill_prediction(cal, P, q, r);
  return IMD_STATUS_SUCCESS;
}

bool eps_valid(double f) { return f < 0 || (f >= 0 && f <= 1); }  // NaN: invalid

bool v7_valid(int32_t regime, int32_t cache, double clock) {
  return regime >= IMD_PERF_REGIME_DEFAULT && regime <= IMD_PERF_REGIME_SUSTAINED &&
         cache >= IMD_PERF_CACHE_COLD && cache <= IMD_PERF_CACHE_REPEAT && std::isfinite(clock) &&
         clock >= 0;
}

bool request_valid(const imd_perf_request_t* req, Req* q) {
  *q = Req{};
  if (req == nullptr) return true;
  if (req->struct_size < kRequestV1Size) return false;
  if ((req->flags & ~(IMD_PERF_BETA_NONZERO | IMD_PERF_ALPHA_ZERO)) != 0) return false;
  if (req->k4_config < -1 || req->k4_config >= IMD_K4_CONFIG_COUNT) return false;
  if (!eps_valid(req->eps_fraction)) return false;
  q->flags = req->flags, q->k4 = req->k4_config, q->eps = req->eps_fraction;
  if (req->struct_size >= kRequestV7Size) {
    if (!v7_valid(req->regime, req->cache, req->clock_mhz)) return false;
    q->regime = req->regime, q->cache = req->cache, q->clock = req->clock_mhz;
  }
  if (req->struct_size >= kRequestV8Size && !err_request_valid(req->accuracy, &q->err))
    return false;
  return true;
}

// The common tail of predict / predict_pack: zero everything but struct_size and status on
// failure, copy struct_size bytes.
imd_status_t finish(imd_perf_t& r, imd_perf_t* out) {
  if (r.status != IMD_STATUS_SUCCESS) {
    const uint32_t size = r.struct_size;
    const int32_t st = r.status;
    std::memset(&r, 0, sizeof r);
    r.struct_size = size, r.status = st;
  }
  std::memcpy(out, &r, std::min<size_t>(out->struct_size, sizeof r));
  return static_cast<imd_status_t>(r.status);
}

}  // namespace
}  // namespace imd::detail

using imd::detail::kPerfV1Size;

extern "C" {

const char* imd_perf_class_name(int32_t cls) {
  return cls >= 0 && cls < IMD_PERF_CLASS_COUNT ? imd::detail::kClassNames[cls] : nullptr;
}

const char* imd_perf_param_name(int32_t cls, int32_t j) {
  if (cls < 0 || cls >= IMD_PERF_CLASS_COUNT || j < 0 || j >= IMD_PERF_MAX_PARAMS) return nullptr;
  return imd::detail::kParamNames[cls][j];
}

const char* imd_perf_phase_name(int32_t phase) {
  return phase >= 0 && phase < IMD_PERF_PHASE_COUNT ? imd::detail::kPhaseNames[phase] : nullptr;
}

const char* imd_perf_kernel_name(int32_t kernel) { return imd::detail::kernel_name(kernel); }

imd_status_t imd_perf_calib_default(imd_perf_calib_t* out) {
  return imd::detail::with_line(nullptr, [&](imd::detail::CalibErr* e) {
    return imd::detail::calib_default(out, e);
  });
}

imd_status_t imd_perf_calib_default_ex(imd_perf_calib_t* out, imd_perf_calib_error_t* err) {
  return imd::detail::with_error(err, [&](imd::detail::CalibErr* e) {
    return imd::detail::calib_default(out, e);
  });
}

imd_status_t imd_perf_calib_parse(const char* text, size_t length, imd_perf_calib_t* out,
                                  int32_t* error_line) {
  return imd::detail::with_line(error_line, [&](imd::detail::CalibErr* e) {
    return imd::detail::calib_parse(text, length, out, e);
  });
}

imd_status_t imd_perf_calib_parse_ex(const char* text, size_t length, imd_perf_calib_t* out,
                                     imd_perf_calib_error_t* err) {
  return imd::detail::with_error(err, [&](imd::detail::CalibErr* e) {
    return imd::detail::calib_parse(text, length, out, e);
  });
}

imd_status_t imd_perf_calib_load(const char* path, imd_perf_calib_t* out, int32_t* error_line) {
  return imd::detail::with_line(error_line, [&](imd::detail::CalibErr* e) {
    return imd::detail::calib_load(path, out, e);
  });
}

imd_status_t imd_perf_calib_load_ex(const char* path, imd_perf_calib_t* out,
                                    imd_perf_calib_error_t* err) {
  return imd::detail::with_error(err, [&](imd::detail::CalibErr* e) {
    return imd::detail::calib_load(path, out, e);
  });
}

const char* imd_perf_calib_error_name(int32_t reason) {
  static const char* const kNames[IMD_PERF_CALIB_ERR_COUNT] = {
      "IMD_PERF_CALIB_OK",
      "IMD_PERF_CALIB_ERR_ARGUMENT",
      "IMD_PERF_CALIB_ERR_FILE",
      "IMD_PERF_CALIB_ERR_SYNTAX",
      "IMD_PERF_CALIB_ERR_UNKNOWN_KEY",
      "IMD_PERF_CALIB_ERR_REPEATED_KEY",
      "IMD_PERF_CALIB_ERR_VALUE",
      "IMD_PERF_CALIB_ERR_TOO_LONG",
      "IMD_PERF_CALIB_ERR_MISSING_KEY",
      "IMD_PERF_CALIB_ERR_FORMAT2_KEY",
      "IMD_PERF_CALIB_ERR_INCONSISTENT"};
  return reason >= 0 && reason < IMD_PERF_CALIB_ERR_COUNT ? kNames[reason] : nullptr;
}

imd_status_t imd_perf_calib_format(const imd_perf_calib_t* calib, char* buf, size_t capacity,
                                   size_t* length) {
  if (calib == nullptr || length == nullptr || (buf == nullptr && capacity > 0) ||
      calib->struct_size < imd::detail::kCalibV1Size)
    return IMD_STATUS_INVALID_VALUE;
  imd_perf_calib_t c;
  std::memset(&c, 0, sizeof c);
  std::memcpy(&c, calib, std::min<size_t>(calib->struct_size, sizeof c));
  if (calib->struct_size < imd::detail::kCalibV7Size && c.format == 2) c.format = 1;
  if (!imd::detail::calib_valid(c)) return IMD_STATUS_INVALID_VALUE;
  if (c.format == 1) imd::detail::to_format1_params(&c);
  for (const auto& [p, cap] : {std::pair<const char*, size_t>{c.gpu, sizeof c.gpu},
                               {c.date, sizeof c.date},
                               {c.version, sizeof c.version},
                               {c.clocks, sizeof c.clocks},
                               {c.notes, sizeof c.notes}}) {
    const size_t len = strnlen(p, cap);
    if (len == cap || std::memchr(p, '\n', len) != nullptr || std::memchr(p, '\r', len) != nullptr)
      return IMD_STATUS_INVALID_VALUE;  // unterminated, or not one line
  }
  return imd::detail::guarded([&] {
    const std::string s = imd::detail::format_calib(c);
    *length = s.size();
    if (capacity <= s.size()) return IMD_STATUS_INSUFFICIENT_WORKSPACE;
    std::memcpy(buf, s.c_str(), s.size() + 1);
    return IMD_STATUS_SUCCESS;
  });
}

imd_status_t imd_perf_predict(const imd_tensor_t* A, const imd_tensor_t* B, const imd_tensor_t* C,
                              const imd_options_t* options, const imd_packing_t* packing,
                              const imd_perf_request_t* request, const imd_perf_calib_t* calib,
                              imd_perf_t* out) {
  if (out == nullptr || out->struct_size < kPerfV1Size) return IMD_STATUS_INVALID_VALUE;
  imd::detail::Req q;
  if (!imd::detail::request_valid(request, &q)) return IMD_STATUS_INVALID_VALUE;
  imd_perf_t r;
  std::memset(&r, 0, sizeof r);
  r.struct_size = out->struct_size;
  r.status = imd::detail::guarded([&] {
    imd_perf_calib_t scratch;
    const imd_perf_calib_t* cal = nullptr;
    if (const imd_status_t st = imd::detail::resolve_calib(calib, &scratch, &cal);
        st != IMD_STATUS_SUCCESS)
      return st;
    return imd::detail::predict_contract(A, B, C, options, packing,
                                         imd::detail::k_align_sel(), q, *cal, &r);
  });
  return imd::detail::finish(r, out);
}

imd_status_t imd_perf_predict_pack(const imd_pack_desc_t* desc,
                                   const imd_perf_request_t* request,
                                   const imd_perf_calib_t* calib, imd_perf_t* out) {
  if (desc == nullptr || out == nullptr || out->struct_size < kPerfV1Size)
    return IMD_STATUS_INVALID_VALUE;
  imd::detail::Req q;
  if (!imd::detail::request_valid(request, &q)) return IMD_STATUS_INVALID_VALUE;
  q.flags = 0, q.k4 = -1;  // of the request only eps, regime, cache and clock are used
  imd_perf_t r;
  std::memset(&r, 0, sizeof r);
  r.struct_size = out->struct_size;
  r.status = imd::detail::guarded([&] {
    imd_perf_calib_t scratch;
    const imd_perf_calib_t* cal = nullptr;
    if (const imd_status_t st = imd::detail::resolve_calib(calib, &scratch, &cal);
        st != IMD_STATUS_SUCCESS)
      return st;
    return imd::detail::predict_pack(*desc, imd::detail::k_align_sel(), q, *cal, &r);
  });
  return imd::detail::finish(r, out);
}

imd_status_t imd_perf_rank(const imd_tensor_t* A, const imd_tensor_t* B, const imd_tensor_t* C,
                           const imd_options_t* base, const imd_perf_rank_request_t* req,
                           const imd_perf_calib_t* calib, imd_perf_candidate_t* out,
                           int32_t capacity, int32_t* num_candidates) {
  using namespace imd::detail;
  if (A == nullptr || B == nullptr || C == nullptr || base == nullptr || req == nullptr ||
      num_candidates == nullptr || req->struct_size < kRankRequestV1Size || capacity < 0 ||
      (capacity > 0 && (out == nullptr || req->candidate_size < kCandidateV1Size)))
    return IMD_STATUS_INVALID_VALUE;
  if (req->num_moduli_count < 1 || req->num_moduli == nullptr || req->residue_group_count < 0 ||
      (req->residue_group_count > 0 && req->residue_groups == nullptr) || req->reuse < 1 ||
      (req->vec_modes & ~(IMD_PERF_VEC_GENERATE | IMD_PERF_VEC_RESIDENT)) != 0 ||
      (req->pack_modes & ~(IMD_PERF_PACK_NONE | IMD_PERF_PACK_A | IMD_PERF_PACK_B |
                           IMD_PERF_PACK_AB)) != 0 ||
      (req->flags & ~IMD_PERF_BETA_NONZERO) != 0 || !eps_valid(req->eps_fraction))
    return IMD_STATUS_INVALID_VALUE;
  Req q;
  q.flags = req->flags, q.eps = req->eps_fraction;
  int32_t order = IMD_PERF_ORDER_TIME;  // a version-6 request: the v6 order
  if (req->struct_size >= kRankRequestV7Size) {
    if (!v7_valid(req->regime, req->cache, req->clock_mhz) ||
        (req->order != IMD_PERF_ORDER_TIES_BY_FOOTPRINT && req->order != IMD_PERF_ORDER_TIME))
      return IMD_STATUS_INVALID_VALUE;
    q.regime = req->regime, q.cache = req->cache, q.clock = req->clock_mhz;
    order = req->order;
  }
  // v8 (§6f): the accuracy request; a target > 0 adds IMD_PERF_INFEASIBLE_ACCURACY.
  if (req->struct_size >= kRankRequestV8Size && !err_request_valid(req->accuracy, &q.err))
    return IMD_STATUS_INVALID_VALUE;
  for (int32_t i = 0; i < req->num_moduli_count; ++i)
    if (req->num_moduli[i] < 2 || req->num_moduli[i] > c::kNMax) return IMD_STATUS_INVALID_VALUE;
  for (int32_t i = 0; i < req->residue_group_count; ++i)
    if (req->residue_groups[i] < 1) return IMD_STATUS_INVALID_VALUE;
  return guarded([&] {
    imd_perf_calib_t scratch;
    const imd_perf_calib_t* calp = nullptr;
    if (const imd_status_t st = resolve_calib(calib, &scratch, &calp); st != IMD_STATUS_SUCCESS)
      return st;
    const imd_perf_calib_t& cal = *calp;
    // DESIGN §4b.5: one hook snapshot for every candidate's plan and packs.
    const KAlignSel sel = k_align_sel();
    const int32_t vec_mask = req->vec_modes != 0 ? req->vec_modes : IMD_PERF_VEC_GENERATE;
    const int32_t pack_mask = req->pack_modes != 0 ? req->pack_modes : IMD_PERF_PACK_NONE;
    struct Cand {
      imd_perf_candidate_t c;
      int64_t order;
    };
    std::vector<Cand> cands;
    for (int32_t ni = 0; ni < req->num_moduli_count; ++ni) {
      const int32_t N = req->num_moduli[ni];
      std::vector<int32_t> gs;
      if (req->residue_group_count == 0)
        for (int32_t g = 1; g <= N; ++g) gs.push_back(g);
      else
        for (int32_t i = 0; i < req->residue_group_count; ++i)
          if (req->residue_groups[i] <= N) gs.push_back(req->residue_groups[i]);
      for (const int32_t g : gs)
        for (const int32_t vec : {IMD_VEC_GENERATE, IMD_VEC_RESIDENT}) {
          if (!(vec_mask & (vec == IMD_VEC_GENERATE ? IMD_PERF_VEC_GENERATE : IMD_PERF_VEC_RESIDENT)))
            continue;
          for (const int32_t pm : {IMD_PERF_PACK_NONE, IMD_PERF_PACK_A, IMD_PERF_PACK_B,
                                   IMD_PERF_PACK_AB}) {
            if (!(pack_mask & pm)) continue;
            const int32_t packed = pm == IMD_PERF_PACK_NONE ? 0
                                   : pm == IMD_PERF_PACK_A  ? 1
                                   : pm == IMD_PERF_PACK_B  ? 2
                                                            : 3;
            imd_options_t o = *base;
            o.num_moduli = N, o.residue_group = g, o.vec_mode = vec;
            Cand cd{};
            imd_perf_candidate_t& c = cd.c;
            cd.order = static_cast<int64_t>(cands.size());
            c.num_moduli = N, c.residue_group = g, c.vec_mode = vec, c.packed = packed;
            // The packed operands' descriptors: the unpacked plan's orders (§6b.2).
            imd_pack_desc_t desc[2];
            std::vector<int32_t> labels[2];
            imd_status_t st = IMD_STATUS_SUCCESS;
            for (int x = 0; x < 2 && st == IMD_STATUS_SUCCESS; ++x) {
              if (!(packed & (1 << x))) continue;
              const imd_tensor_t* X = x == 0 ? A : B;
              labels[x].assign(static_cast<size_t>(std::max(X->rank, 1)), 0);
              std::memset(&desc[x], 0, sizeof desc[x]);
              desc[x].struct_size = sizeof desc[x];
              st = pack_desc_for(A, B, C, &o, sel, x, &desc[x], labels[x].data(),
                                 static_cast<int32_t>(labels[x].size()));
            }
            const imd_packing_t packing = {(packed & 1) ? &desc[0] : nullptr,
                                           (packed & 2) ? &desc[1] : nullptr};
            imd_perf_t pr;
            std::memset(&pr, 0, sizeof pr);
            if (st == IMD_STATUS_SUCCESS)
              st = predict_contract(A, B, C, &o, packed ? &packing : nullptr, sel, q, cal,
                                    &pr);
            double pack_s = 0, pack_eps = 0, pack_lo = 0, pack_hi = 0;
            uint64_t pack_bytes = 0, pack_ws = 0;
            for (int x = 0; x < 2 && st == IMD_STATUS_SUCCESS; ++x) {
              if (!(packed & (1 << x))) continue;
              imd_perf_t pp;
              std::memset(&pp, 0, sizeof pp);
              st = predict_pack(desc[x], sel, q, cal, &pp);
              pack_s += pp.time_s;
              pack_eps += pp.eps_s;
              pack_lo += pp.time_lo_s;
              pack_hi += pp.time_hi_s;
              pack_bytes += pp.pack_bytes;
              pack_ws += pp.pack_ws_bytes;
            }
            c.status = st;
            if (st == IMD_STATUS_SUCCESS) {
              c.k4_config = pr.k4_config, c.dispatch = pr.dispatch;
              c.contract_s = pr.time_s, c.pack_s = pack_s;
              c.time_s = pr.time_s + pack_s / req->reuse;
              c.tflops_eq = c.time_s > 0 ? pr.flops_eq / c.time_s * 1e-12 : 0;
              c.eps_s = pr.eps_s + pack_eps / req->reuse;
              c.eps_fraction = pr.eps_fraction;
              c.workspace_bytes = pr.workspace_bytes;
              c.plan_device_bytes = pr.plan_device_bytes;
              c.pack_bytes = pack_bytes, c.pack_ws_bytes = pack_ws;
              uint64_t total = 0;
              const bool fits = add_u64(c.workspace_bytes, c.plan_device_bytes, &total) &&
                                add_u64(total, pack_bytes, &total) && add_u64(total, pack_ws, &total);
              c.total_bytes = fits ? total : UINT64_MAX;
              if (!(req->budget_bytes == 0 || c.total_bytes <= req->budget_bytes))
                c.infeasible |= IMD_PERF_INFEASIBLE_BUDGET;
              c.kernel_launches = pr.kernel_launches;
              c.time_lo_s = pr.time_lo_s + pack_lo / req->reuse;
              c.time_hi_s = pr.time_hi_s + pack_hi / req->reuse;
              c.clock_mhz = pr.clock_mhz, c.power_w = pr.power_w;
              // v8 (§6f): the contract's estimates (predict_contract, the request's accuracy)
              c.err_bound = pr.err_bound, c.err_typical = pr.err_typical;
              c.err_status = pr.err_status;
              if (pr.err_status == IMD_ERR_MISSED) c.infeasible |= IMD_PERF_INFEASIBLE_ACCURACY;
            } else {
              c.infeasible = IMD_PERF_INFEASIBLE_STATUS;
              c.err_bound = c.err_typical = std::numeric_limits<double>::infinity();  // no plan
            }
            c.feasible = c.infeasible == 0 ? 1 : 0;
            cands.push_back(cd);
          }
        }
    }
    // The v6 order: feasible by time, then infeasible SUCCESS by time, then the rest in
    // enumeration order.
    const auto rank_of = [](const imd_perf_candidate_t& c) {
      return c.status != IMD_STATUS_SUCCESS ? 2 : c.feasible ? 0 : 1;
    };
    const auto tail = [](const imd_perf_candidate_t& x, const imd_perf_candidate_t& y) {
      if (x.num_moduli != y.num_moduli) return x.num_moduli < y.num_moduli;
      if (x.residue_group != y.residue_group) return x.residue_group < y.residue_group;
      if (x.vec_mode != y.vec_mode) return x.vec_mode < y.vec_mode;
      return x.packed < y.packed;
    };
    std::stable_sort(cands.begin(), cands.end(), [&](const Cand& x, const Cand& y) {
      const int rx = rank_of(x.c), ry = rank_of(y.c);
      if (rx != ry) return rx < ry;
      if (rx == 2) return x.order < y.order;
      if (x.c.time_s != y.c.time_s) return x.c.time_s < y.c.time_s;
      if (x.c.total_bytes != y.c.total_bytes) return x.c.total_bytes < y.c.total_bytes;
      return tail(x.c, y.c);
    });
    for (size_t i = 0; i < cands.size(); ++i) cands[i].c.raw_rank = static_cast<int32_t>(i);
    // v7 (DESIGN §6c.1 "Prediction intervals and ties"): in every order, the feasible
    // candidates within tie_ratio x the best's time are ties; ORDER_TIES_BY_FOOTPRINT puts them
    // first, by footprint. (A time ratio, not band overlap: candidates of one contraction share
    // most of the model's error, and overlapping 80 % bands tied candidates 1.2-1.4x apart.)
    if (!cands.empty() && rank_of(cands[0].c) == 0) {
      const double best_tie = cands[0].c.time_s * cal.tie_ratio;
      for (Cand& cd : cands)
        cd.c.tie = rank_of(cd.c) == 0 && cd.c.time_s <= best_tie ? 1 : 0;
      if (order == IMD_PERF_ORDER_TIES_BY_FOOTPRINT)
        std::stable_sort(cands.begin(), cands.end(), [&](const Cand& x, const Cand& y) {
          if (x.c.tie != y.c.tie) return x.c.tie > y.c.tie;
          if (!x.c.tie) return x.c.raw_rank < y.c.raw_rank;
          if (x.c.total_bytes != y.c.total_bytes) return x.c.total_bytes < y.c.total_bytes;
          if (x.c.time_s != y.c.time_s) return x.c.time_s < y.c.time_s;
          return tail(x.c, y.c);
        });
    }
    const size_t n = cands.size();
    if (n > static_cast<size_t>(INT32_MAX)) return IMD_STATUS_INVALID_VALUE;
    auto* dst = reinterpret_cast<unsigned char*>(out);
    for (size_t i = 0; i < std::min(n, static_cast<size_t>(capacity)); ++i)
      std::memcpy(dst + i * req->candidate_size, &cands[i].c,
                  std::min<size_t>(req->candidate_size, sizeof(imd_perf_candidate_t)));
    *num_candidates = static_cast<int32_t>(n);
    return IMD_STATUS_SUCCESS;
  });
}

}  // extern "C"
