// The performance model (DESIGN v1.8 §6c): features of every phase from the plan's lowering,
// the calibration file (format 1), imd_perf_predict, imd_perf_predict_pack, imd_perf_rank.
// Pure: no global or static mutable state, no allocation visible to the caller, thread-safe.
#include "perf.hpp"

#include <algorithm>
#include <charconv>
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
constexpr int64_t kK4TileK = 64;

const char* const kClassNames[IMD_PERF_CLASS_COUNT] = {
    "k0", "kf", "kl", "pack", "k4_128x256", "k4_128x128", "k4_64x128", "k4_64x64",
    "k5", "direct", "beta", "call"};
// Feature names per class (nullptr: the class does not use that slot).
const char* const kParamNames[IMD_PERF_CLASS_COUNT][kP] = {
    {"launch", "byte", nullptr, nullptr, nullptr, nullptr},          // k0
    {"launch", "byte", "op", "row", "op_eps", nullptr},              // kf
    {"launch", "byte", "op", "l2byte", "op_eps", nullptr},           // kl
    {"launch", "byte", "op", "op_eps", nullptr, nullptr},            // pack
    {"round", "mac1", nullptr, nullptr, "byte", "mac_ktail"},        // k4_128x256 (1 CTA per SM)
    {"round", "mac1", "mac2", nullptr, "byte", "mac_ktail"},         // k4_128x128 (2)
    {"round", "mac1", "mac2", nullptr, "byte", "mac_ktail"},         // k4_64x128  (2)
    {"round", "mac1", "mac2", "mac3", "byte", "mac_ktail"},          // k4_64x64   (3)
    {"launch", "byte", "op", nullptr, nullptr, nullptr},             // k5
    {"launch", "byte", "flop", nullptr, nullptr, nullptr},           // direct
    {"launch", "byte", nullptr, nullptr, nullptr, nullptr},          // beta
    {"call", "launch", nullptr, nullptr, nullptr, nullptr},          // call (signed: a correction)
};
const char* const kPhaseNames[IMD_PERF_PHASE_COUNT] = {
    "K0 vec", "KF fused", "KL A", "KL B", "K2 pack A", "K3 pack B", "K4 gemm", "K5 crt",
    "DIRECT", "K5 beta-only", "call", "pack prep", "pack KL"};

static_assert(IMD_PERF_CLASS_COUNT * kP <= IMD_PERF_PARAM_CAPACITY, "param slots");
static_assert(IMD_PERF_CLASS_K4_128x256 + IMD_K4_64x64 == IMD_PERF_CLASS_K4_64x64,
              "K4 classes follow the IMD_K4_* order");

constexpr uint32_t kCalibV1Size = sizeof(imd_perf_calib_t);
constexpr uint32_t kRequestV1Size = sizeof(imd_perf_request_t);
constexpr uint32_t kPerfV1Size = sizeof(imd_perf_t);
constexpr uint32_t kRankRequestV1Size = sizeof(imd_perf_rank_request_t);
constexpr uint32_t kCandidateV1Size = sizeof(imd_perf_candidate_t);
static_assert(sizeof(imd_perf_phase_t) == 96, "imd_perf_phase_t v1 layout changed");
static_assert(sizeof(imd_perf_calib_t) == 1232, "imd_perf_calib_t v1 layout changed");
static_assert(sizeof(imd_perf_t) == 1648, "imd_perf_t v1 layout changed");
static_assert(sizeof(imd_perf_candidate_t) == 128, "imd_perf_candidate_t v1 layout changed");
static_assert(sizeof(imd_perf_rank_request_t) == 64, "imd_perf_rank_request_t v1 layout changed");
static_assert(sizeof(imd_perf_request_t) == 24, "imd_perf_request_t v1 layout changed");

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

// The op_eps feature of a phase: x[slot] = f * ops (f = the call's eps fraction).
void set_eps(PerfPhase& p, int slot, double ops, double f) {
  p.eps_slot = slot;
  p.eps_ops = ops;
  p.x[slot] = f * ops;
}

double u128_to_double(u128 v) { return static_cast<double>(v); }

// KL of one operand (rows, padded rows, g moduli of group 1, KL tile): launch, DRAM bytes
// (read the operand once, write group 1's residues, sigma and the flag), residue ops, the
// L2 re-read of rows longer than two tiles (passes 2 and 3 re-read the middle chunks), and
// the residue ops in the eps > 0 regime.
void add_kl(PerfPhases* P, int32_t phase, double rows, double pad, double k, double k_hat,
            double g, bool along_rows, int64_t k_hat_i, double f) {
  const double tk = static_cast<double>(int64_t{1} << kl_tile_lg_k(along_rows, k_hat_i));
  const double bytes = 8 * rows * k + g * pad * k_hat + 3 * rows;
  const double l2 = 16 * rows * std::max(0.0, k_hat - 2 * tk);
  set_eps(add_phase(P, phase, IMD_PERF_CLASS_KL, 1, {1, bytes, g * rows * k, l2}, bytes), 4,
          g * rows * k, f);
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
                          int32_t sms, double l2, double eps, PerfPhases* out) {
  PerfPhases& P = *out;
  P = PerfPhases{};
  if (e.path == Path::kNoop) return;
  const bool resident = an.options.vec_mode == IMD_VEC_RESIDENT;
  const int64_t v = resident ? 0 : 1;
  const double V = static_cast<double>(e.vec_count);
  const double b = static_cast<double>(e.b), m = static_cast<double>(e.m),
               n = static_cast<double>(e.n), k = static_cast<double>(e.k),
               nh = static_cast<double>(e.n_hat), kh = static_cast<double>(e.k_hat);
  const double out_bytes = 8 * b * m * n * ((flags & IMD_PERF_BETA_NONZERO) ? 2 : 1);
  const auto k0 = [&] {
    if (!resident) add_phase(&P, IMD_PERF_PHASE_K0, IMD_PERF_CLASS_K0, 1, {1, 8 * V}, 8 * V);
  };
  if ((flags & IMD_PERF_ALPHA_ZERO) || e.k == 0) {  // §3.7: K0 and the beta kernel
    k0();
    add_phase(&P, IMD_PERF_PHASE_BETA, IMD_PERF_CLASS_BETA, 1, {1, out_bytes}, out_bytes);
    P.launches = v + 1;
  } else if (!emulates(e)) {  // DIRECT dispatch (mode DIRECT or K-empty)
    k0();
    const double bytes = 8 * b * (m * k + n * k) + out_bytes, flops = 2 * b * m * n * k;
    add_phase(&P, IMD_PERF_PHASE_DIRECT, IMD_PERF_CLASS_DIRECT, 1, {1, bytes, flops}, bytes,
              flops, 2);
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
    if (!fused) k0();
    if (fused) {  // KF: K0 + K1 + group 1's packs of the in-line operands
      double bytes = resident ? 0 : 8 * V, ops = 0, r = 0;
      for (int x = 0; x < 2; ++x) {
        if (!inl[x]) continue;
        bytes += 8 * rows[x] * k + g * pad[x] * kh + 3 * rows[x];
        ops += g * rows[x] * k;
        r += rows[x];
      }
      set_eps(add_phase(&P, IMD_PERF_PHASE_KF, IMD_PERF_CLASS_KF, 1, {1, bytes, ops, r}, bytes),
              4, ops, f);
    } else {  // KL per in-line operand (v1.6: every in-line operand without KF)
      for (int x = 0; x < 2; ++x)
        if (inl[x])
          add_kl(&P, x == 0 ? IMD_PERF_PHASE_KL_A : IMD_PERF_PHASE_KL_B, rows[x], pad[x], k, kh,
                 g, e.pack_along_rows[x], e.k_hat, f);
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
      set_eps(add_phase(&P, x == 0 ? IMD_PERF_PHASE_PACK_A : IMD_PERF_PHASE_PACK_B,
                        IMD_PERF_CLASS_PACK, groups * per,
                        {static_cast<double>(groups * per), bytes, moduli * rows[x] * k}, bytes),
              3, moduli * rows[x] * k, f);
    }
    // K4: the A3 round model of the configuration, over the groups and the K-chunks.
    const int32_t cfg = k4 >= 0 ? k4 : k4_select(e, sms);
    P.k4_config = cfg;
    const K4Counts kc = plan_k4_counts(e, production_launch_limits());
    const int64_t chunk = c::kKcPrime;
    const int64_t gsz[2] = {e.g, e.N % e.g}, gcnt[2] = {e.N / e.g, e.N % e.g ? 1 : 0};
    const int64_t last = e.k_hat - (kc.k_chunks - 1) * chunk;
    const int64_t ksz[2] = {chunk, last}, kcnt[2] = {kc.k_chunks - 1, 1};
    double x[kP] = {0, 0, 0, 0, 0, 0};
    const int occ = kK4Configs[cfg].ctas_per_sm;
    for (int i = 0; i < 2; ++i)
      for (int j = 0; j < 2; ++j) {
        if (gcnt[i] == 0 || kcnt[j] == 0) continue;
        const K4Rounds r = k4_rounds(cfg, e.m, e.n_hat, ksz[j], gsz[i] * e.b, sms);
        const double cnt = static_cast<double>(gcnt[i]) * static_cast<double>(kcnt[j]);
        const double full = u128_to_double(r.full), macs = u128_to_double(r.tile_macs);
        x[0] += cnt * (full + (r.rest > 0 ? 1 : 0));
        x[occ] += cnt * full * occ * macs;
        if (r.rest > 0) x[r.rest] += cnt * r.rest * macs;
        // A chunk whose length is not a multiple of the 64-wide K tile runs a predicated
        // residue tile (CUTLASS 2.x MmaMultistage): its MACs again, at their extra cost.
        if (ksz[j] % kK4TileK != 0) x[kK4TailSlot] += cnt * (full * occ + r.rest) * macs;
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
    const double k4_ops = 2 * N * b * m * nh * kh;
    add_phase(&P, IMD_PERF_PHASE_K4, IMD_PERF_CLASS_K4_128x256 + cfg, kc.launches,
              {x[0], x[1], x[2], x[3], dram, x[kK4TailSlot]}, k4_bytes, k4_ops, 1);
    // K5: the N planes, sigma / tau and the flags read, C written (and read if beta != 0).
    const double k5_bytes = N * b * m * nh + 3 * b * (m + n) + out_bytes;
    add_phase(&P, IMD_PERF_PHASE_K5, IMD_PERF_CLASS_K5, 1,
              {1, k5_bytes, N * c::kTables.L32[e.N] * b * m * n}, k5_bytes);
    P.launches = plan_launches(e, fused, fl, resident, kc.launches);
  }
  add_phase(&P, IMD_PERF_PHASE_CALL, IMD_PERF_CLASS_CALL, 0,
            {1, static_cast<double>(P.launches)}, 0);
}

void perf_pack_phases(const PackLowered& pl, double eps, PerfPhases* out) {
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
  add_phase(&P, IMD_PERF_PHASE_PACK_PREP, IMD_PERF_CLASS_K0, 1, {1, prep}, prep);
  P.eps_fraction = perf_eps_fraction(pl.N, k, eps);
  P.eps_possible = perf_log2_T(pl.N) >= 53;
  add_kl(&P, IMD_PERF_PHASE_PACK_KL, b * r, b * r, k, kh, N, e.pack_along_rows[0], pl.k_hat,
         P.eps_fraction);
  P.launches = 2;
  add_phase(&P, IMD_PERF_PHASE_CALL, IMD_PERF_CLASS_CALL, 0, {1, 2}, 0);
}

double perf_phase_seconds(const imd_perf_calib_t& cal, const PerfPhase& p) {
  const double* th = cal.param + p.cls * kP;
  if (p.cls >= IMD_PERF_CLASS_K4_128x256 && p.cls <= IMD_PERF_CLASS_K4_64x64) {
    // K4: a roofline, max(the A3 round model, the DRAM term).
    double rounds = 0;
    for (int j = 0; j < kP; ++j)
      if (j != kK4ByteSlot) rounds += th[j] * p.x[j];
    return std::max(rounds, th[kK4ByteSlot] * p.x[kK4ByteSlot]) * 1e-9;
  }
  double ns = 0;
  for (int j = 0; j < kP; ++j) ns += th[j] * p.x[j];
  return ns * 1e-9;
}

namespace {

// ---- calibration text -----------------------------------------------------------------------

std::string_view trim(std::string_view s) {
  while (!s.empty() && (s.front() == ' ' || s.front() == '\t')) s.remove_prefix(1);
  while (!s.empty() && (s.back() == ' ' || s.back() == '\t' || s.back() == '\r')) s.remove_suffix(1);
  return s;
}

// A number, optionally followed by whitespace and a '#' comment.
bool parse_number(std::string_view v, double* out) {
  const char* first = v.data();
  const char* last = v.data() + v.size();
  if (first != last && *first == '+') ++first;
  const auto res = std::from_chars(first, last, *out, std::chars_format::general);
  if (res.ec != std::errc() || !std::isfinite(*out)) return false;
  const std::string_view rest = trim(std::string_view(res.ptr, static_cast<size_t>(last - res.ptr)));
  return rest.empty() || rest.front() == '#';
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

bool calib_valid(const imd_perf_calib_t& c) {
  if (c.format != IMD_PERF_CALIB_FORMAT || c.sm_count < 1) return false;
  for (const double pk : {c.peak_dram_bytes_per_s, c.peak_int8_ops_per_s, c.peak_fp64_flops_per_s,
                          c.l2_bytes})
    if (!(pk > 0) || !std::isfinite(pk)) return false;
  for (int cl = 0; cl < IMD_PERF_CLASS_COUNT; ++cl)
    for (int j = 0; j < kP; ++j) {
      const double v = c.param[cl * kP + j];
      if (!std::isfinite(v) || (cl != IMD_PERF_CLASS_CALL && v < 0)) return false;
    }
  return true;
}

imd_status_t parse_calib(std::string_view text, imd_perf_calib_t* out, int32_t* error_line) {
  imd_perf_calib_t r;
  std::memset(&r, 0, sizeof r);
  r.struct_size = sizeof r;
  const auto fail = [&](int32_t line) {
    if (error_line != nullptr) *error_line = line;
    return IMD_STATUS_INVALID_VALUE;
  };
  // Seen keys: 0 format, 1 gpu, 2 sm_count, 3..5 peaks, 6..9 strings, 10 l2_bytes, then the
  // params.
  constexpr int kFixed = 11;
  bool seen[kFixed + IMD_PERF_PARAM_CAPACITY] = {};
  StrField strs[5] = {{"gpu", r.gpu, sizeof r.gpu},
                      {"date", r.date, sizeof r.date},
                      {"version", r.version, sizeof r.version},
                      {"clocks", r.clocks, sizeof r.clocks},
                      {"notes", r.notes, sizeof r.notes}};
  const int str_seen[5] = {1, 6, 7, 8, 9};
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
    if (eq == std::string_view::npos) return fail(line);
    const std::string_view key = trim(s.substr(0, eq)), val = trim(s.substr(eq + 1));
    int slot = -1;
    if (key == "format") {
      int64_t f = 0;
      if (!parse_int(val, &f) || f != IMD_PERF_CALIB_FORMAT) return fail(line);
      r.format = static_cast<int32_t>(f), slot = 0;
    } else if (key == "sm_count") {
      int64_t s2 = 0;
      if (!parse_int(val, &s2) || s2 < 1 || s2 > 1000000) return fail(line);
      r.sm_count = static_cast<int32_t>(s2), slot = 2;
    } else if (key == "peak_dram_bytes_per_s" || key == "peak_int8_ops_per_s" ||
               key == "peak_fp64_flops_per_s" || key == "l2_bytes") {
      double d = 0;
      if (!parse_number(val, &d) || !(d > 0)) return fail(line);
      if (key == "peak_dram_bytes_per_s") r.peak_dram_bytes_per_s = d, slot = 3;
      else if (key == "peak_int8_ops_per_s") r.peak_int8_ops_per_s = d, slot = 4;
      else if (key == "peak_fp64_flops_per_s") r.peak_fp64_flops_per_s = d, slot = 5;
      else r.l2_bytes = d, slot = 10;
    } else {
      for (int i = 0; i < 5 && slot < 0; ++i)
        if (key == strs[i].key) {
          if (val.size() >= strs[i].cap) return fail(line);
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
              if (!parse_number(val, &d) || (cl != IMD_PERF_CLASS_CALL && d < 0)) return fail(line);
              r.param[cl * kP + j] = d;
              slot = kFixed + cl * kP + j;
            }
        }
      }
    }
    if (slot < 0 || seen[slot]) return fail(line);  // unknown or repeated key
    seen[slot] = true;
    if (eol == text.size()) break;
  }
  // Required: format, gpu, sm_count, the peaks and every parameter of every class.
  for (const int i : {0, 1, 2, 3, 4, 5, 10})
    if (!seen[i]) return fail(0);
  for (int cl = 0; cl < IMD_PERF_CLASS_COUNT; ++cl)
    for (int j = 0; j < kP; ++j)
      if (kParamNames[cl][j] != nullptr && !seen[kFixed + cl * kP + j]) return fail(0);
  r.struct_size = out->struct_size;
  std::memcpy(out, &r, std::min<size_t>(out->struct_size, sizeof r));
  if (error_line != nullptr) *error_line = 0;
  return IMD_STATUS_SUCCESS;
}

void append_number(std::string* s, double v) {
  char buf[64];
  const auto res = std::to_chars(buf, buf + sizeof buf, v);  // shortest round-trip form
  s->append(buf, res.ptr);
}

std::string format_calib(const imd_perf_calib_t& c) {
  std::string s;
  s += "# immerdink performance-model calibration (docs/DESIGN.md §6c.2), format 1.\n";
  s += "# Parameters: <class>.<feature> = ns per feature unit (§6c.1).\n";
  s += "format = 1\n";
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
      if (kParamNames[cl][j] == nullptr) continue;
      s += kClassNames[cl];
      s += '.';
      s += kParamNames[cl][j];
      s += " = ";
      append_number(&s, c.param[cl * kP + j]);
      s += '\n';
    }
  return s;
}

// The calibration a function uses: the caller's (validated) or the shipped one.
imd_status_t resolve_calib(const imd_perf_calib_t* in, imd_perf_calib_t* cal) {
  std::memset(cal, 0, sizeof *cal);
  if (in == nullptr) {
    cal->struct_size = sizeof *cal;
    return parse_calib(std::string_view(kPerfCalibDefault, sizeof kPerfCalibDefault - 1), cal,
                       nullptr);
  }
  if (in->struct_size < kCalibV1Size) return IMD_STATUS_INVALID_VALUE;
  std::memcpy(cal, in, sizeof *cal);
  return calib_valid(*cal) ? IMD_STATUS_SUCCESS : IMD_STATUS_INVALID_VALUE;
}

// ---- prediction ---------------------------------------------------------------------------

void fill_prediction(const imd_perf_calib_t& cal, const PerfPhases& P, imd_perf_t* r) {
  r->num_phases = P.n;
  r->k4_config = P.k4_config;
  r->sm_count = cal.sm_count;
  r->kernel_launches = P.launches;
  double t = 0, bound = 0, eps_s = 0;
  for (int i = 0; i < P.n; ++i) {
    const PerfPhase& p = P.ph[i];
    if (p.eps_slot >= 0 && P.eps_possible)
      eps_s += cal.param[p.cls * kP + p.eps_slot] * p.eps_ops * 1e-9;
    imd_perf_phase_t& o = r->phase[i];
    o.phase = p.phase, o.cls = p.cls, o.launches = p.launches;
    o.time_s = perf_phase_seconds(cal, p);
    o.bytes = p.bytes, o.ops = p.ops;
    const double peak = p.peak == 1 ? cal.peak_int8_ops_per_s
                        : p.peak == 2 ? cal.peak_fp64_flops_per_s
                                      : 0;
    o.bound_s = std::max(p.bytes / cal.peak_dram_bytes_per_s, peak > 0 ? p.ops / peak : 0.0);
    for (int j = 0; j < kP; ++j) o.x[j] = p.x[j];
    t += o.time_s;
    bound += o.bound_s;
  }
  r->time_s = t;
  r->bound_s = bound;
  r->eps_s = eps_s;
  r->eps_fraction = P.eps_fraction;
  r->tflops_eq = t > 0 ? r->flops_eq / t * 1e-12 : 0;
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
                              int32_t flags, int32_t k4, double eps,
                              const imd_perf_calib_t& cal, imd_perf_t* r) {
  Analysis an;
  if (const imd_status_t st = analyze_with(A, B, C, options, packing, &an); st != IMD_STATUS_SUCCESS)
    return st;
  ExecPlan e;
  if (const imd_status_t st = plan_data(an, &e); st != IMD_STATUS_SUCCESS) return st;
  PerfPhases P;
  perf_contract_phases(an, e, flags, emulates(e) ? k4 : -1, cal.sm_count, cal.l2_bytes, eps,
                       &P);
  r->dispatch = dispatch_of(e);
  r->workspace_bytes = an.layout.total;
  r->plan_device_bytes = e.device_bytes;
  r->flops_eq = 2.0 * static_cast<double>(e.b) * static_cast<double>(e.m) *
                static_cast<double>(e.n) * static_cast<double>(e.k);
  fill_prediction(cal, P, r);
  return IMD_STATUS_SUCCESS;
}

imd_status_t predict_pack(const imd_pack_desc_t& d, double eps, const imd_perf_calib_t& cal,
                          imd_perf_t* r) {
  PackLowered pl;
  if (const imd_status_t st = pack_lower(d, &pl); st != IMD_STATUS_SUCCESS) return st;
  PerfPhases P;
  perf_pack_phases(pl, eps, &P);
  r->dispatch = IMD_DISPATCH_EMULATE;
  r->pack_bytes = pl.layout.total;
  r->pack_ws_bytes = pl.ws_bytes;
  fill_prediction(cal, P, r);
  return IMD_STATUS_SUCCESS;
}

bool eps_valid(double f) { return f < 0 || (f >= 0 && f <= 1); }  // NaN: invalid

bool request_valid(const imd_perf_request_t* req, int32_t* flags, int32_t* k4, double* eps) {
  *flags = 0, *k4 = -1, *eps = -1;
  if (req == nullptr) return true;
  if (req->struct_size < kRequestV1Size) return false;
  if ((req->flags & ~(IMD_PERF_BETA_NONZERO | IMD_PERF_ALPHA_ZERO)) != 0) return false;
  if (req->k4_config < -1 || req->k4_config >= IMD_K4_CONFIG_COUNT) return false;
  if (!eps_valid(req->eps_fraction)) return false;
  *flags = req->flags, *k4 = req->k4_config, *eps = req->eps_fraction;
  return true;
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

imd_status_t imd_perf_calib_default(imd_perf_calib_t* out) {
  if (out == nullptr || out->struct_size < imd::detail::kCalibV1Size) return IMD_STATUS_INVALID_VALUE;
  return imd::detail::guarded([&] {
    return imd::detail::parse_calib(std::string_view(imd::detail::kPerfCalibDefault,
                                                     sizeof imd::detail::kPerfCalibDefault - 1),
                                    out, nullptr);
  });
}

imd_status_t imd_perf_calib_parse(const char* text, size_t length, imd_perf_calib_t* out,
                                  int32_t* error_line) {
  if (error_line != nullptr) *error_line = 0;
  if ((text == nullptr && length > 0) || out == nullptr ||
      out->struct_size < imd::detail::kCalibV1Size)
    return IMD_STATUS_INVALID_VALUE;
  return imd::detail::guarded([&] {
    return imd::detail::parse_calib(std::string_view(text == nullptr ? "" : text, length), out,
                                    error_line);
  });
}

imd_status_t imd_perf_calib_load(const char* path, imd_perf_calib_t* out, int32_t* error_line) {
  if (error_line != nullptr) *error_line = 0;
  if (path == nullptr || out == nullptr || out->struct_size < imd::detail::kCalibV1Size)
    return IMD_STATUS_INVALID_VALUE;
  return imd::detail::guarded([&] {
    std::FILE* f = std::fopen(path, "rb");
    if (f == nullptr) return IMD_STATUS_INVALID_VALUE;
    std::string text;
    char buf[4096];
    size_t got = 0;
    while ((got = std::fread(buf, 1, sizeof buf, f)) > 0 && text.size() < (size_t{1} << 20))
      text.append(buf, got);
    const bool err = std::ferror(f) != 0;
    std::fclose(f);
    if (err || text.size() >= (size_t{1} << 20)) return IMD_STATUS_INVALID_VALUE;
    return imd::detail::parse_calib(text, out, error_line);
  });
}

imd_status_t imd_perf_calib_format(const imd_perf_calib_t* calib, char* buf, size_t capacity,
                                   size_t* length) {
  if (calib == nullptr || length == nullptr || (buf == nullptr && capacity > 0) ||
      calib->struct_size < imd::detail::kCalibV1Size || !imd::detail::calib_valid(*calib))
    return IMD_STATUS_INVALID_VALUE;
  for (const auto& [p, cap] : {std::pair<const char*, size_t>{calib->gpu, sizeof calib->gpu},
                               {calib->date, sizeof calib->date},
                               {calib->version, sizeof calib->version},
                               {calib->clocks, sizeof calib->clocks},
                               {calib->notes, sizeof calib->notes}}) {
    const size_t len = strnlen(p, cap);
    if (len == cap || std::memchr(p, '\n', len) != nullptr || std::memchr(p, '\r', len) != nullptr)
      return IMD_STATUS_INVALID_VALUE;  // unterminated, or not one line
  }
  return imd::detail::guarded([&] {
    const std::string s = imd::detail::format_calib(*calib);
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
  int32_t flags = 0, k4 = -1;
  double eps = -1;
  if (!imd::detail::request_valid(request, &flags, &k4, &eps)) return IMD_STATUS_INVALID_VALUE;
  imd_perf_t r;
  std::memset(&r, 0, sizeof r);
  r.struct_size = out->struct_size;
  r.status = imd::detail::guarded([&] {
    imd_perf_calib_t cal;
    if (const imd_status_t st = imd::detail::resolve_calib(calib, &cal); st != IMD_STATUS_SUCCESS)
      return st;
    return imd::detail::predict_contract(A, B, C, options, packing, flags, k4, eps, cal, &r);
  });
  if (r.status != IMD_STATUS_SUCCESS) {  // only struct_size and status are meaningful
    const uint32_t size = r.struct_size;
    const int32_t st = r.status;
    std::memset(&r, 0, sizeof r);
    r.struct_size = size, r.status = st;
  }
  std::memcpy(out, &r, std::min<size_t>(out->struct_size, sizeof r));
  return static_cast<imd_status_t>(r.status);
}

imd_status_t imd_perf_predict_pack(const imd_pack_desc_t* desc,
                                   const imd_perf_request_t* request,
                                   const imd_perf_calib_t* calib, imd_perf_t* out) {
  if (desc == nullptr || out == nullptr || out->struct_size < kPerfV1Size)
    return IMD_STATUS_INVALID_VALUE;
  int32_t flags = 0, k4 = -1;
  double eps = -1;
  if (!imd::detail::request_valid(request, &flags, &k4, &eps)) return IMD_STATUS_INVALID_VALUE;
  imd_perf_t r;
  std::memset(&r, 0, sizeof r);
  r.struct_size = out->struct_size;
  r.status = imd::detail::guarded([&] {
    imd_perf_calib_t cal;
    if (const imd_status_t st = imd::detail::resolve_calib(calib, &cal); st != IMD_STATUS_SUCCESS)
      return st;
    return imd::detail::predict_pack(*desc, eps, cal, &r);
  });
  if (r.status != IMD_STATUS_SUCCESS) {
    const uint32_t size = r.struct_size;
    const int32_t st = r.status;
    std::memset(&r, 0, sizeof r);
    r.struct_size = size, r.status = st;
  }
  std::memcpy(out, &r, std::min<size_t>(out->struct_size, sizeof r));
  return static_cast<imd_status_t>(r.status);
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
  for (int32_t i = 0; i < req->num_moduli_count; ++i)
    if (req->num_moduli[i] < 2 || req->num_moduli[i] > c::kNMax) return IMD_STATUS_INVALID_VALUE;
  for (int32_t i = 0; i < req->residue_group_count; ++i)
    if (req->residue_groups[i] < 1) return IMD_STATUS_INVALID_VALUE;
  return guarded([&] {
    imd_perf_calib_t cal;
    if (const imd_status_t st = resolve_calib(calib, &cal); st != IMD_STATUS_SUCCESS) return st;
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
              st = imd_pack_desc_for(A, B, C, &o, x, &desc[x], labels[x].data(),
                                     static_cast<int32_t>(labels[x].size()));
            }
            const imd_packing_t packing = {(packed & 1) ? &desc[0] : nullptr,
                                           (packed & 2) ? &desc[1] : nullptr};
            imd_perf_t pr;
            std::memset(&pr, 0, sizeof pr);
            if (st == IMD_STATUS_SUCCESS)
              st = predict_contract(A, B, C, &o, packed ? &packing : nullptr, req->flags, -1,
                                    req->eps_fraction, cal, &pr);
            double pack_s = 0, pack_eps = 0;
            uint64_t pack_bytes = 0, pack_ws = 0;
            for (int x = 0; x < 2 && st == IMD_STATUS_SUCCESS; ++x) {
              if (!(packed & (1 << x))) continue;
              imd_perf_t pp;
              std::memset(&pp, 0, sizeof pp);
              st = predict_pack(desc[x], req->eps_fraction, cal, &pp);
              pack_s += pp.time_s;
              pack_eps += pp.eps_s;
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
              c.feasible = req->budget_bytes == 0 || c.total_bytes <= req->budget_bytes;
              c.kernel_launches = pr.kernel_launches;
            }
            cands.push_back(cd);
          }
        }
    }
    // Feasible by time, then infeasible SUCCESS by time, then the rest in enumeration order.
    const auto rank_of = [](const imd_perf_candidate_t& c) {
      return c.status != IMD_STATUS_SUCCESS ? 2 : c.feasible ? 0 : 1;
    };
    std::stable_sort(cands.begin(), cands.end(), [&](const Cand& x, const Cand& y) {
      const int rx = rank_of(x.c), ry = rank_of(y.c);
      if (rx != ry) return rx < ry;
      if (rx == 2) return x.order < y.order;
      if (x.c.time_s != y.c.time_s) return x.c.time_s < y.c.time_s;
      if (x.c.total_bytes != y.c.total_bytes) return x.c.total_bytes < y.c.total_bytes;
      if (x.c.num_moduli != y.c.num_moduli) return x.c.num_moduli < y.c.num_moduli;
      if (x.c.residue_group != y.c.residue_group) return x.c.residue_group < y.c.residue_group;
      if (x.c.vec_mode != y.c.vec_mode) return x.c.vec_mode < y.c.vec_mode;
      return x.c.packed < y.c.packed;
    });
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
