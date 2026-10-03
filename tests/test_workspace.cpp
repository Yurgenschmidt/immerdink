// imd_workspace_size (and the C++ lowering behind it) against spec/params.py:
// reads the vectors written by tests/gen_workspace_vectors.py.
#include <cstdint>
#include <cstdio>
#include <fstream>
#include <map>
#include <stdexcept>
#include <string>
#include <vector>

#include "immerdink/immerdink.h"
#include "index_map.hpp"
#include "workspace.hpp"

namespace {

struct Tensor {
  std::vector<int64_t> extent, stride;
  std::vector<int32_t> label;
  imd_tensor_t view() const {
    return {static_cast<int32_t>(extent.size()), extent.data(), stride.data(), label.data()};
  }
};

struct Region {
  std::string name;
  uint64_t start = 0, end = 0;
};

struct Case {
  std::string name;
  imd_options_t opts{};
  Tensor t[3];
  std::string status;
  uint64_t bytes = 0;
  int64_t m = 0, n = 0, k = 0, b = 0;
  int swapped = 0;
  std::vector<Region> regions;
};

const std::map<std::string, imd_status_t> kStatus = {
    {"SUCCESS", IMD_STATUS_SUCCESS},
    {"INVALID_VALUE", IMD_STATUS_INVALID_VALUE},
    {"NOT_SUPPORTED", IMD_STATUS_NOT_SUPPORTED},
};

const char* kRegionName[] = {"sA", "sB", "fA", "fB", "vec", "RA", "RB", "W"};

bool read_case(std::istream& in, Case* c) {
  std::string tok;
  if (!(in >> tok)) return false;
  if (tok != "case") throw std::runtime_error("expected 'case', got " + tok);
  in >> c->name;
  for (;;) {
    in >> tok;
    if (tok == "opts") {
      in >> c->opts.num_moduli >> c->opts.residue_group >> c->opts.rounding >> c->opts.mode >>
          c->opts.vec_mode;
    } else if (tok == "tensor") {
      std::string which;
      int rank = 0;
      in >> which >> rank;
      Tensor& t = c->t[which[0] - 'A'];
      t.extent.resize(rank);
      t.stride.resize(rank);
      t.label.resize(rank);
      for (auto& v : t.extent) in >> v;
      for (auto& v : t.stride) in >> v;
      for (auto& v : t.label) in >> v;
    } else if (tok == "expect") {
      in >> c->status;
      if (c->status == "SUCCESS") in >> c->bytes >> c->m >> c->n >> c->k >> c->b >> c->swapped;
    } else if (tok == "region") {
      Region r;
      in >> r.name >> r.start >> r.end;
      c->regions.push_back(r);
    } else if (tok == "end") {
      return true;
    } else {
      throw std::runtime_error("bad token " + tok);
    }
    if (!in) throw std::runtime_error("truncated case " + c->name);
  }
}

int failures = 0;

template <class... Args>
void fail(const Case& c, const char* fmt, Args... args) {
  if (++failures <= 25) {
    std::fprintf(stderr, "FAIL %s: ", c.name.c_str());
    std::fprintf(stderr, fmt, args...);
    std::fprintf(stderr, "\n");
  }
}

void check(const Case& c) {
  const imd_tensor_t A = c.t[0].view(), B = c.t[1].view(), C = c.t[2].view();
  size_t bytes = 12345;
  const imd_status_t st = imd_workspace_size(&A, &B, &C, &c.opts, &bytes);
  const imd_status_t want = kStatus.at(c.status);
  if (st != want) {
    fail(c, "status %s, expected %s", imd_status_string(st), imd_status_string(want));
    return;
  }
  if (want != IMD_STATUS_SUCCESS) return;
  if (bytes != c.bytes)
    fail(c, "bytes %llu, expected %llu", (unsigned long long)bytes, (unsigned long long)c.bytes);

  imd::detail::Lowered lw;
  if (imd::detail::lower(A, B, C, &lw) != IMD_STATUS_SUCCESS) return fail(c, "lower failed");
  if (lw.m != c.m || lw.n != c.n || lw.k != c.k || lw.b != c.b || int(lw.swapped) != c.swapped)
    fail(c, "lowered (m,n,k,b,swap) = (%lld,%lld,%lld,%lld,%d), expected (%lld,%lld,%lld,%lld,%d)",
         (long long)lw.m, (long long)lw.n, (long long)lw.k, (long long)lw.b, int(lw.swapped),
         (long long)c.m, (long long)c.n, (long long)c.k, (long long)c.b, c.swapped);

  const int g = c.opts.residue_group == 0 ? c.opts.num_moduli : c.opts.residue_group;
  const auto L = imd::detail::workspace_layout(
      {lw.m, lw.n, lw.k, lw.b, c.opts.num_moduli, g, c.opts.mode == IMD_DIRECT,
       c.opts.vec_mode == IMD_VEC_RESIDENT});
  if (!L) return fail(c, "layout overflow");
  std::vector<Region> got;
  for (int r = 0; r < 8; ++r)
    if (L->bytes[r] != 0) got.push_back({kRegionName[r], L->offset[r], L->offset[r] + L->bytes[r]});
  bool same = got.size() == c.regions.size();
  for (size_t i = 0; same && i < got.size(); ++i)
    same = got[i].name == c.regions[i].name && got[i].start == c.regions[i].start &&
           got[i].end == c.regions[i].end;
  if (!same) fail(c, "region offsets differ from params.region_offsets");
}

// Argument checks that the vector file cannot express (null pointers).
void check_null_arguments() {
  const int64_t e[2] = {3, 4}, s[2] = {4, 1};
  const int32_t li[2] = {1, 3}, lb[2] = {3, 2}, lc[2] = {1, 2};
  const imd_tensor_t A{2, e, s, li}, B{2, e + 1, s, lb}, C{2, e, s, lc};
  const imd_options_t o{9, 0, IMD_RN, IMD_EMULATE, IMD_VEC_GENERATE};
  size_t bytes = 0;
  Case c;
  c.name = "null_arguments";
  if (imd_workspace_size(nullptr, &B, &C, &o, &bytes) != IMD_STATUS_INVALID_VALUE) fail(c, "A");
  if (imd_workspace_size(&A, &B, &C, nullptr, &bytes) != IMD_STATUS_INVALID_VALUE) fail(c, "opts");
  if (imd_workspace_size(&A, &B, &C, &o, nullptr) != IMD_STATUS_INVALID_VALUE) fail(c, "bytes");
  const imd_tensor_t Anull{2, nullptr, s, li};
  if (imd_workspace_size(&Anull, &B, &C, &o, &bytes) != IMD_STATUS_INVALID_VALUE) fail(c, "extent");
  const imd_tensor_t Aneg{-1, nullptr, nullptr, nullptr};
  if (imd_workspace_size(&Aneg, &B, &C, &o, &bytes) != IMD_STATUS_INVALID_VALUE) fail(c, "rank");
  if (imd_workspace_size(&A, &B, &C, &o, &bytes) != IMD_STATUS_SUCCESS) fail(c, "valid call");
}

}  // namespace

int main(int argc, char** argv) {
  if (argc != 2) {
    std::fprintf(stderr, "usage: %s workspace_vectors.txt\n", argv[0]);
    return 2;
  }
  std::ifstream in(argv[1]);
  if (!in) {
    std::fprintf(stderr, "cannot open %s\n", argv[1]);
    return 2;
  }
  std::map<std::string, int> by_status;
  int n = 0;
  try {
    for (Case c; read_case(in, &c); c = Case{}) {
      check(c);
      ++by_status[c.status];
      ++n;
    }
  } catch (const std::exception& e) {
    std::fprintf(stderr, "parse error: %s\n", e.what());
    return 2;
  }
  check_null_arguments();
  std::printf("%d cases (", n);
  for (const auto& [s, cnt] : by_status) std::printf(" %s=%d", s.c_str(), cnt);
  std::printf(" ), %d failures\n", failures);
  return failures == 0 && n > 0 ? 0 : 1;
}
