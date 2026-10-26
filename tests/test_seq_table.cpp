// Issue 21 / R2: the per-device-class instruction-sequence defaults of src/host/seq_config.hpp, checked without CUDA (the header is CUDA-free).
// sm_90 (H100, H200) = E1 + F1 + the binary64-limb CRT = 13 (the R2 combination rule; Q1 stays off); sm_89 (RTX 4090) = E1 = 1; sm_120 and every class
// not listed = 0 (the limb CRT is slower on 1/64-rate FP64 cards). With IMMERDINK_SEQ_TWINS=OFF (IMD_NO_SEQ_TWINS) the F1 bit is not a supported set and a class
// default is masked to what the build can run. Also the forcing rules: the hooks force the sets 0..7 and 128 + 4.. (a forced set has no limb-CRT bit), and the K5 hook's
// three states (-1 follow the set, 0 integer, 1 limb) pick the CRT variant. CPU only.
#include <cstdio>

#include "seq_config.hpp"

namespace {
int failures = 0;
#define CHECK(x)                                                         \
  do {                                                                   \
    if (!(x)) {                                                          \
      std::fprintf(stderr, "FAIL line %d: %s\n", __LINE__, #x);         \
      ++failures;                                                        \
    }                                                                    \
  } while (0)
}  // namespace

int main() {
  using namespace imd::detail;
#ifdef IMD_NO_SEQ_TWINS
  CHECK(seq_default_for(9, 0) == (kSeqEmit | kSeqLimbCrt));   // 9: F1 is not compiled
  CHECK(!seq_valid(kSeqFinish) && !seq_valid(3) && seq_valid(kSeqEmit) && seq_valid(0));
#else
  CHECK(seq_default_for(9, 0) == 13);
  CHECK(seq_default_for(9, 0) == (kSeqEmit | kSeqFinish | kSeqLimbCrt));
  CHECK((seq_default_for(9, 0) & kSeqQuant) == 0);             // Q1 is opt-in on sm_90
  for (int b = 0; b <= 7; ++b) CHECK(seq_valid(b) == ((b & kSeqQuant) == 0 || (b & kSeqEmit) != 0));
  // issue 23, k5fin = 128: a set needs F1 (its kernels are F1 kernels); no class row carries it (default OFF until a device class has measured it)
  CHECK(!seq_valid(kSeqK5Fin) && !seq_valid(kSeqK5Fin | kSeqEmit) && !seq_valid(kSeqK5Fin | kSeqQuant | kSeqEmit));
  CHECK(seq_valid(kSeqK5Fin | kSeqFinish) && seq_valid(kSeqK5Fin | kSeqFinish | kSeqEmit) && seq_valid(kSeqK5Fin | 7) && seq_valid(132) && seq_valid(133) && seq_valid(135));
  CHECK(!seq_valid(kSeqK5Fin | kSeqLimbCrt | kSeqFinish) && !seq_valid(256) && !seq_valid(kSeqK5Fin | 2));
  CHECK(seq_effective(13, 133) == 133 && seq_effective(13, 132) == 132 && seq_effective(13 | kSeqK5Fin, -1) == (13 | kSeqK5Fin));
  CHECK(!seq_limb_crt(seq_effective(13, 133), -1) && seq_limb_crt(seq_effective(13, 133), 1));   // a forced 133 runs the integer CRT unless the K5 hook forces limbs
#endif
  CHECK(!seq_valid(kSeqK5Fin) && !seq_valid(kSeqK5Fin | kSeqLimbCrt));   // (in any build: bit 128 alone is not a set)
  CHECK(seq_default_for(8, 9) == 1);                          // RTX 4090: E1 only, no FP64-pipe set
  CHECK(seq_default_for(12, 0) == 0);                         // Blackwell sm_120: the baseline
  CHECK(seq_default_for(10, 0) == 0 && seq_default_for(8, 0) == 0 && seq_default_for(8, 6) == 0 && seq_default_for(7, 5) == 0 && seq_default_for(0, 0) == 0);
  for (const SeqClass& c : kSeqClasses) {                     // no row but sm_90 carries the limb CRT; no row carries Q1
    CHECK((c.bits & kSeqQuant) == 0);
    CHECK(((c.bits & kSeqLimbCrt) != 0) == (c.cc_major == 9 && c.cc_minor == 0));
    CHECK(c.bits < 16);
    CHECK((c.bits & kSeqK5Fin) == 0);   // issue 23: k5fin is off in every class row
  }
  // forcing: the hooks accept -1 and 0..7 only; bit 8 is a class default
  CHECK(!seq_valid(kSeqLimbCrt) && !seq_valid(13) && !seq_valid(-2) && !seq_valid(16) && seq_valid(-1));
  // what a plan runs: the class set unless forced, a forced set is masked to 0..7
  CHECK(seq_effective(seq_default_for(9, 0), -1) == seq_default_for(9, 0));
  CHECK(seq_effective(seq_default_for(9, 0), -1) & kSeqLimbCrt);
  CHECK(seq_effective(13, 0) == 0 && seq_effective(13, 1) == 1 && seq_effective(13, 7) == 7);
  CHECK((seq_effective(13, 5) & kSeqLimbCrt) == 0);
  // the CRT variant: follow / force integer / force limb
  CHECK(seq_limb_crt(13, -1) && seq_limb_crt(9, -1) && !seq_limb_crt(5, -1) && !seq_limb_crt(0, -1));
  CHECK(!seq_limb_crt(13, 0) && !seq_limb_crt(9, 0));         // "seq=default crt=int" on sm_90: the integer CRT
  CHECK(seq_limb_crt(13, 1) && seq_limb_crt(0, 1) && seq_limb_crt(1, 1));
  CHECK(!seq_limb_crt(seq_effective(13, 5), -1));              // a forced seq 5 runs the integer CRT
  CHECK(seq_limb_crt(seq_effective(13, 5), 1));                // ... unless the K5 hook forces the limb CRT
  if (failures) {
    std::fprintf(stderr, "%d failures\n", failures);
    return 1;
  }
  std::printf("PASS  sm_90 %d, sm_89 %d, sm_120 %d\n", seq_default_for(9, 0), seq_default_for(8, 9), seq_default_for(12, 0));
  return 0;
}
