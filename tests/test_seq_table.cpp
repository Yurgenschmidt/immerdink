// Issue 21 / R2: the per-device-class instruction-sequence defaults of src/host/seq_config.hpp, checked without CUDA (the header is CUDA-free).
// sm_90 (H100, H200) = E1 + Q1 + F1 + the binary64-limb CRT + klwarp + klload + k5fin + klf64 = 447 (R2: 13, R4: 173, R5: 189, issue 21 2026-11-08: 447; klshape stays off); sm_89 (RTX 4090) = E1 = 1; sm_120 and every class
// not listed = 0 (the limb CRT is slower on 1/64-rate FP64 cards). With IMMERDINK_SEQ_TWINS=OFF (IMD_NO_SEQ_TWINS) the F1 bit is not a supported set and a class
// default is masked to what the build can run. Also the forcing rules: the hooks force any combination of the bits 1 2 4 16 32 64 and 128 + F1 (a forced set has no limb-CRT bit; Q1 needs E1), and the K5 hook's
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
  CHECK(seq_default_for(9, 0) == (kSeqEmit | kSeqLimbCrt));   // 9: F1, klwarp, klload and k5fin are not compiled
  CHECK(!seq_valid(kSeqFinish) && !seq_valid(3) && seq_valid(kSeqEmit) && seq_valid(0));
#else
  CHECK(seq_default_for(9, 0) == 447);
  CHECK(seq_default_for(9, 0) == (kSeqEmit | kSeqQuant | kSeqFinish | kSeqLimbCrt | kSeqKlWarp | kSeqKlLoad | kSeqK5Fin | kSeqKlF64));
  CHECK((seq_default_for(9, 0) & kSeqKlShape) == 0);   // klshape (slower) stays off; klwarp is on since R5 (docs/NOTES-26.md)
  CHECK((seq_default_for(9, 0) & kSeqQuant) != 0 && seq_valid(seq_default_for(9, 0) & ~kSeqLimbCrt));   // Q1 + klf64 on sm_90 since 2026-11-08; the row minus the limb-CRT bit is a forceable set (its twin, 439)
  // every combination of the bits 1, 2, 4, 16, 32, 64 (issue 23: the KL cuts are independent of the emitter set) is a set unless Q1 lacks E1; with 128 (k5fin) it needs F1 too
  for (int b = 0; b < 256; ++b) {
    const bool bit8 = (b & kSeqLimbCrt) != 0;
    CHECK(seq_valid(b) == (!bit8 && ((b & kSeqQuant) == 0 || (b & kSeqEmit) != 0) && ((b & kSeqK5Fin) == 0 || (b & kSeqFinish) != 0)));
  }
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
  for (const SeqClass& c : kSeqClasses) {                     // no row but sm_90 carries the limb CRT, Q1 or klf64
    CHECK(((c.bits & (kSeqQuant | kSeqKlF64)) != 0) == (c.cc_major == 9 && c.cc_minor == 0));
    CHECK(((c.bits & kSeqLimbCrt) != 0) == (c.cc_major == 9 && c.cc_minor == 0));
    CHECK(((c.bits & (kSeqKlLoad | kSeqK5Fin)) != 0) == (c.cc_major == 9 && c.cc_minor == 0));   // klwarp + klload + k5fin only on sm_90 (R4, R5); klshape in no row
    CHECK((c.bits & kSeqKlShape) == 0);
    CHECK(((c.bits & kSeqKlWarp) != 0) == (c.cc_major == 9 && c.cc_minor == 0));   // klwarp in the sm_90 row only
    CHECK(c.cc_major == 9 || c.bits < 16);
  }
  // the KL bits (issue 23): values, independence from the other bits, the kernel's template value and the launch shape
  CHECK(kSeqKlWarp == 16 && kSeqKlLoad == 32 && kSeqKlShape == 64 && kSeqKl == 112);
  CHECK((kSeqKl & 15) == 0 && (kSeqKl & 128) == 0);
  CHECK(seq_kl_kb(0) == 0 && seq_kl_kb(kSeqKlWarp) == 1 && seq_kl_kb(kSeqKlLoad) == 2 && seq_kl_kb(kSeqKlWarp | kSeqKlLoad) == 3);
  CHECK(seq_kl_kb(kSeqKlShape) == 0 && seq_kl_kb(7) == 0 && seq_kl_kb(13) == 0);
  // klf64 (review 2026-11-08): valid only with klwarp; Kb bit 2 (instances 5 and 7)
  CHECK(seq_valid(kSeqKlF64 | kSeqKlWarp) && !seq_valid(kSeqKlF64 | 1) && seq_kl_kb(kSeqKlF64) == 0);
  CHECK(seq_kl_kb(kSeqKlF64 | kSeqKlWarp) == 5 && seq_kl_kb(kSeqKlF64 | kSeqKlWarp | kSeqKlLoad) == 7);
  CHECK(seq_kl_shape(kSeqKlShape) && !seq_kl_shape(kSeqKlWarp | kSeqKlLoad | 7) && !seq_kl_shape(13));
  // kllong (issue 21, task 15): bit 512, the direct loads for nc > 2 (klload covers nc <= 2); never in seq_kl_kb (the host sets Kb bit 1 per operand)
  CHECK(kSeqKlLong == 512 && (kSeqKl & kSeqKlLong) == 0 && seq_kl_kb(kSeqKlLong) == 0 && seq_kl_kb(kSeqKlLong | kSeqKlWarp) == 1);
  CHECK(seq_kl_direct(kSeqKlLoad, 1) && seq_kl_direct(kSeqKlLoad, 2) && !seq_kl_direct(kSeqKlLoad, 3) && !seq_kl_direct(kSeqKlLong, 2) &&
        seq_kl_direct(kSeqKlLong, 3) && seq_kl_direct(kSeqKlLong, 98) && !seq_kl_direct(447, 8) && seq_kl_direct(447, 2));
  for (const SeqClass& c : kSeqClasses) CHECK((c.bits & kSeqKlLong) == 0);   // off in every row until a class has measured it
  CHECK(seq_pack_s(kSeqKlWarp | kSeqKlLoad | kSeqKlShape, 2) == 0 && seq_pack_s(kSeqEmit | kSeqKl, 2) == 1 && seq_pack_s(kSeqEmit | kSeqQuant | kSeqKl, 2) == 3);
#ifndef IMD_NO_SEQ_TWINS   // (seq_effective masks a forced set to the supported bits: with the twins off only E1 is left)
  CHECK(seq_effective(13, 17) == 17 && seq_effective(13, 113) == 113 && seq_effective(13, -1) == 13 && (seq_effective(13, 125) & kSeqLimbCrt) == 0);
#endif
#ifndef IMD_NO_SEQ_TWINS
  // a class row MAY carry a KL bit later (per device class, with that class's own data): the default mask keeps it and the plan's set runs it
  CHECK(((kSeqEmit | kSeqFinish | kSeqLimbCrt | kSeqKlWarp) & kSeqDefaultMask) == (kSeqEmit | kSeqFinish | kSeqLimbCrt | kSeqKlWarp));
  CHECK(seq_effective(kSeqEmit | kSeqKlLoad, -1) == (kSeqEmit | kSeqKlLoad) && seq_kl_kb(seq_effective(13 | kSeqKlWarp | kSeqKlLoad, -1)) == 3);
#endif
  CHECK(seq_limb_crt(seq_effective(13, 113), 1) && !seq_limb_crt(seq_effective(13, 113), -1));
  // forcing: the hooks accept -1 and 0..7 only; bit 8 is a class default
  CHECK(!seq_valid(kSeqLimbCrt) && !seq_valid(13) && !seq_valid(-2) && !seq_valid(128) && !seq_valid(256) && !seq_valid(24) && seq_valid(-1));
#ifndef IMD_NO_SEQ_TWINS
  CHECK(seq_valid(16) && seq_valid(32) && seq_valid(64) && seq_valid(113) && seq_valid(119) && !seq_valid(2 | 16) && !seq_valid(2 | 112) && !seq_valid(127 + 1));
  CHECK(seq_valid(512) && seq_valid(512 | 439) && seq_valid(512 | 7) && !seq_valid(512 | 2) && !seq_valid(1024) && !seq_valid(512 | 256));
#else
  CHECK(!seq_valid(512));
  CHECK(!seq_valid(16) && !seq_valid(32) && !seq_valid(64) && !seq_valid(113));   // the KL-bit kernels are twins too
#endif
  // what a plan runs: the class set unless forced, a forced set is masked to 0..7
  CHECK(seq_effective(seq_default_for(9, 0), -1) == seq_default_for(9, 0));
  CHECK(seq_effective(seq_default_for(9, 0), -1) & kSeqLimbCrt);
  CHECK(seq_effective(13, 0) == 0 && seq_effective(13, 1) == 1);
#ifndef IMD_NO_SEQ_TWINS
  CHECK(seq_effective(13, 7) == 7);
  CHECK((seq_effective(13, 5) & kSeqLimbCrt) == 0);
#endif
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
