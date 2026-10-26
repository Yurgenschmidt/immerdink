// Instruction-sequence sets of the pack and the CRT finish, one per device class (issue 21, docs/NOTES-21.md
// "Pack emitter and finish"). CUDA-free, header only. Every set computes the same bytes and the same doubles as
// the baseline (the oracle is the gate); a set only changes which instructions do it:
//   kSeqEmit    E1  the fused residue emitter (device/residue.hpp, modp.hpp)       integer pipe, any card
//   kSeqQuant   Q1  quantize by DMUL, DMUL, F2I.U64 (numerics.cuh quantize_cvt)    FP64 pipe; W = 2 only (N <= 16); needs kSeqEmit
//   kSeqFinish  F1  the CRT finish through I2F.F64.U64 (numerics.cuh finish_f1_try) FP64 pipe, both CRT variants
//   kSeqLimbCrt     K5's sums in binary64 limbs (crt_f64_elem) instead of the integer running CRT   FP64 pipe; any card, with or without F1
//   kSeqK5Fin   128 'k5fin': K5 with fewer instructions per output (docs/NOTES-23-k5.md): separate kernels for the limb CRT + F1 combination only (bits 4 and 8 / crt hook 1; with
//               any other combination the shipped K5 runs and the bit does nothing); bit-identical; default OFF in every class row until a device class has measured it
// Bit 8 is a CRT variant, not a pack / finish sequence: it exists only as a class default (the hooks force the sets 0..7 and 128, which
// run the integer CRT, and the K5 hooks force the CRT separately: 0 integer, 1 limb, -1 follow the class set).
// The baseline sequence of each stays in the code and is the choice wherever the new one is not at least as fast:
// kSeqClasses below is a bit set per device class, decided by the per-class SASS counts recorded in docs/NOTES-21.md and,
// for the FP64-pipe sets, by the card's FP64 rate and by the measurements (docs/NOTES-21.md "R2 outcome and the sm_90 default").
// A class that was not counted keeps 0. The test and benchmark hooks (imd_test_set_seq, imd_prof_set_seq) force a set for
// every plan.
#pragma once

#include <cstdint>

#include "immerdink/immerdink_host.h"

namespace imd::detail {

constexpr int kSeqEmit = 1, kSeqQuant = 2, kSeqFinish = 4, kSeqAll = 7;   // kSeqAll: the sets the hooks can force
constexpr int kSeqLimbCrt = 8;                                            // a class default only (see above)
constexpr int kSeqK5Fin = 128;                                            // issue 23: K5's instruction-count cuts (limb CRT + F1 only)
// The sets this build can run: with IMMERDINK_SEQ_TWINS=OFF (IMD_NO_SEQ_TWINS) the Q1 / F1 / k5fin kernel twins are not compiled and the
// hooks refuse them.
#ifdef IMD_NO_SEQ_TWINS
constexpr int kSeqSupported = kSeqEmit;
#else
constexpr int kSeqSupported = kSeqAll | kSeqK5Fin;
#endif
// What a class default may contain in this build: the supported sequences, and the limb CRT (its kernels are not twins; they are always compiled).
constexpr int kSeqDefaultMask = kSeqSupported | kSeqLimbCrt;
// What a forced set (the hooks) may contain: the supported sequences; never the limb-CRT bit (the K5 hook forces the CRT).
constexpr int kSeqForceMask = kSeqSupported;

// The default set per device class, keyed by the compute capability the plan's device reports (the registry of
// host/device_class.cpp has one K4 table per row, but a sequence set depends on the ISA, the register budget and the FP64
// rate, and sm_120 maps onto the sm_89 row there). A class that is not listed gets 0 = the baseline. Evidence per row: the SASS
// counts and register / occupancy figures of docs/NOTES-21.md "Pack emitter and finish" (sm_89, sm_90 and sm_120 builds of
// src/contract.cu). sm_120 WAS counted and stays on the baseline: the E1 loop is shorter per iteration, but the compiler unrolls it
// there and two whole kernels grow (kl_scale_pack<4,*> by 428 / 510 instructions), and nothing was timed. Q1 is opt-in everywhere; F1 and the
// limb CRT are on for sm_90 only, since the Hopper measurements (R2, 2026-10-25, H100 and H200): the pre-registered combination rule gave E1 + F1 + the limb CRT
// for sm_90 and Q1 off (docs/NOTES-21.md "R2 outcome and the sm_90 default"). Walking sm_90 back is this one table row. The limb CRT must stay off
// on cards with a 1/64-rate FP64 pipe (RTX 4090 sm_89, Blackwell sm_120: measured 0.44-0.84x): their rows do not carry kSeqLimbCrt.
struct SeqClass {
  int32_t cc_major, cc_minor;
  int bits;
};
constexpr SeqClass kSeqClasses[] = {
    {8, 9, kSeqEmit},  // RTX 4090: E1 (integer only; fewer instructions in every pack kernel counted). Q1 / F1 / the limb CRT would sit on a 1/64-rate FP64 pipe
    {9, 0, kSeqEmit | kSeqFinish | kSeqLimbCrt},  // H100, H200 (R2): E1 + F1 + limb CRT = 13 (bit-identical). The measured gain is the limb CRT's given E1 + F1 (the crtf1 duels: +4..7% at N = 14 squares, 1.041 / 1.058 / 1.074 at 8192 / 4096 / 2048, r_lowk 1.183); the adopted set was not measured against the old default directly, chaining f1 x crtf1 gives about +6..10% at the same cases. Q1 stays opt-in: alone it ties and is 0.98x at g = 7
};

constexpr int seq_default_for(int32_t cc_major, int32_t cc_minor) {
  for (const SeqClass& c : kSeqClasses)
    if (c.cc_major == cc_major && c.cc_minor == cc_minor) return c.bits & kSeqDefaultMask;
  return 0;
}

// Q1 quantizes into the 64-bit X of the E1 emitter: the Q1 bit without the E1 bit is not a set (it would run the baseline
// silently), so the hooks reject it. The same for k5fin without F1 (its kernels are F1 kernels).
constexpr bool seq_valid(int bits) {
  return bits >= -1 && (bits < 0 || ((bits & ~kSeqForceMask) == 0 && ((bits & kSeqQuant) == 0 || (bits & kSeqEmit) != 0) &&
                                      ((bits & kSeqK5Fin) == 0 || (bits & kSeqFinish) != 0)));
}

// A set as a launch sees it: the plan's bits (the class default, which may carry the limb-CRT bit), or the process-wide override (>= 0; 0..7,
// no limb-CRT bit: a forced set runs the integer CRT unless the K5 hook forces the limb CRT; 128 is allowed). Q1 needs the 64-bit X of W = 2
// and is only offered together with E1: the pack template value S (kernels/pack_emitter.cuh) is
// E1 -> 1, E1 + Q1 -> 3.
constexpr int seq_effective(int plan_bits, int override_bits) {
  return override_bits >= 0 ? (override_bits & kSeqForceMask) : (plan_bits & kSeqDefaultMask);
}
// K5's CRT variant a launch runs: crt_override is the K5 hook (-1 follow the set's kSeqLimbCrt bit, 0 force the integer CRT, 1 force the limb CRT).
constexpr bool seq_limb_crt(int set_bits, int crt_override) {
  return crt_override < 0 ? (set_bits & kSeqLimbCrt) != 0 : crt_override == 1;
}
constexpr int seq_pack_s(int bits, int W) {
  return (bits & kSeqEmit) ? ((bits & kSeqQuant) && W == 2 ? 3 : 1) : 0;
}

// The class table, checked at compile time (tests/test_seq_table.cpp checks the shipped rows, which depend on IMD_NO_SEQ_TWINS).
static_assert(seq_default_for(8, 9) == kSeqEmit, "sm_89 default");
static_assert(seq_default_for(12, 0) == 0 && seq_default_for(7, 5) == 0, "sm_120 and unlisted classes keep the baseline");
static_assert(seq_valid(-1) && seq_valid(0) && seq_valid(kSeqEmit) && !seq_valid(kSeqLimbCrt) && !seq_valid(kSeqLimbCrt | kSeqEmit), "the hooks force the sets 0..7 and 128 only");

}  // namespace imd::detail
