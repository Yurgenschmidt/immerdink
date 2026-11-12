// Instruction-sequence sets of the pack and the CRT finish, one per device class (issue 21, docs/NOTES-21.md
// "Pack emitter and finish"). CUDA-free, header only. Every set computes the same bytes and the same doubles as
// the baseline (the oracle is the gate); a set only changes which instructions do it:
//   kSeqEmit    E1  the fused residue emitter (device/residue.hpp, modp.hpp)       integer pipe, any card
//   kSeqQuant   Q1  quantize by DMUL, DMUL, F2I.U64 (numerics.cuh quantize_cvt)    FP64 pipe; W = 2 only (N <= 16); needs kSeqEmit
//   kSeqFinish  F1  the CRT finish through I2F.F64.U64 (numerics.cuh finish_f1_try) FP64 pipe, both CRT variants
//   kSeqLimbCrt     K5's sums in binary64 limbs (crt_f64_elem) instead of the integer running CRT   FP64 pipe; any card, with or without F1
//   kSeqK5Fin   128 'k5fin': K5 with fewer instructions per output (docs/NOTES-23-k5.md): separate kernels for the limb CRT + F1 combination only (bits 4 and 8 / crt hook 1; with
//               any other combination the shipped K5 runs and the bit does nothing); bit-identical; default OFF in every class row until a device class has measured it
// Bit 8 is a CRT variant, not a pack / finish sequence: it exists only as a class default (the hooks force the sets without it, which
// run the integer CRT, and the K5 hooks force the CRT separately: 0 integer, 1 limb, -1 follow the class set).
// Bit 256 'klf64' (issue 21, docs/NOTES-21-klf64.md): KL's gamma scan and exact square sum replaced by a high-word max and a binary64 sum with a
// rigorous filter (bit-identical; the exact passes run for a tile whose decision is close); FP64 pipe, so sm_90 only; needs klwarp; on in the sm_90 row since 2026-11-08 (H200 1.04-1.09).
// Bit 512 'kllong' (issue 21, task 15): klload's direct loads for the LONG rows too (nc > 2 chunks, 4 values per thread, 1024 threads), where the
// generic shared-memory gather costs about 40 instructions per element and pass, three passes (H200 ncu, r_bigk: 42 % of KL's 282 per element);
// the same gates as klload (unit-stride k read along k-bar, 16-byte aligned base); independent of bit 32, which covers the rows held on chip
// (nc <= 2). Off in every class row until a device class has measured it.
// Bit 1024 'O1' (issue 21, task 17): the E1 emitter on the offset-binary Y = a' + 2^63 (device/residue.hpp PackEmitTableO): the sign rides in Y and in the
// dp4a chain's start value, so the per-element-and-modulus sign multiply (one fmaheavy IMAD, the pipe that binds the emit loop on H200) goes. Needs E1;
// W = 2 only (N <= 16; N > 16 runs E1). Any card (integer pipe; with Q1 the conversion is the signed F2I). Off in every class row until measured.
// Bits 16, 32, 64 (issue 23, docs/NOTES-23.md) are cuts of the FIXED part of the large-row pack KL (everything that does not scale with N), each
// independent of the emitter set and of the others; all off in every class row until the Hopper duels decide:
//   kSeqKlWarp  'klwarp'   row-wide scalar work once per row / tile instead of by every thread / warp (largest_u, tile row setup, cross-warp combines)
//   kSeqKlLoad  'klload'   direct 16-byte global loads of unit-stride-k operands, no shared-memory gather tile (the other operands keep the gather)
//   kSeqKlShape 'klshape'  16 values per thread (256 threads) for the rows held on chip (nc <= 2), instead of 8 (512 threads)
// The baseline sequence of each stays in the code and is the choice wherever the new one is not at least as fast:
// kSeqClasses below is a bit set per device class, decided by the per-class SASS counts recorded in docs/NOTES-21.md and,
// for the FP64-pipe sets, by the card's FP64 rate and by the measurements (docs/NOTES-21.md "R2 outcome and the sm_90 default").
// A class that was not counted keeps 0. The test and benchmark hooks (imd_test_set_seq, imd_prof_set_seq) force a set for
// every plan.
#pragma once

#include <cstdint>

#include "immerdink/immerdink_host.h"

namespace imd::detail {

constexpr int kSeqEmit = 1, kSeqQuant = 2, kSeqFinish = 4;
constexpr int kSeqLimbCrt = 8;                                            // a class default only (see above)
constexpr int kSeqKlWarp = 16, kSeqKlLoad = 32, kSeqKlShape = 64;         // issue 23: KL fixed-part cuts
constexpr int kSeqKl = kSeqKlWarp | kSeqKlLoad | kSeqKlShape;
constexpr int kSeqK5Fin = 128;                                            // issue 23: K5's instruction-count cuts (limb CRT + F1 only)
constexpr int kSeqKlF64 = 256;  // issue 21 'klf64': KL's gamma from the high words and S filtered in binary64 (FP64 pipe; needs klwarp; docs/NOTES-21-klf64.md)
constexpr int kSeqKlLong = 512;  // issue 21 'kllong': klload's direct loads for rows of nc > 2 chunks (the E = 4 kernel)
constexpr int kSeqOffset = 1024;  // issue 21 'O1': the offset-binary emitter input (needs E1; W = 2)
constexpr int kSeqAll = 7 | kSeqKl | kSeqKlF64 | kSeqKlLong | kSeqOffset;   // the bits the hooks can force besides k5fin (never kSeqLimbCrt: the K5 hook forces the CRT)
// The sets this build can run: with IMMERDINK_SEQ_TWINS=OFF (IMD_NO_SEQ_TWINS) the Q1 / F1 / KL-bit / k5fin kernel twins are not compiled and the
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
// there and two whole kernels grow (kl_scale_pack<4,*> by 428 / 510 instructions), and nothing was timed. Q1 is on for sm_90 only since 2026-11-08 (it tied in R2 while KL was co-bound on the FP64-free pipes; after klf64 and the emit-loop cut it gains 3-8%), opt-in elsewhere; F1 and the
// limb CRT are on for sm_90 only, since the Hopper measurements (R2, 2026-10-25, H100 and H200): the pre-registered combination rule gave E1 + F1 + the limb CRT
// for sm_90 and Q1 off (docs/NOTES-21.md "R2 outcome and the sm_90 default"); R4 (2026-10-26) added klload and k5fin (docs/NOTES-24.md). Walking sm_90 back is this one table row. The limb CRT must stay off
// on cards with a 1/64-rate FP64 pipe (RTX 4090 sm_89, Blackwell sm_120: measured 0.44-0.84x): their rows do not carry kSeqLimbCrt.
struct SeqClass {
  int32_t cc_major, cc_minor;
  int bits;
};
constexpr SeqClass kSeqClasses[] = {
    {8, 9, kSeqEmit},  // RTX 4090: E1 (integer only; fewer instructions in every pack kernel counted). Q1 / F1 / the limb CRT would sit on a 1/64-rate FP64 pipe
    {9, 0, kSeqEmit | kSeqQuant | kSeqFinish | kSeqLimbCrt | kSeqKlWarp | kSeqKlLoad | kSeqK5Fin | kSeqKlF64},  // H100, H200: 447 = 189 + Q1 + klf64 (issue 21, 2026-11-08, H200 duels with crt f64, all bit-identical: klf64 437 vs 181 1.04-1.09; Q1 439 vs 437 sq4096 1.035, r_skinny 1.081, r_bigk 1.041, r_mm 1.055, N=8 1.03-1.05; H100: findings/hopper-tuning-20261108.md). Before: E1 + F1 + limb CRT + klwarp + klload + k5fin = 189 (R2 chose 13; R4 added klload + k5fin = 173, bit-identical; R5 added klwarp, docs/NOTES-26.md; set 165 + crt f64 against 13 on both cards: 4096^3 N=14 1.07 / 1.05, 8192^3 N=14 1.05, N=8 8192 1.075, r_lowk 1.05, t_df 1.08; klshape is 4-6% slower and stays off; docs/NOTES-24.md)
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
                                      ((bits & kSeqK5Fin) == 0 || (bits & kSeqFinish) != 0) &&
                                      ((bits & kSeqKlF64) == 0 || (bits & kSeqKlWarp) != 0) &&
                                      ((bits & kSeqOffset) == 0 || (bits & kSeqEmit) != 0)));
}

// A set as a launch sees it: the plan's bits (the class default, which may carry the limb-CRT bit), or the process-wide override (>= 0; any
// supported combination, no limb-CRT bit: a forced set runs the integer CRT unless the K5 hook forces the limb CRT). Q1 needs the 64-bit X of W = 2
// and is only offered together with E1: the pack template value S (kernels/pack_emitter.cuh) is
// E1 -> 1, E1 + Q1 -> 3; O1 adds 4 at W = 2 (E1 + O1 -> 5, E1 + Q1 + O1 -> 7; W = 3 drops it).
constexpr int seq_effective(int plan_bits, int override_bits) {
  return override_bits >= 0 ? (override_bits & kSeqForceMask) : (plan_bits & kSeqDefaultMask);
}
// K5's CRT variant a launch runs: crt_override is the K5 hook (-1 follow the set's kSeqLimbCrt bit, 0 force the integer CRT, 1 force the limb CRT).
constexpr bool seq_limb_crt(int set_bits, int crt_override) {
  return crt_override < 0 ? (set_bits & kSeqLimbCrt) != 0 : crt_override == 1;
}
constexpr int seq_pack_s(int bits, int W) {
  return (bits & kSeqEmit) ? ((bits & kSeqQuant) && W == 2 ? 3 : 1) | ((bits & kSeqOffset) && W == 2 ? 4 : 0) : 0;
}
// The KL kernel's template value Kb (kernels' bit 0: klwarp, bit 1: klload) and whether the launch takes the 16-values-per-thread shape.
constexpr int seq_kl_kb(int bits) {
  return ((bits & kSeqKlWarp) ? 1 : 0) | ((bits & kSeqKlLoad) ? 2 : 0) | ((bits & kSeqKlF64) && (bits & kSeqKlWarp) ? 4 : 0);
}
constexpr bool seq_kl_shape(int bits) { return (bits & kSeqKlShape) != 0; }
// Whether a KL launch takes the direct loads (Kb bit 1), before the operand's own gates (unit-stride k along k-bar, aligned base):
// klload for the rows held on chip (nc <= 2), kllong for the longer rows.
constexpr bool seq_kl_direct(int bits, int64_t nc) { return (bits & (nc > 2 ? kSeqKlLong : kSeqKlLoad)) != 0; }

// The class table, checked at compile time (tests/test_seq_table.cpp checks the shipped rows, which depend on IMD_NO_SEQ_TWINS).
static_assert(seq_default_for(8, 9) == kSeqEmit, "sm_89 default");
static_assert(seq_default_for(12, 0) == 0 && seq_default_for(7, 5) == 0, "sm_120 and unlisted classes keep the baseline");
static_assert(seq_valid(-1) && seq_valid(0) && seq_valid(kSeqEmit) && !seq_valid(kSeqLimbCrt) && !seq_valid(kSeqLimbCrt | kSeqEmit), "the limb CRT bit is not forceable");

}  // namespace imd::detail
