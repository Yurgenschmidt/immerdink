// Instruction-sequence sets of the pack and the CRT finish, one per device class (issue 21, docs/NOTES-21.md
// "Pack emitter and finish"). CUDA-free, header only. Every set computes the same bytes and the same doubles as
// the baseline (the oracle is the gate); a set only changes which instructions do it:
//   kSeqEmit    E1  the fused residue emitter (device/residue.hpp, modp.hpp)       integer pipe, any card
//   kSeqQuant   Q1  quantize by DMUL, DMUL, F2I.U64 (numerics.cuh quantize_cvt)    FP64 pipe; W = 2 only (N <= 16); needs kSeqEmit
//   kSeqFinish  F1  the CRT finish through I2F.F64.U64 (numerics.cuh finish_f1_try) FP64 pipe, both CRT variants
// The baseline sequence of each stays in the code and is the choice wherever the new one is not at least as fast:
// kSeqClasses below is a bit set per device class, decided by the per-class SASS counts recorded in docs/NOTES-21.md and,
// for the FP64-pipe sets, by the card's FP64 rate. A class that was not counted keeps 0. The test and benchmark hooks (imd_test_set_seq, imd_prof_set_seq) force a set for
// every plan.
#pragma once

#include <cstdint>

#include "immerdink/immerdink_host.h"

namespace imd::detail {

constexpr int kSeqEmit = 1, kSeqQuant = 2, kSeqFinish = 4, kSeqAll = 7;
// The sets this build can run: with IMMERDINK_SEQ_TWINS=OFF (IMD_NO_SEQ_TWINS) the Q1 / F1 kernel twins are not compiled and the
// hooks refuse them.
#ifdef IMD_NO_SEQ_TWINS
constexpr int kSeqSupported = kSeqEmit;
#else
constexpr int kSeqSupported = kSeqAll;
#endif

// The default set per device class, keyed by the compute capability the plan's device reports (the registry of
// host/device_class.cpp has one K4 table per row, but a sequence set depends on the ISA, the register budget and the FP64
// rate, and sm_120 maps onto the sm_89 row there). A class that is not listed gets 0 = the baseline. Evidence per row: the SASS
// counts and register / occupancy figures of docs/NOTES-21.md "Pack emitter and finish" (sm_89, sm_90 and sm_120 builds of
// src/contract.cu). sm_120 WAS counted and stays on the baseline: the E1 loop is shorter per iteration, but the compiler unrolls it
// there and two whole kernels grow (kl_scale_pack<4,*> by 428 / 510 instructions), and nothing was timed. Q1 and F1 are opt-in
// everywhere until Hopper measurements (R2) decide: they are reachable by the hooks only.
struct SeqClass {
  int32_t cc_major, cc_minor;
  int bits;
};
constexpr SeqClass kSeqClasses[] = {
    {8, 9, kSeqEmit},  // RTX 4090: E1 (integer only; fewer instructions in every pack kernel counted). Q1 / F1 would also sit on a 1/64-rate FP64 pipe
    {9, 0, kSeqEmit},  // H100, H200: E1. Q1 (+14..30 registers in k23_pack, 2 CTAs per SM instead of 4; 64-bit conversions probably quarter rate) and F1 wait for R2
};

constexpr int seq_default_for(int32_t cc_major, int32_t cc_minor) {
  for (const SeqClass& c : kSeqClasses)
    if (c.cc_major == cc_major && c.cc_minor == cc_minor) return c.bits;
  return 0;
}

// Q1 quantizes into the 64-bit X of the E1 emitter: the Q1 bit without the E1 bit is not a set (it would run the baseline
// silently), so the hooks reject it.
constexpr bool seq_valid(int bits) {
  return bits >= -1 && (bits < 0 || ((bits & ~kSeqSupported) == 0 && ((bits & kSeqQuant) == 0 || (bits & kSeqEmit) != 0)));
}

// A set as a launch sees it: the plan's bits, or the process-wide override (>= 0). Q1 needs the 64-bit X of W = 2
// and is only offered together with E1: the pack template value S (kernels/pack_emitter.cuh) is
// E1 -> 1, E1 + Q1 -> 3.
constexpr int seq_effective(int plan_bits, int override_bits) {
  return override_bits >= 0 ? (override_bits & kSeqAll) : (plan_bits & kSeqAll);
}
constexpr int seq_pack_s(int bits, int W) {
  return (bits & kSeqEmit) ? ((bits & kSeqQuant) && W == 2 ? 3 : 1) : 0;
}

}  // namespace imd::detail
