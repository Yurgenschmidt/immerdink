// Instruction-sequence sets of the pack and the CRT finish, one per device class (issue 21, docs/NOTES-21.md
// "Pack emitter and finish"). CUDA-free, header only. Every set computes the same bytes and the same doubles as
// the baseline (the oracle is the gate); a set only changes which instructions do it:
//   kSeqEmit    E1  the fused residue emitter (device/residue.hpp, modp.hpp)       integer pipe, any card
//   kSeqQuant   Q1  quantize by DMUL, DMUL, F2I.U64 (numerics.cuh quantize_cvt)    FP64 pipe; W = 2 only (N <= 16)
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

// The default set per device class, keyed by the compute capability the plan's device reports (the registry of
// host/device_class.cpp has one K4 table per row, but a sequence set depends on the ISA and the FP64 rate, and sm_120 maps onto
// the sm_89 row there). A class that is not listed, or was not counted, gets 0 = the baseline. Evidence per row: the SASS counts of
// docs/NOTES-21.md "Pack emitter and finish" (sm_89, sm_90 and sm_120 builds of src/contract.cu).
struct SeqClass {
  int32_t cc_major, cc_minor;
  int bits;
};
constexpr SeqClass kSeqClasses[] = {
    {8, 9, kSeqEmit},                 // RTX 4090: E1 (integer only; fewer instructions in every pack kernel). Not Q1 / F1: FP64 at 1/64 of the integer rate there, whatever the instruction count
    {9, 0, kSeqEmit | kSeqQuant},     // H100, H200: E1 and Q1 (fewer instructions in every pack kernel; FP64 pipe as wide as the integer pipe; not timed on Hopper yet)
};

constexpr int seq_default_for(int32_t cc_major, int32_t cc_minor) {
  for (const SeqClass& c : kSeqClasses)
    if (c.cc_major == cc_major && c.cc_minor == cc_minor) return c.bits;
  return 0;
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
