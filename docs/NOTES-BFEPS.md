# Branch-free ε in the pack emitter (ROADMAP A7, DESIGN v1.14 §3.4.1) and the retired eps path of the model (DESIGN v1.15 §6c.1)

Date: 2026-10-06. Branches `bf-eps` (implementation) and `followups-combined` (merge with
`hotclock`, model change, combined refit). Raw data, A/B records and logs are in the evaluation
workspace (`~/repos/immerdink-eval`, `results/followups2/bf-eps/` and
`results/followups2/combined/`; write-up `findings/bf-eps-implementation.md`).

## 1. Kernel change (v1.14)

Up to v1.13 a pack residue of an element with |a'| = μ·2^ε ≥ 2⁵³ (ε > 0) took a second path:
one more modular multiply by `pow2_mod[l][ε]`, a per-thread branch and a lane-divergent table
load; a warp paid it for its whole tile when any lane had ε > 0. So the pack cost depended on
the data (N ≥ 14; dominant-entry rows at every N ≥ 14, typical rows at N ≥ 15 for most k).

v1.14 holds X = |a'| exactly as W(N) little-endian u32 words (W = 2 for N ≤ 16: X = μ << ε as a
u64; W = 3 for N = 17–20: funnel shifts and a word select by ε / 32) and reduces it with W
`dp4a` against byte weights 2^(8i) mod p (bytes 0..9) plus one `umulhi` and a
multiply-subtract (`mod_p_words32<W>`). Every ε costs the same instructions. W is a template
parameter of `k23_pack`, `kf_scale_pack` and `kl_scale_pack`, chosen per plan from N.
- Same int8 bytes (same congruence class): no layout, pack-format, API or TLA+ change.
- Certificates: `cert_pack_words` (|a'| ≤ T and bit lengths 63 / 78 under the product's σ
  rule, RN and TRUNC, on ε-maximizing rows; the s < 12·255² reduction bound).
- Tests: GPU unit test `pack_emitter` (every N, E = 16 / 8 / 4, W = 2 / 3, every ε up to
  eps_max, 46.5 M residues against u128, 0 mismatches), contract suite `eps` (44 contractions
  bit-exact against spec/oracle.py on every launch path), 6 ε-heavy `imd_pack` cases,
  `mod_exhaustive_cpu` widened; a mutated word select fails both GPU tests.
- Registers (ptxas, sm_89): `k23_pack` 114 → 84 (W = 2) / 117 (W = 3); KF and KL stay at their
  launch-bound caps and spill less. `k23_pack`'s residency is set by its 40 768 B of static
  shared memory (2 CTAs per SM) at both W, so the model's occupancy row (`kernel.pack`,
  measured on the W = 3 instantiation) holds for both.

## 2. Measured (A/B, RTX 4090, C bit-identical by hash in every record)

TFLOPS-eq before → after (typical data unless noted), with the call and pack-phase ratios:

| shape | N | TFLOPS-eq | call after/before | pack phases after/before |
|---|---|---|---|---|
| 1024³ | 13 | 16.91 → 17.63 | 0.94–0.98 | 0.90 |
| 1024³ | 15 | 13.65 → 15.65 | 0.85–0.89 | 0.60 |
| 1024³ | 20 | 9.33 → 11.19 | 0.83–0.84 | 0.59–0.61 |
| 1024³, dominant data | 14 | 15.90 → 16.92 | 0.92–0.96 | 0.78–0.81 |
| 1024² × 16384 | 16 | 15.81 → 18.07 | 0.87–0.88 | 0.76–0.79 |
| 1024² × 16384 | 20 | 12.38 → 14.98 | 0.83 | 0.68 |
| 4096³ | 16–20 | +1–5 % | 0.94–1.00 | 0.85–0.95 |
| 16384 × 256 × 4096 | 16–20 | +3–5 % | 0.95–0.97 | 0.93–0.95 |
| DF acQ,bdQ 64⁴ × 2048 | 17–20 | +0.5–5 % | 0.95–0.99 | 0.84–0.87 |
| `imd_pack` 2048² | 16–20 | pack time −18 to −30 % | | |

N ≤ 13: no regression beyond 1 % (tiny calls: three rounds of 3000 calls, median ratio
1.000–1.008). Dominant-entry data now cost what typical data cost.

## 3. The model's eps path retired (v1.15, host API 11 change set)

With no data dependence left in the kernels, the model's eps machinery (the `op_eps` features,
the request's `eps_fraction`, the typical-data D-ramp default, `eps_s` and the bracket, the
`--data dominant` calibration axis) described a cost that no longer exists: with the shipped
θ it predicted N ≥ 15 calls and dominant-data packs too slow by up to the eps terms.

Decisions (DESIGN §6c.1 "The eps path — retired"):
- **No ABI change.** `eps_fraction` in the requests is validated as before (< 0 or 0..1; NaN,
  > 1: INVALID_VALUE) and ignored; `eps_s` / `eps_fraction` out are always 0; the `op_eps`
  feature slots are always 0 and the chains lose their (1 + f) factor. The meaning change is
  part of host API 11 (no further version bump).
- **Calibration format 2 kept.** The `op_eps` keys become optional (absent = 0, formats 1 and
  2; any value ≥ 0 accepted, no effect) and are written only when non-zero, so older files
  keep their text and newer fits have none. A format 3 was not needed: no remaining parameter
  changes meaning, and a calibration is refit with the kernels it describes anyway (as for
  v1.13 stage 2).
- **Calibration data.** `bench/sweep_calib.txt` drops its six `--data dominant` cases (≈ 1.3 min
  of GPU); the hold-out keeps `v_dom`, `w_d800`, `w_dr1` as the check that data no longer
  matter (NOTES-B2 §10).
- Tests: `host_perf` (every fraction predicts bit-identically over N = 2..20 and k across the
  old ramp, op_eps values have no effect, optional keys round-trip), `host_c_api` (NaN still
  rejected, eps out 0).

## 4. Leftovers (DESIGN v1.16, branch `emitter-w1`, 2026-10-06)

Raw data, scripts and logs: eval workspace `results/openitems/emitter/` (write-up
`findings/openitems-emitter.md`).

### 4.1 W = 1 for N ≤ 8: evaluated, not adopted

N = 8 is the largest N with T(N) < 2³² (T(8): 32 bits, T(9): 36; `cert_pack_words`), so for
N ≤ 8 X = |a'| fits one u32 word: `X = u32(μ) << ε` and one `dp4a` per residue. It was
implemented in full (`PackEmitterT<E, 1>`, `mod_p_words32<1>`, W = 1 instantiations of
`k23_pack`, `kf_scale_pack`, `kl_scale_pack`, `pack_words` 1 / 2 / 3, generated
`kPackW1MaxN`, `kPackXBits1`, the certificate, unit tests: 51.2 M residues, 0 mismatches;
contract and `imd_pack` cases bit-exact) and measured against v1.15 (1e8a439).

- **Code** (ptxas, sm_89): `k23_pack` 2496 → 2360 SASS instructions but **104 registers**
  (W = 2: 84, W = 3: 117; still 2 CTAs per SM, shared-memory bound); KF E = 16 126 vs 128
  registers; KL unchanged at its cap (64, spills 4 / 4 B as W = 2).
- **A/B** (RTX 4090, `bench/run_ab.sh`, three interleaved rounds, sweep
  `results/openitems/emitter/sweep_w1.txt`: 512², s256 g = N and g = 3, 1024³ g = N and
  g = 4, 4096³, 1024² × 16384, DF acQ,bdQ 64⁴ × 2048, `imd_pack` 2048²; N = 4, 6, 8 with
  N = 13, 16 as the control where only the v1.16 shift changed). **C bit-identical by hash
  in every record.** Per-phase medians over the rounds, after / before:

  | shape (path) | N | call | pack phases |
  |---|---|---|---|
  | 4096³ (KL E = 8) | 6 / 8 | 0.999 / 1.000 | KL A 0.993 / 0.999, KL B 0.995 / 0.993 |
  | DF acQ,bdQ (KL) | 6 / 8 | 0.997 / 0.999 | KL A 0.986 / 0.987, KL B 0.993 / 1.001 |
  | 1024² × 16384 (KL E = 4) | 6 / 8 | 0.990 / 0.985 | KL A 0.988 / 0.966, KL B 0.985 / 0.965 |
  | `imd_pack` 2048² (KL) | 6 / 8 | 0.972 / 0.965 | pack KL 0.957 / 0.930 |
  | 1024³ (KF E = 16) | 4 / 6 / 8 | 1.03 / 1.02 / 1.01 (±10 % noise) | KF 0.898 / 0.937 / 0.958 |
  | 1024³ g = 4 (K2/K3) | 6 / 8 | 0.996 / 0.991 (noise) | KF 0.983 / 0.941, K2 0.997 / 0.983, K3 0.988 / 0.911 |
  | 512² (KF E = 16) | 4 / 6 / 8 | 0.96 / 0.89 / 1.00 (noise) | KF 0.997 / 0.973 / 1.000 |
  | s256 g = N / g = 3 (KF E = 4, K2/K3 parts) | 6 / 8 | ±5–30 % process noise | no resolvable change |
  | control: same shapes at N = 13 / 16 | | 0.990–1.002 (stable shapes) | 0.992–1.002 (stable shapes) |

  Geometric means without the s256 tiny calls: pack phases **0.973** at N ≤ 8 (control
  0.997), calls 0.987 (control 0.996).
- **Cost of the third instantiation** (Release, one nvcc job, two alternating runs each):
  contract.cu 16.27 / 16.28 s → 18.11 / 18.13 s (+11 %); sm_89 cubin 4.62 → 5.24 MB
  (+13 %); `libimmerdink.so` 6.58 → 7.33 MB (+11 %).
- **Decision:** the adoption bar was a ≥ 3 % pack-phase gain with no call regression above
  0.5 %. The pack phases gained 2.7 % overall (≈ 1 % on the KL phases of the large shapes,
  where the measurement is precise; the larger KF gains at 1024³ sit inside ±10 %
  per-process noise and do not show in the call), so W = 1 is **not adopted**; the code was
  reverted (commit history of `emitter-w1`), the analysis kept here and in DESIGN §3.4.1.
  At N ≤ 8 a residue costs few instructions next to the operand read and the N int8 stores
  of each element (the KL phases are memory-bound), so one `dp4a` less barely shows.
- **For the performance model:** nothing changes at N ≤ 8 (W = 2 stays); the per-phase
  ratios above are the size of the effect a W = 1 kernel would have had.

### 4.2 Defined W = 2 shift (L3)

`X = μ << ε` (u64, ε ≤ 62 in the contract) is undefined C++ for ε ≥ 64, which only a σ bug
could produce, and the debug assert catches it only with asserts on. v1.16 writes the shift
as PTX `shl.b64` (inline asm `shl_clamped`; PTX clamps the count at 64, result 0): defined
for every ε, identity in the contract. Alternatives measured on the sm_89 cubin: `ε & 63`
+8 to +40 SASS instructions per W = 2 kernel, `ε < 64 ? μ << ε : 0` up to +112. PTX shl
keeps registers and spills, instruction counts −2 % (KF E = 16) to +0.2 % (KL E = 8), not
byte-identical (the C++ shift let the compiler assume ε < 64); W = 3 kernels byte-identical
(its shifts were already defined: b ≤ 31, q ≥ 3 selects as q = 2). Timing at N = 13, 16 (the
control rows above, where this is the only change): neutral, C bit-identical.

### 4.3 The s bound (L4)

`s ≤ 10·255·255 = 650,250` in DESIGN §3.4.1 takes every weight as 255. Exact maxima of
s = Σ b_i w_i (cert_pack_words): 210,472 at W = 2 (X < 2⁶³, moduli of N ≤ 16), 264,156 at
W = 3 (X < 2⁷⁸, p = 253); over X ≤ T(N) 258,500 (N = 20, p = 253); with all ten bytes at 255
302,940 (p = 253). (A figure of 254,377 was quoted for this maximum; it matches none of these
definitions and could not be reproduced.) All are < 2¹⁹; the reduction needs s < 2²⁴.

### 4.4 W-aware occupancy (L1)

`kernel_occupancy` measured the W = 3 instantiation of every pack kernel while N ≤ 16 runs
W = 2 (`k23_pack` 84 vs 117 registers). v1.16 adds `kernel_occupancy_w(kernel, W)` (test
hook `imd_test_kernel_occupancy_w`, benchmark hook `imd_prof_kernel_occupancy_w`, imd_bench
occupancy record `kernels_w`) and defines the reported row as the most constrained W
(fewest CTAs per SM, then most registers) = the W = 3 row, unchanged, so the shipped
calibration stays valid. `contract_api` asserts that CTAs per SM are equal for W = 2 and 3
(kf16 2, kf4 4, kl8 2, kl4 1, pack 2), which is all the model's occupancy features use. If a
kernel change ever breaks the equality, the test fails and the model needs per-W rows
(`kernels_w` already records them).

### 4.5 Still open

- `quantize` keeps its per-element s ≥ 0 / s < 0 branch (not per modulus).
