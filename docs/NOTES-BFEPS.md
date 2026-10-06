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

## 4. Open

- W = 1 for N ≤ 8 (T < 2³²: one `dp4a`) is possible and not done (a third instantiation).
- `quantize` keeps its per-element s ≥ 0 / s < 0 branch (not per modulus).
