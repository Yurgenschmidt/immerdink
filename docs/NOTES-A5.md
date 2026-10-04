# A5 notes: small-shape polish

ROADMAP A5. The clean bench-mode sweep (`bench/results/compare-clean-bb6b568.jsonl`, dev
`b6b7e05` = `bb6b568` + docs) left two cases below GEMMul8: s256 (256³) at 0.90× and s512
(512³) at 0.92×, both at N = 8. Goal: ≥ 1.0× there and on the small family (s128, s_cc8,
s_df, s_perm, s_cc16, sq1024) without regressing anything else. Measure first (§1), then fix
by measured payoff (§2), then before/after (§3).

Machine: bench mode (console only, no display server, no other GPU clients, persistence
mode on, PCIe gen4 x16 under load), RTX 4090 at its 450 W cap, i9-13900K (8 P-cores =
CPUs 0–15, 16 E-cores = CPUs 16–31, `powersave` governor with HWP). Every timing through
`tools/run_bench.sh` (quiet gate, machine lock, GPU lock).

New tools (this work):
- `bench/nsys_case.sh` / `bench/nsys_sweep.sh`: nsys 2026.5.1 (`-t cuda --gpuctxsw=true`)
  of one case or a sweep file, one process per (case, N); `bench/nsys_calls.py` turns the
  sqlite export into a per-call breakdown of the bench's timed calls (host time, launch
  latency, GPU span, per-kernel medians, gaps, context switches).
- `imd_bench --methods duel`: imd and GEMMul8 (with the permutations when the GEMM view
  needs them, i.e. perm + GEMMul8, the fair comparison) timed **interleaved, call by call,
  in one process**; reports both medians and the median of the per-pair ratios.
- `IMD_BENCH_TIMES=1`: every timed call's event time on stderr (`IMD_BENCH_CLOCK=1` adds an
  SM-clock probe after each call; its 4 µs window turned out too short to be useful).
- `bench/sweep_a5.txt`: the small family.

## 1. Where the time goes (dev `b6b7e05`, before any change)

### 1.1 Per-kernel GPU time (nsys, 300 timed calls per method, median of the last 295)

`bench/nsys_sweep.sh build/bench/imd_bench bench/sweep_a5.txt build/nsys/base --reps 300`
(GEMMul8 and imd in one process per (case, N), N = 8 and 14). µs. "wall" ≈ the bench's
event time (e0 record → last kernel end; nsys adds ~1 µs of host overhead per API call, so
host-heavy callers look worse under nsys); "lat0" = e0 record → first kernel start (the
launch latency of a call on an idle GPU); "span" = first kernel start → last kernel end =
kernels + gaps.

| case | N | method | launches | host | lat0 | span | kernels | gaps | per kernel |
|---|---|---|---|---|---|---|---|---|---|
| s128 | 8 | GEMMul8 | 6 | 17.3 | 3.1 | 18.1 | 14.7 | 3.5 | scale 2.6 + 2.5, GEMM 3.4 + 3.4, mod 1.5, undo 1.3 |
| s128 | 8 | imd | 3 | 10.1 | 4.5 | 14.1 | 12.8 | 1.3 | KF 6.0, K4 2.8 (64×64), K5 4.0 |
| s256 | 8 | GEMMul8 | 6 | 17.4 | 3.1 | 20.8 | 17.5 | 3.3 | scale 3.7 + 3.7, GEMM 3.4 + 3.5, mod 1.5, undo 1.7 |
| s256 | 8 | imd | 3 | 5.9 | 3.0 | 16.5 | 15.3 | 1.3 | KF 7.1, K4 3.6 (64×64), K5 4.6 |
| s512 | 8 | GEMMul8 | 6 | 16.9 | 3.1 | 33.2 | 29.8 | 3.3 | scale 5.8 + 5.8, GEMM 6.4 + 6.4, mod 2.3, undo 3.1 |
| s512 | 8 | imd | 3 | 5.9 | 3.0 | 30.8 | 29.4 | 1.3 | KF 12.5, K4 7.5 (128×128), **K5 9.4** |
| s512 | 14 | GEMMul8 | 10 | 30.7 | 3.1 | 52.6 | 46.6 | 6.0 | |
| s512 | 14 | imd | 3 | 5.9 | 3.0 | 41.0 | 39.6 | 1.3 | KF 15.3, K4 11.5 (128×256), K5 12.9 |
| s_cc8 | 8 | perm+GEMMul8 | 9 | 22.9 | 3.3 | 43.9 | 38.6 | 5.3 | 3 permutes 8.7 + the s512 kernels |
| s_cc8 | 8 | imd | 3 | 5.8 | 3.0 | 31.2 | 29.9 | 1.3 | KF 13.0, K4 7.4, K5 9.4 |
| s_df | 8 | GEMMul8 | 6 | 17.5 | 3.1 | 67.7 | 64.3 | 3.3 | |
| s_df | 8 | imd | 3 | 6.0 | 3.1 | 47.8 | 46.4 | 1.3 | KF 12.5, K4 15.0, K5 18.9 |
| s_perm | 8 | perm+GEMMul8 | 8 | 20.9 | 3.0 | 75.5 | 70.8 | 4.7 | |
| s_perm | 8 | imd | 3 | 6.0 | 3.1 | 48.2 | 46.9 | 1.3 | KF 13.0, K4 15.0, K5 18.9 |
| s_cc16 | 8 | perm+GEMMul8 | 9 | 22.6 | 3.4 | 365.0 | 359.6 | 5.4 | |
| s_cc16 | 8 | imd | 5 | 9.6 | 3.1 | 291.9 | 289.1 | 2.8 | K0 1.8, KL 68.2 + 64.2, K4 116.8, K5 38.1 |
| sq1024 | 8 | GEMMul8 | 6 | 17.3 | 3.1 | 102.9 | 99.6 | 3.3 | scale 12.3 + 12.3, GEMM 30.7 + 30.5, mod 5.8, undo 8.0 |
| sq1024 | 8 | imd | 3 | 6.0 | 3.1 | 96.3 | 95.0 | 1.4 | KF 40.8, K4 35.4, K5 18.8 |

(N = 14: imd's span is below GEMMul8's on every case, by 1.2–1.9×; full table in
`build/nsys/base/*.txt`, not committed.)

What this says:
- **On the GPU, imd is already ahead on every small case at N = 8**, s512 included (span
  30.8 vs 33.2 µs) and s256 by 1.26× (16.5 vs 20.8 µs). The launch latency (lat0 ≈ 3 µs)
  is the same for both; the gaps are 0.6–0.7 µs per kernel boundary for both, so imd's 3
  launches cost 1.3 µs of gaps against GEMMul8's 3.3 µs (6 launches).
- **The host is not the bottleneck** (as in NOTES-A1): imd_contract spends 0.3 µs before
  its first launch and ~6 µs in total (nsys-inflated) while its GPU span is 14–96 µs.
  GEMMul8's host time is larger (17 µs for 6 launches under nsys) but also below its span.
- **The closest margins are s512 (7 %) and sq1024 (7 %) at N = 8**, and there K5 stands
  out: 9.4 µs for 262 144 outputs at s512, where GEMMul8's whole reconstruction (mod +
  undo) takes 5.4 µs; KF 12.5 µs does both operands' scaling and 8 residues (GEMMul8's two
  scaling kernels: 11.6 µs); K4 (7.5 µs, 128×128, the per-case best of A3) is half of
  GEMMul8's two INT8 GEMMs + mod (15.1 µs).

### 1.2 Why the sweep said 0.90× / 0.92×: the measurement, not the kernels

The clean sweep times 10 calls per method (after 2 warm-up calls), GEMMul8 first, then imd,
in one process. Per-call event times (`IMD_BENCH_TIMES=1`) show that tiny calls drift
between **performance states that last from milliseconds to seconds**, and a 10-call
median samples one of them:

- s256 imd, N = 8, consecutive calls of one process: 47 43 43 43 42 42 41 41 … 40 54 **18
  18 18 18** 68 **17 17 17 …** µs (the state flipped after ~16 calls); another process:
  17.4 µs throughout; another: 20.5–21.5 µs throughout.
- GEMMul8 itself: s512 at N = 14 runs 196 µs per call for 16 calls, then 51 µs; across
  processes 50, 90, 250 or 56 µs (the "bimodal GEMMul8 at N = 14" of NOTES-A1/A3); s256 at
  N = 8 19.5 µs in most processes and 29.7 µs in some.
- Pinning the bench to one core: on a P-core (CPU 2) s256 imd 17.4–18.4 µs, GEMMul8
  21.5 µs; on an E-core (CPU 20) imd 18.4–19.3 µs, **GEMMul8 29.5 µs** (its 6 launches and
  cuBLAS host work become host-bound on an E-core). The unpinned bench thread migrates
  between core types, so this alone moves the ratio by up to 1.5×.
- Under nsys (host calls slower, GPU never idle as long) the states did not appear: every
  kernel's duration was stable to ±1 % over 300 calls.
- Longer warm-up (300 calls instead of 2) did not remove the states (s256 imd 17.2 and
  20.5 µs in two processes with the same settings; GEMMul8 N = 14 56–260 µs).

The causes are outside the kernels (host core type and frequency under `powersave`, GPU
DVFS and PCIe link power states: at idle the link is at gen1, SM 210 MHz, memory 405 MHz),
and they cannot be locked without root (`nvidia-smi -lgc`, governor). A sequential
"GEMMul8 then imd" measurement of a 20 µs call therefore has a ±20 % state error that a
10-call median does not average out. **Interleaving the two methods call by call** puts
both in the same state; the per-pair ratio is stable (p10–p90 within ±3–6 % on most
cases):

`imd_bench --methods duel --reps 400` (`build/nsys/duel-base1.jsonl`; µs medians; ratio =
median over pairs of t(GEMMul8) / t(imd), > 1: imd faster; permuted cases against
perm + GEMMul8):

| case | N | GEMMul8 | imd | ratio [p10, p90] |
|---|---|---|---|---|
| s128 | 8 | 17.4 | 14.3 | 1.20 [1.13, 1.23] |
| s256 | 8 | 19.7 | 16.6 | 1.18 [1.13, 1.24] |
| s512 | 8 | 30.7 | 29.6 | **1.04** [1.02, 1.07] |
| s_cc8 | 8 | 39.9 | 29.7 | 1.35 [1.31, 1.39] |
| s_cc16 | 8 | 341.0 | 274.4 | 1.24 [1.23, 1.26] |
| s_df | 8 | 62.1 | 44.4 | 1.39 [1.36, 1.42] |
| s_perm | 8 | 69.0 | 44.8 | 1.55 [1.52, 1.57] |
| sq1024 | 8 | 94.1 | 94.3 | **1.01** [0.93, 1.07] |
| s128 | 14 | 29.3 | 16.1 | 2.00 [1.78, 4.31] |
| s256 | 14 | 31.1 | 18.5 | 1.67 [1.60, 3.54] |
| s512 | 14 | 47.9 | 38.9 | 1.25 [1.22, 2.08] |
| s_cc8 | 14 | 57.3 | 38.9 | 1.48 [1.44, 1.53] |
| s_cc16 | 14 | 545.8 | 398.0 | 1.38 [1.36, 1.39] |
| s_df | 14 | 98.4 | 66.6 | 1.48 [1.44, 1.52] |
| s_perm | 14 | 107.5 | 66.6 | 1.60 [1.56, 1.64] |
| sq1024 | 14 | 170.8 | 154.5 | 1.11 [1.08, 1.15] |

So in a like-for-like measurement imd is at or above GEMMul8 on the whole small family
before any change; the two thin margins are **s512 (1.04×) and sq1024 (1.01×) at N = 8**,
where a state flip can still push a sequential measurement below 1. Those are what §2
works on: K5 first (9.4 µs at s512, 18.8 µs at sq1024; GEMMul8 reconstructs in 5.4 /
13.8 µs), then KF.

### 1.3 K5 and KF at s512, N = 8 (ncu 2026.3.1, `--set full`, clock locked by ncu at ~2.45 GHz)

| kernel | grid × block | regs | occupancy (theor. / achieved) | waves | warp instr | instr / thread | cycles per issued instr | IPC |
|---|---|---|---|---|---|---|---|---|
| KF `kf_scale_pack<8,4>` | 512 × 256 | 64 | 67 / 56 % | 1.0 | 9.36 M | 2 285 (571 per value) | 13.6 | 1.97 |
| K4 128×128 | 128 × 128 | 250 | 17 / 8 % | 0.5 | 1.45 M | | 7.2 | 0.56 |
| K5 `k5_crt<3,1>` | 1 024 × 256 | 46 | 83 / 67 % | **1.6** | 6.39 M | **780 per output** | 15.5 | 2.04 |

- **K5 is latency-bound with a long per-element chain and 1.6 waves.** 780 instructions per
  output at N = 8 (3 limbs) of which the §3.6 CRT steps are only ~120: the kernel is
  generic in N, so it computes 64-bit addresses for all 12 planes its limb count allows
  (predicated: ~150 instructions), loads every constant of a step with a runtime-indexed
  `LDC` (N is a kernel argument), branches per step on `l < N` and on p = 256, and does two
  32-bit divisions per element for the index split. 46 registers allow 5 blocks of 256 per
  SM, so 1 024 blocks take 1.6 waves.
- **KF is latency-bound too** (one wave, 2 285 instructions per thread): a third in the
  residue emit (8 moduli × 4 values), a quarter of the stall samples on the tile's global
  loads, the rest in the §3.2 K1 integers (u128 squares, shuffles, the 256-bit compare).

## 2. Changes (by measured payoff; all bit-identical, launch structure unchanged)

| # | change | where | nsys effect (N = 8, µs) |
|---|---|---|---|
| 1 | **K5 compiled once per N** (2..20) instead of once per limb count with N as an argument: the plane loop, the plane addresses and every §3.6 constant are compile-time (no runtime-indexed `LDC`, no per-step branches on l < N or p = 256, no predicated addresses for the unused planes of the limb count). The λ-fastest K5 is per N too | `contract.cu` `k5_crt_n<N, E>`, `k5_crt_bfast<N>`, `launch_k5_for` | K5 s128 4.0 → 3.0, s256 4.6 → 3.5, s512 9.4 → 6.6, s_df 18.9 → 16.5, sq1024 18.8 → 16.4 |
| 2 | **K5 with 2 elements per thread** for 1024·SMs < b·m·n ≤ 16384·SMs (131 K – 2 M outputs on the 4090), 1 below, 4 above (measured 1 / 2 / 4 per thread on 16 K – 16 M outputs: 2 is best from s512 to sq1024, 4 from sq2048 up, 1 below s512) | `launch_k5` | s512 6.6 → 6.5 (N = 14: 9.1 → 8.2), s_df 16.5 → 16.3 (N = 14: 26.8 → 25.6), sq1024 16.4 → 16.1, x768 11.5 → 10.7; sq2048+ unchanged (4) |
| 3 | **Pack residues with the 32-bit magic**: the byte-weighted sum s < 7·255² is reduced by one `umulhi` with ⌈2³²/p⌉ (2²⁴ for p = 256: branch-free) instead of a 64-bit multiply-high, and the sign is applied after SMod (SMod(−r) = −SMod(r) for odd p; for p = 256 both give the byte (−r) mod 256). Exhaustive CPU test of the reduction over [0, 7·255²] for every modulus, end to end on 2²⁰ μ per modulus | `modp.hpp` `mod_p_bytes32`, `PackEmitterT::emit` (KF, KL, K2/K3) | KF s512 12.5 → 11.4 (N = 14: 15.1 → 13.1), sq1024 40.8 → 35.3 (N = 14: 55.8 → 43.3) |
| 4 | **§3.2 term in 64-bit words**: ⌈m²·2ᵗ⌉ = ⌈m′²/2ˢ⌉ with the significand normalized to 53 bits, s ≥ 44: one 64×64 product and 64-bit shifts instead of generic 128-bit shifts. GPU test against the reference on 2²⁸ (x, γ) pairs (every exponent, subnormals, γ − ilogb x from 0 to 2100) | `numerics.cuh` `square_term_n` (K1, KF, KL) | KF s512 11.4 → 11.2, s_df 12.5 → 10.8, sq1024 35.3 → 33.8 |

Tests: every contract suite bit-exact (the forced-KL runs now also take K5's 2-per-thread
mapping, the unfused ones the 4-per-thread one), and a new suite `contract_every_n` runs a
matrix and a batch-fastest case for every N in 2..20 (each through every fused mode, K4
configuration and K5 mapping of `run_case`), since K5 is now a separate kernel per N.
Build: 3 × 19 j̄-fastest + 19 λ-fastest K5 instantiations instead of 3 × 5; `contract.cu`
compiles in 16 s instead of 6.5 s (one job; it builds in parallel with the CUTLASS K4 units).

Not done, with the numbers that say why:
- **K4 selection rule**: at s256 / s512 the rule already picks the per-case best of A3
  (64×64 3.6 µs, within 0.1 µs of 64×128; 128×128 7.5 µs). Its 5 % margin leaves ~8 % of K4
  at sq1024 N = 14 (128×256 66.8 µs vs 128×128 ≈ 62), where imd is already 1.25× GEMMul8;
  a rule change touches every shape (A3's fit), so it stays as it is.
- **K5 fused into a small-tile K4 (2 launches)**: the fixed cost per launch is ~2 µs (gap
  0.6–0.7 µs + fill/drain: GEMMul8's 1.3 µs kernels are the floor), i.e. at most ~2 µs per
  call. But one CTA would have to hold all N moduli of its output tile: at s512 that is 16
  CTAs of 128×128 (or 64 of 64×64) doing 8 GEMMs each in sequence plus the CRT, against
  today's 128 CTAs and a 6.5 µs K5 on all SMs: slower, not faster. Only s128-sized problems
  (≤ 4 output tiles) could gain, and s128 is already 1.3× GEMMul8. Not needed; no spec
  change made.
- **KF instruction count**: after changes 3 and 4 KF is still the largest kernel at s512
  (11.2 µs, close to issue-bound: ~2 000 instructions per thread in one wave); GEMMul8's two
  scaling kernels take 11.6 µs for the same work. Further cuts (quantize, the per-row
  256-bit compare) are a few % each.

## 3. Before / after

### 3.1 nsys, per call (after = `efbb01e`; same tool and settings as §1.1; µs)

| case | N | GEMMul8 span | imd span before → after | imd kernels after | imd wall before → after | GEMMul8 wall |
|---|---|---|---|---|---|---|
| s128 | 8 | 18.4 | 14.1 → **12.8** | KF 5.7, K4 2.8, K5 3.0 | 19.7 → 16.5 | 22.3 |
| s256 | 8 | 20.7 | 16.5 → **14.8** | KF 6.5, K4 3.6, K5 3.5 | 20.3 → 18.7 | 24.6 |
| s512 | 8 | 33.2 | 30.8 → **26.5** | KF 11.2, K4 7.5, K5 6.5 | 34.5 → 30.2 | 37.0 |
| s512 | 14 | 52.5 | 41.0 → **34.2** | KF 13.1, K4 11.6, K5 8.2 | 44.7 → 37.9 | 56.3 |
| s_cc8 | 8 | 44.1 (perm) | 31.2 → **27.2** | KF 11.7, K4 7.5, K5 6.7 | 34.9 → 30.9 | 48.1 |
| s_df | 8 | 67.9 | 47.8 → **43.5** | KF 10.8, K4 15.0, K5 16.4 | 51.6 → 47.5 | 71.9 |
| s_perm | 8 | 75.5 (perm) | 48.2 → **43.7** | KF 11.3, K4 14.9, K5 16.2 | 52.1 → 47.4 | 79.2 |
| s_cc16 | 8 | 346.9 (perm) | 291.9 → **262.6** | K0 1.7, KL 60.6 + 58.0, K4 109.1, K5 29.7 | 295.8 → 266.3 | 350.9 |
| sq1024 | 8 | 102.4 | 96.3 → **86.6** | KF 33.7, K4 35.3, K5 16.3 | 100.2 → 90.4 | 106.3 |
| sq1024 | 14 | 187.6 | 157.7 → **136.5** | KF 42.9, K4 66.8, K5 25.5 | 161.8 → 140.3 | 191.4 |

Host time, launch latency and gaps are unchanged (5.8–6.3 µs, 3.0 µs, 1.3 µs for 3
launches): the gain is all kernel time, mostly K5 and KF.

### 3.2 Interleaved duel (`imd_bench --methods duel --reps 400`, same as §1.2)

| case | N | ratio before [p10, p90] | **ratio after** [p10, p90] |
|---|---|---|---|
| s128 | 8 | 1.20 [1.13, 1.23] | **1.30** [1.22, 1.39] |
| s256 | 8 | 1.18 [1.13, 1.24] | **1.30** [1.21, 1.38] |
| s512 | 8 | 1.04 [1.02, 1.07] | **1.20** [1.17, 1.25] |
| s_cc8 | 8 | 1.35 [1.31, 1.39] | **1.53** [1.50, 1.59] |
| s_cc16 | 8 | 1.24 [1.23, 1.26] | **1.29** [1.28, 1.31] |
| s_df | 8 | 1.39 [1.36, 1.42] | **1.53** [1.48, 1.57] |
| s_perm | 8 | 1.55 [1.52, 1.57] | **1.70** [1.66, 1.73] |
| sq1024 | 8 | 1.01 [0.93, 1.07] | **1.09** [1.01, 1.18] |
| s128 | 14 | 2.00 [1.78, 4.31] | **2.12** [1.93, 3.32] |
| s256 | 14 | 1.67 [1.60, 3.54] | **1.91** [1.79, 2.69] |
| s512 | 14 | 1.25 [1.22, 2.08] | **1.47** [1.44, 1.62] |
| s_cc8 | 14 | 1.48 [1.44, 1.53] | **1.74** [1.70, 1.78] |
| s_cc16 | 14 | 1.38 [1.36, 1.39] | **1.43** [1.42, 1.45] |
| s_df | 14 | 1.48 [1.44, 1.52] | **1.64** [1.59, 1.70] |
| s_perm | 14 | 1.60 [1.56, 1.64] | **1.78** [1.73, 1.85] |
| sq1024 | 14 | 1.11 [1.08, 1.15] | **1.25** [1.19, 1.31] |

(`build/nsys/duel-{base1,after1}.jsonl`, not committed: the A/B below is the committed
record.)

### 3.3 Same-session A/B (`bench/run_ab.sh`, `AFTER_METHODS=gemmul8,imd`, `bench/sweep_a5.txt`: 300 timed calls after 20 warm-up calls)

`bench/results/a5-small-{before-b6b7e05,after-efbb01e}.jsonl`; each side runs GEMMul8 and
then imd in its own quiet-gated process, interleaved per case; "vs G8" divides by the
GEMMul8 (perm + GEMMul8 for the permuted cases) time of the same process; summary by
`bench/ab_ratio.py`.

| case | N | G8 µs (b / a) | imd before | imd after | after / before | before vs G8 | **after vs G8** |
|---|---|---|---|---|---|---|---|
| s128 | 8 | 17.3 / 18.6 | 19.4 | 15.2 | 0.783 | 0.89 | **1.23** |
| s128 | 14 | 28.4 / 31.4 | 20.2 | 16.4 | 0.811 | 1.41 | **1.91** |
| s256 | 8 | 21.5 / 21.5 | 18.7 | 16.4 | 0.877 | 1.15 | **1.31** |
| s256 | 14 | 34.4 / 34.5 | 23.2 | 21.2 | 0.911 | 1.48 | **1.63** |
| s512 | 8 | 30.7 / 30.7 | 29.9 | 27.0 | 0.904 | 1.03 | **1.14** |
| s512 | 14 | 48.1 / 48.9 | 42.0 | 34.9 | 0.832 | 1.15 | **1.40** |
| s_cc8 (perm) | 8 | 66.4 / 40.6 | 35.9 | 29.7 | 0.828 | 1.85 ¹ | **1.37** |
| s_cc8 (perm) | 14 | 62.8 / 57.3 | 42.8 | 36.8 | 0.859 | 1.47 | **1.56** |
| s_cc16 (perm) | 8 | 343.0 / 343.3 | 275.5 | 264.2 | 0.959 | 1.25 | **1.30** |
| s_cc16 (perm) | 14 | 545.8 / 545.8 | 401.2 | 378.2 | 0.943 | 1.36 | **1.44** |
| s_df | 8 | 61.4 / 61.4 | 45.3 | 40.9 | 0.902 | 1.36 | **1.50** |
| s_df | 14 | 98.2 / 97.4 | 66.6 | 61.4 | 0.923 | 1.48 | **1.59** |
| s_perm (perm) | 8 | 69.6 / 69.6 | 49.2 | 41.0 | 0.833 | 1.42 | **1.70** |
| s_perm (perm) | 14 | 105.5 / 105.5 | 68.6 | 59.4 | 0.866 | 1.54 | **1.78** |
| sq1024 | 8 | 93.5 / 93.2 | 88.1 | 79.9 | 0.907 | 1.06 | **1.17** |
| sq1024 | 14 | 172.8 / 172.9 | 146.1 | 127.0 | 0.869 | 1.18 | **1.36** |

¹ GEMMul8 in a slow state in that process (66 µs; 40.6 in the after process).

The s128 before process shows the effect of §1.2 itself: 0.89× with 300 calls in one
process, 1.20× interleaved. After: every small case at N = 8 is ≥ 1.14× GEMMul8 in this
sequential measurement too (s512 1.14×, sq1024 1.17×), with margins larger than the state
error of the clean sweep (≈ 10 % on s256 / s512).

ncu after (s512, N = 8, same settings as §1.3): K5 `k5_crt_n<8,2>` 512 × 256 threads, 40
registers, 0.67 waves, **417 instructions per output** (was 780, 46 registers, 1.6 waves);
KF 8.37 M warp instructions (was 9.36 M: 2 044 per thread, still one wave at 64 registers).

### 3.4 Full comparison sweep (`bench/sweep_compare.txt`, `bench/run_ab.sh`, `AFTER_METHODS=gemmul8,imd`, the sweep's own 10 timed calls)

`bench/results/a5-compare-{before-b6b7e05,after-efbb01e}.jsonl`. As in §3.3, each side runs
GEMMul8 then imd in its own quiet-gated process; "vs G8" is same-process.

| case | N | G8 µs (b / a) | imd before | imd after | after / before | before vs G8 | **after vs G8** |
|---|---|---|---|---|---|---|---|
| s128 | 8 | 17.1 / 17.4 | 20.6 | 16.3 | 0.791 | 0.83 | **1.07** |
| s128 | 14 | 255.4 / 259.8 | 16.1 | 44.8 | 2.779 | 15.85 | **5.80** |
| s256 | 8 | 19.5 / 19.5 | 21.2 | 14.3 | 0.675 | 0.92 | **1.36** |
| s256 | 14 | 93.3 / 93.1 | 20.9 | 17.4 | 0.833 | 4.47 | **5.35** |
| s512 | 8 | 30.7 / 33.8 | 33.7 | 27.6 | 0.819 | 0.91 | **1.22** |
| s512 | 14 | 253.2 / 136.9 | 39.0 | 40.7 | 1.044 | 6.49 | **3.36** |
| s_lowk | 8 | 302.1 / 301.7 | 103.7 | 94.2 | 0.908 | 2.91 | **3.20** |
| s_lowk | 14 | 483.2 / 495.9 | 184.8 | 167.9 | 0.909 | 2.62 | **2.95** |
| s_batch | 8 | 4030.5 / 4030.5 | 66.6 | 58.4 | 0.877 | 60.55 | **69.03** |
| s_batch | 14 | 6869.5 / 6871.5 | 113.3 | 93.5 | 0.825 | 60.62 | **73.50** |
| s_cc8 (perm) | 8 | 42.0 / 46.0 | 32.8 | 28.6 | 0.874 | 1.28 | **1.61** |
| s_cc8 (perm) | 14 | 298.0 / 101.8 | 50.9 | 40.8 | 0.800 | 5.85 | **2.50** |
| s_cc16 (perm) | 8 | 351.2 / 365.2 | 278.9 | 285.3 | 1.023 | 1.26 | **1.28** |
| s_cc16 (perm) | 14 | 548.8 / 578.6 | 408.0 | 396.3 | 0.971 | 1.35 | **1.46** |
| s_df | 8 | 65.5 / 61.4 | 59.3 | 39.9 | 0.674 | 1.11 | **1.54** |
| s_df | 14 | 107.0 / 99.2 | 73.7 | 59.2 | 0.802 | 1.45 | **1.68** |
| s_perm (perm) | 8 | 77.9 / 77.1 | 50.2 | 46.1 | 0.919 | 1.55 | **1.67** |
| s_perm (perm) | 14 | 122.2 / 120.4 | 77.1 | 70.5 | 0.914 | 1.58 | **1.71** |
| sq1024 | 8 | 93.5 / 93.6 | 92.2 | 79.6 | 0.864 | 1.01 | **1.18** |
| sq1024 | 14 | 176.2 / 259.4 | 149.9 | 129.1 | 0.861 | 1.18 | **2.01** |
| sq2048 | 8 | 582.2 / 583.0 | 476.8 | 467.9 | 0.981 | 1.22 | **1.25** |
| sq2048 | 14 | 959.9 / 966.1 | 751.2 | 733.1 | 0.976 | 1.28 | **1.32** |
| sq4096 | 8 | 3469.5 / 3471.1 | 2654.6 | 2647.1 | 0.997 | 1.31 | **1.31** |
| sq4096 | 14 | 5973.2 / 5978.1 | 4353.4 | 4303.2 | 0.988 | 1.37 | **1.39** |
| sq8192 | 8 | 20881.8 / 20902.3 | 17937.9 | 17964.2 | 1.001 | 1.16 | **1.16** |
| sq8192 | 14 | 35895.5 / 35694.2 | 30717.6 | 30481.9 | 0.992 | 1.17 | **1.17** |
| sq16384 | 14 | 244313.0 / 245042.0 | 224820.0 | 222691.0 | 0.991 | 1.09 | **1.10** |
| r_lowk | 8 | 6681.1 / 6684.5 | 3612.7 | 3586.6 | 0.993 | 1.85 | **1.86** |
| r_lowk | 14 | 11898.8 / 11901.4 | 5695.2 | 5472.4 | 0.961 | 2.09 | **2.17** |
| r_bigk | 8 | 7253.6 / 7252.3 | 6862.9 | 6550.9 | 0.955 | 1.06 | **1.11** |
| r_bigk | 14 | 11121.9 / 11128.2 | 10457.1 | 10109.5 | 0.967 | 1.06 | **1.10** |
| r_skinny | 8 | 2810.1 / 2810.2 | 2368.9 | 2305.9 | 0.973 | 1.19 | **1.22** |
| r_skinny | 14 | 4511.6 / 4512.0 | 3590.0 | 3538.4 | 0.986 | 1.26 | **1.28** |
| r_mm | 8 | 2119.9 / 2128.3 | 2015.5 | 1883.6 | 0.935 | 1.05 | **1.13** |
| r_mm | 14 | 3099.1 / 3108.9 | 2779.1 | 2627.6 | 0.945 | 1.12 | **1.18** |
| t_nn | 8 | 3634.7 / 3632.5 | 2818.5 | 2764.5 | 0.981 | 1.29 | **1.31** |
| t_nn | 14 | 6158.8 / 6170.9 | 4551.3 | 4426.0 | 0.972 | 1.35 | **1.39** |
| t_batch | 8 | 1737.7 / 1738.9 | 1561.7 | 1468.4 | 0.940 | 1.11 | **1.18** |
| t_batch | 14 | 2747.5 / 2750.5 | 2223.6 | 2134.4 | 0.960 | 1.24 | **1.29** |
| t_ikl (perm) | 8 | 2634.4 / 2633.3 | 1678.4 | 1584.9 | 0.944 | 1.57 | **1.66** |
| t_ikl (perm) | 14 | 3651.8 / 3646.5 | 2377.0 | 2237.9 | 0.941 | 1.54 | **1.63** |
| t_df | 8 | 2307.4 / 2303.9 | 1528.3 | 1523.1 | 0.997 | 1.51 | **1.51** |
| t_df | 14 | 4041.7 / 4035.1 | 2464.1 | 2442.2 | 0.991 | 1.64 | **1.65** |
| t_cc (perm) | 8 | 4378.0 / 4380.5 | 2673.2 | 2659.9 | 0.995 | 1.64 | **1.65** |
| t_cc (perm) | 14 | 6896.7 / 6896.6 | 4449.2 | 4321.2 | 0.971 | 1.55 | **1.60** |
| t_permout (perm) | 8 | 3929.1 / 3928.6 | 2820.6 | 2757.5 | 0.978 | 1.39 | **1.42** |
| t_permout (perm) | 14 | 6465.0 / 6465.3 | 4561.4 | 4427.3 | 0.971 | 1.42 | **1.46** |

- **No regression**: after / before 0.935–1.001 on every medium and large case (sq8192 N = 8 1.001;
  K5 per N and the cheaper pack residues and §3.2 terms help the large shapes by 1–6 %:
  r_bigk 0.955–0.967, r_mm 0.935–0.945, t_batch 0.940–0.960, t_ikl 0.941–0.944, r_lowk
  N = 14 0.961, t_nn / t_cc / t_permout N = 14 0.971–0.972); vs GEMMul8 ≥ 1.07× everywhere.
- Three small records with after / before > 1 are performance states of §1.2, not the
  code: s128 N = 14 (44.8 µs in a slow state for all 10 calls, min 43.1; the 300-call A/B
  of §3.3 has 20.2 → 16.4), s512 N = 14 and s_cc16 N = 8 (the after process in the 2535 MHz
  clock state, `gpu_after`; GEMMul8 in the same process is slower too, so the same-process
  ratios rise: s_cc16 1.26 → 1.28×). A re-run of those cases
  (`bench/results/a5-recheck-{before-b6b7e05,after-efbb01e}.jsonl`): s128 N = 14 21.6 → 13.5
  µs, s512 N = 14 49.5 → 36.7 µs, s_cc16 N = 8 273.4 → 285.9 µs with GEMMul8 347 → 374 µs in
  the same processes (1.27 → 1.31×; nsys span 291.9 → 262.6 µs, §3.1); s512 N = 8 there:
  0.94 → 1.03× (10 calls).
- GEMMul8 at N = 14 is bimodal on the smallest shapes as before (s128 255 µs, s512 253 /
  137 µs, sq1024 259 µs in single processes), so those ratios mean little.

## 4. Where it stands

- **All small cases ≥ 1× GEMMul8 at N = 8 and N = 14** in every measurement made here:
  interleaved 1.09–1.70× (N = 8), sequential 300-call A/B 1.14–1.70×, the 10-call sweep
  1.03–1.67× (s512 1.03× in the noisiest re-run, 1.22× in the main one).
- What is left per call at s512, N = 8 (nsys): KF 11.2 + K4 7.5 + K5 6.5 + gaps 1.3 + ~3
  launch latency ≈ 30 µs against GEMMul8's 37 µs. The 3 µs launch latency and ~0.65 µs per
  kernel boundary are the same for both libraries; imd has 3 launches to GEMMul8's 6.
- The measurement itself (§1.2) is now the largest uncertainty for tiny calls: a 20–40 µs
  call moves by ±10–20 % between per-process states (host core type under `powersave`, GPU
  DVFS). For paper-grade small-shape numbers use `--methods duel` (interleaved) or pin the
  bench to a P-core and lock clocks (root).
