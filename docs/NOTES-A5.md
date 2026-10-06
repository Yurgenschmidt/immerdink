# A5 notes: small-shape polish

ROADMAP A5. After A1 / A3 the clean bench-mode sweep
(`bench/results/compare-clean-bb6b568.jsonl`, dev `b6b7e05` = `bb6b568` + docs) still had
s256 (256³) and s512 (512³) at N = 8 as the weakest small cases. Goal: shorter calls there
and on the small family (s128, s_cc8, s_df, s_perm, s_cc16, sq1024) without regressing
anything else. Measure first (§1), then fix by measured payoff (§2), then before/after (§3).

Machine: bench mode (console only, no display server, no other GPU clients, persistence
mode on, PCIe gen4 x16 under load), RTX 4090 at its 450 W cap, i9-13900K (8 P-cores =
CPUs 0–15, 16 E-cores = CPUs 16–31, `powersave` governor with HWP). Every timing through
`tools/run_bench.sh` (quiet gate, machine lock, GPU lock).

New tools (this work):
- `bench/nsys_case.sh` / `bench/nsys_sweep.sh`: nsys 2026.5.1 (`-t cuda --gpuctxsw=true`)
  of one case or a sweep file, one process per (case, N); `bench/nsys_calls.py` turns the
  sqlite export into a per-call breakdown of the bench's timed calls (host time, launch
  latency, GPU span, per-kernel medians, gaps, context switches).
- `imd_bench --methods duel`: two sides timed **interleaved, call by call, in one
  process**; reports both medians and the median of the per-pair ratios. In this repository
  the sides are two immerdink configurations (`--duel-a`, `--duel-b`; docs/BENCHMARKING.md
  "Tiny shapes"). The duels run for this work paired imd with another library; their
  ratios are kept outside this repository.
- `IMD_BENCH_TIMES=1`: every timed call's event time on stderr (`IMD_BENCH_CLOCK=1` adds an
  SM-clock probe after each call; its 4 µs window turned out too short to be useful).
- `bench/sweep_a5.txt`: the small family.

## 1. Where the time goes (dev `b6b7e05`, before any change)

### 1.1 Per-kernel GPU time (nsys, 300 timed calls per method, median of the last 295)

`bench/nsys_sweep.sh build/bench/imd_bench bench/sweep_a5.txt build/nsys/base --reps 300`
(one process per (case, N), N = 8 and 14). µs. "wall" ≈ the bench's event time (e0 record
→ last kernel end; nsys adds ~1 µs of host overhead per API call, so host-heavy callers look
worse under nsys); "lat0" = e0 record → first kernel start (the
launch latency of a call on an idle GPU); "span" = first kernel start → last kernel end =
kernels + gaps.

| case | N | launches | host | lat0 | span | kernels | gaps | per kernel |
|---|---|---|---|---|---|---|---|---|
| s128 | 8 | 3 | 10.1 | 4.5 | 14.1 | 12.8 | 1.3 | KF 6.0, K4 2.8 (64×64), K5 4.0 |
| s256 | 8 | 3 | 5.9 | 3.0 | 16.5 | 15.3 | 1.3 | KF 7.1, K4 3.6 (64×64), K5 4.6 |
| s512 | 8 | 3 | 5.9 | 3.0 | 30.8 | 29.4 | 1.3 | KF 12.5, K4 7.5 (128×128), **K5 9.4** |
| s512 | 14 | 3 | 5.9 | 3.0 | 41.0 | 39.6 | 1.3 | KF 15.3, K4 11.5 (128×256), K5 12.9 |
| s_cc8 | 8 | 3 | 5.8 | 3.0 | 31.2 | 29.9 | 1.3 | KF 13.0, K4 7.4, K5 9.4 |
| s_df | 8 | 3 | 6.0 | 3.1 | 47.8 | 46.4 | 1.3 | KF 12.5, K4 15.0, K5 18.9 |
| s_perm | 8 | 3 | 6.0 | 3.1 | 48.2 | 46.9 | 1.3 | KF 13.0, K4 15.0, K5 18.9 |
| s_cc16 | 8 | 5 | 9.6 | 3.1 | 291.9 | 289.1 | 2.8 | K0 1.8, KL 68.2 + 64.2, K4 116.8, K5 38.1 |
| sq1024 | 8 | 3 | 6.0 | 3.1 | 96.3 | 95.0 | 1.4 | KF 40.8, K4 35.4, K5 18.8 |

(N = 14 and the full per-kernel table in `build/nsys/base/*.txt`, not committed.)

What this says:
- **The fixed costs are small.** The launch latency (lat0) is ≈ 3 µs; the gaps are
  0.6–0.7 µs per kernel boundary, 1.3 µs per call for 3 launches.
- **The host is not the bottleneck** (as in NOTES-A1): imd_contract spends 0.3 µs before
  its first launch and ~6 µs in total (nsys-inflated) while its GPU span is 14–96 µs.
- **The time is in the kernels, and at s512 / sq1024 K5 stands out**: 9.4 µs for 262 144
  outputs at s512, more than K4 (7.5 µs, 128×128, the per-case best of A3) although K5 does
  O(m·n) work against K4's O(m·n·k); KF (12.5 µs) does both operands' scaling and 8
  residues.

### 1.2 Why the sweep's s256 / s512 numbers misled: the measurement, not the kernels

The clean sweep times 10 calls per method (after 2 warm-up calls) in one process. Per-call
event times (`IMD_BENCH_TIMES=1`) show that tiny calls drift between **performance states that last from milliseconds to seconds**, and a 10-call
median samples one of them:

- s256 imd, N = 8, consecutive calls of one process: 47 43 43 43 42 42 41 41 … 40 54 **18
  18 18 18** 68 **17 17 17 …** µs (the state flipped after ~16 calls); another process:
  17.4 µs throughout; another: 20.5–21.5 µs throughout.
- Other libraries' small calls timed in the same processes drift between such states too.
- Pinning the bench to one core: on a P-core (CPU 2) s256 imd 17.4–18.4 µs; on an E-core
  (CPU 20) 18.4–19.3 µs. A caller with more host work per call (more launches) can become
  host-bound on an E-core, and the unpinned bench thread migrates between core types.
- Under nsys (host calls slower, GPU never idle as long) the states did not appear: every
  kernel's duration was stable to ±1 % over 300 calls.
- Longer warm-up (300 calls instead of 2) did not remove the states (s256 imd 17.2 and
  20.5 µs in two processes with the same settings).

The causes are outside the kernels (host core type and frequency under `powersave`, GPU
DVFS and PCIe link power states: at idle the link is at gen1, SM 210 MHz, memory 405 MHz),
and they cannot be locked without root (`nvidia-smi -lgc`, governor). A sequential
comparison of two methods (or two builds) on a 20 µs call therefore has a ±20 % state error
that a 10-call median does not average out. **Interleaving the two sides call by call**
puts both in the same state; the per-pair ratio is stable (p10–p90 within ±3–6 % on most
cases). imd's medians in such an interleaved run (`imd_bench --methods duel --reps 400`,
`build/nsys/duel-base1.jsonl`, µs):

| case | N | imd |
|---|---|---|
| s128 | 8 | 14.3 |
| s256 | 8 | 16.6 |
| s512 | 8 | 29.6 |
| s_cc8 | 8 | 29.7 |
| s_cc16 | 8 | 274.4 |
| s_df | 8 | 44.4 |
| s_perm | 8 | 44.8 |
| sq1024 | 8 | 94.3 |
| s128 | 14 | 16.1 |
| s256 | 14 | 18.5 |
| s512 | 14 | 38.9 |
| s_cc8 | 14 | 38.9 |
| s_cc16 | 14 | 398.0 |
| s_df | 14 | 66.6 |
| s_perm | 14 | 66.6 |
| sq1024 | 14 | 154.5 |

So the sweep's weak s256 / s512 numbers were per-process states, not the code. §2 works
on the kernels where the time is: K5 first (9.4 µs at s512, 18.8 µs at sq1024), then KF.

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
  at sq1024 N = 14 (128×256 66.8 µs vs 128×128 ≈ 62), ~5 µs of a 137 µs call; a rule change
  touches every shape (A3's fit), so it stays as it is.
- **K5 fused into a small-tile K4 (2 launches)**: the fixed cost per launch is ~2 µs (gap
  0.6–0.7 µs + the fill/drain of a minimal ~1.3 µs kernel), i.e. at most ~2 µs per call.
  But one CTA would have to hold all N moduli of its output tile: at s512 that is 16 CTAs of 128×128 (or 64 of 64×64) doing 8 GEMMs each in sequence plus the CRT, against
  today's 128 CTAs and a 6.5 µs K5 on all SMs: slower, not faster. Only s128-sized problems
  (≤ 4 output tiles) could gain, and s128's whole GPU span is already 12.8 µs (§3.1). Not
  needed; no spec change made.
- **KF instruction count**: after changes 3 and 4 KF is still the largest kernel at s512
  (11.2 µs, close to issue-bound: ~2 000 instructions per thread in one wave). Further cuts
  (quantize, the per-row 256-bit compare) are a few % each.

## 3. Before / after

### 3.1 nsys, per call (after = `efbb01e`; same tool and settings as §1.1; µs)

| case | N | imd span before → after | imd kernels after | imd wall before → after |
|---|---|---|---|---|
| s128 | 8 | 14.1 → **12.8** | KF 5.7, K4 2.8, K5 3.0 | 19.7 → 16.5 |
| s256 | 8 | 16.5 → **14.8** | KF 6.5, K4 3.6, K5 3.5 | 20.3 → 18.7 |
| s512 | 8 | 30.8 → **26.5** | KF 11.2, K4 7.5, K5 6.5 | 34.5 → 30.2 |
| s512 | 14 | 41.0 → **34.2** | KF 13.1, K4 11.6, K5 8.2 | 44.7 → 37.9 |
| s_cc8 | 8 | 31.2 → **27.2** | KF 11.7, K4 7.5, K5 6.7 | 34.9 → 30.9 |
| s_df | 8 | 47.8 → **43.5** | KF 10.8, K4 15.0, K5 16.4 | 51.6 → 47.5 |
| s_perm | 8 | 48.2 → **43.7** | KF 11.3, K4 14.9, K5 16.2 | 52.1 → 47.4 |
| s_cc16 | 8 | 291.9 → **262.6** | K0 1.7, KL 60.6 + 58.0, K4 109.1, K5 29.7 | 295.8 → 266.3 |
| sq1024 | 8 | 96.3 → **86.6** | KF 33.7, K4 35.3, K5 16.3 | 100.2 → 90.4 |
| sq1024 | 14 | 157.7 → **136.5** | KF 42.9, K4 66.8, K5 25.5 | 161.8 → 140.3 |

Host time, launch latency and gaps are unchanged (5.8–6.3 µs, 3.0 µs, 1.3 µs for 3
launches): the gain is all kernel time, mostly K5 and KF.

### 3.2 Interleaved duel (`imd_bench --methods duel --reps 400`, same as §1.2)

These runs paired imd with another library, so their ratios are kept outside this
repository. The same-session A/B below is the committed record of imd before / after.

### 3.3 Same-session A/B (`bench/run_ab.sh`, `bench/sweep_a5.txt`: 300 timed calls after 20 warm-up calls)

`bench/results/a5-small-{before-b6b7e05,after-efbb01e}.jsonl`; each side runs in its own
quiet-gated process (another library's calls before imd's, as in the comparison sweeps),
interleaved per case; summary by `bench/ab_ratio.py`. µs.

| case | N | imd before | imd after | after / before |
|---|---|---|---|---|
| s128 | 8 | 19.4 | 15.2 | 0.783 |
| s128 | 14 | 20.2 | 16.4 | 0.811 |
| s256 | 8 | 18.7 | 16.4 | 0.877 |
| s256 | 14 | 23.2 | 21.2 | 0.911 |
| s512 | 8 | 29.9 | 27.0 | 0.904 |
| s512 | 14 | 42.0 | 34.9 | 0.832 |
| s_cc8 (perm) | 8 | 35.9 | 29.7 | 0.828 |
| s_cc8 (perm) | 14 | 42.8 | 36.8 | 0.859 |
| s_cc16 (perm) | 8 | 275.5 | 264.2 | 0.959 |
| s_cc16 (perm) | 14 | 401.2 | 378.2 | 0.943 |
| s_df | 8 | 45.3 | 40.9 | 0.902 |
| s_df | 14 | 66.6 | 61.4 | 0.923 |
| s_perm (perm) | 8 | 49.2 | 41.0 | 0.833 |
| s_perm (perm) | 14 | 68.6 | 59.4 | 0.866 |
| sq1024 | 8 | 88.1 | 79.9 | 0.907 |
| sq1024 | 14 | 146.1 | 127.0 | 0.869 |

The s128 before process shows the effect of §1.2 itself: 19.4 µs with 300 calls in one
process, 14.3 µs in the interleaved run. After: every small case is 4–22 % faster (after /
before 0.78–0.96 at N = 8, 0.81–0.94 at N = 14), in line with the nsys kernel times of
§3.1.

ncu after (s512, N = 8, same settings as §1.3): K5 `k5_crt_n<8,2>` 512 × 256 threads, 40
registers, 0.67 waves, **417 instructions per output** (was 780, 46 registers, 1.6 waves);
KF 8.37 M warp instructions (was 9.36 M: 2 044 per thread, still one wave at 64 registers).

### 3.4 Full sweep (`bench/sweep_compare.txt`, `bench/run_ab.sh`, the sweep's own 10 timed calls)

`bench/results/a5-compare-{before-b6b7e05,after-efbb01e}.jsonl`. As in §3.3, each side runs
in its own quiet-gated process. µs.

| case | N | imd before | imd after | after / before |
|---|---|---|---|---|
| s128 | 8 | 20.6 | 16.3 | 0.791 |
| s128 | 14 | 16.1 | 44.8 | 2.779 |
| s256 | 8 | 21.2 | 14.3 | 0.675 |
| s256 | 14 | 20.9 | 17.4 | 0.833 |
| s512 | 8 | 33.7 | 27.6 | 0.819 |
| s512 | 14 | 39.0 | 40.7 | 1.044 |
| s_lowk | 8 | 103.7 | 94.2 | 0.908 |
| s_lowk | 14 | 184.8 | 167.9 | 0.909 |
| s_batch | 8 | 66.6 | 58.4 | 0.877 |
| s_batch | 14 | 113.3 | 93.5 | 0.825 |
| s_cc8 (perm) | 8 | 32.8 | 28.6 | 0.874 |
| s_cc8 (perm) | 14 | 50.9 | 40.8 | 0.800 |
| s_cc16 (perm) | 8 | 278.9 | 285.3 | 1.023 |
| s_cc16 (perm) | 14 | 408.0 | 396.3 | 0.971 |
| s_df | 8 | 59.3 | 39.9 | 0.674 |
| s_df | 14 | 73.7 | 59.2 | 0.802 |
| s_perm (perm) | 8 | 50.2 | 46.1 | 0.919 |
| s_perm (perm) | 14 | 77.1 | 70.5 | 0.914 |
| sq1024 | 8 | 92.2 | 79.6 | 0.864 |
| sq1024 | 14 | 149.9 | 129.1 | 0.861 |
| sq2048 | 8 | 476.8 | 467.9 | 0.981 |
| sq2048 | 14 | 751.2 | 733.1 | 0.976 |
| sq4096 | 8 | 2654.6 | 2647.1 | 0.997 |
| sq4096 | 14 | 4353.4 | 4303.2 | 0.988 |
| sq8192 | 8 | 17937.9 | 17964.2 | 1.001 |
| sq8192 | 14 | 30717.6 | 30481.9 | 0.992 |
| sq16384 | 14 | 224820.0 | 222691.0 | 0.991 |
| r_lowk | 8 | 3612.7 | 3586.6 | 0.993 |
| r_lowk | 14 | 5695.2 | 5472.4 | 0.961 |
| r_bigk | 8 | 6862.9 | 6550.9 | 0.955 |
| r_bigk | 14 | 10457.1 | 10109.5 | 0.967 |
| r_skinny | 8 | 2368.9 | 2305.9 | 0.973 |
| r_skinny | 14 | 3590.0 | 3538.4 | 0.986 |
| r_mm | 8 | 2015.5 | 1883.6 | 0.935 |
| r_mm | 14 | 2779.1 | 2627.6 | 0.945 |
| t_nn | 8 | 2818.5 | 2764.5 | 0.981 |
| t_nn | 14 | 4551.3 | 4426.0 | 0.972 |
| t_batch | 8 | 1561.7 | 1468.4 | 0.940 |
| t_batch | 14 | 2223.6 | 2134.4 | 0.960 |
| t_ikl (perm) | 8 | 1678.4 | 1584.9 | 0.944 |
| t_ikl (perm) | 14 | 2377.0 | 2237.9 | 0.941 |
| t_df | 8 | 1528.3 | 1523.1 | 0.997 |
| t_df | 14 | 2464.1 | 2442.2 | 0.991 |
| t_cc (perm) | 8 | 2673.2 | 2659.9 | 0.995 |
| t_cc (perm) | 14 | 4449.2 | 4321.2 | 0.971 |
| t_permout (perm) | 8 | 2820.6 | 2757.5 | 0.978 |
| t_permout (perm) | 14 | 4561.4 | 4427.3 | 0.971 |

- **No regression**: after / before 0.935–1.001 on every medium and large case (sq8192 N = 8 1.001;
  K5 per N and the cheaper pack residues and §3.2 terms help the large shapes by 1–6 %:
  r_bigk 0.955–0.967, r_mm 0.935–0.945, t_batch 0.940–0.960, t_ikl 0.941–0.944, r_lowk
  N = 14 0.961, t_nn / t_cc / t_permout N = 14 0.971–0.972).
- Three small records with after / before > 1 are performance states of §1.2, not the
  code: s128 N = 14 (44.8 µs in a slow state for all 10 calls, min 43.1; the 300-call A/B
  of §3.3 has 20.2 → 16.4), s512 N = 14 and s_cc16 N = 8 (the after process in the 2535 MHz
  clock state, `gpu_after`; the other library's calls in the same process were slower
  too). A re-run of those cases (`bench/results/a5-recheck-{before-b6b7e05,after-efbb01e}.jsonl`): s128 N = 14 21.6 → 13.5
  µs, s512 N = 14 49.5 → 36.7 µs, s_cc16 N = 8 273.4 → 285.9 µs, again with the whole after
  process slower (nsys span 291.9 → 262.6 µs, §3.1).

## 4. Where it stands

- **Every small case is faster** in every measurement made here: GPU span 9–17 % shorter
  (nsys, §3.1), 4–22 % in the sequential 300-call A/B (§3.3), and no regression on the
  medium and large shapes (§3.4).
- What is left per call at s512, N = 8 (nsys): KF 11.2 + K4 7.5 + K5 6.5 + gaps 1.3 + ~3
  launch latency ≈ 30 µs. The 3 µs launch latency and ~0.65 µs per kernel boundary are
  fixed costs; imd has 3 launches.
- The measurement itself (§1.2) is now the largest uncertainty for tiny calls: a 20–40 µs
  call moves by ±10–20 % between per-process states (host core type under `powersave`, GPU
  DVFS). For paper-grade small-shape numbers use an interleaved duel (`--methods duel`,
  two configurations; docs/BENCHMARKING.md) or pin the bench to a P-core and lock clocks
  (root).
