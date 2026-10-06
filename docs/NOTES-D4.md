# D4 notes (part 1): fused K1 + pack for large operands (KL)

ROADMAP D4, first part (user decision: K1 + pack fusion for large shapes, not pack inside the
GEMM). Design: DESIGN v1.6 §4 "Fused large path KL", §6a version 4; spec: `Pipeline.tla`
`FusedLarge` (5 positive, 4 negative models). Before = dev `a572d85`, after = `197cd0f`
(this branch; later commits change documentation only).

Tooling as in NOTES-A3: every timing and profile through `tools/run_bench.sh` (quiet gate,
memcap, the shared GPU lock); ncu 2026.3.1 with counters (`gpu__time_duration`,
`dram__bytes_read/write`, `lts__t_sector_hit_rate`) on the scale and pack kernels of the
first `imd_contract` of a bench process (ncu locks the SM clock: its durations compare with
each other, not with event timings); nsys 2026.5.1 `cuda_gpu_kern_sum` over 10 timed calls
(native clocks); the bench's phase marks for the per-phase split. PCIe is stuck at gen1 on
this box; nothing timed here includes a host transfer.

## 1. The problem (before)

Without KF (k̂ > 1024) the operand x streams from DRAM twice before its first GEMM: K1's
pass 1 (flag, γ; pass 2 for S mostly hits L2) and group 1's pack, and once more per later
residue group. K4 is at 89–99 % of its bound on r_skinny / r_bigk / r_mm, so what was left
there was K1 + the pack (NOTES-A3 §1).

## 2. Design and the launch shape

KL (DESIGN v1.6 §4) replaces K1 of an operand and its group-1 pack (and, for B, the launch
zeroing RB rows n..n̂−1) with one launch per operand; K0 stays a separate launch, so KL reads
the scatter vectors from the vec region (or `h·stride` when K is one fused mode). One CTA owns
whole rows: the k23_pack tile (4096 FP64: th rows × tk, tk = pow2 ≥ k̂ ≤ 4096 along k̄,
th = 8 along rows), a row = nc chunks of tk; pass 1 (γ, flag; chunk 0 kept in registers, the
last chunk left in shared memory), pass 2 (S; middle chunks re-read, most recent first),
pass 3 (group 1's residues; middle chunks re-read). Rows of nc ≤ 2 (k̂ ≤ 8192 along k̄) are
read from DRAM exactly once and never re-read; longer rows re-read nc − 2 chunks from L2
twice.

Values per thread (a launch choice, same results): KL A / KL B phase ms, N = 8, imd only,
one gated process each (`bench/results/d4-kl-threads-variants.jsonl`, records in the order
E = 16, 8, 4 per case):

| case (rows × k, chunks) | E = 16 (256 thr, 2 CTA/SM) | E = 8 (512 thr, 2 CTA/SM) | E = 4 (1024 thr, 1 CTA/SM) |
|---|---|---|---|
| r_skinny A (16384 × 4096, nc 1) | 1.346 | **1.222** (87 % of bound) | 1.707 |
| sq8192 A (8192 × 8192, nc 2) | 1.451 | **1.223** (87 %) | 1.640 |
| sq2048 A (2048 × 2048, nc 1) | 0.112 | **0.098** | 0.121 |
| t_nn A (4096 × 4096, nc 1) | 0.360 | **0.328** | 0.488 |
| r_bigk A (2048 × 32768, nc 8) | 1.903 | 1.801 | **1.713** |
| r_mm A (1000 × 50000, nc 13) | 2.142 | 1.567 | **1.420** |
| r_mm B (64 × 50000, nc 13) | 0.246 | 0.169 | **0.146** |
| t_nn B (along rows, th 8, nc 8) | 0.527 | 0.544 | **0.486** |

The first version (E = 16 everywhere) left KL at 37–74 % of its bound; r_mm got slower than
the unfused path (2.14 vs 0.59 + 0.93 ms). With 256 threads per CTA and (for re-read rows)
one CTA per SM, the emit of 8–14 residues per value ran at a low IPC (8 warps per SM).
Adopted: **E = 8 for nc ≤ 2, E = 4 for nc > 2** (contract.cu `launch_kl`). With one 1024-thread
CTA per SM the rows in flight are SMs × th × 8k̂ (r_bigk 32 MB, r_mm 51 MB), inside K1's 64 MiB
L2 budget (A3), without a separate grid cap.

Registers: E = 8 and E = 4 compile to 64 registers (`__launch_bounds__` 512×2 / 1024×1) with
24–36 bytes of spill; E = 16 (not used) to 128 without spill.

## 3. DRAM traffic and time of the scale + pack kernels (ncu, one call)

Before: K1 of A and B and every pack (K2/K3); after: KL A, KL B and the packs of groups
2.. . g = N (one group), 7 and 2. Read / write MB from `dram__bytes_read/write.sum`, µs from
`gpu__time_duration` (locked clocks).

| case | N | g | before: kernels | read MB | write MB | µs | after: kernels | read MB | write MB | µs | read after/before | time after/before |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| r_skinny | 8 | 8 | 2 K1 + 2 pack | 1108 | 510 | 1926 | 2 KL | 554 | 515 | 1228 | 0.50 | 0.64 |
| r_skinny | 8 | 7 | 2 K1 + 4 pack | 1662 | 504 | 2593 | 2 KL + 2 pack | 1108 | 508 | 1899 | 0.67 | 0.73 |
| r_skinny | 8 | 2 | 2 K1 + 8 pack | 2770 | 488 | 3867 | 2 KL + 6 pack | 2216 | 485 | 3330 | 0.80 | 0.86 |
| r_skinny | 14 | 14 | 2 K1 + 2 pack | 1108 | 914 | 2299 | 2 KL | 554 | 933 | 1724 | 0.50 | 0.75 |
| r_skinny | 14 | 7 | 2 K1 + 4 pack | 1662 | 882 | 3006 | 2 KL + 2 pack | 1108 | 889 | 2430 | 0.67 | 0.81 |
| r_skinny | 14 | 2 | 2 K1 + 14 pack | 4432 | 855 | 6297 | 2 KL + 12 pack | 3878 | 853 | 5749 | 0.87 | 0.91 |
| r_bigk | 8 | 8 | 2 K1 + 2 pack | 2217 | 1019 | 4006 | 2 KL | 1098 | 1083 | 3418 | 0.50 | 0.85 |
| r_bigk | 8 | 7 | 2 K1 + 4 pack | 3279 | 1008 | 5371 | 2 KL + 2 pack | 2165 | 1053 | 4641 | 0.66 | 0.86 |
| r_bigk | 8 | 2 | 2 K1 + 8 pack | 5435 | 976 | 7747 | 2 KL + 6 pack | 4297 | 973 | 7105 | 0.79 | 0.92 |
| r_bigk | 14 | 14 | 2 K1 + 2 pack | 2214 | 1824 | 4881 | 2 KL | 1206 | 1911 | 4215 | 0.54 | 0.86 |
| r_bigk | 14 | 7 | 2 K1 + 4 pack | 3282 | 1765 | 6373 | 2 KL + 2 pack | 2165 | 1810 | 5512 | 0.66 | 0.86 |
| r_bigk | 14 | 2 | 2 K1 + 14 pack | 8650 | 1706 | 12506 | 2 KL + 12 pack | 7521 | 1703 | 11701 | 0.87 | 0.94 |
| r_mm | 8 | 8 | 2 K1 + 2 pack | 853 | 373 | 1706 | 2 KL | 530 | 395 | 1486 | 0.62 | 0.87 |
| r_mm | 8 | 7 | 2 K1 + 4 pack | 1280 | 367 | 2253 | 2 KL + 2 pack | 939 | 388 | 2031 | 0.73 | 0.90 |
| r_mm | 8 | 2 | 2 K1 + 8 pack | 2133 | 351 | 3304 | 2 KL + 6 pack | 1712 | 354 | 3075 | 0.80 | 0.93 |
| r_mm | 14 | 14 | 2 K1 + 2 pack | 856 | 688 | 2034 | 2 KL | 605 | 710 | 1957 | 0.71 | 0.96 |
| r_mm | 14 | 7 | 2 K1 + 4 pack | 1280 | 643 | 2623 | 2 KL + 2 pack | 938 | 664 | 2466 | 0.73 | 0.94 |
| r_mm | 14 | 2 | 2 K1 + 14 pack | 3413 | 612 | 5269 | 2 KL + 12 pack | 2992 | 615 | 4920 | 0.88 | 0.93 |
| sq8192 | 8 | 8 | 2 K1 + 2 pack | 2148 | 1019 | 3761 | 2 KL | 1074 | 1029 | 2394 | 0.50 | 0.64 |
| sq8192 | 8 | 7 | 2 K1 + 4 pack | 3222 | 1007 | 5029 | 2 KL + 2 pack | 2148 | 1014 | 3625 | 0.67 | 0.72 |
| sq8192 | 8 | 2 | 2 K1 + 8 pack | 5371 | 977 | 7537 | 2 KL + 6 pack | 4297 | 970 | 6371 | 0.80 | 0.85 |
| sq8192 | 14 | 14 | 2 K1 + 2 pack | 2148 | 1820 | 4516 | 2 KL | 1074 | 1883 | 3407 | 0.50 | 0.75 |
| sq8192 | 14 | 7 | 2 K1 + 4 pack | 3222 | 1765 | 5968 | 2 KL + 2 pack | 2148 | 1771 | 4626 | 0.67 | 0.78 |
| sq8192 | 14 | 2 | 2 K1 + 14 pack | 8593 | 1706 | 12171 | 2 KL + 12 pack | 7519 | 1699 | 10971 | 0.87 | 0.90 |
| t_cc | 8 | 8 | 2 K1 + 2 pack | 537 | 214 | 932 | 2 KL | 269 | 215 | 650 | 0.50 | 0.70 |
| t_cc | 8 | 7 | 2 K1 + 4 pack | 806 | 203 | 1294 | 2 KL + 2 pack | 537 | 203 | 989 | 0.67 | 0.76 |
| t_cc | 8 | 2 | 2 K1 + 8 pack | 1343 | 173 | 1983 | 2 KL + 6 pack | 1075 | 167 | 1718 | 0.80 | 0.87 |
| t_cc | 14 | 14 | 2 K1 + 2 pack | 537 | 410 | 1141 | 2 KL | 269 | 420 | 851 | 0.50 | 0.75 |
| t_cc | 14 | 7 | 2 K1 + 4 pack | 806 | 358 | 1493 | 2 KL + 2 pack | 537 | 357 | 1174 | 0.67 | 0.79 |
| t_cc | 14 | 2 | 2 K1 + 14 pack | 2150 | 299 | 3221 | 2 KL + 12 pack | 1881 | 293 | 2948 | 0.88 | 0.92 |

- **g = N: the DRAM reads of the scale + pack phase halve** (0.50 on r_skinny, sq8192, t_cc;
  0.50 / 0.54 on r_bigk at N = 8 / 14; 0.62 / 0.71 on r_mm, whose 400 KB rows partly miss L2
  in passes 2/3); the writes (group 1's residues) are unchanged. Time (locked clocks) 0.64–0.75
  where rows stay on chip, 0.85–0.96 for the long-row shapes, whose KL is instruction- and
  latency-bound rather than at its DRAM bound (r_bigk KL 66–69 % of its one-read bound,
  r_mm 56–61 %).
- **g < N: exactly one read of each operand is saved**, whatever g: the later groups' packs
  still read the operand once each, so the read ratio is ⌈N/g⌉ / (1 + ⌈N/g⌉) (g = 7: 2/3,
  measured 0.66–0.73; N = 14, g = 2: 7/8, measured 0.87–0.88) and the time ratio approaches 1
  as g shrinks (0.86–0.94 at g = 2).

## 4. Per call, native clocks (nsys, g = N, 10 timed calls)

`K1 + pack` before vs `KL` after (µs per call, both operands), and every kernel of the call:

| case | N | before K1 + pack µs | after KL (+ packs) µs | after / before | before call µs (K0..K5) | after call µs |
|---|---|---|---|---|---|---|
| r_skinny | 8 | 1922 (659 + 1263) | 1272 | 0.66 | 2969 | 2321 |
| r_skinny | 14 | 2334 (648 + 1686) | 1733 | 0.74 | 4139 | 3541 |
| r_bigk | 8 | 3972 (1522 + 2450) | 3402 | 0.86 | 7406 | 6853 |
| r_bigk | 14 | 4915 (1522 + 3392) | 4318 | 0.88 | 10920 | 10330 |
| r_mm | 8 | 1645 (642 + 1003) | 1480 | 0.90 | 2173 | 2008 |
| r_mm | 14 | 1997 (642 + 1356) | 1927 | 0.96 | 2897 | 2828 |
| sq8192 | 8 | 3749 (1280 + 2468) | 2428 | 0.65 | 18420 | 17340 |
| sq8192 | 14 | 4573 (1268 + 3304) | 3416 | 0.75 | 30272 | 28814 |
| t_cc | 8 | 968 (362 + 606) | 656 | 0.68 | 2952 | 2641 |
| t_cc | 14 | 1177 (357 + 820) | 866 | 0.74 | 4712 | 4399 |

## 5. Same-session before / after (all of `bench/sweep_compare.txt`)

`bench/run_ab.sh` with before = dev `a572d85`, after = `197cd0f`, interleaved per case, both
processes running another library's calls first (as in the comparison sweeps; the clock
states of NOTES-A3 §5): `bench/results/d4-before-a572d85.jsonl`,
`bench/results/compare-197cd0f.jsonl`. Median ms.

| case | N | imd before | imd after | after/before | TFLOPS-eq after |
|---|---|---|---|---|---|
| s128 | 8 | 0.03891 | 0.01946 | 0.500 | 0.2156 |
| s128 | 14 | 0.02253 | 0.02253 | 1.000 | 0.1862 |
| s256 | 8 | 0.01946 | 0.04352 | 2.237 | 0.771 |
| s256 | 14 | 0.0215 | 0.0215 | 1.000 | 1.56 |
| s512 | 8 | 0.03686 | 0.03072 | 0.833 | 8.738 |
| s512 | 14 | 0.04301 | 0.04659 | 1.083 | 5.761 |
| s_lowk | 8 | 0.1055 | 0.1096 | 1.039 | 4.9 |
| s_lowk | 14 | 0.1838 | 0.1838 | 1.000 | 2.921 |
| s_batch | 8 | 0.06963 | 0.07578 | 1.088 | 1.771 |
| s_batch | 14 | 0.1147 | 0.1198 | 1.045 | 1.12 |
| s_cc8 | 8 | 0.03686 | 0.0471 | 1.278 | 5.699 |
| s_cc8 | 14 | 0.04608 | 0.04864 | 1.056 | 5.519 |
| s_cc16 | 8 | 0.2918 | 0.2775 | 0.951 | 26.12 |
| s_cc16 | 14 | 0.4234 | 0.3999 | 0.944 | 18.13 |
| s_df | 8 | 0.05018 | 0.05427 | 1.082 | 9.892 |
| s_df | 14 | 0.07424 | 0.07373 | 0.993 | 7.282 |
| s_perm | 8 | 0.05171 | 0.0512 | 0.990 | 10.49 |
| s_perm | 14 | 0.07373 | 0.07526 | 1.021 | 7.133 |
| sq1024 | 8 | 0.1085 | 0.1029 | 0.948 | 20.87 |
| sq1024 | 14 | 0.1628 | 0.1628 | 1.000 | 13.19 |
| sq2048 | 8 | 0.5827 | 0.4772 | 0.819 | 36 |
| sq2048 | 14 | 0.8545 | 0.7511 | 0.879 | 22.87 |
| sq4096 | 8 | 2.975 | 2.622 | 0.882 | 52.41 |
| sq4096 | 14 | 4.608 | 4.295 | 0.932 | 32 |
| sq8192 | 8 | 18.64 | 17.19 | 0.922 | 63.96 |
| sq8192 | 14 | 30.62 | 29.6 | 0.967 | 37.14 |
| sq16384 | 14 | 232.4 | 223.1 | 0.960 | 39.43 |
| r_lowk | 8 | 3.569 | 3.566 | 0.999 | 38.55 |
| r_lowk | 14 | 5.5 | 5.492 | 0.999 | 25.02 |
| r_bigk | 8 | 7.391 | 6.669 | 0.902 | 41.22 |
| r_bigk | 14 | 10.92 | 10.19 | 0.933 | 26.96 |
| r_skinny | 8 | 3.022 | 2.392 | 0.791 | 28.73 |
| r_skinny | 14 | 4.174 | 3.565 | 0.854 | 19.28 |
| r_mm | 8 | 2.186 | 2.104 | 0.962 | 3.042 |
| r_mm | 14 | 2.909 | 2.869 | 0.986 | 2.231 |
| t_nn | 8 | 3.104 | 2.79 | 0.899 | 49.25 |
| t_nn | 14 | 4.726 | 4.447 | 0.941 | 30.9 |
| t_batch | 8 | 1.57 | 1.547 | 0.985 | 22.21 |
| t_batch | 14 | 2.233 | 2.215 | 0.992 | 15.51 |
| t_ikl | 8 | 1.671 | 1.666 | 0.997 | 20.62 |
| t_ikl | 14 | 2.354 | 2.361 | 1.003 | 14.55 |
| t_df | 8 | 1.74 | 1.514 | 0.871 | 45.37 |
| t_df | 14 | 2.667 | 2.43 | 0.911 | 28.27 |
| t_cc | 8 | 2.974 | 2.673 | 0.899 | 51.41 |
| t_cc | 14 | 4.616 | 4.3 | 0.932 | 31.96 |
| t_permout | 8 | 3.102 | 2.778 | 0.896 | 49.47 |
| t_permout | 14 | 4.726 | 4.43 | 0.937 | 31.02 |

- **Every case with k̂ > 1024 is faster** (after/before at N = 8 / 14): r_skinny 0.79 / 0.85
  (2.39 / 3.57 ms, 28.7 / 19.3 TFLOPS-eq), r_bigk 0.90 / 0.93, r_mm 0.96 / 0.99,
  sq2048–sq8192 0.82–0.97, sq16384 0.96, t_nn / t_df / t_cc / t_permout 0.87–0.94, s_cc16
  0.95. r_lowk, t_batch, t_ikl (k̂ ≤ 1024: KF, unchanged) within ±2 %.
- **Small shapes (KF path, code unchanged) only move by noise.** The event timings of
  20–60 µs calls swing between processes (the first sweep had s256 N = 8 at 0.019 → 0.044 ms;
  an interleaved rerun with 40 timed calls, `bench/results/d4-small-{before-a572d85,
  after-197cd0f}.jsonl`, had it 0.039 → 0.022 ms, i.e. the other way). nsys kernel times
  (per call, µs, 40 timed calls) are identical before and after:

| case | N | before (KF + K4 + K5) | after |
|---|---|---|---|
| s256 | 8 | 7.1 + 3.6 + 4.6 = 15.4 | 7.1 + 3.6 + 4.6 = 15.4 |
| s256 | 14 | 8.0 + 4.0 + 5.6 = 17.6 | 8.0 + 4.0 + 5.6 = 17.6 |
| s512 | 8 | 12.6 + 7.5 + 9.4 = 29.5 | 12.1 + 7.2 + 9.0 = 28.3 |
| s512 | 14 | 15.4 + 11.6 + 12.5 = 39.5 | 15.4 + 11.5 + 12.4 = 39.3 |
| s_cc8 | 8 | 13.1 + 7.8 + 9.4 = 30.2 | 13.1 + 7.8 + 9.4 = 30.3 |
| s_cc8 | 14 | 16.0 + 11.7 + 12.7 = 40.3 | 16.0 + 11.6 + 12.6 = 40.2 |
| s_df | 8 | 12.5 + 15.1 + 18.7 = 46.3 | 12.6 + 15.2 + 18.8 = 46.6 |
| s_df | 14 | 14.7 + 20.3 + 27.7 = 62.7 | 16.4 + 22.8 + 30.7 = 69.8 ¹ |

¹ every kernel of that process 8–12 % slower, K4 and K5 included (unchanged code): the
process's clock state.

## 6. Residue groups (g < N): what KL can and cannot save

- KL saves one DRAM read of the operand for every g (§3). Cutting the later groups' reads too
  would need all N residues of the operand at once, i.e. N RA (RB) slots: that is the g = N
  workspace term for the operand, and for the large operand it is exactly the memory g < N
  was chosen to save (e.g. r_skinny A at N = 14: 940 MB of residues at g = 14, 470 MB at
  g = 7). No smaller stand-in exists: the quantized a' (|μ| ≤ 2⁵³) takes as many bytes as the
  FP64 input, and the FP64 row is itself smaller than the remaining residues when N − g > 8.
- So the workspace formula (§4a) is unchanged, and `g` stays the caller's only
  memory/speed knob; KL makes small g relatively cheaper only by the one saved read.
- Unchanged structure for groups 2..: K2/K3 read σ/τ stored by KL (spec: `pos_large_g1`,
  `neg_large_no_sigma`).

## 7. Tests

`IMMERDINK_CUTLASS_DIR=... tools/test.sh`: 6 + 23 tests pass. Every emulated contract case
runs, per vec mode, with the plan-time rule, unfused, without the small variants, and (new)
with KL forced on both operands (fmode 2) and with KL tiles of 64 rows (fmode 3: rows of
k̂ > 64 span several chunks, so the register / shared-memory / re-read chunks all run on
small shapes); each run bit-exact vs `spec/oracle.py`, all C bit-identical, the workspace
stage-checked at g = N (KL's σ/τ, flags, RA/RB including padding), the launches of every call
counted by the LD_PRELOAD shim against the plan and `imd_lowering`. The production rule takes
KL in `contract_bigk` (k up to 262212, 150000-long rows, two and three K-chunks) and in the
k̂ > 1024 cases of e2e / replay. `graph_capture` captures and replays a KL shape (A along k̄,
two chunks; B along rows, ten chunks, zero rows) in both vec modes. Host queries:
`host_c_api`, `host_python`, `host_vs_plan` check `fused_large` and the v4 launch count.

## 8. Not done / next

- **Long rows are not at their bound** (r_bigk KL 66–69 %, r_mm 56–61 %): with nc > 2 the
  three passes run back to back per row in one 1024-thread CTA per SM, and the re-read chunks
  partly miss L2 when the residue writes evict them (r_bigk N = 14: 1206 MB read for a
  1074 MB minimum; r_mm 530–605 MB for 426 MB). A finer split (several CTAs per row with a
  grid-wide combine of γ and S) would need co-residency guarantees (cooperative launch) and a
  new visibility rule (the `neg_large_row_split` negative shows why a plain split is wrong); not
  attempted.
- **Few long rows** (r_mm B, 64 rows × 50000: 64 CTAs on 128 SMs): KL B 0.147 ms vs
  K1 B + K3 0.138 ms before, i.e. neutral; a per-operand rule could keep K1 + K3 there, but
  the difference is within noise and the rule stays "both operands".
- A one-launch front end for every shape (KF extended to multi-chunk rows, computing offsets
  from the descriptor) was considered: KL's structure (vec region reads after K0, per-operand
  launches) left the KF small-problem path untouched (identical kernel times, §5) and needed
  no descriptor decoding per element for non-affine K; it is not pursued here.
- KF and K2/K3 still use E = 16; the E = 8 result suggests trying it there (separate item).
