# Issue 21: pack and CRT on Hopper — what is done offline, and what R1 must tell us

Status: the offline part is done (profile on the RTX 4090, the L2 knobs, a bandwidth microbenchmark and its analysis). The Hopper limiter is
**not named yet**: that is what R1 is for. Written for a reader new to the project.

## The question

Around the INT8 multiply (K4) sit two kinds of memory-bound kernels. The **pack** (kernels KF / KL, K2, K3) reads the FP64 operand and writes N
narrow integer planes (one residue per modulus). The **CRT** (K5) reads the N planes of K4's result and writes the FP64 answer. At 4096³ with
N = 14 they take 1.2 ms of an H100 call (38%) and 61% at 1536³. On the H200, whose memory is 43% faster (datasheet), they take the same time within 1%,
so bandwidth is probably not what limits them there. Issue 14's arithmetic says K4 alone cannot reach parity at 4096³, so pack and CRT are on the
critical path.

## On the RTX 4090 they ARE bandwidth-bound (ncu, 2026-10-15)

`ncu --set full` on the first call of 4096³, N = 14 (`imd_bench`; ncu fixes the clock, so durations compare with each other):

| kernel | time | DRAM throughput | DRAM bytes | SM throughput | L2 hit |
|---|---|---|---|---|---|
| KL pack of A | 437 µs | 85% of peak (837 GB/s) | ~370 MB | 41% | 65% |
| KL pack of B | 437 µs | 85% | ~370 MB | 41% | 65% |
| K5 CRT | 429 µs | 87% (858 GB/s) | ~370 MB | 66% | 50% |
| K4 (for reference) | 3.19 ms | 23% | | 94% | 98% |

The bytes are exactly what the algorithm must move (operand 134 MB + 14 planes × 16.8 MB). So on the 4090 pack and CRT take 1.3 ms because the memory
system delivers 0.85–0.9 TB/s, and nothing in the kernels needs fixing there.

## What a Hopper number must be compared with

If the same kernels also ran at ~0.86 TB/s on an H100 (3.35 TB/s datasheet) they would use a quarter of the memory system: **the same time on three
cards with very different memories is not a coincidence of bandwidth**. The candidates are (a) the launch shape asks for too little data at once
(Little's law: bytes in flight per SM = bandwidth × latency / SMs; a kernel of 64 registers and 8 values per thread at the occupancy ncu reports may
not cover H100's latency), (b) a per-launch or per-tile fixed cost, (c) the instruction stream (SM throughput was 66% for the CRT already on the 4090),
(d) L2 effects (the K1 rows-in-flight budget below). The bandwidth microbenchmark separates (a) from the rest.

## `bench/membw/imd_membw.cu` (R1) and its analysis

Patterns shaped like the library's kernels, on buffers far larger than L2: `rd`, `wr`, `cp`, `fan` (read 16 B, write 2 B into each of 14 planes: the
pack) and `fanin` (read 2 B from each of 14 planes, write 16 B: the CRT), swept over blocks per SM {1..16}, threads {128..1024}, loads in flight per
thread {1..8}; plus `rd_small`, a read of L2/4 … 8 × L2 (the L2 cliff). One JSON line per shape with GB/s and % of the datasheet-derived peak
(memory clock × bus width × 2). `tools/build_membw.sh OUT run` builds it with one nvcc call.

`bench/membw/analyze.py MEMBW.jsonl BENCH.jsonl` takes the pack / CRT phase times of `imd_bench` records, converts them to bandwidth with the bytes the phase
must move, and compares with the best the card delivers for the pattern (or for a plain copy, if higher): **BANDWIDTH-BOUND** (≥ 85% of it; only fewer
bytes help), **SHAPE-BOUND** (< 85% and the card does ≥ 1.15× more: the kernel's launch shape leaves bandwidth unused), SMALL (< 32 MiB: latency, not judged).

On the RTX 4090 (full sweep, 68 shapes per pattern): rd 953 GB/s (94.6% of the 1008 GB/s peak), wr 923, cp 894, fan 819, fanin 885; the launch shape matters by
up to 20% (fanin 623–885). The library's phases at 4096³: KL A 867 GB/s, KL B 876, K5 922, i.e. **97–103% of the best of the microbenchmark: bandwidth-bound**.
L2-resident reads run at 3.0–3.6 TB/s up to 72 MiB and fall to 0.94 TB/s at 144 MiB.

## K1's L2 budgets are now derived from the card's L2

K1 (the scaling kernel) reads a row twice; pass 2 hits L2 only while the rows in flight fit. Two byte budgets decide how many rows it keeps in flight, and
both were constants for the 4090: 40 MiB (the grid cap of a two-pass kernel) and 64 MiB (the long-row variants; "the L2 less what the rest of the GPU keeps there").
They are now **5/9 and 8/9 of the device's L2** (`host/analysis.hpp` `k1_l2_fit_bytes`, `k1_l2_budget_bytes`; the plan reads the L2 size with the device class of issue 18):

| card | L2 | fit (was 40) | budget (was 64) |
|---|---|---|---|
| RTX 4090 | 72 MiB | 40.0 MiB | 64.0 MiB, exactly the old values (`test_l2_knobs`), so every 4090 launch is unchanged |
| H100 | 50 MiB | 27.8 | 44.4 |
| H200 | 60 MiB | 33.3 | 53.3 |
| RTX PRO 6000 | 128 MiB | 71.1 | 113.8 |

The change affects only the schedule of K1, never its result (S and γ are exact and order-independent), so output hashes are unchanged by construction. The 8/9 and 5/9
are the 4090's ratios, not fitted for the other cards: R2's grid tests them (below). The other launch settings are already derived from the SM count
(`kKfSmallBlocksPerSm = 16` per SM, K5's one-element-per-SM rule) or from the shape (KL's tile).

## The target X

"H100 pack plus CRT at sq4096, t_df, r_lowk and s_cc16 is at most X ms": X per case is the time left once K4 is as fast as the design reaches:
X = T_target − N·2mnk / R_K4, with T_target the time at which immerdink matches GEMMul8 fast (61.6 TFLOPS-eq at 4096³: 2.23 ms) and R_K4 the K4 rate the wgmma
decision of issue 16 projects (`bench/campaign/required_rate.py`). With the 1.5× faster K4 of the issue text this gives about 0.9 ms at 4096³ (1.2 ms now). It is written
here as a formula because R_K4 comes from issue 16's verdict; R2 fixes the number before its grid runs.

## R1 and R2

* **R1**: `membw.main` (the microbenchmark, about 1 minute per card) and the phase marks already recorded by every sweep. `analyze.py` names the limiter per card.
* **R2** (grid): the settings the analysis points at, on sq4096, sq8192, t_df, r_lowk and three small shapes; adopt a setting if the call time improves ≥ 3% on 2 of the 4
  named cases and none is more than 2% slower. If R1 does not name the limiter, a repair needs the spare run R4 (issue 14).
* The emulation-versus-native crossover per card for issue 18 is in `docs/SUPPORT.md`.

## Not done here

Everything on Hopper. The microbenchmark has run only on the RTX 4090. Whether Hopper's limiter is (a)–(d) is open until R1.
