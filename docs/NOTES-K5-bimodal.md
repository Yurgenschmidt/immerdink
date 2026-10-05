# K5 "bimodal" times: GPU time-slice preemption by the desktop, not a kernel effect

Symptom (reported on dev 0613e49 and by the D1 agent): the `K5 crt` phase of sq4096 N = 14
(`ik,jk->ij`, i = j = k = 4096) reads ~0.43 ms in some benchmark processes and 0.8–0.97 ms in
others, with K4 at 3.1–3.4 ms and similar SM clocks; sq8192 K5 2.24 ms against ~1.8 ms
earlier. Investigated 2026-10-03 on dev 0613e49 and re-checked on e843a8e (D1 merge; K5 code
unchanged since b2c45cd).

**Result.** K5 itself is not bimodal. The extra ~0.4–0.5 ms is a GPU **context switch**:
when another GPU context has work pending, the GPU time-slices between contexts (slice
≈ 2.2 ms here) and preempts ours. On this box the other contexts are the desktop's graphics
clients — the Claude desktop app's renderer (`claude-desktop --type=gpu-process`) and
`gnome-shell`. One switch costs our kernel ~0.4–0.5 ms (the other contexts run only
~45–60 µs). Because the preemptions are phase-locked to the start of each `imd_contract` call,
they keep landing in the same phase, which makes them look like a property of that phase. With
the D4 KL path (48f2e6b on), the second preemption of a call falls in K5. With the older
K1 + K2 + K3 path (b2c45cd, b3e9b47) it falls in K4. **No kernel change is warranted.** The
fixes are in how and when the benchmarks run (§6).

Tooling: every timing through `tools/run_bench.sh` / `bench/run_sweep.sh` (quiet gate, memcap,
GPU lock). nsys 2026.5.1 with `--gpuctxsw=true` (GPU context-switch trace) and ncu 2026.3.1
(counters, `--cache-control none --clock-control none`) were run under `flock` on the GPU
lock. The per-call probe is `bench/k5bimodal/k5probe.cu`. The trace analysis is
`bench/k5bimodal/ctxstat.py` (per kernel: clean vs. switched-out calls) and `ctxphase.py`
(switch-out offsets within each call). Data: `bench/results/k5bimodal-*.jsonl`.

## 1. Distribution (task 1)

**Ten gated processes on dev 0613e49** (`bench/run_sweep.sh`, another library's calls before
imd's as in the comparison sweeps, warm-up 3 s), in `bench/results/k5bimodal-dev-0613e49.jsonl`. The commit field reads `0613e49-dirty`
because a temporary stderr print of the buffer addresses was in `imd_bench.cu`; it has been
removed and is not committed.

| run | K5 phase (median of 5) | K4 | call median / min (timed loop) | spread | SM clock after | gate GPU util mean/peak |
|---|---|---|---|---|---|---|
| 1 | 0.895 | 3.317 | 4.977 / 4.310 | 18.3 % | 2805 | 8.4 / 13 % |
| 2 | 0.835 | 3.335 | 4.860 / 4.300 | 17.7 % | 2805 | 9.7 / 13 % |
| 3 | 0.952 | 3.118 | 5.090 / 4.312 | 22.7 % | 2685 | 8.2 / 9 % |
| 4 | 0.882 | 3.135 | 4.944 / 4.306 | 27.1 % | 2715 | 7.8 / 10 % |
| 5 | 0.863 | 3.139 | 4.878 / 4.303 | 26.2 % | 2640 | 8.8 / 12 % |
| 6 | 0.806 | 3.130 | 5.135 / 4.311 | 19.5 % | 2805 | 7.7 / 8 % |
| 7 | 0.847 | 3.147 | 5.209 / 4.401 | 19.4 % | 2805 | 7.3 / 8 % |
| 8 | 0.828 | 3.153 | 4.972 / 4.410 | 17.1 % | 2805 | 8.0 / 8 % |
| 9 | 0.864 | 3.425 | 5.033 / 4.315 | 20.0 % | 2805 | 8.0 / 9 % |
| 10 | 0.785 | 3.387 | 5.083 / 4.315 | 19.4 % | 2805 | 7.9 / 9 % |

All ten processes were "slow" (all times in ms). The timed loop's **minimum** was
4.30–4.41 ms in every process, the same as the clean 48f2e6b sample (4.286 ms). So every
process had clean calls, and no code path got slower.

**Within one process the time is per call, not per process.** `k5probe` (gated, 15 calls per
buffer combination, two workspaces × two C buffers, two processes) gave the following K5
sequences in ms:

```
ws0 c0: 0.921 1.025 0.979 0.446 0.446 0.445 0.956 0.822 0.506 0.449 0.446 0.938 1.087 1.102 0.593
ws0 c1: 0.447 0.450 0.447 1.066 1.077 0.446 0.451 0.447 1.057 1.025 0.524 0.447 0.447 0.447 1.067
ws1 c0: 0.552 0.451 0.425 1.046 0.980 0.595 0.424 0.428 1.015 1.126 0.433 0.433 0.434 1.007 0.995
```

The same buffers (identical C, W, scatter-vector addresses) produce both modes on consecutive
calls. The phase record's "mode" is whatever the median of 5 calls happens to be.

**Addresses.** The virtual layout is identical in every process (cudaMalloc is
deterministic): W = ws + 469 983 488 (W mod 2 MiB = 0x36100, the vec region pushes RA, RB and W
off 2 MiB alignment), C mod 2 MiB = 0, and W − C = −0x0FFC9F00 in both the bench and the probe.
Physical placement does not matter either, because fast and slow calls alternate on the same
allocations. **Not correlated** with address alignment or offset, the clock (2640–2820 MHz,
both modes at each), power, the warm-up, or the L2 state left by K4.

## 2. Root cause: time-slice preemption (nsys `--gpuctxsw`)

nsys records every GPU context save and restore on the device, including other processes'.
Over 18 profiled calls of `k5probe`, **every** slow K5 overlaps a switch-out of our context and
every fast one does not (`ctxstat.py`):

```
k5_crt      calls 18  clean  5 median 443.1 us
    switched out 1 x: 13 calls  median dur 833.6 us (+390.4), off-GPU gap 138.9 us, residual 251.6 us
k4_kernel   calls 18  clean  7 median 3164.4 us
    switched out 1 x: 10 calls  median dur 3518.1 us (+353.7), off-GPU gap 241.3 us, residual 112.4 us
    switched out 2 x:  1 calls  median dur 3802.2 us (+637.8), off-GPU gap 431.1 us, residual 206.7 us
kl_scale_pack calls 36 clean 36 median 432.1 us
```

The same analysis of the real bench (`imd_bench --methods imd --reps 20 --phase-reps 10` under
nsys, `bench/results/k5bimodal-nsys-ctxsw.jsonl`):

| build | K5 clean / switched | K5 clean → switched (µs) | K4 clean / switched | K4 clean → switched (µs) |
|---|---|---|---|---|
| dev 0613e49 | 8 / 24 | 430.5 → 912.7 | 14 / 18 | 3116 → 3410 |
| e843a8e (D1) | 10 / 22 | 441.0 → 942.0 | 17 / 15 | 3148 → 3522 |
| b2c45cd (A1) | 32 / 0 | 442.9 | 3 / 29 (17 of them twice) | 3060 → 3524 / 3767 |

A typical slow K5 (offsets from the kernel start):

```
+222.2 us  SAVE_END       ./k5probe          (our context preempted)
+269.9 us  RESTORE_START  claude-desktop
+314.4 us  SAVE_END       claude-desktop     (it ran ~45 us)
+360.1 us  RESTORE_START  ./k5probe
+831.0 us  kernel end                        (clean: 443 us)
```

With two foreign contexts (gnome-shell, then claude-desktop) the gap is ~260 µs and K5 takes
0.91–0.98 ms. That explains the spread of the slow mode: 0.80–0.84 ms with one foreign
context, 0.91–1.10 ms with two, and up to 1.3 ms with two switches.

**Why it lands in K5 (phase locking).** Between calls the host synchronizes, so our context
idles and the desktop's work runs without a preemption. Each call then starts a fresh time
slice. The switch-outs within a call (`ctxphase.py`, dev bench) fall at +2.18 ms and then at
+4.33…4.7 ms after the call's first kernel. Slices are 2.18–2.27 ms long, or about twice that
when the desktop had nothing pending at the first expiry. On the KL path, K5 runs from about
+4.35 ms to +4.8 ms, right on the second expiry. The K1 + K2 + K3 path (before D4) spends
~0.5 ms longer before K4, so both expiries fall inside its longer K4 and K5 escapes. That is
the whole "bisect" signal (§3).

**sq8192.** In the probe trace K5 measured 1765 µs clean and 2254 µs switched once (11 of 15
calls). That matches the reported 1.8 vs 2.24 ms: a 1.8 ms kernel almost always contains one
expiry of a 2.2 ms slice. K4 at sq8192 (~27 ms) was switched out 8–12 times in every call
(~1.5–2 ms off the GPU per call, plus residuals): the desktop costs the large shapes ~7–9 %
end to end.

**Residual.** Beyond the off-GPU gap (our SAVE_END to our RESTORE_START), a preempted kernel
loses extra time. That is ~250–320 µs for K5 and KL, and ~65–200 µs for K4. The trace has no
preemption-request timestamps, so the split between "before SAVE_END" (draining in-flight
loads, saving state) and "after RESTORE_START" (restoring state, refilling the memory pipeline)
is unknown. Hypothesis, untested: the save and restore cost grows with the resident state.
k5_crt<4,4> is register-limited at full occupancy (62 registers, 4 CTAs × 256 threads per SM:
a full 256 KB register file per SM, ~32 MiB across the GPU), and it streams memory with up
to 14 outstanding loads per thread.

## 3. Bisect (task 2)

Six gated `bench/run_sweep.sh` processes per commit, each built in a temporary worktree (since
removed), at the same session and conditions (desktop active, gate GPU utilization 8–12 %).
Data in `bench/results/k5bimodal-bisect.jsonl`.

| commit | path before K4 | K5 phase, 6 processes (ms) | K4 phase (ms) | call min (ms) | spread |
|---|---|---|---|---|---|
| b2c45cd (A1+A4) | K1, K2, K3 | 0.435–0.443 (6 fast) | 3.53–3.73 | 4.63–5.03 | 8.7–16.3 % |
| b3e9b47 (A3) | K1, K2, K3 | 0.436–0.446 (6 fast) | 3.52–3.80 | 4.63–5.01 | 13.5–17.9 % |
| 48f2e6b (D4 part 1) | KL | 0.80–0.96 (5 slow), 0.436 (1) | 3.10–3.44 | 4.33–4.70 | 15.1–23.1 % |
| 0613e49 (dev, §1) | KL | 0.785–0.952 (10 slow) | 3.12–3.43 | 4.30–4.41 | 17.1–27.1 % |

The K5 phase "regresses" at D4, but `k5_crt` is byte-identical from b2c45cd to e843a8e, and
`src/` is identical between 48f2e6b and 0613e49. What D4 changed is the call's timeline: KL
made the pre-K4 phases ~0.5 ms shorter, which moved the second slice expiry from K4 into K5.
On the older commits the same contamination appears as an inflated K4 (3.5–3.8 ms against a
clean 3.04 ms) and a higher call minimum, because every call there is preempted at least
once. The older commits are not faster; their K5 is just not where the preemption falls.

**Why the earlier sweeps were clean.** `bench/results/compare-b2c45cd`, `-b3e9b47` and
`-48f2e6b` each recorded K5 0.425–0.433 ms, K4 3.02–3.07 ms, a timed-loop spread of 0.4–2.3 %,
and a quiet-gate log of **GPU utilization 0.0 % (peak 0 %)**. Every run today had 7–12 %
utilization at the gate (the desktop drawing at idle clocks) and a spread of 8.7–27 %. Every
fast-K5 record in `bench/results` from 2026-10-03 (A1, A2, A3, D4) has a spread ≤ 2.3 %; the
two older records with gate utilization > 0 (`a2-before-c63117a`, 7.4 %; `compare-1e99b65`,
30.4 %) have spreads of 9.2 % and 12.7 % and inflated K4. The gate passed today's runs because
since fddc944 it gates on *clock-weighted* GPU load (< 5 %), which a desktop drawing at P8
clocks stays under. That was deliberate (desktop compositing had blocked runs), but a context
that submits any work at all forces time-slice preemption of the benchmark, however little
work it does.

## 4. ncu comparison (task 3)

`ncu --set full -k regex:k5_crt --launch-skip 3 --launch-count 8 --cache-control none
--clock-control none` on `k5probe`. Under ncu's serialized replay the launches took
0.445–0.608 ms rather than 0.44 / 0.85. The counters describe the same work in every launch:

| metric | fastest launch (0.445 ms) | slowest launches (0.59–0.61 ms) |
|---|---|---|
| dram__bytes_read.sum | 244.1 MB (others 235.0 MB) | 235.0 MB |
| dram__bytes_write.sum | 132.8 MB | 134.8–137.2 MB |
| DRAM bytes read per FBPA, max / min | 1.0005 | 1.0005 |
| lts__t_sector_hit_rate | 49.5 % (reads 30.9 %, writes 100 %) | 50.5–50.7 % (reads 31.1 %, writes 100 %) |
| global load bytes per sector | 22.35 of 32 | 22.35 of 32 |
| global store bytes per sector | 32 of 32 | 32 of 32 |
| L2 slice sectors, max / min | 1.46 | 1.46 |
| stall reasons per issue (math pipe throttle, not selected, wait, dispatch) | 2.19, 2.62, 1.66, 0.67 | 2.18–2.23, 2.57–2.63, 1.66, 0.68 |
| long-scoreboard stall per issue | 2.51 | 4.07–4.81 |
| sm__throughput % of peak | 73.1 | 50.7–52.6 |

- **DRAM:** 224 MiB of W planes read once. C is 134 MB of writes, a few MB of which stay in
  L2. Traffic is balanced across the 12 memory partitions (no partition camping).
- **Sector efficiency:** stores are fully coalesced. The 70 % load efficiency comes from the
  small c_col / sB / fB gathers, which hit L1/L2.
- **L2:** the slice imbalance (max/min 1.46) is the same in every launch, so it is a constant
  cost, not the bimodality.
- **Elapsed time:** the longer launches did the same work with fewer instructions per cycle
  and more long-scoreboard wait. That fits the preemption cost, but ncu cannot attribute it.
  nsys `--gpuctxsw` (§2) is the tool that can.

## 5. Is there a kernel-level fix? (task 4)

No. K5 runs at 0.430–0.443 ms clean, 83–85 % of its DRAM lower bound of 0.366 ms. That is
unchanged since b2c45cd and independent of addresses, alignment and tile order. Changes to
store hints, the plane walk order or the W alignment would not touch a time-slice preemption.
§4a W-region alignment is not implicated, so no design change is proposed. The only
kernel-side lever is the preemption residual (§2: ~0.25–0.3 ms per switch for K5, under the
untested resident-state hypothesis). It only exists while another context is active, so it is
not worth the cost (a lower occupancy for K5) for the library.

## 6. Proposal (benchmark methodology and environment; not implemented — outside kernel scope)

1. **Benchmark with the desktop quiet on this GPU.** The Claude desktop window renders
   continuously while an agent session streams output. Minimize or cover it during sweeps
   (Chromium stops drawing hidden windows), or move the display to another GPU. gnome-shell
   draws on any screen change. A longer compute timeslice
   (`nvidia-smi compute-policy --set-timeslice`, if this GeForce driver supports it) is a system
   setting for the user to decide; untested here.
2. **Gate on activity, not load.** tools/quiet_check.sh already measures raw
   `utilization.gpu`. Any non-zero raw utilization while no benchmark is running means another
   context submits work and will preempt the benchmark. Gating on raw mean ≤ ~1 % (or waiting
   for it) would have rejected every contaminated run in §1/§3 and accepted every clean
   historic one (0.0 %). If that blocks too often, record it and flag the result instead
   (item 3).
3. **Detect and report contamination in the bench.** The timed loop's `spread_pct` separates
   the two cleanly in all data here: ≤ 2.3 % clean, ≥ 8.7 % contaminated. Proposed:
   - `bench/summarize.py` flags records with spread > 5 %;
   - the phase record carries each phase's **min** over the phase reps next to the median
     (the min is the clean value whenever at least one rep was clean; with 5 reps and ~70 %
     contamination, the median is not);
   - a few more phase reps (10).
4. **For investigations,** `nsys profile --gpuctxsw=true` plus `bench/k5bimodal/ctxstat.py`
   shows directly which kernels were preempted and by whom.

## 7. Before / after

No code changed. The clean values are: sq4096 N = 14 K5 0.430–0.443 ms, K4 3.04–3.16 ms,
call 4.30 ms (KL path). Contaminated values are: K5 +0.39–0.50 ms per preemption, K4 +0.29–0.46
ms per preemption, and the call median +0.6–0.9 ms. BENCHMARKS / ROADMAP numbers taken while
the desktop was drawing (gate GPU utilization > 0, spread > 5 %) should be read with this in
mind. The D1 phase records (`bench/results/d1-phases-73b62ee.jsonl`) were taken with gate GPU
utilization 7.5–8.0 % and timed-loop spreads of 1–22 %, so they are affected too.
