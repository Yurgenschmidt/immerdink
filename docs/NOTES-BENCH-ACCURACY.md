# imd_bench: the accuracy a number of moduli N buys (fields `acc_*`)

Purpose: report the speed of `imd` at a given N (for example N = 8 at 4096^3 and 8192^3 on H100 / H200) together with the
accuracy that N gives, from the same record. Nothing here touches the library, the timed loops or the hot path.

## What existed before

Every timed record that carries an accuracy (methods imd, dgemm, emu, gemmul8, cutensor, duel-a/b, xduel-imd, energy) already had
`max_relerr`, `max_normerr`, `err_samples`, `nonfinite` (and `c_hash` for imd), from `check()` in `bench/imd_bench.cu`:
the reference is a double-double (Dot2) dot product computed on the device at `--samples` (default 256) randomly chosen
output entries (all of them if C is smaller), the host holds only the samples.
`max_relerr` = max |c - r| / |r| over the samples (blows up where r is near zero); `max_normerr` = max |c - r| / (||a_i||_2 ||b_j||_2)
(DESIGN section 3.8). The imd accuracy is read after the timed loop, from the C of the last timed call. duel checks both sides
after the timed rep loop, xduel the immerdink side in a fresh call into a poisoned C. No native-DGEMM comparison existed.

## What was added

1. `max_cwerr` (all methods that carry an accuracy): the componentwise error max |c - r| / (|A||B|)_ij over the samples, where
   (|A||B|)_ij = sum_k |a_ik b_kj| is accumulated in the same device reference kernel. This is the standard GEMM accuracy
   measure (a correctly rounded DGEMM has about k*u/2 worst case, in practice about 1e-16..4e-16).
2. Method `imd` records get the `acc_*` fields below, computed in `Bench::accuracy_fields` after the timed loop and the profiled
   calls, immediately before the record is written. Untimed.
3. `--accuracy 0|1`: switches the full-matrix DGEMM comparison (below). Default: on when its extra device buffers fit.

## Field definitions (method imd)

Sampled, always present (reference = double-double Dot2 at `acc_n_sampled` entries, `--samples`, default 256; use 4096 for
the large-square runs; the reference is ~106 bits, so it is exact for this purpose):

| field | meaning |
|---|---|
| `acc_ref` | `dd-dot2-sampled` |
| `acc_n_sampled` | number of sampled entries |
| `acc_max_relerr_cw` | max over samples of \|c - r\| / (\|A\|\|B\|)_ij (componentwise). Same value as `max_cwerr`. |
| `acc_bits_cw` | -log2(acc_max_relerr_cw): bits of componentwise accuracy |
| `acc_max_relerr_elem` | max over samples of \|c - r\| / \|r\| (same as `max_relerr`; dominated by entries with cancellation, so large at small N) |
| `acc_bits_elem` | -log2 of the above |
| `acc_max_rel_to_norm` | max \|c - r\| / (\|\|a_i\|\|_2 \|\|b_j\|\|_2) (normwise; same value as `max_normerr`) |

Full-matrix comparison with native cuBLAS DGEMM (`CUBLAS_COMPUTE_64F_PEDANTIC`, same inputs), present when `acc_dgemm` is
`cublas-dgemm-pedantic`:

| field | meaning |
|---|---|
| `acc_dgemm` | `cublas-dgemm-pedantic`, or why it was not done: `off (--accuracy 0)`, `skipped: case needs permutations ...`, `skipped: extra buffers N MiB do not fit ...` |
| `acc_dgemm_max_relerr_cw` | native DGEMM against the same double-double reference at the same samples: the error of the FP64 reference itself, to be read next to `acc_max_relerr_cw` |
| `acc_dgemm_max_relerr_elem` | the same, relative to \|r\| |
| `acc_vs_dgemm_max_cw` | max over ALL entries of \|c_imd - c_dgemm\| / (\|A\|\|B\|)_ij, with (\|A\|\|B\|) from a DGEMM on \|A\|, \|B\| |
| `acc_vs_dgemm_mean_cw` | the mean of the same over all entries |
| `acc_vs_dgemm_n` / `acc_vs_dgemm_nonfinite` | entries compared (entries with a zero scale are skipped) / non-finite entries of the emulation |

Reading them: `acc_max_relerr_cw` is the emulation's error against the truth, at the sampled entries. `acc_vs_dgemm_max_cw` is a bound
on the whole matrix (the max over about m*n entries, against DGEMM's own error, which is floor about 3e-16..9e-16 over millions of
entries); once N is large (N >= 14) it saturates at DGEMM's own error, which is the sign that the emulation is as accurate as
native FP64 there. Below that it tracks `acc_max_relerr_cw` times a small factor (the max over more entries).

Memory of the comparison: A + B + 2C bytes extra (|A|, |B|, C_dgemm, |A||B|), 2.1 GB at 8192^3. Auto mode (no `--accuracy`)
runs it when that is at most min(free device memory / 2, 8 GiB); `--accuracy 1` runs it whenever it fits in free memory minus
256 MiB; otherwise the field `acc_dgemm` says `skipped: ...` and the sampled fields remain. Cases that need cuTENSOR permutations
to reach a GEMM view skip it. The comparison costs two DGEMMs (about 1 s each at 8192^3 on an RTX 4090), outside the timed region.

## N range and sweeps

The library accepts N = 2..20 (`kNMin`, `kNMax` in `include/immerdink/detail/constants.hpp`); `--N 6,7,8,9,10,11,14,18` emits one
record per N, each with its own `acc_*` fields. On the RTX 4090 every N in 2..20 ran at 2048^3 and 4096^3 (below). The Hopper
wgmma K4 backend and set 13 are not exercised by an sm_89 run; on H100 / H200 check `k4b_run` / `k4_wgmma_served` in the same
records.

## Sample values, RTX 4090, 4096^3, `--samples 256`, one run (not a timing claim: the machine was loaded)

`acc_bits_cw` is -log2 of the componentwise error. `acc_vs_dgemm_max_cw` saturates near DGEMM's own error from N = 14.

| N | acc_max_relerr_cw | bits | acc_max_rel_to_norm | acc_vs_dgemm_max_cw |
|---|---|---|---|---|
| 6 | 3.3e-07 | 21.5 | 1.9e-07 | 6.1e-07 |
| 7 | 2.4e-08 | 25.3 | 1.4e-08 | 4.0e-08 |
| 8 | 1.4e-09 | 29.4 | 8.2e-10 | 2.4e-09 |
| 9 | 8.6e-11 | 33.4 | 5.0e-11 | 1.6e-10 |
| 10 | 5.2e-12 | 37.5 | 3.0e-12 | 9.4e-12 |
| 11 | 3.4e-13 | 41.4 | 2.0e-13 | 5.8e-13 |
| 12 | 2.0e-14 | 45.5 | 1.1e-14 | 3.8e-14 |
| 14 | 1.0e-16 | 53.1 | 5.9e-17 | 9.0e-16 |
| 16 | 5.0e-18 | 57.5 | 3.0e-18 | 9.0e-16 |

(native DGEMM against the reference at the same samples: 2.5e-16 componentwise.) 8192^3 at N = 8, 4096 samples: 1.2e-09
componentwise, 1.8e-09 against DGEMM over all entries; N = 14: 1.5e-16 and 8.6e-16.

## Test

`bench/test_accuracy_fields.py --bench build/bench/imd_bench` (ctest `bench_accuracy_fields`, registered when
`IMMERDINK_BENCH_BIN` is set, like `bench_seq_crt`): a 256^3 case at N = 6, 8, 11, 14: the fields exist, the componentwise error decreases
strictly with N, the sampled and full-matrix figures agree, `--accuracy 0` drops the DGEMM fields and leaves C (c_hash) unchanged.
