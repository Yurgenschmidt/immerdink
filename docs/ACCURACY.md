# Accuracy on realistic data (M5)

A sanity check of the DESIGN §3.8 error model on contraction-shaped workloads with
data chosen to stress the scaling. It is not a coupled-cluster implementation: the
shapes come from [CC_NOTES.md](CC_NOTES.md) and are only used as realistic fixtures.

**Result in one line.** No element anywhere exceeded the §3.8 bound: 520 runs, 1.66 M
elements checked against an exact reference, and 1.13 G elements checked against FP64.
Normwise, the bound is 5–100× pessimistic until the final rounding dominates.
**Elementwise** accuracy is the real limit: it depends on `‖a‖‖b‖ / |x|`, and data whose
magnitudes are anti-correlated along the contracted index need up to 6 more moduli than
benign data.

Library commit: `8359856` (dev tip after M4). Data:
[bench/accuracy/results/](../bench/accuracy/results/).

## 1. Method

### Fixtures

| name | einsum | extents | GEMM view (m, n, k) |
|---|---|---|---|
| `df_acQ` | `acQ,bdQ->acbd` (DF integral build) | a=b=c=d=24, Q=600 | (576, 576, 600) |
| `df_ijdkQ` | `ijdkQ,ldQ->klij` | i=j=k=l=8, d=24, Q=150 | (512, 8, 3600) |
| `ring_kjbc` | `kjbc,kiac->abij` (ring/CCD) | i=j=k=12, a=b=c=48 | (576, 576, 576) |
| `T_bdck` | `bdck,adij->abcijk` ((T)-style, 6-index permuted output) | a=b=c=d=40, i=j=k=5 | (8000, 1000, 40) |

### Data

Each distribution is generated on the GEMM-view matrices `A (m×k)` and `B (k×n)`. These
are then folded back into the fixture's tensor layouts, so the library sees permuted,
interleaved operands.

| data | construction | median `|x| / (‖a‖‖b‖)` |
|---|---|---|
| `uniform` | U(−1, 1) | 1e−2 – 1e−1 |
| `decay_same` | random sign · U(½, 1) · 10^(−1−9h/(k−1)), so magnitudes go from 1e−1 to 1e−10 along the contracted index h (the outermost K label varies slowest), times 3 decades along rows/columns; A and B decay in the same direction | 5e−2 – 7e−1 |
| `decay_opp` | the same, but B decays in the opposite direction along h. Orbital-energy-like: large integrals meet small amplitudes, so `Σ|a_h b_h| ≪ ‖a‖‖b‖` | 5e−10 – 3e−9 |
| `outliers` | N(0, 1), with 2 entries per row of A and 2 per column of B scaled by 10^U(3,8) | 1e−6 – 2e−6 |
| `cancel` | U(−1, 1); `a_{h+k/2} = a_h(1 + 1e−6·U(−1,1))`, `b_{h+k/2} = −b_h`, so pairs cancel to 1e−6 relative | 5e−9 – 4e−8 |

### Checks per run (N = 8…20, RN and TRUNC, α = 1, β = 0)

- **Exact grid.** A random 64 × 64 grid of GEMM rows × columns (64 × 8 for `df_ijdkQ`).
  - Reference: every product is split exactly (Dekker TwoProd), then `math.fsum` gives
    the correctly rounded `hi` and the remainder `lo`, so `|x − (hi + lo)| ≤ 2⁻⁵³|lo|`.
    This was validated against `oracle.exact_dot` (rational arithmetic).
  - The bound is §3.8 evaluated per element with the **exact** shifts σ_i and τ_j. These
    come from the §3.2 integer rule, and on random rows they match `oracle.scale_vector`
    bit for bit.
  - Reported:
    - normwise error `|D − x| / (‖a_i‖‖b_j‖)`;
    - elementwise error `|D − x| / |x|`;
    - observed/bound, max and median.
- **Full output.** Every element must satisfy `|D − D_fp64| ≤ bound + γ_k Σ|a||b|`.
  This is a necessary condition for the bound, where D_fp64 is a numpy/OpenBLAS FP64
  matmul.
- **Native FP64.** The numpy result on the same grids, for scale.

Run it:

```sh
tools/memcap.sh python3 bench/accuracy/contractions.py --out bench/accuracy/results/contractions.jsonl  # 27 s, 0.8 GB RSS
python3 bench/accuracy/summarize.py > bench/accuracy/results/summary.md
tools/memcap.sh python3 bench/accuracy/iterate.py --out bench/accuracy/results/iterate.jsonl          # 17 s
```

A small version runs as the ctest `accuracy_bound_quick` (`contractions.py --quick`,
about 1 s). It fails on any bound violation.

## 2. Results: single contractions

### Observed / bound, max over the 4 fixtures

| data | rnd | N=8 | N=9 | N=10 | N=11 | N=12 | N=13 | N=14 | N=15 | N=16 | N=17 | N=18 | N=19 | N=20 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| uniform | RN | 0.18 | 0.15 | 0.18 | 0.21 | 0.19 | 0.20 | 0.07 | 0.51 | 0.89 | 0.95 | 0.98 | 0.99 | 0.99 |
| uniform | TRUNC | 0.23 | 0.19 | 0.20 | 0.23 | 0.24 | 0.22 | 0.06 | 0.35 | 0.84 | 0.94 | 0.97 | 0.99 | 0.99 |
| decay_same | RN | 0.15 | 0.19 | 0.16 | 0.16 | 0.16 | 0.17 | 0.20 | 0.72 | 0.94 | 0.97 | 0.98 | 0.98 | 0.98 |
| decay_same | TRUNC | 0.20 | 0.20 | 0.22 | 0.20 | 0.18 | 0.17 | 0.12 | 0.58 | 0.91 | 0.97 | 0.98 | 0.98 | 0.98 |
| decay_opp | RN | 0.18 | 0.16 | 0.19 | 0.16 | 0.16 | 0.20 | 0.18 | 0.16 | 0.18 | 0.16 | 0.15 | 0.21 | 0.22 |
| decay_opp | TRUNC | 0.17 | 0.15 | 0.15 | 0.17 | 0.16 | 0.16 | 0.17 | 0.18 | 0.18 | 0.17 | 0.16 | 0.17 | 0.17 |
| outliers | RN | 0.14 | 0.13 | 0.13 | 0.14 | 0.17 | 0.16 | 0.17 | 0.70 | 0.89 | 0.96 | 0.97 | 0.97 | 0.97 |
| outliers | TRUNC | 0.14 | 0.14 | 0.14 | 0.14 | 0.14 | 0.15 | 0.16 | 0.57 | 0.87 | 0.94 | 0.97 | 0.97 | 0.97 |
| cancel | RN | 0.14 | 0.11 | 0.10 | 0.16 | 0.20 | 0.15 | 0.10 | 0.05 | 0.02 | 0.00 | 0.00 | 0.02 | 0.19 |
| cancel | TRUNC | 0.06 | 0.05 | 0.06 | 0.07 | 0.08 | 0.07 | 0.07 | 0.03 | 0.01 | 0.00 | 0.00 | 0.01 | 0.11 |

Two regimes:
- **N ≤ 14: the emulation error dominates.** The max ratio is about 0.15–0.2, and it
  comes from `T_bdck` (k = 40). For k = 576–3600 it is only 0.01–0.06, and the median
  ratio is about 1e−2.
  - The observed normwise error is almost independent of k: 7–9e−10 at N = 8 for both
    k = 40 and k = 3600. The bound's leading term `2√k/T` grows with √k.
  - The Cauchy–Schwarz step (ℓ1 → ℓ2) costs √k in the worst case, while the
    quantization errors add like a random walk. Normwise, the observed max error is
    2–3 · ‖a‖‖b‖/T at every N ≤ 14 and every k here.
- **N ≥ 15–16: the final rounding dominates.** The ratio climbs to 0.9–0.99, because the
  emulated integer result is exact and the only error left is the single final rounding,
  which attains the `u·|D|` term.
  - This is the **tight** part of the bound, not a near-violation. Emulated results there
    are correctly rounded.
  - The exceptions are `decay_opp` and `cancel`: there `|x| ≪ ‖a‖‖b‖`, so `u|D|` stays
    negligible.

### Max normwise error |D − x| / (‖a‖‖b‖), over the 4 fixtures

| data | rnd | N=8 | N=9 | N=10 | N=11 | N=12 | N=13 | N=14 | N=15 | N=16 | N=17 | N=18 | N=19 | N=20 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| uniform | RN | 9e-10 | 5e-11 | 4e-12 | 3e-13 | 2e-14 | 2e-15 | 1e-16 | 4e-17 | 4e-17 | 4e-17 | 4e-17 | 4e-17 | 4e-17 |
| uniform | TRUNC | 2e-09 | 1e-10 | 9e-12 | 7e-13 | 5e-14 | 4e-15 | 2e-16 | 4e-17 | 4e-17 | 4e-17 | 4e-17 | 4e-17 | 4e-17 |
| decay_same | RN | 9e-10 | 6e-11 | 4e-12 | 3e-13 | 2e-14 | 1e-15 | 1e-16 | 1e-16 | 1e-16 | 1e-16 | 1e-16 | 1e-16 | 1e-16 |
| decay_same | TRUNC | 3e-09 | 2e-10 | 1e-11 | 7e-13 | 5e-14 | 3e-15 | 2e-16 | 1e-16 | 1e-16 | 1e-16 | 1e-16 | 1e-16 | 1e-16 |
| decay_opp | RN | 1e-09 | 6e-11 | 4e-12 | 3e-13 | 2e-14 | 1e-15 | 9e-17 | 7e-18 | 5e-19 | 3e-20 | 3e-21 | 2e-22 | 2e-23 |
| decay_opp | TRUNC | 2e-09 | 1e-10 | 8e-12 | 5e-13 | 4e-14 | 2e-15 | 2e-16 | 1e-17 | 1e-18 | 7e-20 | 6e-21 | 5e-22 | 3e-23 |
| outliers | RN | 7e-10 | 5e-11 | 3e-12 | 2e-13 | 2e-14 | 1e-15 | 1e-16 | 1e-16 | 1e-16 | 1e-16 | 1e-16 | 1e-16 | 1e-16 |
| outliers | TRUNC | 2e-09 | 9e-11 | 7e-12 | 4e-13 | 3e-14 | 2e-15 | 2e-16 | 1e-16 | 1e-16 | 1e-16 | 1e-16 | 1e-16 | 1e-16 |
| cancel | RN | 7e-10 | 4e-11 | 3e-12 | 2e-13 | 2e-14 | 1e-15 | 7e-17 | 2e-18 | 4e-20 | 1e-21 | 7e-23 | 2e-23 | 2e-23 |
| cancel | TRUNC | 6e-10 | 4e-11 | 3e-12 | 2e-13 | 2e-14 | 1e-15 | 7e-17 | 3e-18 | 5e-20 | 2e-21 | 7e-23 | 2e-23 | 2e-23 |
| *§3.8 RN term 1, k=3600* | | *5e-08* | *3e-09* | *2e-10* | *1e-11* | *9e-13* | *6e-14* | *4e-15* | *3e-16* | *2e-17* | *2e-18* | *1e-19* | *9e-21* | *7e-22* |

- Each modulus buys about 15× (log₂T grows by about 3.9 bits per modulus).
- Normwise, every distribution behaves the same. The scaling is per row and per column,
  so decay, outliers and cancellation do not hurt the **normwise** error.
- TRUNC is 1.5–3× worse than RN, about ¼ modulus. That matches DESIGN §3.3 for
  zero-mean data; all of this data has random signs.

### Max elementwise error |D − x| / |x|, over the 4 fixtures

| data | rnd | N=8 | N=9 | N=10 | N=11 | N=12 | N=13 | N=14 | N=15 | N=16 | N=17 | N=18 | N=19 | N=20 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| uniform | RN | 3e-05 | 2e-06 | 3e-07 | 2e-08 | 2e-09 | 1e-10 | 2e-12 | 1e-16 | 1e-16 | 1e-16 | 1e-16 | 1e-16 | 1e-16 |
| uniform | TRUNC | 7e-05 | 2e-06 | 3e-07 | 3e-08 | 3e-09 | 1e-10 | 7e-12 | 1e-16 | 1e-16 | 1e-16 | 1e-16 | 1e-16 | 1e-16 |
| decay_same | RN | 3e-05 | 2e-06 | 8e-08 | 2e-08 | 5e-10 | 4e-11 | 4e-12 | 5e-14 | 1e-16 | 1e-16 | 1e-16 | 1e-16 | 1e-16 |
| decay_same | TRUNC | 5e-05 | 2e-06 | 1e-07 | 2e-08 | 7e-10 | 5e-11 | 4e-12 | 1e-13 | 1e-16 | 1e-16 | 1e-16 | 1e-16 | 1e-16 |
| decay_opp | RN | 2e+03 | 4e+02 | 2e+01 | 5e-01 | 8e-02 | 1e-02 | 6e-04 | 7e-06 | 2e-06 | 2e-07 | 1e-08 | 9e-10 | 7e-11 |
| decay_opp | TRUNC | 6e+03 | 6e+02 | 4e+01 | 2e+00 | 1e-01 | 1e-02 | 7e-04 | 6e-05 | 5e-06 | 2e-07 | 2e-08 | 2e-09 | 2e-10 |
| outliers | RN | 6e+00 | 1e+00 | 3e-01 | 5e-03 | 9e-05 | 1e-04 | 8e-06 | 1e-07 | 4e-08 | 2e-10 | 3e-11 | 2e-11 | 2e-12 |
| outliers | TRUNC | 5e+00 | 1e+00 | 4e-01 | 4e-02 | 8e-05 | 9e-05 | 4e-06 | 4e-07 | 4e-08 | 2e-10 | 4e-11 | 8e-12 | 3e-13 |
| cancel | RN | 1e+02 | 7e+00 | 4e-01 | 2e-02 | 2e-03 | 3e-04 | 1e-05 | 2e-07 | 5e-09 | 1e-11 | 9e-13 | 5e-15 | 1e-16 |
| cancel | TRUNC | 1e+02 | 8e+00 | 5e-01 | 1e-02 | 5e-04 | 3e-04 | 7e-06 | 2e-07 | 7e-09 | 1e-11 | 9e-13 | 9e-15 | 1e-16 |

### Native FP64 (numpy/OpenBLAS) on the same grids

| data | normwise | elementwise |
|---|---|---|
| uniform | 2e-16 | 5e-12 |
| decay_same | 2e-15 | 2e-12 |
| decay_opp | 9e-24 | 6e-12 |
| outliers | 9e-16 | 4e-13 |
| cancel | 1e-16 | 1e-05 |

Per-fixture breakdown, median ratios and the raw records:
[bench/accuracy/results/summary.md](../bench/accuracy/results/summary.md) and
`contractions.jsonl`.

## 3. Results: an iterative calculation

`bench/accuracy/iterate.py` runs a Jacobi fixed-point iteration for
`(Δ + W) x = r`, with `x[a,b,i,j]` and v = 40, o = 10.
- `Δ_abij = e_a + e_b − e_i − e_j` lies in [2, 10].
- W is a dense symmetric `W[a,b,c,d]` with decaying couplings. It is applied through
  `abcd,cdij->abij`, a GEMM view of (1600, 100, 1600).
- W is scaled so that ρ(Δ⁻¹W) = 0.7.
- The iteration is `x ← (r − W·x)/Δ` from x = 0, for 100 iterations.
- It runs once with the contraction in native FP64 (numpy) and once with `imd_contract`
  at each N. The final residual `‖r − (Δ+W)x‖/‖r‖` is evaluated in 80-bit extended
  precision.

Drift `‖x_N^t − x_fp64^t‖ / ‖x_fp64^t‖`:

| run | it 5 | it 10 | it 20 | it 40 | it 60 | it 80 | it 100 | max | final residual |
|---|---|---|---|---|---|---|---|---|---|
| FP64 (numpy) | | | | | | | | | 1.1e-16 |
| RN N=8 | 1.28e-09 | 1.32e-09 | 1.31e-09 | 1.35e-09 | 1.36e-09 | 1.36e-09 | 1.36e-09 | 1.38e-09 | 1.0e-09 |
| RN N=10 | 5.95e-12 | 6.10e-12 | 6.13e-12 | 6.20e-12 | 6.38e-12 | 6.42e-12 | 6.41e-12 | 6.43e-12 | 4.8e-12 |
| RN N=12 | 2.86e-14 | 2.90e-14 | 2.90e-14 | 2.91e-14 | 2.96e-14 | 3.00e-14 | 3.01e-14 | 3.02e-14 | 2.3e-14 |
| RN N=14 | 1.85e-16 | 1.87e-16 | 1.88e-16 | 1.88e-16 | 1.90e-16 | 1.94e-16 | 1.96e-16 | 1.96e-16 | 1.3e-16 |
| RN N=16 | 1.29e-16 | 1.30e-16 | 1.30e-16 | 1.31e-16 | 1.33e-16 | 1.37e-16 | 1.38e-16 | 1.39e-16 | 7.0e-17 |
| RN N=20 | 1.29e-16 | 1.30e-16 | 1.30e-16 | 1.31e-16 | 1.33e-16 | 1.37e-16 | 1.38e-16 | 1.39e-16 | 7.0e-17 |
| TRUNC N=8 | 4.36e-09 | 4.43e-09 | 4.46e-09 | 4.47e-09 | 4.47e-09 | 4.47e-09 | 4.47e-09 | 4.48e-09 | 2.6e-09 |
| TRUNC N=12 | 9.61e-14 | 9.69e-14 | 9.71e-14 | 9.70e-14 | 9.69e-14 | 9.68e-14 | 9.68e-14 | 9.74e-14 | 5.7e-14 |
| TRUNC N=16 | 1.30e-16 | 1.31e-16 | 1.31e-16 | 1.33e-16 | 1.38e-16 | 1.38e-16 | 1.39e-16 | 1.39e-16 | 7.0e-17 |

- **No accumulation.** The drift is set within the first few iterations and then stays
  flat: +6 % from iteration 5 to iteration 100 at N = 8. A contracting iteration damps
  each step's error. The steady state is the fixed point of the perturbed map, so it sits
  at about one contraction's error divided by (1 − ρ).
- The converged result is as good as the contraction:
  - N = 12 gives a residual of 2e−14;
  - N = 14 matches native FP64;
  - N ≥ 16 has a **lower** residual than native FP64: 7.0e−17 against 1.1e−16. The
    emulated contraction is correctly rounded, the BLAS one is not.
  - The 1.3e−16 drift floor at N ≥ 16 is the gap between two FP64 roundings of the same
    iteration, not an emulation error.
- TRUNC costs about 3× in the converged error at N ≤ 12. That is a little more than in
  the single contractions: TRUNC's bias toward zero does not average out across
  iterations.

## 4. Recommended N

The **normwise** target ε is relative to ‖a_i‖‖b_j‖. "Guaranteed" means the §3.8 terms
1 + 2 (`params.error_guidance`) are ≤ ε. "Observed" is the smallest N whose max normwise
error over every run here (k ≤ 3600) is ≤ ε. All columns are for RN.

| target ε | §3.8, k = 1000 | §3.8, k = 10⁴ | §3.8, k = 2¹⁷ | observed (k ≤ 3600) |
|---|---|---|---|---|
| 1e-6 | 7 | 8 | 8 | ≤ 8 |
| 1e-8 | 9 | 9 | 10 | ≤ 8 |
| 1e-10 | 11 | 11 | 11 | 9 |
| 1e-12 | 12 | 13 | 13 | 11 |
| 1e-14 | 14 | 14 | 15 | 13 |
| 1e-15 | 15 | 15 | 16 | 14 |
| u = 2⁻⁵³ | 16 | 16 | 17 | 15 |

The guaranteed columns follow from the `2√k/T` column of `params.py --table`.

- **FP64-comparable accuracy on typical data: N = 14 (RN).** The normwise error there is
  ≤ 1e−16, the same as or better than native FP64 (1e−16 to 2e−15 here). The converged
  iteration matches FP64.
- **Guaranteed FP64-level normwise accuracy: N = 16** (k ≤ 10⁴), or 17 at k = 2¹⁷.
- **Rule of thumb.** For an **elementwise** target ε_rel on outputs with
  `|x| ≈ r·‖a‖‖b‖`, take the normwise N for ε_rel·r. That is about
  `log₁₀(1/r) / 1.15` more moduli, since each modulus is worth about 15×. The data
  confirms it:
  - `outliers` (r ≈ 1e−6) needs N = 18 for 1e−10 elementwise, where `uniform` needs 13;
  - `decay_opp` (r ≈ 1e−9) needs N = 20 for 1e−10, against 13.
- Use RN. TRUNC is never better here, and it costs about ¼ modulus in single
  contractions and more in iterations.

## 5. Surprises and notes

1. **No bound violation.**
   - 520 runs: 4 fixtures × 5 distributions × N = 8…20 × {RN, TRUNC}.
   - 1.66 M exactly-referenced elements, max ratio 0.99.
   - 1.13 G elements in the full-output check, 0 failures.
2. **The bound is loose by about √k in the emulation regime.** The max observed
   normwise error is 2–3·‖a‖‖b‖/T, independent of k. The bound is 2√k‖a‖‖b‖/T. So, for these data
   distributions, recommendations from observation are 1–2 moduli cheaper than the
   guarantee at k ≈ 10³–10⁴. The guarantee itself is worst-case and cannot be tightened
   without assumptions on the data.
3. **Anti-correlated magnitudes are the realistic worst case** (`decay_opp`).
   - Native FP64's error scales with `Σ|a_h||b_h|`, which is tiny here, so native gets
     6e−12 elementwise.
   - Ozaki's error scales with ‖a‖‖b‖, which is about 1e9× larger.
   - To match native FP64 elementwise on such data takes N = 20. N = 14 gives only
     6e−4.
   - This is the "normwise per row and column" caveat in CC_NOTES, made quantitative.
     Integral × amplitude contractions where large integrals meet small amplitudes, and
     vice versa, are exactly this case.
4. **Heavy cancellation favours emulation.** Native FP64 loses `u·Σ|a||b|/|x|` (1e−5
   elementwise for `cancel`). The emulated integer product is exact, so from N = 15 on
   the elementwise error is smaller than native's: 2e−7 at N = 15, 5e−9 at N = 16, 1e−16
   at N = 20.
5. **Outliers** do not hurt normwise accuracy. A row with one 1e8 entry, though, quantizes
   all its other entries coarsely, so elementwise accuracy suffers as `‖a‖‖b‖/|x|`
   predicts.
6. **The `u·|D|` term of §3.8 is attained.** At N ≥ 16 the ratio reaches 0.99 on benign
   data, because the result is the correctly rounded exact value. A future tightening of
   the bound must keep this term in full.

No design issue was found. The adaptive-precision item in PLAN could use the elementwise
rule above: choose N from an estimate of `‖a‖‖b‖/|x|`, which a cheap low-N pass could
supply.
