# Memory model

Explicit equations for the memory footprint (and compulsory memory traffic) of an
Ozaki-II tensor contraction in immerdink. They are written to be lifted directly into
other code. A `workspace_size()` query in the library will implement §3 exactly, and the
pipeline TLA+ spec will carry §3 as an invariant.

All sizes are in **bytes**. `⌈x⌉_a` means "round x up to a multiple of a".

---

## 1. Symbols

Contraction: `C[I,J,L] = α · Σ_K A[I,K,L] · B[K,J,L] + β · C[I,J,L]`
(labels may appear in any order and with any strides in each tensor).

| Symbol | Meaning |
|---|---|
| `m = ∏_{i∈I} e_i` | combined extent of indices free in A (rows of the GEMM view) |
| `n = ∏_{j∈J} e_j` | combined extent of indices free in B (columns) |
| `k = ∏_{κ∈K} e_κ` | combined extent of contracted indices |
| `b = ∏_{λ∈L} e_λ` | combined extent of batch indices (in A, B and C); `b = 1` if none |
| `s_T` | bytes per user element: 8 (FP64), 16 (complex128), 4 (FP32) |
| `N` | number of moduli (≈ 14 for FP64-level accuracy, see §6) |
| `s_e` | bytes per scaling exponent: 2 (int16 suffices for FP64) |
| `p` | bytes per stored modular product: 4 (raw INT32) or 1 (reduced mod `m_l`, u8) |
| `s_acc` | bytes per CRT accumulator element (16 for int128 / double-double) |
| `g` | moduli processed per group (`1 ≤ g ≤ N`; `g = N` means ungrouped) |
| `a_m, a_n, a_k` | alignment of the INT8 GEMM operands (typically `a_k = 16`, `a_m = a_n = 1`) |
| `m̂, n̂, k̂` | `⌈m⌉_{a_m}`, `⌈n⌉_{a_n}`, `⌈k⌉_{a_k}` |

## 2. User tensors (owned by the caller)

Dense:

```
T_user = s_T · b · (m·k + k·n + m·n)
```

Strided: for each tensor X with extents `e_d` and strides `σ_d` (in elements), the
allocation is at least its span:

```
span(X) = s_T · (1 + Σ_d (e_d − 1)·|σ_d|)
T_user  = span(A) + span(B) + span(C)
```

## 3. Library workspace

```
W = W_E + W_R + W_P (+ W_gemm)
```

**Scaling exponents** (per `(I,L)` row of A, per `(J,L)` column of B):

```
W_E = s_e · b · (m + n)
```

**INT8 residue operands** (`g` moduli resident at a time):

```
W_R = b · g · (m̂·k̂ + k̂·n̂)
```

**Modular products**, one variant per pipeline design:

| Variant | Description | `W_P` |
|---|---|---|
| **V0** | each INT8 GEMM writes raw INT32; one CRT pass at the end | `4 · b · N · m · n` |
| **V1** | GEMM epilogue reduces mod `m_l` and stores u8 | `1 · b · N · m · n` |
| **V2** | groups of `g` moduli; CRT accumulated into a running buffer | `b · m · n · (g·p + s_acc)` |
| **V3** | CRT fused into the GEMM (register-resident accumulator) | `0` |

With V2 or V3 the residues need only be resident for one group (`g < N` in `W_R`). The
price is re-reading A and B (FP64) `⌈N/g⌉` times (§5).

**`W_gemm`**: workspace internal to the INT8 GEMM. It is 0 for a non-split-K CUTLASS
GEMM; split-K adds its partials or semaphores.

### Consequence worth noting

V1 costs `N` bytes per C element and V2 costs `g·p + s_acc`. So
**V1 needs less memory than every grouped variant whenever `N < g·p + s_acc`**. With
`s_acc = 16` that covers all FP64-level configurations (`N ≈ 14`). Grouping only pays off together with
residue grouping (`W_R`), or for V0 (`p = 4`).

### K splitting (exactness of INT32)

The INT32 accumulator is exact (in any accumulation order) iff

```
k ≤ K_max = ⌊(2^31 − 1) / max_l ⌊m_l/2⌋²⌋        (= 131071 when m_l = 256 is used)
```

This is `NoAccOverflow` in `spec/OzakiCRT.tla`. For larger `k`, split into
`⌈k / K_max⌉` sequential GEMMs. In V1 each later GEMM's epilogue adds into the stored u8
residue mod `m_l`, so **no extra memory** is needed, only `N·b·m·n` extra reads per
additional split.

## 4. Peak device memory

```
P = T_user + W
```

### Worked example: `m = n = k = 8192`, `b = 1`, `N = 14`, FP64, no padding

| Term | Bytes | |
|---|---|---|
| `T_user` | 8·3·2²⁶ | 1.50 GiB |
| `W_E` | 2·2·8192 | 32 KiB |
| `W_R` (g = N) | 14·2·2²⁶ | 1.75 GiB |
| `W_P` V0 | 4·14·2²⁶ | 3.50 GiB |
| `W_P` V1 | 14·2²⁶ | 0.875 GiB |
| `W_P` V2 (g=2, p=1, s_acc=16) | 18·2²⁶ | 1.125 GiB |
| **P (V0)** | | **6.75 GiB** |
| **P (V1)** | | **4.125 GiB** |

For comparison, native DGEMM needs only `T_user` = 1.5 GiB.

## 5. Compulsory global-memory traffic (lower bound)

Phases run in sequence, so a time lower bound is the sum over phases of
`max(ops / peak_ops, bytes / bandwidth)`.

| Phase | Bytes moved | Work |
|---|---|---|
| scaling pre-pass (reductions over K) | `s_T·b·(m·k + k·n)` | small |
| residue pack (gather + scale + residues) | `s_T·b·(m·k + k·n) · ⌈N/g⌉` read + `b·N·(m̂·k̂ + k̂·n̂)` write | `O(N·(mk+kn))` |
| INT8 GEMMs | `ρ · b·N·(m̂·k̂ + k̂·n̂)` read + `p·b·N·m·n` write | `2·b·N·m·n·k̂` int8 ops |
| CRT + unscale + α/β + scatter | `p·b·N·m·n` read + `s_T·b·m·n·(1 + [β≠0])` | `O(N·m·n)` (V1/V0) |

`ρ ≥ 1` is the operand re-read factor of the GEMM tiling (1 if L2 captures all reuse).

Worked example above, V1, β = 0, ρ = 1: about **7.75 GiB** of traffic (≈ 8 ms at the
4090's ~1 TB/s) and `2·14·8192³ ≈ 1.54·10¹³` int8 ops (≈ 23 ms at the 660 TOPS dense
peak). The paper's measured ~9.8 TFLOPS-equivalent is ≈ 112 ms for this size, so there is
a lot of headroom between measured and bound. This is where the performance work is.

## 6. Choosing N (drives every term above)

The CRT range must hold the exact integer product. With `M = ∏_l m_l` and the
crude (any-order, worst-case) bound,

```
k · max_k|A'[i,k]| · max_k|B'[k,j]|  ≤  (M − 1)/2            (Preconditions in OzakiCRT.tla)
```

If the scaling keeps `β_A` bits per row of A and `β_B` bits per column of B, this gives

```
log2 M ≥ 1 + log2 k + β_A + β_B     ⇒     N ≈ ⌈(1 + log2 k + β_A + β_B) / 7.9⌉
```

using the greedy pairwise-coprime moduli `256, 255, 253, 251, 247, 241, 239, 233, 229,
227, 223, 217, 211, 199, 197, 193, …` (`log2 M` = 47.8 at N=6, 102.5 at N=13, 110.2 at
N=14, 125.4 at N=16). The paper's *fast mode* uses a Cauchy–Schwarz bound (`‖a_i‖₂‖b_j‖₂`)
instead of the `k·max·max` bound, which is tighter and keeps more bits for the same N.

**CRT accumulator width.** Reconstruction needs integers up to:
- `N·M·255` for a naive sum: fits in signed int128 for `N ≤ 14`;
- `M·256` with a running mod-`M` reduction: fits for `N ≤ 15`.

Larger N needs wider integers or Garner mixed-radix reconstruction (open design question).

## 7. On-chip resources per CTA (GPU, CUTLASS-style INT8 GEMM)

Tile `BM × BN × BK`, `S` pipeline stages, `T` threads:

```
smem_CTA      = S · (BM·BK + BK·BN)                  (int8 operands)
regs_acc      = BM·BN / T                            (INT32 accumulators per thread)
regs_crt (V3) = BM·BN·s_acc / (4·T)                  (register-resident CRT accumulator)
CTAs/SM       = min( ⌊smem_SM / smem_CTA⌋,
                     ⌊65536 / (regs_thread · T)⌋,
                     ⌊1536 / T⌋,  24 )               (sm_89 limits; smem_SM ≈ 100 KiB)
```

Example: `128×128×64`, `S = 3`, `T = 256` gives `smem_CTA` = 48 KiB and `regs_acc` = 64.
Adding a V3 int128 accumulator costs another 256 registers per thread, which exceeds
the 255 limit. **V3 is infeasible at this tile size.** At `64×128` it costs 160 registers
before addressing overhead, so one CTA per SM. This is why V1 is the starting design.
