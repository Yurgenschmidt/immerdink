# Memory model

These are explicit equations for an immerdink contraction's device memory, memory traffic
and on-chip resources, written so they can be lifted into other code.

- §2–§3 are **exact**. They equal `imd_workspace_size`, are certified by
  `spec/params.py` (`workspace_size`, `region_offsets`, `cert_workspace`), and are
  model-checked by `spec: Pipeline LayoutExact`. The design source is DESIGN §4a.
- §4–§6 are **models** (lower bounds and estimates), not guarantees.

All sizes are in **bytes**. `⌈x⌉_a` means "round x up to a multiple of a".

---

## 1. Symbols

The contraction is `C[I,J,L] = α · Σ_K A[I,K,L] · B[K,J,L] + β · C[I,J,L]`, with labels in
any order and any strides. Extents are counted **after IndexMap lowering** (DESIGN §5):
diagonals merged, extent-1 modes dropped, and A-only/B-only labels moved into K.

| Symbol | Meaning |
|---|---|
| `m = ∏_{i∈I} e_i` | rows of the GEMM view (labels in A and C only) |
| `n = ∏_{j∈J} e_j` | columns (labels in B and C only) |
| `k = ∏_{κ∈K} e_κ` | contracted extent (labels in A and B, plus single-operand labels) |
| `b = ∏_{λ∈L} e_λ` | batch extent (labels in A, B and C); `b = 1` if there are none |
| `N` | number of moduli (2…20) |
| `g` | moduli per residue group (1…N; the API default is N) |
| `k̂, n̂` | `⌈k⌉₁₆`, `⌈n⌉₁₆`, the CUTLASS int8 alignment |
| `align(x)` | `⌈x⌉₂₅₆`, the workspace region alignment |
| `s_T` | bytes per user element (8 for FP64) |

## 2. User tensors (owned by the caller)

```
dense:    T_user = s_T · b · (m·k + k·n + m·n)
strided:  T_user = Σ_{X∈{A,B,C}} s_T · (1 + Σ_d (e_d − 1)·|σ_d|)      (span of each tensor)
```

## 3. Library workspace: exact

```
W_sA  = 2·b·m                      int16 row shifts of A
W_sB  = 2·b·n                      int16 column shifts of B
W_fA  = 1·b·m                      u8 non-finite flags (A rows)
W_fB  = 1·b·n                      u8 non-finite flags (B columns)
W_vec = 8·(2m + 2n + 2k + 3b)      int64 scatter vectors
W_RA  = g·b·m·k̂                    int8 residues of A, one group
W_RB  = g·b·n̂·k̂                    int8 residues of B, one group
W_W   = N·b·m·n̂                    u8 modular products, all N planes

workspace_size =  Σ align(W_r) over r ∈ (sA, sB, fA, fB, vec, RA, RB, W), in this order
               =  0                if b·m·n = 0                (empty output)
               =  align(W_vec)     if mode = DIRECT, k = 0, or K-empty (k = 1)
```

Rules:
- Offsets are cumulative sums of the aligned sizes, so `offset(r) = Σ_{r'<r} align(W_r')`.
- The caller must pass `ws_bytes ≥ workspace_size`.
- An overflow of uint64 returns INVALID_VALUE.
- **K-chunking costs no memory.** Each later chunk adds into the stored u8 planes.
- **The batch-launch split costs no memory.**

### Scatter vectors: per call or per plan (DESIGN v1.2 `vec_mode`)

```
IMD_VEC_GENERATE (default):  W_vec is in the workspace (above); the plan owns 0 device
                             bytes and 0 pinned bytes (O(rank) host bytes)
IMD_VEC_RESIDENT:            the workspace has NO vec region (drop align(W_vec) above:
                             DIRECT, k = 0 and K-empty need 0 bytes), and the plan owns
                             D_plan = W_vec = 8·(2m + 2n + 2k + 3b) device bytes
                             (imd_plan_device_bytes; 0 for an empty output)
```

`D_plan` is the size requested from `cudaMalloc`; the driver rounds allocations up to its
own granularity, which is not counted. Neither mode pins host memory.

Peak device memory, for one call and the plans alive at that time:

```
P_dev = T_user + workspace_size + Σ_{live plans} D_plan
```

With GENERATE the last term is 0, so `P_dev = T_user + workspace_size` as in v1.1.
Example: m = 10⁸ rows, n = k = 64, b = 1 gives `W_vec` ≈ 1.6 GB; RESIDENT keeps that per
plan for the plan's lifetime, GENERATE only inside the workspace of a running call (which
the caller may share between plans used one after another).

### Worked examples (exact, from `params.workspace_size`)

| Shape (m, n, k, b) | N | g | workspace | user FP64 | ws / user |
|---|---|---|---|---|---|
| 8192³, 1 | 14 | 14 | 2,819,014,912 B (2.625 GiB) | 1.500 GiB | 1.75 |
| 8192³, 1 | 14 | 1 | 1,074,184,448 B (1.000 GiB) | 1.500 GiB | 0.67 |
| 8192³, 1 | 9 | 9 | 1,812,381,952 B (1.688 GiB) | 1.500 GiB | 1.13 |
| 4096³, 1 | 14 | 14 | 704,864,512 B (0.656 GiB) | 0.375 GiB | 1.75 |
| 2500³, 10 | 10 | 10 | 1,887,285,248 B (1.758 GiB) | 1.397 GiB | 1.26 |
| 1000 × 64 × 50000, 1 | 12 | 12 | 639,988,736 B (0.596 GiB) | 0.397 GiB | 1.50 |

For large square problems the three big terms are about `N·m·k`, `N·n·k` and `N·m·n` bytes.

### Knobs for reducing memory
- **`g`** shrinks the residue terms by `N/g`. The cost is that A and B are re-read
  `⌈N/g⌉` times (§4).
- **The caller can tile I, J or L** (sub-views are valid inputs). The workspace then scales
  with the tile's m, n and b.
  - Results stay **bitwise identical**, because shifts depend only on full-K rows and
    columns.
  - Caller tiling over K changes results; avoid it.
- **N** scales the residue and product terms linearly. Choose it from §6.

## 4. Compulsory global-memory traffic (lower bound)

The kernels run in sequence, so a time lower bound is the sum over kernels of
`max(ops / peak_ops, bytes / bandwidth)`.

| Kernel | Bytes moved | Work |
|---|---|---|
| K0 vec (GENERATE; none with RESIDENT) | `W_vec` written, nothing read (v1.1: an H2D copy from pinned memory) | `O(W_vec/8 · rank)` integer ops |
| K1 scale | `s_T·b·(m·k + k·n)` read | `O(b·(mk + kn))` integer ops |
| K2/K3 pack | `⌈N/g⌉·s_T·b·(m·k + k·n)` read + `N·b·(m·k̂ + n̂·k̂)` written | `O(N·b·(mk+kn))` |
| K4 GEMM | `ρ·N·b·(m·k̂ + n̂·k̂)` read + `N·b·m·n̂·(2·n_chunks − 1)` u8 written/read | `2·N·b·m·n̂·k̂` int8 ops |
| K5 CRT | `N·b·m·n̂` read + `s_T·b·m·n·(1 + [β≠0])` | `O(N·b·m·n)` integer ops |

Here:
- `ρ ≥ 1` is the GEMM tiling's operand re-read factor (1 if L2 captures all reuse).
- `n_chunks = ⌈k / 131056⌉`.
- The flagged-element native path adds `s_T·k·(n·#flaggedRows + m·#flaggedCols)` bytes.

**Example: 8192³, N = 14, g = 14, β = 0, ρ = 1.** About 7.75 GiB of traffic (≈ 8 ms at
~1 TB/s) and `2·14·8192³ ≈ 1.54·10¹³` int8 ops (≈ 23 ms at the 4090's 660 TOPS dense
peak). Published GEMMul8 runs take about 85 ms (4090 Laptop), so the headroom is in the
non-GEMM kernels and GEMM efficiency.

## 5. On-chip resources per CTA (CUTLASS 2.x int8 GEMM on sm_89)

For tile `BM × BN × BK`, `S` stages and `T` threads:

```
smem_CTA  = S · (BM·BK + BK·BN)                     int8 operands (≤ 99 KB per block)
regs_acc  = BM·BN / T                               int32 accumulators per thread
CTAs/SM   = min(⌊smem_SM/smem_CTA⌋, ⌊65536/(regs_thread·T)⌋, ⌊1536/T⌋, 24)
            smem_SM ≈ 100 KB
```

Example: `128×128×64`, S = 3, T = 256 → `smem_CTA` = 48 KiB and `regs_acc` = 64. The
prototype epilogue kernels compiled to about 226–228 registers per thread at 128×256, so 1
CTA per SM.

## 6. Choosing N

Use the worst-case normwise error guidance (DESIGN §3.8):

```
RN:  |D − Σab| ≲ 2√k·s/(T − κ/2) · ‖a_ī‖₂‖b_j̄‖₂ ,   T = ⌊√⌊(P−1)/2⌋⌋,   s = √(1 + k·2⁻⁶⁰)
```

`python3 spec/params.py --table` prints `log2 P`, `log2 T`, the RN-feasible k limit and
this guidance for every N. Measured with the oracle at k = 1000, the emulated error
matches native FP64 dot-product error at N = 14–15 for well-scaled data (φ ≤ 1), and at
N = 16 for wide dynamic range (φ = 4).

## 7. Alternatives considered (for comparison)

| Variant | Product storage | Why not v1 |
|---|---|---|
| V0: raw int32 planes | `4·N·b·m·n` | 4× V1 for nothing once the epilogue reduces mod p |
| V2: grouped CRT accumulator | `b·m·n·(g + 16)` | more than V1's `N·b·m·n` for every N ≤ 16 |
| V3: register-resident CRT | 0 | a 128-bit accumulator adds `BM·BN·16/(4T)` registers per thread (256 at 128×128/256 threads), which exceeds the 255 limit |
| GEMMul8 v3.5.2 | grouped uint32 planes, padded to 256 | ≈ 3.5 GiB at 8192³, N = 14, against our 2.625 GiB |
