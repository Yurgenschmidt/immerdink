# Memory model

These are explicit equations for an immerdink contraction's device memory, memory traffic
and on-chip resources, written so they can be lifted into other code.

- §2–§3 are **exact**. They equal `imd_workspace_size`, are certified by
  `spec/params.py` (`workspace_size`, `region_offsets`, `cert_workspace`), and are
  model-checked by `spec: Pipeline LayoutExact`. The design source is DESIGN §4a.
- Programs get these numbers from `libimmerdink_host` (§3a): the CUDA-free library whose
  code libimmerdink itself uses to size plans and workspaces.
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

### Packed operands (DESIGN v1.7 §6b, D1)

A packed operand X (b batches, r rows, k contracted; k̂ = ⌈k⌉₁₆) lives in a caller-owned
buffer for as long as the caller keeps it:

```
P_sig  = 2·b·r                                  int16 shifts
P_flag = 1·b·r                                  u8 non-finite flags
P_res  = N·b·r·k̂ + (⌈r⌉₁₆ − r)·k̂               int8 residues, all N planes, + zeroed tail
pack_bytes    = align(P_sig) + align(P_flag) + align(P_res)      (0 if b·r·k = 0)
pack_ws_bytes = align(8·(r + k + b))            imd_pack's workspace: X's scatter vectors,
                                                only during imd_pack (0 for an empty pack)
```

A plan that reads X from its pack has no sX, fX, RX regions in its workspace (X's post-swap
role); vec and W are unchanged:

```
workspace_size_packed = workspace_size − align(W_sX) − align(W_fX) − align(W_RX)  (per packed X)
P_dev = T_user + workspace_size_packed + Σ_{kept packs} pack_bytes + Σ_{live plans} D_plan
        (+ pack_ws_bytes while an imd_pack runs, if not taken from the workspace)
```

The pack holds all N planes (`≈ N/8` of the FP64 operand: 1.75× at N = 14); the workspace
saves `g/N` of that per contraction (all of it at g = N). Example (8192³, N = 14, g = 14):
one operand packed: workspace 1.75 GiB (−0.88 GiB), pack 0.88 GiB; both packed: workspace
0.875 GiB (W + vec), packs 2 × 0.88 GiB. Exact numbers: `imd_pack_layout`,
`imd_workspace_size_packed` (§3a), spec/params.py `pack_layout`, `pack_ws_bytes`,
`workspace_size(packed=...)`.

## 3a. Getting exact footprints from another program (no GPU)

`libimmerdink_host.so` (DESIGN §6a; header `immerdink/immerdink_host.h`) answers every
question of §3 for concrete descriptors and options, with no GPU, CUDA runtime or driver.
It is the code libimmerdink runs at plan creation, so the answers are what the kernels use
(`host_vs_plan` checks real plans against it); spec/params.py only verifies it.

| Question | C | Python (`python/immerdink/host.py`) |
|---|---|---|
| workspace bytes for (A, B, C, N, g, rounding, mode, vec_mode) | `imd_workspace_size` | `workspace_size` |
| the §3 region table (offset, bytes per region) | `imd_workspace_layout` | `workspace_layout` |
| plan device bytes `D_plan`, plan host payload | `imd_plan_bytes` | `plan_bytes` |
| GEMM view, dispatch, groups, K4 launches, all sizes, plan-time status | `imd_lowering` | `lowering` |
| K4 configuration for an SM count; temporary bytes of a `k4_tune` plan creation (DESIGN v1.5) | `imd_lowering` (`sm_count` in, `k4_config`, `k4_tune_bytes`) | `lowering(..., sm_count=)` |
| workspace for every g = 1..N | (loop over g) | `g_tradeoff`, `python3 -m immerdink.host` |
| a packed operand's bytes, layout and imd_pack workspace (DESIGN v1.7) | `imd_pack_layout` | `pack_layout` |
| the descriptor reproducing one contraction's orders | `imd_pack_desc_for` | `pack_desc_for` |
| workspace / region table / summary of a plan with packed operands | `imd_workspace_size_packed`, `imd_workspace_layout_packed`, `imd_lowering_packed` | `packing=` on `workspace_size`, `workspace_layout`, `lowering` |

The **g trade-off** for a CC-shaped contraction (o = 40, v = 200; N = 14):

```
$ PYTHONPATH=python python3 -m immerdink.host "kjbc,kiac->abij" a=200,b=200,c=200,i=40,j=40,k=40 -N 14
kjbc,kiac->abij  N=14  EMULATE  m=8000 n=8000 k=8000 b=1  (swapped)  status SUCCESS
plan: 0 device bytes, 2,352 host bytes
  g  workspace bytes      GiB  vs g=N groups launches
  1    1,024,432,896    0.954     38%     14       14
  2    1,152,432,896    1.073     43%      7        7
  4    1,408,432,896    1.312     52%      4        4
  7    1,792,432,896    1.669     67%      2        2
 14    2,688,432,896    2.504    100%      1        1
```
(rows g = 3, 5, 6, 8–13 omitted here). The part independent of g is the W planes, `N·b·m·n̂` =
896 MB; each step of g adds `b·(m + n̂)·k̂` = 128 MB of residues. BENCHMARKS ("Residue-group
size g") gives the speed side: g = 7 cost ≤ 12 % on every benchmarked shape, g ≤ 2 is
a last resort.

From C (link `-limmerdink_host`; no CUDA include path needed):

```c
#include "immerdink/immerdink_host.h"
/* A, B, C: imd_tensor_t descriptors as for imd_plan_create */
for (int32_t g = 1; g <= 14; ++g) {
  const imd_options_t o = {14, g, IMD_RN, IMD_EMULATE, IMD_VEC_GENERATE};
  size_t ws = 0, dev = 0, host = 0;
  if (imd_workspace_size(&A, &B, &C, &o, &ws) != IMD_STATUS_SUCCESS) break;
  imd_plan_bytes(&A, &B, &C, &o, &dev, &host);   /* NOT_SUPPORTED if kappa >= T (RN) */
  printf("g=%d workspace %zu plan device %zu host %zu\n", g, ws, dev, host);
}
```

Peak device memory for a set of live plans and one call at a time is then exactly
`T_user + max workspace_size + Σ plan_device_bytes` (§3), all from these queries.

## 4. Compulsory global-memory traffic (lower bound)

The kernels run in sequence, so a time lower bound is the sum over kernels of
`max(ops / peak_ops, bytes / bandwidth)`.

| Kernel | Bytes moved | Work |
|---|---|---|
| K0 vec (GENERATE; none with RESIDENT) | `W_vec` written, nothing read (v1.1: an H2D copy from pinned memory) | `O(W_vec/8 · rank)` integer ops |
| K1 scale | `s_T·b·(m·k + k·n)` read | `O(b·(mk + kn))` integer ops |
| K2/K3 pack | `⌈N/g⌉·s_T·b·(m·k + k·n)` read + `N·b·(m·k̂ + n̂·k̂)` written | `O(N·b·(mk+kn))` |
| KL (v1.6, per operand x, replaces K1 of x and group 1's pack of x) | `s_T·b·rows_x·k` read + `g·b·rows_pad_x·k̂` written; L2 re-reads only for rows longer than two 4096-value chunks | as K1 + one group's pack |
| K4 GEMM | `ρ·N·b·(m·k̂ + n̂·k̂)` read + `N·b·m·n̂·(2·n_chunks − 1)` u8 written/read | `2·N·b·m·n̂·k̂` int8 ops |
| K5 CRT | `N·b·m·n̂` read + `s_T·b·m·n·(1 + [β≠0])` | `O(N·b·m·n)` integer ops |

Here:
- With KL for operand x (DESIGN v1.6 §4), x is read `⌈N/g⌉` times from DRAM in total
  instead of `1 + ⌈N/g⌉` (K1's own second pass hits L2 in both cases).
- `ρ ≥ 1` is the GEMM tiling's operand re-read factor (1 if L2 captures all reuse).
- `n_chunks = ⌈k / 131056⌉`.
- The flagged-element native path adds `s_T·k·(n·#flaggedRows + m·#flaggedCols)` bytes.

**Example: 8192³, N = 14, g = 14, β = 0, ρ = 1.** About 7.75 GiB of traffic (≈ 8 ms at
~1 TB/s) and `2·14·8192³ ≈ 1.54·10¹³` int8 ops (≈ 23 ms at the 4090's 660 TOPS dense
peak). The measured call takes ≈ 30 ms (NOTES-A5 §3.4), so what is left above the bound
is in the non-GEMM kernels and GEMM efficiency.

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
