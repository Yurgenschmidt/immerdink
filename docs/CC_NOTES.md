# Background: coupled-cluster contractions

This is context only. CC (t1-dressed DF-CCSD, CCSD(T)) was the original motivation, but
immerdink is a **general** Ozaki-II TBLIS-style contraction kernel, and nothing in its
design is CC-specific. The shapes below are useful as realistic test fixtures.

## Naming
- immerdink's "Ozaki-II" is the **CRT scheme** (arXiv:2504.08009).
- ehrenfest's `OzakiSpec(scheme="II")` is a different method: slice-pair truncation of
  Ozaki-I.

## Representative contraction shapes (GEMM view m, n, k)

| Class | Example | (m, n, k) | Notes |
|---|---|---|---|
| DF integral builds | `acQ,bdQ->acbd`, `B^Q_ia B^Q_jb` | (ov, ov, nQ), (v, v·nb, nQ) | permuted outputs; A ≡ B common |
| ring / CCD | `kjbc,kiac->abij` | (ov, ov, ov) | k > 131071 at large o·v → K-chunking |
| pp ladder | `τ[ij,cd] V[cd,b]` | (o², v, v²/2) | k > 131071 for v ≥ 512 |
| (T) particle | `Σ_d (bd\|ck) t^{ad}_{ij}` | (v², v, v) | six permuted outputs, β = 1 |
| (T) hole / Fock-like | `Σ_l …` | small k or small m·n | usually not worth emulating |

Typical sizes: o ≈ 20–300, v ≈ 200–3000, nQ ≈ 3–5× the basis size.

## Takeaways that apply generally
- **Operands are usually permuted, strided sub-tensors.** A fast gather-pack for
  interleaved permutations matters more than anything else. ehrenfest measured strided
  slicing at 0.6 G elem/s against 17 G elem/s coalesced.
- **Truncation biases sums of products** toward zero. That is the general argument for
  round-to-nearest quantization (DESIGN §3.3). Small PySCF experiments showed 15–37×
  lower energy error at equal N.
- **Ozaki errors are normwise per row and column.** Contractions with heavy cancellation
  lose relative accuracy. This is inherent to every Ozaki scheme.
