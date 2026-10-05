# immerdink
I want a GETT/T-BLIS-style GETT kernel. Until that happens, I'll make AI do it myself ;-)

TBLIS-style tensor contractions computed with Ozaki scheme II (CRT-based) FP64 emulation
on INT8 tensor cores (CUDA, sm_89 first).

- [docs/SCOPE.md](docs/SCOPE.md): goal, decisions
- [docs/DESIGN.md](docs/DESIGN.md): what is computed and how
- [docs/PLAN.md](docs/PLAN.md): roadmap and milestones
- [docs/MEMORY_MODEL.md](docs/MEMORY_MODEL.md): exact workspace / traffic equations
- [docs/PROOFS.md](docs/PROOFS.md): proofs of the core lemmas (Lean blueprint)
- [spec/README.md](spec/README.md): TLA+ specs, real-width certificate, bit-exact oracle

Build and run all correctness tests: `tools/test.sh` (conda env `immerdink`; see
[docs/NOTES-M0.md](docs/NOTES-M0.md)). Status: M1 (end-to-end emulated contraction on
sm_89, bit-exact against the oracle) is described in [docs/NOTES-M1.md](docs/NOTES-M1.md).

Two libraries: `libimmerdink_host.so` (CUDA-free: lowering, legality and the exact memory
model; header `immerdink/immerdink_host.h`) and `libimmerdink.so` (the CUDA kernels; links
the host library and sizes everything with it). See DESIGN §6a.

### Exact footprints without a GPU

Callers that manage memory themselves (e.g. ehrenfest) ask the host library what a
contraction needs, for each residue-group size g (the workspace/speed knob):

```python
import sys; sys.path.insert(0, "python")    # or PYTHONPATH=python
from immerdink import host                  # finds build/libimmerdink_host.so, or set
                                            # IMMERDINK_HOST_LIB / IMMERDINK_LIB_DIR
spec, ext = "kjbc,kiac->abij", dict(a=200, b=200, c=200, i=40, j=40, k=40)
host.workspace_size(spec, ext, N=14, g=7)                 # 1792432896
host.plan_bytes(spec, ext, N=14, vec_mode="RESIDENT")     # (384024, 2352): device, host
lw = host.lowering(spec, ext, N=14, g=7)                  # GEMM view, dispatch, launches
(lw.dispatch, lw.m, lw.n, lw.k, lw.swapped, lw.num_groups)  # ('EMULATE', 8000, 8000, 8000, True, 2)
print(host.workspace_layout(spec, ext, N=14, g=7))        # the region table
for row in host.g_tradeoff(spec, ext, N=14):              # workspace for g = 1..14
    print(row["g"], row["workspace"])
```

or `PYTHONPATH=python python3 -m immerdink.host "kjbc,kiac->abij" a=200,b=200,c=200,i=40,j=40,k=40 -N 14`
for the g table ([docs/MEMORY_MODEL.md](docs/MEMORY_MODEL.md) §3a has its output and the C
version). Strides default to dense (last mode fastest); pass `strides=(sA, sB, sC)` or raw
`(extent, stride, label)` descriptors otherwise.

### Contractions from Python

`immerdink.Plan` / `immerdink.contract` (DESIGN §6c) run contractions on caller-owned float64
CUDA arrays: anything with `__cuda_array_interface__` or DLPack (CuPy, PyTorch, Numba, JAX),
or raw pointers (`immerdink.DeviceArray(ptr, shape, strides)`). No array library is
required, nothing is copied between host and device, and immerdink allocates nothing: the
output and the workspace are yours.

```python
import immerdink as imd                     # finds build/libimmerdink.so, or set IMMERDINK_LIB
import cupy as cp                           # any CUDA array library works the same way
A, B = cp.random.rand(64, 32, 48), cp.random.rand(48, 40)
C = cp.empty((40, 64, 32))
plan = imd.Plan("ijk,kl->lij", A, B, C, N=14)                # imd_plan_create
ws = cp.empty(plan.workspace_bytes, cp.uint8)                # exact bytes (also: imd.workspace_size)
plan.contract(A, B, C, alpha=1.0, beta=0.0, workspace=ws,
              stream=cp.cuda.get_current_stream())           # enqueue only: capture-safe
imd.contract("ijk,kl->lij", A, B, C, N=14, workspace=ws)     # einsum front end, cached plans
pa = imd.Pack(plan.pack_spec(0))                             # packed operands (DESIGN §6b)
```

Day-to-day work lands on `dev`; `main` only changes via pull request.

## License and credits
MIT (see [LICENSE](LICENSE)). Third-party code keeps its own license: CUTLASS
(`third_party/cutlass`, BSD-3-Clause).

Built on published ideas, not code:
- Ozaki, Uchino, Imamura, *Ozaki Scheme II* (arXiv:2504.08009), with its error analysis
  (arXiv:2602.02549);
- GEMMul8 (RIKEN-RCCS);
- Matthews, *High-Performance Tensor Contraction without Transposition* (TBLIS).
