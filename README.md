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

### Predicted time and configuration ranking (no GPU)

The same library predicts what a call costs on a calibrated GPU (the RTX 4090 calibration
ships with it; `tools/calibrate.sh` makes one for another GPU) and ranks the configurations a
caller may choose inside a memory budget, each with its exact footprint (DESIGN §6c):

```python
p = host.predict(spec, ext, N=14, g=7)            # per-phase and total time, TFLOPS-eq
(p.tflops_eq, p.time_s, [(q.name, q.time_s) for q in p.phases])
for c in host.rank(spec, ext, N=[12, 14], vec_modes=("GENERATE", "RESIDENT"),
                   packs=("NONE", "A", "B"), reuse=4, budget=2 * 2**30)[:5]:
    print(c.num_moduli, c.residue_group, c.vec_mode, c.packed, c.time_s, c.total_bytes)
```

or `python3 -m immerdink.host <spec> <extents> -N 14 --rank 12,14 --budget 2147483648`. From C:
`imd_perf_predict`, `imd_perf_rank`, `imd_perf_calib_load` (`immerdink_host.h`).

Day-to-day work lands on `dev`; `main` only changes via pull request.

## License and credits
MIT (see [LICENSE](LICENSE)). Third-party code keeps its own license: CUTLASS
(`third_party/cutlass`, BSD-3-Clause).

Built on published ideas, not code:
- Ozaki, Uchino, Imamura, *Ozaki Scheme II* (arXiv:2504.08009), with its error analysis
  (arXiv:2602.02549);
- GEMMul8 (RIKEN-RCCS);
- Matthews, *High-Performance Tensor Contraction without Transposition* (TBLIS).
