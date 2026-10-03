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
[docs/NOTES-M0.md](docs/NOTES-M0.md)).

Day-to-day work lands on `dev`; `main` only changes via pull request.

## License and credits
MIT (see [LICENSE](LICENSE)). Third-party code keeps its own license: CUTLASS
(`third_party/cutlass`, BSD-3-Clause).

Built on published ideas, not code:
- Ozaki, Uchino, Imamura, *Ozaki Scheme II* (arXiv:2504.08009), with its error analysis
  (arXiv:2602.02549);
- GEMMul8 (RIKEN-RCCS);
- Matthews, *High-Performance Tensor Contraction without Transposition* (TBLIS).
