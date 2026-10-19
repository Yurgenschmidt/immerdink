# Issue 19: the test gate on every GPU

This note says what the test suite checks, how its pass rules are fixed, and how to run it on a
cluster GPU. It is written for someone new to the project.

## Why a test gate

immerdink emulates 64-bit floating-point matrix products with 8-bit integer tensor-core products
(Ozaki scheme II). The integer sums are exact. So the result matrix `C` of a contraction is a function
of the inputs and the options alone: the same on an RTX 4090, an H100, an H200 and an RTX PRO 6000.
If two GPUs disagree by even one bit, something is wrong. That is not noise.

Before issue 19 the suite (63 tests on the feature branch) had only run on the RTX 4090. With this issue it registers 68 tests: 25 that never touch the GPU and 43 that do (67 when no compute-sanitizer is found, which drops `contract_initcheck_pad`). The cluster runs
checked a sampled error and one output fingerprint (a hash) per benchmark case. This issue adds the
whole suite on the other GPUs, plus a committed list of output hashes.

## What changed

1. **`tools/test.sh` no longer restarts itself forever.** It re-launches itself once inside the
   memory budget wrapper `tools/memcap.sh`. Without a systemd user manager (every cluster node) the
   wrapper cannot create the budget, so the test "am I inside the budget?" never became true and the
   script re-launched itself about 200 times in 6 seconds. The wrapper now sets `IMMERDINK_MEMCAP=1`
   and the scripts that use it (`tools/test.sh`, `tools/check_proofs.sh`, `spec/check.sh`,
   `spec/apalache/check.sh`) treat that as "already handled". The ctest `tools_memcap_loop_check`
   runs both with `systemctl` stubbed to fail and requires that they end.
2. **The suite runs on a cluster GPU.** `IMMERDINK_ENV=none` uses the `cmake`, `nvcc`, `g++` and
   `python3` on `PATH`; `IMMERDINK_CUDA_ARCH` (or the `CMAKE_CUDA_ARCHITECTURES` the cluster jobs
   export) picks the architecture (90 for H100 and H200, 120 for the RTX PRO 6000; the default is
   89). `IMMERDINK_GPU_LOCK=none` skips the lock that protects the shared desktop GPU (**cluster jobs must set it**: the default path is this desktop's `ehrenfest-ipdps/_wip/gpu.lock`, with `$XDG_RUNTIME_DIR` or `/tmp` as the fallback). Both `90` and
   `120` build the library and all tests on the 4090 machine (checked on 2026-10-15).
3. **Host-only tests simulate the cluster cards.** The planner rules were already checked with 132 SMs (H100,
   H200); 188 (RTX PRO 6000) is added to the host tests (`test_host_python`), alongside 16, 46, 82 and 170.
4. **Committed output hashes** (`tests/data/output_hashes.json`, `tests/test_output_hashes.py`).
5. **A failed test leaves its evidence.** After a failed run `tools/rerun_failed.sh` reruns each failed
   test with full output and then under `compute-sanitizer`, in the same job (a rerun costs no queue
   wait). Tests that load the allocation shim (`contract_*`) are rerun without the sanitizer, because
   both hook the CUDA runtime.

## The hash list

Each entry gives a contraction (shapes, labels, number of moduli N, rounding, alpha, beta), a data
family, and the hash of the whole result `C`. The hash is `imd_bench`'s `c_hash`: the sum over `C`'s
elements of `splitmix64(bits(C[i]) ^ (0x9E3779B97F4A7C15 * (i + 1)))`, modulo 2^64.

Inputs are built on the host with integer steps and exact floating-point steps only (no `exp` or
`log`), so every machine builds the same bits. Family `u` is uniform in (-1/2, 1/2); family `pow2`
multiplies each element by a power of two between 2^-16 and 2^15, which gives a wide dynamic range
and still no rounding in the data.

Where the hash came from (`source`):

| source | meaning |
|---|---|
| `cpu` | the pure-Python oracle `spec/oracle.py` (exact rational arithmetic). Independent of any GPU. Shapes up to 256 x 256 x 256. |
| `rtx4090` | the library on the RTX 4090. For shapes the oracle cannot do. The small `cpu` cases tie the 4090 run to the oracle. |

Tags select cases: `small` and `mid` run in the normal suite (about a second on a 4090), `large`
(4096-cubed class, a K longer than one chunk, 16 batches of 1024-cubed) in `output_hashes_large`
(about 10 seconds), `huge` (8192-cubed, needs about 4 GB) only by hand.

| ctest | cases | note |
|---|---|---|
| `output_hashes` | small, mid | GPU result against the committed hash |
| `output_hashes_oracle` | small, mid | also recomputes the CPU hashes with the oracle (`sq256`, the one CPU-source case tagged mid, included) |
| `output_hashes_large` | large | GPU result against the committed hash |

To change or add a case: edit the case list in `tests/data/output_hashes.json` (leave `c_hash`
empty) and run `python3 tests/test_output_hashes.py --regen NAME --write` on the 4090. For a `cpu`
case it also checks that the GPU agrees with the oracle and refuses to write if not. A hash changes
only when the arithmetic changes. Say why in the commit message.

## Pass rules, fixed before R1

These are written into `R1-measure.decisions.toml` (generalWork) before any cluster data exist.

* `tests-pass`: every test passes. A test that was skipped, or missing (the suite runs fewer than the
  67 expected, the count without the sanitizer test; 68 with it), counts as failed. GO means all passed; anything else is NO-GO and the analyst classes
  each failure as a *test assumption* (the test expects 4090 behaviour) or a *library bug*.
* `hashes-match`: every committed hash is reproduced. A mismatch is a bug, never noise.
* H200 runs a smoke subset (`output_hashes`, three contract suites, two host suites): `smoke-pass` and
  `smoke-hashes-match`.

The experiment `tests.suite` builds the library in a scratch directory of the job (not copied back),
runs `tools/test.sh` with JUnit output, and `bench/campaign/summarize_tests.py` turns the JUnit files
and the hash results into the record the decisions read. Rerun logs (full output and sanitizer) are
copied back when something failed.

## Limits of the hash gate (from the independent review)

* A hash whose `source` is `rtx4090` is the 4090 library's own output, tied to the oracle only through the small shapes: a bug that was already in the 4090 library when the hash was taken is baked
  in. No independent accuracy bound is checked on the mid and large cases yet (a follow-up: an error bound against an FP64 GEMM).
* The cases use finite inputs, one seed and one truncation case. NaN, Inf, subnormals, zero rows and the flagged-output (K5 native) paths are not covered.
* A deliberate change of the arithmetic needs the hashes regenerated on the 4090 (`--regen`), which is awkward for Hopper-only work: say so in the pull request.
* A mismatch of a large hash on Hopper or Blackwell fails `output_hashes_large`; the failure message says how to tell a library bug from a bad baseline (compare with the 4090 library run and the oracle on a small shape).
* `tests.suite` expects every hash case: `summarize_tests.py --expect-hashes N` makes a suite that died halfway, or a case that raised, count as a mismatch, not as a smaller denominator.
* The harness loads libcudart 13, then 12, then the unversioned name (a CUDA 12 node has no `.so.13`); the allocation shim takes its soname from the toolkit CMake found.

## Not done here

* R1 itself (needs the cluster). R2 reruns a subset after kernel changes, R3 the full suite.
* A hash list measured on Hopper or Blackwell: the committed hashes are the contract those GPUs are
  checked against, not something they define.
