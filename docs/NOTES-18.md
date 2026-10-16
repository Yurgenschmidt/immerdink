# Issue 18: device class, per-GPU tables and plan status

Status: **implemented** (branch `port/18-device-class`), after the independent review of 2026-10-15 (see "Review outcome"; "Results" at the end). Normative text: DESIGN §6h; user-facing: SUPPORT.md. Written for a reader new to the project.

## The problem in plain words

immerdink was tuned on one card, the RTX 4090. Three things inside it silently assume that card:

1. **The planner reads one number from the GPU: its SM count** (an SM is one compute unit; the 4090 has 128).
   Everything else, such as which of the four K4 tile shapes is fastest and what each costs, comes
   from constants fitted on the 4090 (`src/host/k4_config.hpp`, `kK4Costs`; the performance model's
   shipped calibration `src/host/calib/rtx4090_sm89.calib`).
2. **The binary is built for sm_89 only** (CMake default `89`). CMake then also embeds portable code
   (PTX), which the driver compiles again for a different GPU the first time a kernel runs. On an
   H100 (sm_90) the library therefore "works", but nobody is told that it runs recompiled code and
   4090 tuning.
3. **A GPU the library cannot run on fails late**: at the first `imd_contract` call, with a CUDA
   error, not when the plan is created.

The cluster cards are the H100 and H200 (sm_90, 132 SMs), and the RTX PRO 6000 (sm_120, 188 SMs).
Later cards (RTX 5090: also sm_120, 170 SMs) must be addable by adding table rows, not by editing code.

## Decisions already taken (owner, 2026-10-11, tracking issue 14)

* An unknown GPU falls back to the table of its **architecture generation**, else sm_89. The plan is
  flagged **approximate**. Recompiling at first use is allowed. Native FP64 is never a fallback
  (cuTENSOR covers that).
* wgmma code (Hopper's matrix instruction) runs only on compute capability 9.0.
* The support matrix gets each card's crossover between emulation and native FP64.

## Facts measured on the dev box (2026-10-15, CUDA 13.4 toolkit, driver 595.99, RTX 4090)

A one-kernel probe, `cudaFuncGetAttributes` after building for one target:

| built for | result | `binaryVersion` / `ptxVersion` |
|---|---|---|
| sm_89 (native) | ok | 89 / 89 |
| sm_80 or sm_86 SASS | ok, runs on the 4090 | 80 / 80 and 86 / 86 |
| sm_90 SASS only | error 209 "no kernel image is available" | - |
| compute_80 PTX only | error 222 "PTX compiled with an unsupported toolchain" | - |

The last row is the driver being older than the toolkit (it cannot read PTX written by nvcc 13.4).
It shows that "PTX only" is not a safe fallback in general. To see what a successful recompile looks
like I loaded hand-edited PTX (ISA version lowered) through the driver API: the function reports
**binary 89, ptx 80**. So a kernel that was recompiled is recognised by `binaryVersion != ptxVersion`.
The error path of the attribute call is also the earliest point at which a missing kernel image shows
(CUDA lazy module loading means nothing fails at library load).

## Proposed design

### D1. The device class (host library, CUDA-free)

```c
typedef struct {
  uint32_t struct_size;
  int32_t  cc_major, cc_minor;     /* compute capability                              */
  int32_t  sm_count;
  int32_t  reserved;
  uint64_t l2_bytes;
  uint64_t smem_optin_bytes;       /* largest dynamic shared memory per block          */
  char     name[64];
} imd_device_class_t;
```

Read per plan creation, for the *current* device (`cudaGetDevice`, as `plan_create` already does for the SM count), with `cudaDeviceGetAttribute` (not `cudaGetDeviceProperties`: slow, layout changes between toolkits), cached per device id behind a mutex. The handle's stored device is not used. The host library never touches
CUDA; it takes a class as an argument, so every rule can be tested with synthetic cards.

### D2. Tables are rows in a registry

A *table set* is what is fitted per architecture: the K4 cost table and margin, and the performance
calibration. A registry row says which cards a set was fitted on:

`{ table_id, cc (major*10+minor), sm_count (0 = any), name_prefix, K4 costs, calibration }`.

Today there is one row: `{sm89_4090, 89, 128, "NVIDIA GeForce RTX 4090"}`. The rows for sm_90 (#20, #23),
sm_120 (#24) are added by those issues. Until then a Hopper or Blackwell card gets the sm89 set. No row
may be added without the data that fitted it (the row names its calibration file and run).

### D3. Selection rule (pure function `imd_device_select(class) -> selection`)

Evaluated in this order, first match wins:

1. `cc < 8.0`: **unsupported** (the kernels need `cp.async` and INT8 `mma.sync` m16n8k32).
2. A row with the same `cc`, the same `sm_count` (or row value 0) and a matching name prefix:
   match `EXACT`, tables **tuned**.
3. A row with the same `cc` (same architecture, another card, e.g. RTX 6000 Ada on sm_89, RTX 5090 on
   sm_120): match `ARCH`, approximate.
4. A row with the same `cc_major`: the row with the nearest `cc_minor`, ties to the lower one
   ("its architecture generation", e.g. sm_86 uses sm_89): match `GENERATION`, approximate.
5. Otherwise the sm_89 row: match `DEFAULT`, approximate.

The SM count is always the card's own: the K4 rule takes it as an input today, so an approximate
set still adapts the tile choice to 132 or 188 SMs; only the cost constants are borrowed.

### D4. Kernel load check at plan creation

For every kernel the plan will launch (the plan already lists its launches; see `kernel_launches` in
`imd_lowering_t`), call `cudaFuncGetAttributes` while creating the plan. This makes the plan
fail early, forces lazy loading, and keeps `imd_contract` free of first-call surprises (it must stay
graph-capture safe). Per kernel, with `dev = 10*cc_major + cc_minor`:

* call fails: **unsupported**, `imd_plan_create` returns `IMD_STATUS_NOT_SUPPORTED` and the reason carries
  the CUDA error (209 no image, 222 PTX too new, ...).
* `binary == ptx == dev`: `NATIVE` code.
* `binary == ptx < dev`, same major: `COMPATIBLE`, prebuilt code of an older minor version
  (sm_86 code on sm_89).
* `binary != ptx`: `RECOMPILED` (the driver compiled PTX at first use).

The plan's binary kind is the worst over its kernels (`RECOMPILED` > `COMPATIBLE` > `NATIVE`).

### D5. Plan status

One value, in this precedence:

| status | when |
|---|---|
| `UNSUPPORTED` | D3 step 1, or a kernel failed D4. No plan is created. |
| `RECOMPILED` | some kernel was compiled from PTX at plan creation |
| `APPROXIMATE` | tables are not `EXACT` (D3 steps 3-5), or the code is `COMPATIBLE` |
| `NATIVE` | `EXACT` tables and `NATIVE` code |

`imd_plan_info(plan, imd_plan_info_t*)` (versioned by `struct_size` like the other structs) returns the
status, the binary kind, the match level, the table id and the device class the plan was made for.
`imd_device_info(handle, ...)` gives the same without a plan (used by the cluster jobs to print each
card's properties once). Python: `Plan.status`, `Plan.info`.

### D6. Fixed limits new configurations must fit

Recorded as constants with `static_assert`s and one test, and in the support matrix:
4 K4 configurations (`IMD_K4_CONFIG_COUNT`) and a 256-byte parameter block (the K4 launch parameter
struct). #22 (wgmma) must add its configurations inside these limits or change them together with the
stored plan layout.

### D7. Multi-arch build

CMake option `IMMERDINK_CUDA_ARCHS` (a list, default `89`; test.sh already passes
`CMAKE_CUDA_ARCHITECTURES`) plus `IMMERDINK_EMBED_PTX` (default `ON`, adds the PTX of the lowest
listed architecture). The release recipe is `89-real;90-real;120-real` plus PTX, so every shipped card
runs `NATIVE` code. (sm_90a / sm_120a variants are a #22 matter and only for their own kernels.)

### D8. Support matrix (`docs/SUPPORT.md`)

One table per card: compute capability, SMs, L2, status the library reports today, which K4 path, which
table set, and the measured crossover between emulation and native FP64 at DGEMM-level accuracy. The
crossover numbers come from the issue 11 runs (cited in the file); #21 supplies updates.

## Tests ("done when" of the issue)

1. **Selection model** `spec/devices.py` (Python, exact reference) and the C++ rule cross-checked on a
   grid of synthetic cards: every cc from 3.0 to 12.1, sm counts {1, 46, 82, 108, 128, 132, 142, 170, 188}, the
   real names of the shipped cards and strings that are not: the function is total, each card gets the
   documented match level, `cc < 8.0` is the only `UNSUPPORTED`, a synthetic RTX 5090 (12.0, 170 SMs)
   gets the sm_120 row if one exists else sm89, flagged approximate. Negative cases: a mutated rule
   (swapped order of steps 2 and 3, ties upward, name ignored) must fail the grid.
2. **Kernel classification** (pure function of `(error, binary, ptx, dev)`), exhaustively over the
   observed and invented values, including the measured rows above.
3. **Three 4090 builds**: native `89` gives `NATIVE`; foreign `90`-only gives plan creation
   `NOT_SUPPORTED` (error 209); PTX-only `compute_80` gives `NOT_SUPPORTED` (error 222 on this driver).
   `RECOMPILED` cannot be produced on this box with a real build (driver older than toolkit); it is
   covered by test 2 plus the driver-API probe (binary 89, ptx 80) recorded above.
   Every one of them loads (or fails to load) every kernel of the plan at plan creation.
4. **4090 output hashes unchanged** (`tests/data/output_hashes.json`, the benchmark `c_hash`es) and time
   within max(2%, 2 x spread): the check adds one `cudaFuncGetAttributes` per kernel at plan creation only.
5. Host tests with simulated SM counts 132 and 188 (already in the suite after issue 19).

## Questions for the reviewer

1. Is the precedence in D5 right, in particular `COMPATIBLE` code counting as approximate, and
   `RECOMPILED` outranking `APPROXIMATE`?
2. Is step 4 of D3 (nearest minor in the same major) the right reading of "table of its architecture
   generation"? Should sm_90 fall back to sm_89 (same "generation" word, different major) or only to
   `DEFAULT`, which is also sm_89 today? (Same answer now; they differ once a row for sm_80 or sm_100 exists.)
3. Should an unsupported device fail `imd_create` (early) instead of `imd_plan_create`? `imd_create`
   knows the device but not the kernels the plan needs.
4. D4 forces the load of kernels in `imd_plan_create`. Is there any reason this breaks graph capture,
   multi-threaded plan creation, or `compute-sanitizer` runs?
5. Anything in the "done when" list not covered?

## Review outcome (independent reviewer, 2026-10-15) and what changed

The reviewer read the code. Verdicts on the five questions, and the changes they cause:

1. **D5: agree with change.** The status mixes two axes. The plan info now carries them separately,
   `table_match` (EXACT / ARCH / GENERATION / DEFAULT) and `code_kind` (NATIVE / COMPATIBLE /
   RECOMPILED), and `status` is only their summary (precedence of D5 unchanged). A 4090 running a
   PTX-only build is EXACT tables with RECOMPILED code.
2. **D3 step 4: agree.** sm_90 does not use "generation" to reach sm_89 (Hopper and Ada are different
   generations): it gets `DEFAULT`, reported as such. Steps 4 and 5 stay distinguishable.
3. **Fail `imd_create`: disagree, accepted.** `imd_handle_st::device` is never read; `plan_create` asks
   `cudaGetDevice` itself, so the handle's device and the plan's can differ. The authoritative
   classification is per plan, on the current device. `imd_create` only rejects `cc < 8.0` early as a hint.
4. **D4 safety: agree with change.** Plan creation already allocates and creates events, so it is not
   capture-safe today, and tests/test_graph.cpp creates plans before capturing. D4 adds no hazard and
   removes one (a lazy first-load inside a capture). Added tests: the first contraction of a fresh plan
   captured in a graph while another thread creates plans; the negative builds are excluded from the
   compute-sanitizer registration. With `CUDA_MODULE_LOADING=EAGER` the error appears at context
   creation, so errors 209 and 222 are mapped from `cudaGetDevice` / `cudaDeviceGetAttribute` too.
   Recompiling many CUTLASS kernels at first use can take seconds: a latency note in SUPPORT.md.

Defects found, with the fix:

* **"The plan already lists its launches" was false** (`kernel_launches` is a count). The kernel set
  depends on path, N, W, fused flags, K1 and K5 variants, and the K5 variant is chosen at contract
  time. Fix: check a **superset**. `kernel_occupancy` / `kernel_occupancy_w` (src/contract.cu) already
  call `cudaFuncGetAttributes` for every `IMD_PERF_KERNEL_*` and both W instantiations. The first plan
  creation on a device runs them all once, classifies every kernel, and caches the result for the
  device. A source check (`tests/check_kernel_sites.py`) confirms every `<<<` site in src/ launches a
  kernel that the occupancy table covers. K4 is checked per configuration in `k4_prepare`.
* **A foreign architecture already fails at plan creation for emulated plans** (`k4_prepare` ->
  `cudaFuncSetAttribute` returns `IMD_STATUS_CUDA_ERROR`), and only the non-emulated paths fail late.
  The change is explicit: both become `IMD_STATUS_NOT_SUPPORTED`, with the CUDA error in the reason.
  `k4_tune` must pass the classification through (it aborts on any failure other than NOT_SUPPORTED).
* **The reason had no channel.** New: `imd_device_info(handle, imd_device_info_t*)` (in immerdink.h) returns
  the device class, the selection, the code kind, the failing kernel id and the CUDA error of the last
  classification. Test (b) asserts the error code through it.
* **`imd_lowering` cannot select tables.** Host API 13 -> 14: `imd_lowering_t` gets a `table_id` input
  (0 = the reference table) next to `sm_count`, and `k4_select` takes the table. The perf model keeps
  using the SM count of its calibration; a calibration is per card (#23), so for a borrowed table the
  model's SM count is the donor's and the plan is flagged approximate. D3's sentence "the SM count is
  always the card's own" is true for the K4 rule, not for `imd_perf_*`.
* **Test (b) depends on the driver.** On a driver as new as the toolkit a PTX-only build is `RECOMPILED`,
  not unsupported. The test accepts the set {RECOMPILED, NOT_SUPPORTED with error 222} and prints which it
  got; the support matrix says "PTX-only builds need a driver at least as new as the toolkit".
* **D3 never used `smem_optin_bytes`.** It is informational. The plan-time check (occupancy of the K4
  configuration below 1 is already `NOT_SUPPORTED`) is the authority.
* **The 256-byte limit has no CUDA-free test.** `kK4BlobBytes` lives in src/k4/k4_ops.hpp and is already
  `static_assert`ed per configuration. The host test pins `IMD_K4_CONFIG_COUNT`; the 256 is recorded in
  the docs and checked by a source test (`tests/check_kernel_sites.py` greps the constant).
* **Struct hygiene.** Explicit sizes and `offsetof` `static_assert`s as in host_api.cpp:53-59; bump
  `IMD_HOST_API_VERSION`; Python binding rows; `imd_plan_info` and `imd_device_info` live in immerdink.h
  because they need a handle or plan.
* **D7.** `89` already embeds SASS and PTX. The release recipe `89-real;90-real;120-real` plus PTX of the
  lowest architecture still makes sm_86 run recompiled code unless sm_80 SASS is added. Documented. Build
  time and size grow by about three times for the four CUTLASS configurations.
* **D8.** Cards with no measurement get the cell "not measured", never an extrapolation.
* **`ctas_per_sm` in `K4ConfigDesc` is a 4090 assumption.** The table (`K4Table`) a registry row points
  to carries its own `ctas_per_sm[4]`, overriding the descriptor for the cost model.
* **(c) reading.** Agreed: with no sm_120 row the 5090 gets `DEFAULT` (sm_89), flagged approximate. The
  registry is replaceable in the pure function (a pointer to rows), and a test injects a synthetic sm_120
  row and asserts `ARCH` for the 5090 and `EXACT` for a card matching it.

## Results (RTX 4090, 2026-10-15)

* **Rule and classification.** `tests/test_devices.py`: 63 756 synthetic cards over six registries
  (built-in, plus a sm_120 row, two sm_89 rows, an Ampere+Hopper registry, wildcards, ties, a registry
  without sm_89), 12 named cards, 2000 kernel classifications, invalid arguments, and eight mutants of the rule
  (min cc 9.0, name ignored, SM count ignored, arch before exact, generation across majors, ties upward,
  compatible code across majors, approximate outranking recompiled), every one killed by the grid. A synthetic
  RTX 5090 (12.0, 170 SMs) gets `DEFAULT` and `approximate`; with an injected sm_120 row, `ARCH`; the RTX PRO 6000
  with that row `EXACT`.
* **Status of this card.** `NVIDIA GeForce RTX 4090`, sm_89, 128 SMs: tables `exact`, code `native`, status
  `native`; 110 kernels checked (99 with a perf id, 11 more) in 75-130 ms on the first call, microseconds after.
  Same with `CUDA_MODULE_LOADING=EAGER`.
* **Three builds** (`tools/device_status_check.sh all`; log kept in the PR): native (`89-real`) `native`;
  PTX only (`80-virtual`) `unsupported`, CUDA error 222 (the driver cannot compile PTX written by toolkit 13.4;
  on a driver as new as the toolkit it would be `recompiled`, which this machine cannot show); foreign
  architecture (`90-real`) `unsupported`, CUDA error 209. In both, `imd_plan_create` returns `NOT_SUPPORTED`
  and `Plan(...)` raises `NotSupportedError`; `imd_device_info` names the failing kernel and the error.
* **Not shown here.** A real `recompiled` status on a card (needs a driver newer than the toolkit; the
  driver-API probe above gives binary 89, ptx 80 for hand-lowered PTX and the classification test covers that
  row), and anything on Hopper or Blackwell (R1 prints each card's status; the cluster build recipe is in the
  SUPPORT.md table).
* **4090 results unchanged.** The whole suite is bit-exact against the oracle (68 tests); plan creation is the
  only code path that changed (one cached call on the first plan of a device). The benchmark output hashes
  and the 2%-or-2x-spread timing gate were not rerun for this change.
* **Reviewer points left open:** a sm_120 / sm_90 table row (issues 20, 23, 24); `imd_create`'s own early
  `cc < 8.0` hint was not added (the plan-time check is authoritative); the first plan on a `recompiled` device
  compiles the whole library, which can take seconds (SUPPORT.md says so).
