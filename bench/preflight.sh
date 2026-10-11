#!/usr/bin/env bash
# Check that this machine can build and run the metrics (bench/run_metrics.sh) and say what to fix.
#   bench/preflight.sh            (run it on the GPU node; the git/CUTLASS check wants network, so
#                                  run `git submodule update --init --depth 1 third_party/cutlass`
#                                  on a login node first if compute nodes are offline)
# Exit status: 0 = ready (warnings allowed), 1 = something blocks the run.
here="$(cd "$(dirname "$0")" && pwd)"; root="$(dirname "$here")"
fail=0
ok()   { printf '  ok    %s\n' "$*"; }
warn() { printf '  WARN  %s\n' "$*"; }
bad()  { printf '  FAIL  %s\n' "$*"; fail=1; }

echo "toolchain:"
env_dir="${IMMERDINK_ENV:-$HOME/miniforge3/envs/immerdink}"
if [ -x "$env_dir/bin/nvcc" ]; then nvcc="$env_dir/bin/nvcc"; else nvcc="$(command -v nvcc || true)"; fi
[ -z "$nvcc" ] && [ -x "${CUDA_HOME:-/usr/local/cuda}/bin/nvcc" ] && nvcc="${CUDA_HOME:-/usr/local/cuda}/bin/nvcc"
if [ -n "$nvcc" ]; then
  v="$("$nvcc" --version | sed -n 's/.*release \([0-9.]*\).*/\1/p')"
  ok "nvcc $v ($nvcc)"
  awk -v v="$v" 'BEGIN{exit !(v+0 < 12.8)}' && bad "nvcc $v is too old for Blackwell (sm_120 needs >= 12.8; 13.x preferred): module load a newer cuda"
else bad "no nvcc: module load cuda (or set CUDA_HOME / IMMERDINK_ENV)"; fi
if command -v cmake >/dev/null; then
  cv="$(cmake --version | sed -n '1s/.* //p')"; ok "cmake $cv"
  [ "$(printf '3.28\n%s\n' "$cv" | sort -V | head -1)" = 3.28 ] || bad "cmake $cv < 3.28: module load cmake, or pip install --user cmake"
else bad "no cmake (module load cmake, or pip install --user cmake)"; fi
command -v ninja >/dev/null && ok "ninja" || warn "no ninja: make will be used (slower; pip install --user ninja)"
if command -v g++ >/dev/null; then
  gv="$(g++ -dumpversion)"; ok "g++ $gv"
  [ "${gv%%.*}" -ge 11 ] 2>/dev/null || bad "g++ $gv < 11 (C++20 needed): module load gcc"
else bad "no g++ (module load gcc)"; fi
command -v python3 >/dev/null && ok "python3 (summary tables)" || warn "no python3: summary.md will be skipped"
if [ -f "$root/third_party/cutlass/include/cutlass/cutlass.h" ] || [ -f "${IMMERDINK_CUTLASS_DIR:-/nonexistent}/include/cutlass/cutlass.h" ]; then ok "CUTLASS present"
else warn "CUTLASS missing: bench/build.sh will fetch it (needs network); or run git submodule update --init --depth 1 third_party/cutlass"; fi

ct_hdr=""
for d in "${CUTENSOR_ROOT:-}" "${CUTENSOR_DIR:-}" "${CUTENSOR_HOME:-}" "${CUTENSOR_PATH:-}" "${CUDA_HOME:-}" "${CUDA_PATH:-}" "$env_dir" "${CONDA_PREFIX:-}" /usr/local/cuda /usr /usr/local; do
  [ -n "$d" ] || continue
  for sub in include include/libcutensor libcutensor/include; do
    [ -f "$d/$sub/cutensor.h" ] && { ct_hdr="$d/$sub/cutensor.h"; break 2; }
  done
done
if [ -n "$ct_hdr" ]; then
  tv=""
  for h in "$ct_hdr" "$(dirname "$ct_hdr")/cutensor/types.h"; do
    [ -f "$h" ] || continue
    ma="$(sed -n 's/^#define CUTENSOR_MAJOR[[:space:]]*\([0-9]*\).*/\1/p' "$h" | head -1)"
    mi="$(sed -n 's/^#define CUTENSOR_MINOR[[:space:]]*\([0-9]*\).*/\1/p' "$h" | head -1)"
    [ -n "$ma" ] && { tv="$ma.${mi:-0}"; break; }
  done
  ok "cuTENSOR ${tv:-?} ($ct_hdr): the cutensor / cutensor-emu baselines will be built"
  if [ -n "$tv" ]; then
    [ "$(printf '2.4\n%s\n' "$tv" | sort -V | head -1)" = 2.4 ] \
      || warn "cuTENSOR $tv < 2.4 has no emulated FP64 (CUTENSOR_COMPUTE_DESC_8XINT8): the cutensor-emu records will be errors"
    [ "$(printf '2.6\n%s\n' "$tv" | sort -V | head -1)" = 2.6 ] \
      || warn "cuTENSOR $tv < 2.6: emulation is not guaranteed to fall back to native FP64 where it does not apply"
  fi
else
  warn "cuTENSOR not found: the cuTENSOR baselines are skipped (module load cutensor, or export CUTENSOR_ROOT=<dir with include/cutensor.h and lib/libcutensor.so>, then rebuild in a fresh IMMERDINK_BENCH_BUILD)"
fi

echo "comparison baselines (bench/sweep_compare.txt; each one optional, a missing one records an error):"
cb_hdr=""
if [ -n "$nvcc" ]; then
  nroot="$(cd "$(dirname "$nvcc")/.." && pwd)"
  for d in "$nroot/include" "$nroot/targets/x86_64-linux/include" "$nroot/targets/sbsa-linux/include"; do
    [ -f "$d/cublas_api.h" ] && { cb_hdr="$d/cublas_api.h"; break; }
  done
fi
if [ -z "$cb_hdr" ]; then warn "cublas_api.h not found next to nvcc: cannot tell whether cuBLAS has the FP64 emulation API (the build tests it)"
elif grep -q CUBLAS_COMPUTE_64F_EMULATED_FIXEDPOINT "$cb_hdr"; then ok "cuBLAS FP64 emulation API ($cb_hdr): the emu baseline will be built"
else warn "cuBLAS in $(dirname "$cb_hdr") has no FP64 emulation API (CUDA 13 needed): the emu baseline records 'needs CUDA 13 cuBLAS'; cuTENSOR's emulation may not engage either"; fi
if [ -n "${GEMMUL8_DIR:-}" ] && [ -f "$GEMMUL8_DIR/lib/libgemmul8.a" ] && [ -f "$GEMMUL8_DIR/include/gemmul8.hpp" ]; then
  ok "GEMMul8 ($GEMMUL8_DIR): the gemmul8 baseline will be built; pinned: $(grep -v '^#' "$here/GEMMUL8_PIN" | head -1)"
else warn "GEMMUL8_DIR not set (or incomplete): the gemmul8 baseline is compiled out; bench/build_gemmul8.sh <GEMMul8 source at the pinned commit> <arch> <dir> builds it"; fi

echo "GPU:"
if command -v nvidia-smi >/dev/null && nvidia-smi -L >/dev/null 2>&1; then
  nvidia-smi --query-gpu=index,name,compute_cap,driver_version,power.limit,clocks.max.sm,persistence_mode --format=csv,noheader | sed 's/^/  gpu   /'
  [ "$(nvidia-smi -L | wc -l)" -gt 1 ] && warn "several GPUs: the benchmark uses device 0; pin it with CUDA_VISIBLE_DEVICES=<n> (and do the same for the quiet check)"
  e="$(nvidia-smi --query-gpu=power.draw --format=csv,noheader 2>/dev/null | head -1)"; ok "power readable ($e)"
  if nvidia-smi --query-gpu=clocks_event_reasons.sw_power_cap --format=csv,noheader >/dev/null 2>&1; then ok "clock event reasons readable"; else warn "clock event reasons not readable (older driver): records lose power-cap flags"; fi
else bad "nvidia-smi sees no GPU (are you on a GPU node? srun --gres=gpu:1 ...)"; fi

echo "energy counter (NVML):"
if ldconfig -p 2>/dev/null | grep -q libnvidia-ml.so.1 || [ -e /usr/lib64/libnvidia-ml.so.1 ] || [ -e /usr/lib/x86_64-linux-gnu/libnvidia-ml.so.1 ]; then ok "libnvidia-ml.so.1 found"
else warn "libnvidia-ml.so.1 not on the default path: the energy sweep dlopens it; add its directory to LD_LIBRARY_PATH"; fi
echo "  (the energy method checks the counter itself and stops with a clear error if it does not step)"

echo "calibration:"
echo "  bench/run_metrics.sh calibrates first (bench/calibrate.sh: sustained dense load until the clock and"
echo "  temperature are stable, steady clock/power, recommended lock); run it alone with: bench/calibrate.sh"
echo "clocks:"
lock="$(nvidia-smi --query-gpu=clocks.sm,clocks.max.sm --format=csv,noheader,nounits 2>/dev/null | head -1)"
echo "  sm / max sm MHz (idle): $lock"
if sudo -n true 2>/dev/null; then ok "sudo available: lock clocks with  sudo nvidia-smi -lgc <f>,<f>   (undo: -rgc)"
else warn "no passwordless sudo: ask the admins to lock clocks (nvidia-smi -lgc), or run unlocked and say so in the report"; fi

echo "machine quiet (the run itself waits for this):"
if "$root/tools/quiet_check.sh" >/tmp/imd-preflight-quiet.$$ 2>&1; then ok "quiet"
else warn "not quiet (see below); set QUIET_LOAD / QUIET_CPU / QUIET_GPU_IGNORE or run on an exclusive node"; sed 's/^/        /' /tmp/imd-preflight-quiet.$$ | tail -15; fi
rm -f /tmp/imd-preflight-quiet.$$
echo "other:"
[ -w "${XDG_RUNTIME_DIR:-/tmp}" ] && ok "lock directory writable" || bad "${XDG_RUNTIME_DIR:-/tmp} not writable (set XDG_RUNTIME_DIR / TLC_LOCK / IMMERDINK_GPU_LOCK)"
command -v flock >/dev/null && ok "flock" || bad "no flock (util-linux)"
command -v systemctl >/dev/null && systemctl --user show-environment >/dev/null 2>&1 && ok "systemd user manager (memory cap active)" || warn "no systemd user manager: runs uncapped (fine on an HPC node with its own memory limit)"

echo
if [ $fail = 0 ]; then echo "READY: bench/run_metrics.sh <outdir outside the repo>"; else echo "NOT READY: fix the FAIL lines"; fi
exit $fail
