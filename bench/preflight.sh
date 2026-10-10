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
