#!/bin/bash
# explore/syrk-big: C = A A^T (SYRK) at large n on ONE H200, phases of one PACE embers batch job (syrk_job.sh runs
# build check bench). Two immerdink paths: dense imd (B = A, method imd) and the block-SYRK prototype (method imd_syrk:
# the packed API, lower block pairs + a mirror pass); baselines cuBLAS DSYRK (+ mirror), DGEMM and FP64 fixed-point
# emulation (fix55). Each phase writes $O/<phase>/step.log and its JSONL; syrk_job.sh prints the tables
# (bench/explore/syrk/syrk_tab.py) into $O/<phase>/tab.txt after each phase, so a phase cut by its timeout still has them.
#   build   bench/build.sh through pace_immerdink_build.sbatch (ARCH=90, profiling build, no GEMMul8) into build90syrk
#   check   correctness BEFORE any timing: whole-C bit identity of imd_syrk against dense immerdink (--N 8,14; both round to nearest) at
#           n = 4096 k = 2048 and at n not a multiple of the block (5000 x 1999 with block 2048, 4100 x 2048 with block
#           1024), both stream counts; the error of imd / imd_syrk / dsyrk against native DGEMM (max relative, normwise)
#   bench   the timing sweep; TFLOPS-eq uses the dense count 2 n^2 k for every method
# Budget (H200, model in the job notes): every timed configuration stops after 3 repetitions once 2 s are spent (10 at
# most), so the largest case (n = 65536, k = 32768) costs about 4 calls per method; the whole bench phase is ~8 GPU-minutes
# plus process setups. Writes only under $HOME/scratch/claude-imd/oz/ and $TMPDIR. No mail options.
set -u
ph="${1:?phase}"
E=$HOME/scratch/miniforge3/envs/immerdink-cuda13
S=$HOME/scratch/claude-imd
D=$S/oz
W=$D/srcsyrk                 # git worktree of explore/syrk-big (syrk_job.sh checks it out from the inbox bundle)
B=$D/build90syrk
O=$D/out-h200-syrk/$ph
mkdir -p "$O"
exec > "$O/step.log" 2>&1
echo "phase $ph host $(hostname) job ${SLURM_JOB_ID:-?} start $(date -Is) worktree $(git -C "$W" rev-parse --short HEAD)"
export TMPDIR=${TMPDIR:-/tmp}
export CUDA_CACHE_PATH=$TMPDIR/nvcache IMMERDINK_GPU_LOCK=none
export XDG_RUNTIME_DIR=$TMPDIR/xdg XDG_CACHE_HOME=$TMPDIR/xdgcache; mkdir -p "$XDG_RUNTIME_DIR" "$XDG_CACHE_HOME"
cd "$TMPDIR" || exit 3

if [ "$ph" = build ]; then
  CUT=$S/src/third_party/cutlass
  pin="$(git -C "$W" ls-tree HEAD third_party/cutlass | awk '{print $3}')"; have="$(git -C "$CUT" rev-parse HEAD 2>/dev/null)"
  echo "cutlass pin $pin, shared checkout $have"; [ "$pin" = "$have" ] || { echo "ERROR: CUTLASS pin mismatch"; exit 3; }
  rm -f "$B/imd_bench"   # no stale binary can pass the check below
  t0=$SECONDS
  ARCH=90 SRC_DIR="$W" CUTLASS_DIR="$CUT" BUILD_DIR="$B" BUILD_ENV_FILE="$B.env" GEMMUL8_SRC= HPC_ENV=$E JOBS=8 \
    bash $HOME/repos/generalWork/phoenix/immerdink/pace_immerdink_build.sbatch > "$O/build.log" 2>&1
  rc=$?; echo "build exit $rc in $((SECONDS - t0)) s"
  grep -E "IMMERDINK_PROFILE|IMMERDINK_K4_WGMMA|IMD_CUBLAS_HAS_EMULATION" "$B/CMakeCache.txt" 2>/dev/null
  [ $rc = 0 ] && [ -x "$B/imd_bench" ] || { tail -40 "$O/build.log"; echo "ERROR: build"; exit 3; }
  echo "end $(date -Is)"; exit 0
fi

export HPC_ENV=$E
source $HOME/repos/generalWork/phoenix/immerdink/hpc_common.sh
hpc_toolchain_setup || { echo "ERROR: toolchain"; exit 4; }
[ -f "$B.env" ] && source "$B.env"
export LD_LIBRARY_PATH=$E/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
BIN=$B/imd_bench
[ -x "$BIN" ] || { echo "ERROR: no $BIN"; exit 6; }
nvidia-smi --query-gpu=name,clocks.sm,clocks.max.sm,clocks.mem,temperature.gpu,power.draw,memory.total,driver_version --format=csv,noheader
J=$O/$ph.jsonl
rm -f "$J"   # imd_bench appends: a re-run phase must not double its records
# one imd_bench process per case: <timeout s> <name> <n> <k> <args...>
run() {
  local to=$1 nm=$2 n=$3 k=$4; shift 4
  local t0=$SECONDS
  timeout -k 20 "$to" "$BIN" --name "$nm" --spec "ik,jk->ij" --ext "i=$n,j=$n,k=$k" --self 1 --seed 1 --phi 0.5 --clock-probe 1 \
    --out "$J" "$@" > "$O/$nm.log" 2>&1
  local rc=$?
  echo "$nm (n=$n k=$k) exit $rc in $((SECONDS - t0)) s"
  grep -E "imd_syrk check|imd_syrk N=|setup failed|error|Error" "$O/$nm.log" | head -30
}

case "$ph" in
check)
  # --reps 2 --max-seconds 0.1: correctness is the point, timing is incidental; --accuracy 1: the full-C DGEMM comparison
  c=(--N 8,14 --syrk-check 1 --syrk-streams 1,2 --accuracy 1 --reps 2 --warmup 1 --max-seconds 0.1 --samples 4096 --phase-reps 1)
  run 200 chk_n4096 4096 2048 --methods imd,imd_syrk,dsyrk,dgemm --syrk-block 1024,2048 "${c[@]}"
  run 120 chk_n5000 5000 1999 --methods imd,imd_syrk --syrk-block 2048 "${c[@]}"
  run 120 chk_n4100 4100 2048 --methods imd_syrk --syrk-block 1024 "${c[@]}" ;;
bench)
  # Headline shape first (n = 32768, k = 16384); the lead's predicted TFLOPS-eq there: dense 185-215 (N=8) / 115-125 (N=14),
  # imd_syrk 330-380 / 200-230, native DSYRK <= ~110-120. Per-case estimated wall (H200 ~1.7-1.8 GHz) in the comments.
  t=(--reps 10 --warmup 1 --max-seconds 2 --samples 1024 --phase-reps 2 --N 8,14 --emu fix55)
  # n=32768 k=16384: imd 0.18/0.31 s, imd_syrk ~0.10/0.17 s x 12 configs, dgemm 0.6 s, dsyrk 0.3 s x 2, emu ~0.4 s: ~90 s
  run 420 b_n32768_k16384 32768 16384 --methods imd_syrk,imd,dsyrk,dgemm,emu --syrk-block 2048,4096,8192 --syrk-streams 1,2 \
    --emu-ws-mib 32768 "${t[@]}"
  # n=16384: small enough for the full-C DGEMM comparison at the bench scale (A + B + 2C <= 8.6 GB)
  run 240 b_n16384_k4096 16384 4096 --methods imd_syrk,imd,dsyrk,dgemm,emu --syrk-block 2048,4096 --syrk-streams 1,2 \
    --emu-ws-mib 16384 --accuracy 1 "${t[@]}"
  run 300 b_n16384_k16384 16384 16384 --methods imd_syrk,imd,dsyrk,dgemm,emu --syrk-block 2048,4096 --syrk-streams 1,2 \
    --emu-ws-mib 16384 --accuracy 1 "${t[@]}"
  run 300 b_n32768_k4096 32768 4096 --methods imd_syrk,imd,dsyrk,dgemm,emu --syrk-block 4096 --syrk-streams 1,2 \
    --emu-ws-mib 32768 "${t[@]}"
  # n=32768 k=32768: imd 0.36/0.62 s, syrk 0.2/0.34 s, dgemm 1.2 s: ~90 s
  run 420 b_n32768_k32768 32768 32768 --methods imd_syrk,imd,dsyrk,dgemm,emu --syrk-block 4096,8192 --syrk-streams 2 \
    --emu-ws-mib 32768 "${t[@]}"
  # n=65536 k=16384 (C = 34 GB): dense N=14 needs ~133 GB (may fail: recorded as an error); syrk 0.4/0.7 s, dgemm 2.4 s: ~120 s
  run 480 b_n65536_k16384 65536 16384 --methods imd_syrk,imd,dsyrk,dgemm,emu --syrk-block 4096,8192 --syrk-streams 2 \
    --emu-ws-mib 32768 "${t[@]}"
  # n=65536 k=32768 (A 17 GB, C 34 GB): dense imd does not fit at N=14 (~171 GB) and barely at N=8 (~120 GB): SYRK paths,
  # dsyrk, dgemm (4.7 s per call) and emu; ~150 s
  run 540 b_n65536_k32768 65536 32768 --methods imd_syrk,dsyrk,dgemm,emu,imd --syrk-block 8192 --syrk-streams 2 \
    --emu-ws-mib 40960 --reps 4 --warmup 1 --max-seconds 2 --samples 1024 --phase-reps 1 --N 8,14 --emu fix55 ;;
*) echo "unknown phase $ph"; exit 2 ;;
esac
nvidia-smi --query-gpu=clocks.sm,temperature.gpu,power.draw --format=csv,noheader
echo "end $(date -Is)"
