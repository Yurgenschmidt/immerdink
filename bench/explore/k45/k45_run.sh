#!/bin/bash
# K45 single-warp CRT gate + G2b native comparators (immerdink-eval analysis/20261110-dataflow/plan.md §6), phases of ONE H200
# embers batch job (k45_job.sh runs: build gate ncugate g2b probe imdb g2b2). Each phase writes $O/<phase>/step.log.
#   build    nvcc k45_gate.cu (verbatim immerdink K5 'k5fin' code from dev 91a5cbe, see K5_SOURCE) and g2b_bench.cu, ptxas -v logged
#   gate     k45_gate timing: ref k5f_crt_n<14,4> (256 thr, today's grid) + single-warp gates E1 E2 E2pf E4 E4pf E4x2 at r_lowk and
#            t_df, interleaved; bit-exact vs ref and ref vs exact host CRT; pad 120 KiB (L1 ~100 KiB) and pad 200 KiB (L1 ~24 KiB)
#   ncugate  ncu of the same kernels (one launch each per shape, 14 kernels): instructions, cycles, issue, warp stalls
#   g2b      g2b_bench round 1 (dsyrk, dsyrk+mirror, dgemm, emu55, emumath, cutensor; R1 permuted) + emulation-routing check
#   probe    ncu kernel-name listing: which kernels do cublasDsyrk / Dsyrkx / Dgemm run on a FP64_EMULATED_FIXEDPOINT_MATH handle
#   imdb     immerdink imd_bench (build90dev = dev 91a5cbe, if present) at t_df, r_lowk, R1: imd N=14,15, cutensor, emu fix55, dgemm
#   g2b2     g2b_bench round 2 (after imdb: exposes clock / thermal drift between rounds)
# No mail options. Writes only under $HOME/scratch/claude-imd/oz/ and $TMPDIR.
set -u
ph="${1:?phase}"
E=$HOME/scratch/miniforge3/envs/immerdink-cuda13
S=$HOME/scratch/claude-imd
W=$HOME/scratch/claude-imd/oz/srck45     # git worktree of explore/k45-gate (k45_job.sh checks it out from the inbox bundle)
IN=$W/bench/explore/k45
D=$S/oz
B=$D/k45build
O=$D/out-h200-k45/$ph
mkdir -p "$O" "$B"
exec > "$O/step.log" 2>&1
echo "phase $ph host $(hostname) job ${SLURM_JOB_ID:-?} start $(date -Is)"
nvidia-smi --query-gpu=name,clocks.sm,clocks.max.sm,clocks.mem,temperature.gpu,power.draw,driver_version --format=csv,noheader
export HPC_ENV=$E
source $HOME/repos/generalWork/phoenix/immerdink/hpc_common.sh
hpc_toolchain_setup || { echo "ERROR: toolchain"; exit 4; }
export TMPDIR=${TMPDIR:-/tmp}
export CUDA_CACHE_PATH=$TMPDIR/nvcache IMMERDINK_GPU_LOCK=none
export XDG_RUNTIME_DIR=$TMPDIR/xdg; mkdir -p "$XDG_RUNTIME_DIR"
CTR="$(find_cutensor 2>/dev/null)" || CTR=$E
[ -n "$CTR" ] || CTR=$E
CTL="$(_cutensor_lib "$CTR")" || CTL=$E/lib
export LD_LIBRARY_PATH=$CTL:$E/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
find_ncu() {
  for c in "$E/bin/ncu" $(ls -d "$E"/nsight-compute*/ncu "$E"/pkgs/nsight-compute*/ncu 2>/dev/null) /usr/local/cuda/bin/ncu /usr/local/cuda-*/bin/ncu /opt/nvidia/nsight-compute/*/ncu; do
    [ -x "$c" ] && { echo "$c"; return 0; }; done
  _hpc_module_init >/dev/null 2>&1
  ( module load cuda >/dev/null 2>&1 && command -v ncu ) && return 0
  ( module load nvhpc >/dev/null 2>&1 && command -v ncu ) && return 0
  return 1
}
case "$ph" in
build)
  echo "nvcc $(command -v nvcc): $(nvcc --version | tail -1); cutensor root $CTR lib $CTL"
  cat "$IN/K5_SOURCE"; echo "worktree at $(git -C "$W" rev-parse --short HEAD)"
  rm -f "$B/k45_gate" "$B/g2b_bench"   # no stale binary can pass the check below
  # gate: default cap 96 registers (__maxnreg__); the ptxas table goes to ptxas-gate.txt
  nvcc -std=c++17 -O3 -DNDEBUG --expt-relaxed-constexpr -diag-suppress=20012 -arch=sm_90 -I"$W/include" -I"$W/src" \
    -Xptxas -v "$IN/k45_gate.cu" -o "$B/k45_gate" > "$O/ptxas-gate.txt" 2>&1
  rc1=$?; echo "k45_gate nvcc exit $rc1"; grep -E "error|registers|spill" "$O/ptxas-gate.txt" | head -30
  nvcc -std=c++17 -O3 -arch=sm_90 -I"$CTR/include" -I"$E/include" -Xptxas -v "$IN/g2b_bench.cu" -o "$B/g2b_bench" \
    -L"$CTL" -L"$E/lib" -lcublas -lcutensor -Wl,-rpath,"$CTL" -Wl,-rpath,"$E/lib" > "$O/ptxas-g2b.txt" 2>&1
  rc2=$?; echo "g2b_bench nvcc exit $rc2"; grep -E "error|registers|spill" "$O/ptxas-g2b.txt" | head -20
  [ $rc1 = 0 ] && [ $rc2 = 0 ] && [ -x "$B/k45_gate" ] && [ -x "$B/g2b_bench" ] || { echo "ERROR: build"; exit 3; } ;;
gate)
  for pad in 120 200; do
    timeout 140 "$B/k45_gate" --mode all --reps 51 --pad-kb $pad --shapes r_lowk,t_df --out "$O/gate-pad$pad.jsonl" > "$O/gate-pad$pad.log" 2>&1
    echo "gate pad $pad exit $?"; grep -E "mismatch|distinct|validation|ERROR|CUDA" "$O/gate-pad$pad.log" | head -40
  done ;;
ncugate)
  NCU="$(find_ncu)" || { echo "ERROR: no ncu"; exit 5; }
  echo "ncu $NCU: $("$NCU" --version | tail -1)"
  M=smsp__inst_executed.sum,sm__inst_executed.sum,smsp__cycles_active.avg,smsp__cycles_elapsed.avg,sm__cycles_elapsed.avg,sm__cycles_elapsed.avg.per_second,smsp__issue_active.avg.pct_of_peak_sustained_active,smsp__issue_active.sum,smsp__warps_active.avg.per_cycle_active,launch__registers_per_thread,launch__grid_size,launch__block_size,gpu__time_duration.sum,dram__bytes_read.sum,dram__bytes_write.sum,lts__t_sectors_srcunit_tex_op_read.sum,l1tex__t_sector_hit_rate.pct
  # -k: only the ref and the gates (fill_W / cmp_bits excluded); 7 kernels x 2 shapes = 14 launches; clocks unlocked like the timing run
  timeout 660 "$NCU" -k 'regex:k5f_crt_n|k45g_crt_n' -c 14 --clock-control none --section WarpStateStats --section SourceCounters \
    --section LaunchStats --section Occupancy --metrics "$M" --import-source no -f -o "$O/gate" \
    "$B/k45_gate" --mode ncu --pad-kb 120 --shapes r_lowk,t_df > "$O/gate-ncu.log" 2>&1
  echo "ncu gate exit $?"
  "$NCU" -i "$(ls "$O"/gate.ncu-rep* | head -1)" --page raw --csv > "$O/gate.raw.csv" 2> "$O/gate.csv.err"
  echo "raw csv exit $? lines $(wc -l < "$O/gate.raw.csv")"
  "$NCU" -i "$(ls "$O"/gate.ncu-rep* | head -1)" --page details --section WarpStateStats --csv > "$O/gate.stalls.csv" 2>> "$O/gate.csv.err"
  echo "stalls csv exit $? lines $(wc -l < "$O/gate.stalls.csv")" ;;
g2b|g2b2)
  timeout 280 "$B/g2b_bench" --mode all --reps 41 --cases t_df,r_lowk,r1_q1024,r1_q2048 --out "$O/g2b.jsonl" > "$O/g2b.log" 2>&1
  echo "g2b exit $?"; grep -E "routing|ERROR|CUDA|cuBLAS|cuTENSOR" "$O/g2b.log" | head -20 ;;
probe)
  NCU="$(find_ncu)" || { echo "ERROR: no ncu"; exit 5; }
  timeout 220 "$NCU" --metrics gpu__time_duration.sum -c 300 --csv --print-units base "$B/g2b_bench" --mode probe > "$O/probe.csv" 2> "$O/probe.err"
  echo "probe exit $? lines $(wc -l < "$O/probe.csv")"
  # one line per kernel: marker id / kernel name (markers separate the six calls)
  grep -E '"gpu__time_duration.sum"' "$O/probe.csv" | awk -F'","' '{print $5 " | " $NF}' | head -120 ;;
imdb)
  BIN=$D/build90dev/imd_bench
  [ -x "$BIN" ] || { echo "ERROR: no imd_bench at $BIN (phase skipped)"; exit 6; }
  [ -f "$D/build90dev.env" ] && source "$D/build90dev.env"
  common=(--methods imd,cutensor,emu,dgemm --N 14,15 --emu fix55 --phi 0.5 --reps 15 --warmup 2 --max-seconds 1 --samples 262144 --phase-reps 0 --seed 1)
  # specs as in bench/sweep_compare.txt at dev 91a5cbe (t_df's acbd output is the row-major 4096 x 4096 matrix)
  timeout 140 "$BIN" --name t_df --spec "acQ,bdQ->acbd" --ext "a=64,b=64,c=64,d=64,Q=2048" "${common[@]}" --out "$O/imdb.jsonl" > "$O/imdb-t_df.log" 2>&1
  echo "imd_bench t_df exit $?"
  timeout 140 "$BIN" --name r_lowk --spec "ik,jk->ij" --ext "i=8192,j=8192,k=1024" "${common[@]}" --out "$O/imdb.jsonl" > "$O/imdb-r_lowk.log" 2>&1
  echo "imd_bench r_lowk exit $?"
  for q in 1024 2048; do   # R1-like permuted output, a=b=128, i=j=32: the same contraction as g2b r1_q$q
    timeout 140 "$BIN" --name "r1_q$q" --spec "aiQ,bjQ->abij" --ext "a=128,b=128,i=32,j=32,Q=$q" "${common[@]}" --out "$O/imdb.jsonl" > "$O/imdb-r1_q$q.log" 2>&1
    echo "imd_bench r1_q$q exit $?"
  done ;;
*) echo "unknown phase $ph"; exit 2 ;;
esac
nvidia-smi --query-gpu=clocks.sm,temperature.gpu,power.draw --format=csv,noheader
echo "end $(date -Is)"
