#!/usr/bin/env bash
# nsys profiles of a fixed list of short imd_bench runs, summarized per run: where the GPU time goes, and
# whether INT8 tensor-core GEMM kernels actually ran (decisive for cuTENSOR's FP64 emulation).
#   bench/profile_nsys.sh <output directory>          (bench/run_metrics.sh calls it as SWEEPS element "profile")
# Writes <output directory>/profile/:
#   index.json              [{tag, case, method, N, variant, nsys_rep, summary, engaged_int8}]
#   <tag>/summary.json      {tag, case, method, N, variant, gpu, wall_ms, kernels:[{name, category, time_ms,
#                           share_pct, count}], engaged_int8, int8_share_pct, windows, notes, ...}
#   <tag>/records.jsonl     the imd_bench records of the profiled run (status, errors, timing under nsys);
#   <tag>/run.log           its console output and the nsys messages
#   <tag>/<tag>.nsys-rep    only with PROFILE_KEEP=1 (open it with nsys-ui); without it the report and the
#                           sqlite export are deleted after the summary is written (they are tens of MB each)
# Categories, windows and the engaged_int8 rule: bench/nsys_summarize.py (docstring). The list of runs:
# bench/sweep_profile.txt (case | method | argument; specs from bench/sweep_compare.txt).
# Each run is one imd_bench process with few repetitions (a few seconds), started by nsys through
# tools/run_bench.sh (quiet gate, shared GPU lock), so it never overlaps another timed run. Profiling
# changes the timings slightly: wall_ms is for orientation; the throughput numbers are the sweeps'.
# nsys is looked for in: $NSYS, PATH, $IMMERDINK_ENV/bin, $CUDA_HOME/bin and /usr/local/cuda*/bin,
# /opt/nvidia/nsight-systems*/ and $HOME/nsight-systems-*/ (the newest). If there is none, or it fails, the
# script logs it, writes index.json and summaries with engaged_int8 null and the reason in the notes, and
# exits 0 (a missing profiler must not fail the metrics run).
# Environment (besides what bench/build.sh and tools/run_bench.sh read):
#   NSYS=<path to nsys>        PROFILE_KEEP=1   keep the .nsys-rep files
#   PROFILE_TAGS=<regex>       only the runs whose tag matches (grep -E), e.g. 'cutensor-emu|imd-N14'
#   PROFILE_LIST=<file>        another list (default bench/sweep_profile.txt)
#   PROFILE_REPS=5 PROFILE_WARMUP=2   timed repetitions / warm-up calls of each run (at least 3 are taken)
#   PROFILE_TMP=<dir>          where the reports are written before the summary (default $TMPDIR or /tmp;
#                              node-local scratch on an HPC node)
# Exit: 0 (also when nsys is missing or a run failed: see the summaries); 2 usage; 3 the machine never got quiet.
set -uo pipefail
here="$(cd "$(dirname "$0")" && pwd)"
root="$(dirname "$here")"
[ $# -eq 1 ] || { echo "usage: $0 <output directory outside the repository>" >&2; exit 2; }
outdir="$(realpath -m -- "$1")"
rootp="$(cd "$root" && pwd -P)"
case "$outdir/" in
  "$rootp"/*) echo "refusing: $outdir is inside the repository $rootp; give a directory outside it" >&2; exit 2 ;;
esac
prof="$outdir/profile"
mkdir -p "$prof" || exit 2
list="${PROFILE_LIST:-$here/sweep_profile.txt}"
compare="$here/sweep_compare.txt"
env_dir="${IMMERDINK_ENV:-$HOME/miniforge3/envs/immerdink}"
build="${IMMERDINK_BENCH_BUILD:-$root/build/bench}"
reps="${PROFILE_REPS:-5}"; warm="${PROFILE_WARMUP:-2}"
summ="$here/nsys_summarize.py"

# --- the planned runs: tag, case, method, N, variant, spec, extents, imd_bench method args ------------
plan="$(mktemp "${TMPDIR:-/tmp}/imd-profile-plan.XXXXXX")"
trap 'rm -f "$plan"' EXIT
n_total=0
trim() { local v="${1#"${1%%[![:space:]]*}"}"; printf '%s' "${v%"${v##*[![:space:]]}"}"; }
while IFS='|' read -r case_ method arg _rest; do
  case_="$(trim "$case_")"
  case "$case_" in ''|'#'*) continue ;; esac
  method="$(trim "$method")"; arg="$(trim "$arg")"
  line="$(awk -F'|' -v c="$case_" '{n=$1; gsub(/^[ \t]+|[ \t]+$/, "", n); if (n == c) {print; exit}}' "$compare")"
  if [ -z "$line" ]; then echo "profile list: case '$case_' is not in $compare; skipped" >&2; continue; fi
  spec="$(echo "$line" | cut -d'|' -f2 | xargs)"; ext="$(echo "$line" | cut -d'|' -f3 | xargs)"
  N=""; variant=""; margs=""; tag="$case_-$method"
  case "$method" in
    imd)     N="$arg"; margs="--N $arg"; tag="$tag-N$arg" ;;
    gemmul8) mode="${arg%%:*}"; N="${arg##*:}"; variant="$mode"; margs="--g8mode $mode --N $N"; tag="$tag-$mode-N$N" ;;
    emu)     variant="$arg"; margs="--emu $arg"; tag="$tag-$arg" ;;
    cutensor-emu)  # argument: - or comma tokens ws<MiB>|wsest, eager|performant, b<bits> (e.g. ws2048 or eager,b55)
      if [ "$arg" != - ] && [ -n "$arg" ]; then
        ce=""; for t in ${arg//,/ }; do
          case "$t" in
            ws*) margs="$margs --ct-ws-mib ${t#ws}" ;;
            eager|performant) ce="${ce:+$ce,}strategy=$t" ;;
            b[0-9]*) ce="${ce:+$ce,}bits=${t#b}" ;;
            *) echo "profile list: bad cutensor-emu argument '$t'; skipped" >&2; continue 2 ;;
          esac
        done
        [ -n "$ce" ] && margs="$margs --ct-emu $ce"; variant="$arg"; tag="$tag-${arg//,/-}"
      fi ;;
    cutensor|dgemm) ;;
    *) echo "profile list: unknown method '$method'; skipped" >&2; continue ;;
  esac
  if [ -n "${PROFILE_TAGS:-}" ] && ! echo "$tag" | grep -Eq "$PROFILE_TAGS"; then continue; fi
  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' "$tag" "$case_" "$method" "${N:--}" "${variant:--}" "$spec" "$ext" "$margs" >> "$plan"
  n_total=$((n_total + 1))
done < "$list"
echo "profile: $n_total run(s) -> $prof"
[ "$n_total" -gt 0 ] || { echo "profile: nothing to run (PROFILE_TAGS?)"; [ -f "$prof/index.json" ] || echo "[]" > "$prof/index.json"; exit 0; }
[ -z "${PROFILE_TAGS:-}" ] && rm -f "$prof/index.json"   # (a PROFILE_TAGS run adds to / replaces entries of an existing index)

gpu="$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)"
gpu_arg=(); [ -n "$gpu" ] && gpu_arg=(--gpu "$gpu")
missing() {  # missing <reason>: summaries without data, exit 0
  echo "profile: $1" >&2
  echo "profile: $1" > "$prof/NSYS_MISSING.txt"
  cut -f1-5 "$plan" > "$plan.entries"   # (N / variant "-" = none; nsys_summarize.py reads that)
  python3 "$summ" missing --outdir "$prof" --nsys-note "$1" --entries "$plan.entries" ${gpu_arg[@]+"${gpu_arg[@]}"}
  rm -f "$plan.entries"
  python3 "$summ" table "$prof" > "$outdir/profile.md" 2>/dev/null
  exit 0
}

# --- find nsys -----------------------------------------------------------------------------------
nsys="${NSYS:-}"
if [ -z "$nsys" ]; then
  nsys="$(command -v nsys || true)"
  if [ -z "$nsys" ]; then
    for c in "$env_dir/bin/nsys" "${CUDA_HOME:-/nonexistent}/bin/nsys" /usr/local/cuda*/bin/nsys \
             /opt/nvidia/nsight-systems*/bin/nsys /opt/nvidia/nsight-systems*/*/target-linux-x64/nsys \
             "$HOME"/nsight-systems-*/bin/nsys; do
      [ -x "$c" ] && nsys="$c"   # the last (lexically newest) wins
    done
  fi
fi
[ -n "$nsys" ] && [ -x "$nsys" ] || missing "nsys not found (put it on PATH, set NSYS=<path>, or install nsight-systems into IMMERDINK_ENV): no profiles"
nsys_ver="$("$nsys" --version 2>/dev/null | head -1 | sed 's/^NVIDIA Nsight Systems version //')"
echo "profile: nsys $nsys_ver ($nsys)"
command -v python3 >/dev/null || missing "no python3: the nsys reports cannot be summarized"

[ -d "$env_dir/lib" ] && export LD_LIBRARY_PATH="$env_dir/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
# a profile is not a throughput measurement: do not wait minutes for a desktop to stop drawing on the GPU
export QUIET_RAW_WAIT="${QUIET_RAW_WAIT:-10}"
export QUIET_GPU_IGNORE="${QUIET_GPU_IGNORE:-gnome-remote-desktop-daemon|Xorg|gnome-shell|claude-desktop --type=gpu-process}"
"$here/build.sh" || { echo "profile: build failed" >&2; exit 1; }
[ -x "$build/imd_bench" ] || { echo "profile: no $build/imd_bench" >&2; exit 1; }

tmp="$(mktemp -d "${PROFILE_TMP:-${TMPDIR:-/tmp}}/imd-nsys.XXXXXX")"
trap 'rm -rf "$tmp"; rm -f "$plan"' EXIT
nfail=0; i=0
while IFS=$'\t' read -r tag case_ method N variant spec ext margs; do
  i=$((i + 1))
  [ "$N" = - ] && N=""; [ "$variant" = - ] && variant=""   # (tab is IFS whitespace: empty fields were written as -)
  d="$prof/$tag"; mkdir -p "$d"
  echo "== [$i/$n_total] $tag $(date -Is)"
  # --sample=none --cpuctxsw=none: no CPU sampling (needs perf_event access; not wanted); CUDA trace only
  QUIET_LOG="$d/quiet.log" "$root/tools/run_bench.sh" "$nsys" profile -t cuda --sample=none --cpuctxsw=none \
    -o "$tmp/$tag" --force-overwrite true --export sqlite \
    "$build/imd_bench" --name "$case_" --spec "$spec" --ext "$ext" --methods "$method" $margs \
    --reps "$reps" --warmup "$warm" --max-seconds 0.5 --samples 4096 --phase-reps 0 --phi 0.5 \
    --out "$d/records.jsonl" > "$d/run.log" 2>&1 < /dev/null
  rc=$?
  rm -f "$d/records.jsonl.tmp"
  # nsys draws progress bars with carriage returns: keep the log readable
  tr '\r' '\n' < "$d/run.log" | grep -v '^\[[0-9]*/[0-9]*\] \[' | grep -v '^$' > "$d/run.log.clean" && mv "$d/run.log.clean" "$d/run.log"
  if [ $rc = 3 ]; then
    echo "machine never got quiet; stopping the profile pass before $tag" >&2
    exit 3
  fi
  note=""
  [ $rc != 0 ] && { note="nsys / imd_bench exit $rc: $(tail -c 300 "$d/run.log" | tr '\n' ' ')"; nfail=$((nfail + 1)); }
  rep="$tmp/$tag.nsys-rep"; db="$tmp/$tag.sqlite"
  if [ ! -s "$db" ] && [ -s "$rep" ]; then   # nsys without --export support at profile time
    "$nsys" export --type sqlite --force-overwrite true -o "$db" "$rep" > "$d/export.log" 2>&1 || true
  fi
  keep=""
  if [ "${PROFILE_KEEP:-0}" = 1 ] && [ -s "$rep" ]; then mv "$rep" "$d/$tag.nsys-rep"; keep="$tag/$tag.nsys-rep"; fi
  if [ -s "$db" ]; then
    python3 "$summ" summarize --db "$db" --tag "$tag" --case "$case_" --method "$method" --N "$N" --variant "$variant" \
      --outdir "$prof" --nsys-rep "$keep" --nsys-version "$nsys_ver" --runlog "$d/records.jsonl" --notes "$note" \
      ${gpu_arg[@]+"${gpu_arg[@]}"}
  else
    echo "$tag	$case_	$method	$N	$variant" > "$tmp/one.entries"
    python3 "$summ" missing --outdir "$prof" --entries "$tmp/one.entries" ${gpu_arg[@]+"${gpu_arg[@]}"} \
      --nsys-note "no nsys report for this run (nsys / imd_bench exit $rc): $(tail -c 300 "$d/run.log" | tr '\n' ' ')"
  fi
  rm -f "$rep" "$db"
done < "$plan"
python3 "$summ" table "$prof" > "$outdir/profile.md" 2>/dev/null
echo "profile: done ($n_total run(s), $nfail with a non-zero exit): $prof/index.json"
exit 0
