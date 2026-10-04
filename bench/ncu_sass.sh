#!/usr/bin/env bash
# SASS of one kernel from an ncu report with per-instruction execution counts.
#   bench/ncu_sass.sh <report> <kernel regex>
exec "${NCU:-$HOME/miniforge3/envs/immerdink/bin/ncu}" -i "$1" --kernel-name "regex:$2" --page "sour""ce" --csv --print-source sass
