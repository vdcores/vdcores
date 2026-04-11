#!/usr/bin/env bash

set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  run_fresh_median.sh [-n runs] [-v] -- <command ...>

Runs the command in a fresh process N times, extracts the first
"Average execution time (ns): ..." line from each run, and reports min/median/max.

Example:
  tests/script/run_fresh_median.sh -n 15 -- \
    python app/python/llama32_1b/sched.py -N 1 --debug-num-layers 1 -b 1 --debug-stop-after attn
EOF
}

runs=11
verbose=0

while [[ "$#" -gt 0 ]]; do
  case "${1:-}" in
    -n)
      if [[ "$#" -lt 2 ]]; then
        echo "missing value for -n" >&2
        usage >&2
        exit 2
      fi
      runs="$2"
      shift 2
      ;;
    -v)
      verbose=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    --)
      shift
      break
      ;;
    *)
      echo "unknown option: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

if [[ "$#" -eq 0 ]]; then
  usage >&2
  exit 2
fi

if ! [[ "$runs" =~ ^[0-9]+$ ]] || [[ "$runs" -le 0 ]]; then
  echo "runs must be a positive integer" >&2
  exit 2
fi

tmp_values="$(mktemp)"
trap 'rm -f "$tmp_values"' EXIT

echo "[fresh-median] command: $*"
echo "[fresh-median] runs: $runs"

for ((i = 1; i <= runs; i++)); do
  echo "[fresh-median] run $i/$runs"

  output="$("$@" 2>&1)"
  if [[ "$verbose" -eq 1 ]]; then
    printf '%s\n' "$output"
  fi

  value="$(printf '%s\n' "$output" | awk -F': ' '/Average execution time \(ns\):/ {print $2; exit}')"
  if [[ -z "$value" ]]; then
    echo "[fresh-median] failed to find 'Average execution time (ns):' in run $i output" >&2
    exit 1
  fi

  printf '%s\n' "$value" >> "$tmp_values"
done

sort -n "$tmp_values" > "${tmp_values}.sorted"
count="$(wc -l < "${tmp_values}.sorted")"
mid=$(( (count + 1) / 2 ))
min_value="$(sed -n '1p' "${tmp_values}.sorted")"
median_value="$(sed -n "${mid}p" "${tmp_values}.sorted")"
max_value="$(sed -n "${count}p" "${tmp_values}.sorted")"
mean_value="$(awk '{sum += $1} END {printf "%.2f", sum / NR}' "${tmp_values}.sorted")"

echo "[fresh-median] summary (ns)"
echo "[fresh-median]   min:    $min_value"
echo "[fresh-median]   median: $median_value"
echo "[fresh-median]   mean:   $mean_value"
echo "[fresh-median]   max:    $max_value"
