#!/usr/bin/env bash
# Remaining i7-8086K CPU runs. This file is executed inside WSL, so shell
# variables here are not expanded by the host command wrapper.
set -o pipefail

RAW=/mnt/c/Users/ICH/.freebuff/worktrees/f0e51299/kindi/docs/benchmarks/2026-11-09-i7-8086k-cpu/raw
PROJ=/mnt/c/Users/ICH/.freebuff/worktrees/f0e51299/kindi
DEPS=/home/beefc0de/kindi-cpu-deps
CACHE=/home/beefc0de/.cache/kindi-cpu-i7
PACKS=/mnt/c/Users/ICH/.kindi/packs

cd "$PROJ" || exit 1
mkdir -p "$RAW"
echo "=== remaining CPU runs $(date -u +%FT%TZ) ===" | tee "$RAW/bench-remaining-master.log"
free -h | tee -a "$RAW/bench-remaining-master.log"

verify_6() {
  model=$1
  tag=$2
  log="$RAW/verify-$tag.log"
  echo "=== $model CPU verify, 6 threads ===" | tee "$log"
  PYTHONPATH="$DEPS" KINDI_DEVICE=cpu KINDI_CPU_CACHE="$CACHE" \
    OMP_NUM_THREADS=6 OMP_PROC_BIND=close OMP_PLACES=cores \
    python3 -m kindi --pack "$PACKS/bonsai-$model" verify 2>&1 | tee -a "$log"
  echo "=== $model verify finished ===" | tee -a "$RAW/bench-remaining-master.log"
}

bench_pinned_6() {
  model=$1
  tag=$2
  log="$RAW/bench-$tag.log"
  echo "=== $model CPU bench ctx 24576, 6 threads, close/cores ===" | tee "$log"
  PYTHONPATH="$DEPS" KINDI_DEVICE=cpu KINDI_CPU_CACHE="$CACHE" \
    OMP_NUM_THREADS=6 OMP_PROC_BIND=close OMP_PLACES=cores \
    python3 -m kindi --pack "$PACKS/bonsai-$model" --ctx 24576 bench 2>&1 | tee -a "$log"
  echo "=== $model $tag finished ===" | tee -a "$RAW/bench-remaining-master.log"
}

bench_default_12() {
  model=$1
  tag=$2
  log="$RAW/bench-$tag.log"
  echo "=== $model CPU bench ctx 24576, 12 threads, default binding ===" | tee "$log"
  PYTHONPATH="$DEPS" KINDI_DEVICE=cpu KINDI_CPU_CACHE="$CACHE" \
    OMP_NUM_THREADS=12 \
    python3 -m kindi --pack "$PACKS/bonsai-$model" --ctx 24576 bench 2>&1 | tee -a "$log"
  echo "=== $model $tag finished ===" | tee -a "$RAW/bench-remaining-master.log"
}

verify_6 2b 2b
verify_6 4b 4b
bench_default_12 2b 2b-12t-default
bench_default_12 4b 4b-12t-default
bench_pinned_6 9b 9b-6t
bench_default_12 9b 9b-12t-default

echo "=== all remaining CPU runs finished $(date -u +%FT%TZ) ===" | tee -a "$RAW/bench-remaining-master.log"
free -h | tee -a "$RAW/bench-remaining-master.log"
