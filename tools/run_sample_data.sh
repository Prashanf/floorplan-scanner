#!/usr/bin/env bash
# Run every tier on the three provided sample captures. Run from the repo root.
#   [CAPTURES="single_room"] tools/run_sample_data.sh [lidar|photo|video ...]   (default: all tiers, all captures)
# Expects sample_data_run/{lidar,video,photos}/<capture>/ (see README, "Sample data").
set -u
PY="${PYTHON:-python}"
CAPTURES="${CAPTURES:-single_room single_scan_floor_only single_scan_with_ceiling}"
TIERS="${*:-lidar photo video}"
mkdir -p output
for tier in $TIERS; do
  for n in $CAPTURES; do
    dir="sample_data_run/$([ "$tier" = photo ] && echo photos || echo "$tier")/$n"
    out="output/sample_${tier}_$n"
    echo "=== $tier $n"
    $PY run.py "$dir" --tier "$tier" --output-dir "$out" --verbose > "output/sample_${tier}_${n}_log.txt" 2>&1 \
      && tail -n 12 "output/sample_${tier}_${n}_log.txt" | grep -E "^(tier|  room|no rooms|  warning)" || echo "  failed, see output/sample_${tier}_${n}_log.txt"
    # drift ablation: only meaningful when the first run produced rooms
    if [ "$tier" = photo ] && grep -q '"room_count": [1-9]' "$out/report.json" 2>/dev/null; then
      $PY run.py "$dir" --tier photo --output-dir "${out}_no_drift" --no-drift-correction --verbose \
        > "output/sample_${tier}_${n}_no_drift_log.txt" 2>&1 || echo "  no-drift run failed"
    fi
  done
done
