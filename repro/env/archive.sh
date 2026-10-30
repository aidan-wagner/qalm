#!/usr/bin/env bash
# Archive raw campaign outputs to the Chameleon object store (container qalm-repro), incrementally.
#   repro/env/archive.sh [runs_dir]
# 1) `run.py collect --gzip-logs` summarizes finished jobs into repro/results/ and gzips their logs
# 2) finished job directories (and the runs-level state files) are copied with checksums;
#    already-uploaded files are skipped, so this can run repeatedly during the campaign
# 3) `rclone check` verifies the copy; the local file list + sha256 goes to repro/results/ARTIFACTS.sha256
set -euo pipefail
REPO=/home/cc/mingkuan/qalm
RUNS=${1:-$REPO/repro/runs}
DEST=rclone_s3:qalm-repro/$(basename "$RUNS")
cd "$REPO"
REPRO_RUNS_DIR="$RUNS" python3 repro/harness/run.py collect --plan "$RUNS/plan.jsonl" --gzip-logs
# only jobs that are finished (done.json present); running jobs are uploaded on a later pass
LIST=$(mktemp)
( cd "$RUNS/jobs" && for d in *; do [ -f "$d/done.json" ] || [[ "$d" == *.* ]] && echo "$d/**"; done ) > "$LIST"
rclone copy "$RUNS/jobs" "$DEST/jobs" --checksum --include-from "$LIST" --transfers 16 --log-level ERROR
rclone copy "$RUNS" "$DEST" --checksum --max-depth 1 --log-level ERROR
rclone check "$RUNS/jobs" "$DEST/jobs" --one-way --include-from "$LIST" --log-level ERROR
rm -f "$LIST"
( cd "$RUNS" && find jobs -type f \( -path '*/done.json' -o -name '*.gz' -o -name '*.qasm' -o -name 'result.json' \) \
    -newer plan.jsonl -print0 | sort -z | xargs -0 sha256sum ) > "$REPO/repro/results/ARTIFACTS.sha256"
echo "archived to $DEST; manifest repro/results/ARTIFACTS.sha256 ($(wc -l < "$REPO/repro/results/ARTIFACTS.sha256") files)"
