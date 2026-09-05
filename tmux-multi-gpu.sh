#!/bin/bash
# tmux-multi-gpu.sh - open an 8-pane tmux session that runs wsinsight once per GPU.
#
# Hardcoded for 8 GPUs in a 4 rows x 2 cols layout. Pane N (0..7) executes
# `wsinsight run` with CUDA_VISIBLE_DEVICES=N against a per-shard slides list.
#
# Why this script calls `wsinsight` directly, not the `./wsinsight.sh` wrapper:
# the wrapper's only job here would be env / docker-backend selection, but this
# script is for native runs (one GPU per shell). The wrapper would be a no-op
# wrapper-around-the-real-wrapper. The one caveat: it shares the `-b` short
# flag with wsinsight's `--batch-size`. Calling `./wsinsight.sh -b native run
# -b 20 ...` would silently fail because the wrapper would interpret the
# second `-b 20` as `--backend 20`. Use `--` to disambiguate if migrating:
# `./wsinsight.sh -b native -- run -b 20 ...`.
#
# `-z` (--zoo-model-dir) vs `-m` (--model): they are NOT alternatives. `-m`
# resolves a registry name via HuggingFace; `-z` is a path to a local directory
# already containing config.json + weights (offline). This script uses `-z`
# because /app/zoo/huangch/... is a baked-in local path; switching to `-m`
# would require a registry identifier, NOT a directory path. Keep `-z`.
#
# Idempotency: if a tmux session named `wsinsight` already exists, this script
# attaches to it without rebuilding the layout.

set -euo pipefail

if tmux has-session -t wsinsight 2>/dev/null; then
    echo "tmux session 'wsinsight' already exists; attaching (no rebuild)." >&2
    exec tmux attach -t wsinsight
fi

tmux new-session -d -s wsinsight

# Build left column: 4 rows
tmux split-window -v -t wsinsight:0.0
tmux split-window -v -t wsinsight:0.1
tmux split-window -v -t wsinsight:0.2

# Equalize row heights (25% each)
tmux select-layout -t wsinsight:0 even-vertical

# Split each row in half — target steps by 2 because each split inserts a pane
tmux split-window -h -t wsinsight:0.0
tmux split-window -h -t wsinsight:0.4
tmux split-window -h -t wsinsight:0.6

# Layout (after all splits):
#   [pane 0 | pane 1]   GPU 0 / GPU 4
#   [pane 2 | pane 3]   GPU 1 / GPU 5
#   [pane 4 | pane 5]   GPU 2 / GPU 6
#   [pane 6 | pane 7]   GPU 3 / GPU 7

ZOO=/app/zoo/huangch/10x-brca-CellViT-SAM-H-x40/main
OUT=results/primary-brca
BATCH=20

for i in 0 1 2 3 4 5 6 7; do
    part=$(printf "%02d" "$i")
    tmux send-keys -t "wsinsight:0.$i" \
        "CUDA_VISIBLE_DEVICES=$i wsinsight run -b $BATCH -i datasets/slides_part_${part}.txt -z $ZOO -o $OUT" Enter
done

tmux attach -t wsinsight
