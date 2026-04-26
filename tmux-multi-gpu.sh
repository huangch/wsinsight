#!/bin/bash

tmux new-session -d -s wsinsight

# Build left column: 4 rows
tmux split-window -v -t wsinsight:0.0
tmux split-window -v -t wsinsight:0.1
tmux split-window -v -t wsinsight:0.2

# Equalize row heights (25% each)
tmux select-layout -t wsinsight:0 even-vertical

# Split each row in half — target steps by 2 because each split inserts a pane
tmux split-window -h -t wsinsight:0.0   # row 1: 0(L), 1(R)
tmux split-window -h -t wsinsight:0.4   # row 3: 4(L), 5(R) 
tmux split-window -h -t wsinsight:0.6   # row 4: 6(L), 7(R)

# Layout:
# [pane 0 | pane 1]   GPU 0 / GPU 4
# [pane 2 | pane 3]   GPU 1 / GPU 5
# [pane 4 | pane 5]   GPU 2 / GPU 6
# [pane 6 | pane 7]   GPU 3 / GPU 7

ZOO=/app/zoo/huangch/10x-brca-CellViT-SAM-H-x40/main
OUT=results/primary-brca
BATCH=20

for i in 0 1 2 3 4 5 6 7; do
    part=$(printf "%02d" "$i")
    tmux send-keys -t wsinsight:0.$i \
        "CUDA_VISIBLE_DEVICES=$i wsinsight run -b $BATCH -i datasets/slides_part_${part}.txt -z $ZOO -o $OUT" Enter
done

tmux attach -t wsinsight
(ws