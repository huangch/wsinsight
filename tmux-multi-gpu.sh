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

tmux send-keys -t wsinsight:0.0 "CUDA_VISIBLE_DEVICES=0 wsinsight run -b 20 -i datasets/slides_part_00.txt -z /app/zoo/huangch/10x-brca-CellViT-SAM-H-x40/main -o results/primary-brca" Enter
tmux send-keys -t wsinsight:0.1 "CUDA_VISIBLE_DEVICES=4 wsinsight run -b 20 -i datasets/slides_part_04.txt -z /app/zoo/huangch/10x-brca-CellViT-SAM-H-x40/main -o results/primary-brca" Enter
tmux send-keys -t wsinsight:0.2 "CUDA_VISIBLE_DEVICES=1 wsinsight run -b 20 -i datasets/slides_part_01.txt -z /app/zoo/huangch/10x-brca-CellViT-SAM-H-x40/main -o results/primary-brca" Enter
tmux send-keys -t wsinsight:0.3 "CUDA_VISIBLE_DEVICES=5 wsinsight run -b 20 -i datasets/slides_part_05.txt -z /app/zoo/huangch/10x-brca-CellViT-SAM-H-x40/main -o results/primary-brca" Enter
tmux send-keys -t wsinsight:0.4 "CUDA_VISIBLE_DEVICES=2 wsinsight run -b 20 -i datasets/slides_part_02.txt -z /app/zoo/huangch/10x-brca-CellViT-SAM-H-x40/main -o results/primary-brca" Enter
tmux send-keys -t wsinsight:0.5 "CUDA_VISIBLE_DEVICES=6 wsinsight run -b 20 -i datasets/slides_part_06.txt -z /app/zoo/huangch/10x-brca-CellViT-SAM-H-x40/main -o results/primary-brca" Enter
tmux send-keys -t wsinsight:0.6 "CUDA_VISIBLE_DEVICES=3 wsinsight run -b 20 -i datasets/slides_part_03.txt -z /app/zoo/huangch/10x-brca-CellViT-SAM-H-x40/main -o results/primary-brca" Enter
tmux send-keys -t wsinsight:0.7 "CUDA_VISIBLE_DEVICES=7 wsinsight run -b 20 -i datasets/slides_part_07.txt -z /app/zoo/huangch/10x-brca-CellViT-SAM-H-x40/main -o results/primary-brca" Enter

tmux attach -t wsinsight
(ws