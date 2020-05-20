#!/bin/bash
./gen.sh 8 8 all_iter_less_wired 1 4
./gen.sh 8 8 all_iter_less_wired 1 2
./gen.sh 8 8 all_iter_less_wired 1 5
./gen.sh 8 8 all_iter_less_wired 3 4

mkdir -p all_iter_less_wired/processed
mkdir -p all_iter_less_wired_1_2/processed
mkdir -p all_iter_less_wired_1_5/processed
mkdir -p all_iter_less_wired_3_4/processed

mkdir -p bfs_less_wired/processed

mkdir -p graph_only_less_wired/processed
mkdir -p graph_only_less_wired_1_2/processed
mkdir -p graph_only_less_wired_1_5/processed
mkdir -p graph_only_less_wired_3_4/processed

mkdir -p graph_only_BFS_less_wired/processed
mkdir -p graph_only_BFS_less_wired_1_2/processed
mkdir -p graph_only_BFS_less_wired_1_5/processed
mkdir -p graph_only_BFS_less_wired_3_4/processed

ln -s ../all_iter_less_wired/raw ./bfs_less_wired/

ln -s ../all_iter_less_wired/raw ./graph_only_less_wired/
ln -s ../all_iter_less_wired_1_2/raw ./graph_only_less_wired_1_2/
ln -s ../all_iter_less_wired_1_5/raw ./graph_only_less_wired_1_5/
ln -s ../all_iter_less_wired_3_4/raw ./graph_only_less_wired_3_4/

ln -s ../all_iter_less_wired/raw ./graph_only_BFS_less_wired/raw
ln -s ../all_iter_less_wired_1_2/raw ./graph_only_BFS_less_wired_1_2/
ln -s ../all_iter_less_wired_1_5/raw ./graph_only_BFS_less_wired_1_5/
ln -s ../all_iter_less_wired_3_4/raw ./graph_only_BFS_less_wired_3_4/
