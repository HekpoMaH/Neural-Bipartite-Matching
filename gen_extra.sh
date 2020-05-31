#!/bin/bash

if [ "$4" -eq "1" -a "$5" -eq "4" ]; then
    for i in {300..500}; do
        echo "$i"
        ./gen.py $1 $2 $4 $5 >t.txt
        ./deterministic.py --flip <t.txt >"bfs_less_wired/raw/train/$i.txt"
    done
    echo
fi
