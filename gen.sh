#!/bin/bash

# exit 0

FOLDER="$3"
if ! [ "$4" == "1" -a "$5" -eq "4" ]; then
    FOLDER="${FOLDER}_$4_$5"
fi

if [ "$3" == "all_iter_less_wired" ]; then 

    if [ "$4" -eq "1" -a "$5" -eq "4" ]; then
        mkdir -p $FOLDER/raw/train
        for i in {1..300}; do
            echo "$i"
            ./gen.py $1 $2 $4 $5 >t.txt
            ./deterministic.py <t.txt >"all_iter_less_wired/raw/train/$i.txt"
        done
        echo
        mkdir -p $FOLDER/raw/val
        for i in {1..50}; do
            echo "$i"
            ./gen.py $1 $2 $4 $5 >t.txt
            ./deterministic.py <t.txt >"all_iter_less_wired/raw/val/$i.txt"
        done
        echo

        mkdir -p $FOLDER/raw/test_4x
        for i in {1..50}; do
            echo "$i"
            ./gen.py $(($1*4)) $(($2*4)) $4 $5 >t.txt
            ./deterministic.py <t.txt >"all_iter_less_wired/raw/test_4x/$i.txt"
        done

        mkdir -p $FOLDER/raw/test_8x
        for i in {1..50}; do
            echo "$i"
            ./gen.py $(($1*8)) $(($2*8)) $4 $5 >t.txt
            ./deterministic.py <t.txt >"all_iter_less_wired/raw/test_8x/$i.txt"
        done
    fi

    mkdir -p $FOLDER/raw/test
    for i in {1..50}; do
        echo "$i"
        ./gen.py $1 $2 $4 $5 >t.txt
        ./deterministic.py <t.txt >"$FOLDER/raw/test/$i.txt"
    done
    echo

    mkdir -p $FOLDER/raw/test_2x
    for i in {1..50}; do
        echo "$i"
        ./gen.py $(($1*2)) $(($2*2)) $4 $5 >t.txt
        ./deterministic.py <t.txt >"$FOLDER/raw/test_2x/$i.txt"
    done

fi

