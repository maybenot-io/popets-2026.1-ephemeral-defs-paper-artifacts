#!/bin/bash

if [ -z "$MAYBENOT" ]; then
    echo "Error: MAYBENOT environment variable is not set."
    echo "Please set it to the path of the maybenot binary, e.g.:"
    echo "export MAYBENOT=/path/to/maybenot"
    exit 1
fi

for s in $(seq 0 9);
do
    $MAYBENOT sim -c scrambler-inf-default.toml -i ../scrambler-120interval-1500count-400min-1000max.def -o $TMP_DATASET_FOLDER/scrambler-inf-default/ -e -s $s
    $MAYBENOT sim -c scrambler-inf-bw1.toml -i ../scrambler-120interval-1500count-400min-1000max.def -o $TMP_DATASET_FOLDER/scrambler-inf-bw1/ -e -s $s
    $MAYBENOT sim -c scrambler-inf-bw2.toml -i ../scrambler-120interval-1500count-400min-1000max.def -o $TMP_DATASET_FOLDER/scrambler-inf-bw2/ -e -s $s
    $MAYBENOT sim -c scrambler-inf-bw4.toml -i ../scrambler-120interval-1500count-400min-1000max.def -o $TMP_DATASET_FOLDER/scrambler-inf-bw4/ -e -s $s
    $MAYBENOT sim -c scrambler-inf-bw8.toml -i ../scrambler-120interval-1500count-400min-1000max.def -o $TMP_DATASET_FOLDER/scrambler-inf-bw8/ -e -s $s
done

