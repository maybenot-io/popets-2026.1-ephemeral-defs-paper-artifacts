#!/bin/bash

if [ -z "$MAYBENOT" ]; then
    echo "Error: MAYBENOT environment variable is not set."
    echo "Please set it to the path of the maybenot binary, e.g.:"
    echo "export MAYBENOT=/path/to/maybenot"
    exit 1
fi

if [ -z "$TMP_DATASET_FOLDER" ]; then
    echo "Error: TMP_DATASET_FOLDER environment variable is not set."
    echo "Please set it to the path of the temporary dataset folder, e.g.:"
    echo "export TMP_DATASET_FOLDER=/tmp/datasets"
    exit 1
fi


$MAYBENOT sim -c scrambler-inf-default.toml -i scrambler-120interval-1500count-400min-1000max.def -o $TMP_DATASET_FOLDER/scrambler-inf-default/ -e -f
$MAYBENOT sim -c scrambler-inf-bw1.toml -i scrambler-120interval-1500count-400min-1000max.def -o $TMP_DATASET_FOLDER/scrambler-inf-bw1/ -e -f
$MAYBENOT sim -c scrambler-inf-bw2.toml -i scrambler-120interval-1500count-400min-1000max.def -o $TMP_DATASET_FOLDER/scrambler-inf-bw2/ -e -f
$MAYBENOT sim -c scrambler-inf-bw4.toml -i scrambler-120interval-1500count-400min-1000max.def -o $TMP_DATASET_FOLDER/scrambler-inf-bw4/ -e -f
$MAYBENOT sim -c scrambler-inf-bw8.toml -i scrambler-120interval-1500count-400min-1000max.def -o $TMP_DATASET_FOLDER/scrambler-inf-bw8/ -e -f

