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


$MAYBENOT sim -c eph-blocking-inf-default.toml -i eph-blocking-1k-c2-100k-april8-use-scale0.75 -o $TMP_DATASET_FOLDER/eph-blocking-inf-default/ -e -f
$MAYBENOT sim -c eph-blocking-inf-bw1.toml -i eph-blocking-1k-c2-100k-april8-use-scale0.75 -o $TMP_DATASET_FOLDER/eph-blocking-inf-bw1/ -e -f
$MAYBENOT sim -c eph-blocking-inf-bw2.toml -i eph-blocking-1k-c2-100k-april8-use-scale0.75 -o $TMP_DATASET_FOLDER/eph-blocking-inf-bw2/ -e -f
$MAYBENOT sim -c eph-blocking-inf-bw4.toml -i eph-blocking-1k-c2-100k-april8-use-scale0.75 -o $TMP_DATASET_FOLDER/eph-blocking-inf-bw4/ -e -f
$MAYBENOT sim -c eph-blocking-inf-bw8.toml -i eph-blocking-1k-c2-100k-april8-use-scale0.75 -o $TMP_DATASET_FOLDER/eph-blocking-inf-bw8/ -e -f

