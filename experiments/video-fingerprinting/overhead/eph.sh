#!/bin/bash

if [ -z "$MAYBENOT" ]; then
    echo "Error: MAYBENOT environment variable is not set."
    echo "Please set it to the path of the maybenot binary, e.g.:"
    echo "export MAYBENOT=/path/to/maybenot"
    exit 1
fi

# hypothesis: seed 2 with bw8 is an outlier, probably due to simulator bug. We do 100 seeds to confirm.
for s in $(seq 0 99);
do
    $MAYBENOT sim -c eph-blocking-inf-default.toml -i ../eph-blocking-1k-c2-100k-april8-use-scale0.75 -o $TMP_DATASET_FOLDER/eph-blocking-inf-default-seed$s/ -e -s $s
    $MAYBENOT sim -c eph-blocking-inf-bw1.toml -i ../eph-blocking-1k-c2-100k-april8-use-scale0.75 -o $TMP_DATASET_FOLDER/eph-blocking-inf-bw1-seed$s/ -e -s $s
    $MAYBENOT sim -c eph-blocking-inf-bw2.toml -i ../eph-blocking-1k-c2-100k-april8-use-scale0.75 -o $TMP_DATASET_FOLDER/eph-blocking-inf-bw2-seed$s/ -e -s $s
    $MAYBENOT sim -c eph-blocking-inf-bw4.toml -i ../eph-blocking-1k-c2-100k-april8-use-scale0.75 -o $TMP_DATASET_FOLDER/eph-blocking-inf-bw4-seed$s/ -e -s $s
    $MAYBENOT sim -c eph-blocking-inf-bw8.toml -i ../eph-blocking-1k-c2-100k-april8-use-scale0.75 -o $TMP_DATASET_FOLDER/eph-blocking-inf-bw8-seed$s/ -e -s $s
done

