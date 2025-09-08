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

$MAYBENOT search -c combo-10.toml -o def-base -s 0 -d "seed 0"
$MAYBENOT sim -c combo-10.toml -i def-base -o $TMP_DATASET_FOLDER/def-base -e

for i in {1..10}; do
    $MAYBENOT combo -c combo-10.toml -i def-base -o def-base-c${i} -h ${i}
    $MAYBENOT sim -c combo-10.toml -i def-base-c${i} -o $TMP_DATASET_FOLDER/def-base-c${i} -e
done
