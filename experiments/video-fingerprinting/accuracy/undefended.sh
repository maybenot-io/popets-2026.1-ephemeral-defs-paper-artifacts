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


for f in $(seq 0 9);
do
    $MAYBENOT eval -c undefended-default.toml -i LongEnough/none-none/ -f $f
    $MAYBENOT eval -c undefended-bw1.toml -i LongEnough-variable-extended/none-none-bw1/ -f $f
    $MAYBENOT eval -c undefended-bw2.toml -i LongEnough-variable-extended/none-none-bw2/ -f $f
    $MAYBENOT eval -c undefended-bw4.toml -i LongEnough-variable-extended/none-none-bw4/ -f $f
    $MAYBENOT eval -c undefended-bw8.toml -i LongEnough-variable/none-none-bw8/ -f $f
done

