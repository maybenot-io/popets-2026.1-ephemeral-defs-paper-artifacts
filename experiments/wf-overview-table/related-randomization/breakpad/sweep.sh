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
tmp_file=tmp.def

run_simulation() {
    local cfg=../$1

    output_file=$TMP_DATASET_FOLDER/break_pad-$1
    $MAYBENOT fixed -c "break_pad_client" -s "break_pad_server" -o $tmp_file

    $MAYBENOT sim -c $cfg -i $tmp_file -o $output_file
    $MAYBENOT eval -c $cfg -i $output_file
    rm -rf $output_file
    rm $tmp_file
}

cfgs=("infinite.toml" "bottleneck.toml")
for cfg in "${cfgs[@]}"; do
    run_simulation "$cfg"
done

