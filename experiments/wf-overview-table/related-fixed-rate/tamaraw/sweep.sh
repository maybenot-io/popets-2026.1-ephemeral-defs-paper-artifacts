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

# defense parameters to explore
pc_values=(0.005 0.010 0.015 0.020 0.025 0.030)
ps_values=(0.0015 0.0020 0.0025 0.0030 0.0035 0.0040 0.0045 0.0050 0.0055 0.0060 0.0065)
window_values=(1000000.0 1500000.0 2000000.0 2500000.0 4000000.0 6000000.0)

run_simulation() {
    local cfg=../$1

    for pc in "${pc_values[@]}"; do
        for ps in "${ps_values[@]}"; do
            for w in "${window_values[@]}"; do
                output_file="$TMP_DATASET_FOLDER/tamaraw-pc$pc-ps$ps-w$w"
                $MAYBENOT fixed -c "tamaraw $pc $w" -s "tamaraw $ps $w" -o $tmp_file

                $MAYBENOT sim -c $cfg -i $tmp_file -o $output_file
                $MAYBENOT eval -c $cfg -i $output_file
                rm -rf $output_file
                rm $tmp_file
            done
        done
    done
}

cfgs=("small-normal.toml" "small-bottleneck.toml")
for cfg in "${cfgs[@]}"; do
    run_simulation "$cfg"
done
