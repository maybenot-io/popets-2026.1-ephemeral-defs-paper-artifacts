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
wmins=(1 5 10)
wmax=(12 15 20)
budgets=(1500 1700 2000 2500 3000 6000)
states=(1 10 100)

run_simulation() {
    local cfg=../$1

    for min in "${wmins[@]}"; do
        for max in "${wmax[@]}"; do
            for budget in "${budgets[@]}"; do
                for s in "${states[@]}"; do

                    output_file=$TMP_DATASET_FOLDER/front-budget$budget-wmin$min-wmax$max-states$s
                    $MAYBENOT fixed -c "front $budget $min $max $s" -s "front $budget $min $max $s" -o $tmp_file -n 19000

                    $MAYBENOT sim -c $cfg -i $tmp_file -o $output_file
                    $MAYBENOT eval -c $cfg -i $output_file
                    rm -rf $output_file
                    rm $tmp_file
                done
            done
        done
    done
}

# we want to run DL on all parameters, unfortunately, to find the most flattering combination
cfgs=("infinite.toml" "bottleneck.toml")
for cfg in "${cfgs[@]}"; do
    run_simulation "$cfg"
done

