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

# regulator heavy params (baseline)
us=(3.95)
cs=(1.77)
rs=(277)
ds=(0.94)
ts=(3.55)
ns=(3550)

# set parameters in earlier phases
bs=(10)
us=(4.5)
cs=(1.0)
rs=(600)
ns=(5000)

# this phase
ds=(0.9 0.92 0.94 0.96 0.98)
ts=(1.0 1.5 2.0 2.5 3.0 3.55 4.0)

run_simulation() {
    local cfg=$1

    for B in "${bs[@]}"; do
        for U in "${us[@]}"; do
            for C in "${cs[@]}"; do
                for N in "${ns[@]}"; do
                    for R in "${rs[@]}"; do
                        for D in "${ds[@]}"; do
                            for T in "${ts[@]}"; do
                                output_file="$TMP_DATASET_FOLDER/regulator-u$U-c$C-r$R-d$D-t$T-n$N-b$B"
                                $MAYBENOT fixed -c "regulator_client $U $C" -s "regulator_server $R $D $T $N $B" -o $tmp_file
                                $MAYBENOT sim -c $cfg -i $tmp_file -o $output_file -e
                                rm $tmp_file
                        done
                        done
            done
                        done
            done
            done
    done
}

cfgs=("../infinite.toml")
for cfg in "${cfgs[@]}"; do
    run_simulation "$cfg"
done

