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

# regulator infinite params (baseline)
bs=(10)
us=(4.5)
cs=(0.5)
rs=(1200)
ds=(0.98)
ts=(2.5)
ns=(5600)

# previous phases
rs=(220)

# this phase
ns=(500 1000 1500 2000 2500 3000 3500 4000 4500 5000)

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

cfgs=("../bottleneck.toml")
for cfg in "${cfgs[@]}"; do
    run_simulation "$cfg"
done

