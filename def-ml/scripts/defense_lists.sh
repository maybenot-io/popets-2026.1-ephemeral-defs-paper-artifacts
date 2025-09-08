#!/bin/bash
#

ephemeral_bottle=("ephemeral-block-bottle-sc0.75" "ephemeral-pad-bottle-sc0.5")
ephemeral_inf=("ephemeral-block-inf-sc0.75" "ephemeral-pad-inf-sc0.75")
other_bottle=("no_defence" "regulator-bottleneck" "front-bottleneck" "interspace" "breakpad" "tamaraw-bottleneck")
other_inf=("no_defence" "regulator-infinite" "front-infinite" "interspace" "breakpad" "tamaraw-infinite")

TRUNCATED=false

# Loop through all arguments
for arg in "$@"; do
    case "$arg" in
        --truncated)
            TRUNCATED=true
            ;;
    esac
done


if [ "$TRUNCATED" = true ]
then
    echo "Truncated mode: trimming defences list..."
    defences_bottle=("no_defence") # "front-bottleneck" "ephemeral-pad-bottle-sc0.5")
    defences_inf=("no_defence" "front-infinite" "ephemeral-pad-inf-sc0.75")
else
    defences_bottle=("${ephemeral_bottle[@]}" "${other_bottle[@]}")
    defences_inf=("${ephemeral_inf[@]}" "${other_inf[@]}")
fi

