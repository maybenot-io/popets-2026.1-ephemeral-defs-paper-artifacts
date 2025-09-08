#!/bin/bash

if [ -z "$MAYBENOT" ]; then
    echo "Error: MAYBENOT environment variable is not set."
    echo "Please set it to the path of the maybenot binary, e.g.:"
    echo "export MAYBENOT=/path/to/maybenot"
    exit 1
fi

$MAYBENOT search -c bottleneck.toml -o eph-padding-bottle-1k -s 0 -d "seed 0"

def_hash=$(sha256sum "eph-padding-bottle-1k" | awk '{print $1}')
expected_hash="d6e539183f24634523bafd98d56949e8d290726dd84d2fc5ac5afd2a2aaed67d"
if [[ "$def_hash" != "$expected_hash" ]]; then
        echo "WARNING: unexpected defense hash"
fi

$MAYBENOT combo -c bottleneck.toml -i eph-padding-bottle-1k -o eph-padding-bottle-1k-c2-100k

def_hash=$(sha256sum "eph-padding-bottle-1k-c2-100k" | awk '{print $1}')
expected_hash="1a754ea83f1ca71f2530883cb7d704c100962c32504d7cb72fec1a383ea310a2"
if [[ "$def_hash" != "$expected_hash" ]]; then
        echo "WARNING: unexpected defense hash"
fi
