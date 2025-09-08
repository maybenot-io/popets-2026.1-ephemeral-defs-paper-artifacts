#!/bin/bash

if [ -z "$MAYBENOT" ]; then
    echo "Error: MAYBENOT environment variable is not set."
    echo "Please set it to the path of the maybenot binary, e.g.:"
    echo "export MAYBENOT=/path/to/maybenot"
    exit 1
fi

$MAYBENOT search -c bottleneck.toml -o eph-blocking-bottle-1k -s 0 -d "seed 0"

def_hash=$(sha256sum "eph-blocking-bottle-1k" | awk '{print $1}')
expected_hash="a09392ed033502ff59ab9edf4382e1b1d9ae9226d6e52e31fbf309f2c2c39ea6"
if [[ "$def_hash" != "$expected_hash" ]]; then
        echo "WARNING: unexpected defense hash"
fi

$MAYBENOT combo -c bottleneck.toml -i eph-blocking-bottle-1k -o eph-blocking-bottle-1k-c2-100k

def_hash=$(sha256sum "eph-blocking-bottle-1k-c2-100k" | awk '{print $1}')
expected_hash="106e70fa7da34168127d33e0ee5cb1a20bff8b81ee550df5a27c2faa08c89bd7"
if [[ "$def_hash" != "$expected_hash" ]]; then
        echo "WARNING: unexpected defense hash"
fi
