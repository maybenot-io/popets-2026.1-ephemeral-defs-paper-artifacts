#!/bin/bash

if [ -z "$MAYBENOT" ]; then
    echo "Error: MAYBENOT environment variable is not set."
    echo "Please set it to the path of the maybenot binary, e.g.:"
    echo "export MAYBENOT=/path/to/maybenot"
    exit 1
fi

$MAYBENOT search -c infinite.toml -o eph-blocking-1k -s 0 -d "seed 0"

def_hash=$(sha256sum "eph-blocking-1k" | awk '{print $1}')
expected_hash="0b45918b3a261ff10e8475ab17ea23a6ed95907e8adb2d9d036a0270d49565ca"
if [[ "$def_hash" != "$expected_hash" ]]; then
        echo "WARNING: unexpected defense hash"
fi

$MAYBENOT combo -c infinite.toml -i eph-blocking-1k -o eph-blocking-1k-c2-100k

def_hash=$(sha256sum "eph-blocking-1k-c2-100k" | awk '{print $1}')
expected_hash="af97b1307745ebfd003313018fa84a20794dc66b26d606c0213728ef4356985e"
if [[ "$def_hash" != "$expected_hash" ]]; then
        echo "WARNING: unexpected defense hash"
fi
