#!/bin/bash

if [ -z "$MAYBENOT" ]; then
    echo "Error: MAYBENOT environment variable is not set."
    echo "Please set it to the path of the maybenot binary, e.g.:"
    echo "export MAYBENOT=/path/to/maybenot"
    exit 1
fi

$MAYBENOT search -c infinite.toml -o eph-padding-1k -s 0 -d "seed 0"

def_hash=$(sha256sum "eph-padding-1k" | awk '{print $1}')
expected_hash="9b0317543bbe302cbc493f95a5b6ea39e2fd222827684eed8765a580b1007295"
if [[ "$def_hash" != "$expected_hash" ]]; then
        echo "WARNING: unexpected defense hash"
fi

$MAYBENOT combo -c infinite.toml -i eph-padding-1k -o eph-padding-1k-c2-100k

def_hash=$(sha256sum "eph-padding-1k-c2-100k" | awk '{print $1}')
expected_hash="1f2a60bd20b295856d99a2cfc6b92449c2d4d092035d600d9d2d9ce9c21288b5"
if [[ "$def_hash" != "$expected_hash" ]]; then
        echo "WARNING: unexpected defense hash"
fi
