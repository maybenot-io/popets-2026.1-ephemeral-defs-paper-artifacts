#!/bin/bash
#

cd def-ml/wfdata

# CircuitFP-General
curl https://dart.cse.kau.se/popets-2026.1-ephemeral-defs-paper-artifacts/circuitfp-general-rend.tar.gz -o circuitfp-general-rend.tar.gz
tar -xvzf circuitfp-general-rend.tar.gz

# Bigenough
curl https://dart.cse.kau.se/popets-2026.1-ephemeral-defs-paper-artifacts/bigenough-95x10x20-standard-rngsubpages.tar.gz -o bigenough.tar.gz
tar -xvzf bigenough.tar.gz
cd ..
uv run python convert_data.py --dataset bigenough
cd wfdata

# Gong-Surakav
curl https://dart.cse.kau.se/popets-2026.1-ephemeral-defs-paper-artifacts/gong-surakav-undefended-cw.tar.gz -o gong-surakav.tar.gz
tar -xvzf gong-surakav.tar.gz
cp ../gong-surakav_rename.sh gong-surakav/
cd gong-surakav
bash gong-surakav_rename.sh
rm gong-surakav_rename.sh
cd ../../
uv run python convert_data.py --dataset gong-surakav
cd wfdata

# LongEnough
curl https://dart.cse.kau.se/popets-2026.1-ephemeral-defs-paper-artifacts/LongEnough.zip -o LongEnough.zip
unzip LongEnough.zip -d LongEnough
curl https://dart.cse.kau.se/popets-2026.1-ephemeral-defs-paper-artifacts/longenough-rename.py -o LongEnough/longenough-rename.py
cd LongEnough
python longenough-rename.py LongEnough
