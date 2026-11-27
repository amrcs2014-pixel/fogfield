#!/bin/bash
# Fading-channel calibration: broadcast PDR versus distance (single attempt, 45 B).
#   bash calib.sh <txPowerDbm> <dmin> <dmax> <dstep> <tag>
TXP=${1:-0}; DMIN=${2:-25}; DMAX=${3:-1500}; DSTEP=${4:-25}; TAG=${5:-0dBm}
OUT="$HOME/fogfield-emu-out"
mkdir -p "$OUT"
EMU="$(cd "$(dirname "$0")" && pwd)"
bash "$EMU/run_emu.sh" --mode=calib --txPowerDbm="$TXP" --nakagamiM=2 --calibMin="$DMIN" \
     --calibMax="$DMAX" --calibStep="$DSTEP" --calibN=2000 --out="$OUT/calib_$TAG.json"
cp "$OUT/calib_$TAG.json" "$EMU/calib_$TAG.json"
cat "$EMU/calib_$TAG.json"
