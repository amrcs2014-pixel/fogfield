#!/bin/bash
# Single-attempt broadcast PDR exactly at R_comm, with the 21-byte beacon payload (and 45 B for
# comparison with the earlier sweep), on the calibrated fading channel.
EMU="$(cd "$(dirname "$0")" && pwd)"
OUT="$HOME/fogfield-emu-out"
mkdir -p "$OUT"
for spec in "MEL 17.1 820" "BEJ 21.8 1400"; do
  set -- $spec
  for b in 21 45; do
    bash "$EMU/run_emu.sh" --mode=calib --txPowerDbm=$2 --nakagamiM=2 --calibMin=$3 --calibMax=$3 \
         --calibStep=1 --calibN=10000 --calibBytes=$b --out="$OUT/calib_rcomm_$1_$b.json"
    cp "$OUT/calib_rcomm_$1_$b.json" "$EMU/calib_rcomm_$1_$b.json"
    echo "$1 ${b}B: $(cat "$OUT/calib_rcomm_$1_$b.json")"
  done
done
