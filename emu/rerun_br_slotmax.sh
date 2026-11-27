#!/bin/bash
# Re-run FogField-BR's gossip runs that the controller comparison uses (main grid: n = 50, MEL and
# BEJ, both channels; scalability from slot 5 on: n = 50 ... 800, both channels) with the binary
# that also logs every slot's peak transmit time over all nodes (slot_max_tx_ms), so that
# FogField-BR's busiest node is measured per slot like the controller. Removes the old records
# (WSL copy and Windows mirror) so that run_grid.py redoes exactly these runs.
HERE="$(cd "$(dirname "$0")" && pwd)"
OUT="$HOME/fogfield-emu-out/runs"
for tb in MEL BEJ; do
  for s in 0 1 2 3 4; do
    for ch in disk fading; do
      f="main_${tb}_static_FogField-BR_n50_s${s}__gossip_${ch}_w1.json"
      rm -f "$OUT/$f" "$HERE/runs/$f"
    done
  done
done
for f in "$HERE"/runs/scalability_BEJ_static_FogField-BR_n*__gossip_*_w1_o5_r*.json; do
  b="$(basename "$f")"
  rm -f "$OUT/$b" "$HERE/runs/$b"
done
python3 "$HERE/run_grid.py" 4 --v2 &
( python3 "$HERE/run_grid.py" 2 && python3 "$HERE/run_grid.py" 2 --v2b ) &
wait
