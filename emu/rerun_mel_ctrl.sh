#!/bin/bash
# Re-run the 10 MEL collection-mode runs at n = 50 with the current binary, which also logs the
# controller's transmit time per slot (ctrl_slots). Removes the old records (WSL copy and the
# Windows mirror) so that run_grid.py's first pass redoes exactly these runs.
HERE="$(cd "$(dirname "$0")" && pwd)"
for s in 0 1 2 3 4; do
  for ch in disk fading; do
    f="main_MEL_static_D-POA-style_n50_s${s}__controller_${ch}_w1.json"
    rm -f "$HOME/fogfield-emu-out/runs/$f" "$HERE/runs/$f"
  done
done
python3 "$HERE/run_grid.py" 4
