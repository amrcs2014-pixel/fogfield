#!/bin/bash
# Run the compiled fogfield-emu binary with the given arguments (no rebuild).
NS3="$HOME/ns-3-fogfield"
export LD_LIBRARY_PATH="$NS3/build/lib:${LD_LIBRARY_PATH:-}"
exec "$NS3/build/scratch/ns3.48-fogfield-emu-optimized" "$@"
