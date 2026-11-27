#!/bin/bash
# Copy fogfield-emu.cc into the private ns-3 tree and build it.
set -euo pipefail
export PATH="$HOME/.local/bin:$PATH"
NS3="$HOME/ns-3-fogfield"
HERE="$(cd "$(dirname "$0")" && pwd)"
cp "$HERE/ns3/fogfield-emu.cc" "$NS3/scratch/fogfield-emu.cc"
cd "$NS3"
./ns3 build fogfield-emu 2>&1 | grep -v "^\[" | tail -40
ls -la build/scratch/ | grep fogfield
