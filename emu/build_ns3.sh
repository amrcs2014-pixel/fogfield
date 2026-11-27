#!/bin/bash
# Build an unmodified ns-3.48 in ~/ns-3-fogfield for the FogField emulation tier.
# Set NS3_SRC to a local ns-3-dev clone to avoid the download; a clone takes only the committed
# tree, so local modifications in that clone are not carried over. ns-3.48 needs CMake >= 3.25.
set -euo pipefail
export PATH="$HOME/.local/bin:$PATH"   # e.g. a pip-installed CMake, if the system one is older
cmake --version | head -1
SRC="${NS3_SRC:-https://gitlab.com/nsnam/ns-3-dev.git}"   # or a local clone
DST="$HOME/ns-3-fogfield"
COMMIT=ns-3.48          # tag ns-3.48 = commit d2add90b4
if [ ! -d "$DST/.git" ]; then
  git clone --quiet "$SRC" "$DST"
fi
cd "$DST"
git checkout --quiet "$COMMIT"
echo "tree status (should be empty):"; git status --short | head -5
cat VERSION
rm -rf cmake-cache build      # leftovers of a failed configure in this private clone only
./ns3 configure -d optimized --enable-modules "wifi;internet;applications;mobility;flow-monitor" \
    --disable-tests --disable-examples
./ns3 build -j 8
echo BUILD_DONE
