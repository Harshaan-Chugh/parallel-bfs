#!/bin/bash
# download_snap.sh — Download Stanford SNAP road network datasets
#
# Downloads three road networks into snap-graphs/ directory:
#   roadNet-PA  (~1.1M vertices, ~1.5M edges)
#   roadNet-TX  (~1.4M vertices, ~1.9M edges)
#   roadNet-CA  (~2.0M vertices, ~2.8M edges)
#
# Usage: ./scripts/download_snap.sh

set -e

SNAP_DIR="$(cd "$(dirname "$0")/.." && pwd)/snap-graphs"
mkdir -p "$SNAP_DIR"

echo "=== Downloading SNAP road networks to $SNAP_DIR ==="

for dataset in roadNet-PA roadNet-TX roadNet-CA; do
    URL="https://snap.stanford.edu/data/${dataset}.txt.gz"
    GZ="$SNAP_DIR/${dataset}.txt.gz"
    TXT="$SNAP_DIR/${dataset}.txt"

    if [ -f "$TXT" ]; then
        echo "  $dataset.txt already exists, skipping"
        continue
    fi

    echo "  Downloading $dataset..."
    wget -q --show-progress "$URL" -O "$GZ"
    echo "  Decompressing..."
    gunzip "$GZ"
    echo "  Done: $(wc -l < "$TXT") lines"
done

echo ""
echo "=== All SNAP datasets ready in $SNAP_DIR ==="
ls -lh "$SNAP_DIR"/*.txt
