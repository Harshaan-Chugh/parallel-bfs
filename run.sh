#!/bin/bash
# ============================================================
# run.sh - Build and test both BFS implementations
#
# Usage:
#   ./run.sh build          Build both implementations
#   ./run.sh test-tiny      Run on all tiny graphs (correctness check)
#   ./run.sh test-medium    Run on medium graphs
#   ./run.sh test-large     Run on large graphs
#   ./run.sh test-all       Run on everything
#   ./run.sh validate       Run + validate against reference BFS
#   ./run.sh bench          Benchmark both approaches on large graphs
# ============================================================

set -e

ROOT_DIR="$(cd "$(dirname "$0")" && pwd)"
BUILD_DIR="$ROOT_DIR/build"
TEST_DIR="$ROOT_DIR/test-graphs"
VALIDATE="$TEST_DIR/validate_bfs.py"

LINALG="$BUILD_DIR/bfs_linalg"
GRAPHFIRST="$BUILD_DIR/bfs_graphfirst"

SOURCE=0  # Default source vertex

# Use only 1 GPU
export CUDA_VISIBLE_DEVICES=0

# ---- Helpers ----

run_bfs() {
    local binary="$1"
    local name="$2"
    local graph="$3"
    local src="${4:-0}"

    echo "  [$name] $(basename "$graph") (source=$src)"
    "$binary" "$graph" "$src"
    echo ""
}

run_and_validate() {
    local binary="$1"
    local name="$2"
    local graph="$3"
    local src="${4:-0}"
    local tmpfile
    tmpfile=$(mktemp "$BUILD_DIR/bfs_output_XXXXXX.txt")

    echo "  [$name] $(basename "$graph") (source=$src)"
    "$binary" "$graph" "$src" > "$tmpfile"

    # Extract depth lines for validation (lines matching "vertex N: depth D")
    grep "vertex" "$tmpfile" | awk '{print $NF}' > "${tmpfile}.depths"

    if [ -s "${tmpfile}.depths" ]; then
        python3 "$VALIDATE" "$graph" "$src" "${tmpfile}.depths" 2>&1 | sed 's/^/    /'
    else
        echo "    (large graph - skipping line-by-line validation)"
    fi

    rm -f "$tmpfile" "${tmpfile}.depths"
    echo ""
}

# ---- Commands ----

cmd_build() {
    echo "=== Building ==="
    make -C "$ROOT_DIR" all
    echo ""
}

cmd_test_tiny() {
    echo "=== Testing on tiny graphs ==="
    for graph in "$TEST_DIR"/tiny_*.edgelist; do
        for binary_info in "$LINALG:linalg" "$GRAPHFIRST:graphfirst"; do
            binary="${binary_info%%:*}"
            name="${binary_info##*:}"
            if [ -x "$binary" ]; then
                run_bfs "$binary" "$name" "$graph" 0
            else
                echo "  [$name] not built, skipping"
            fi
        done
        echo "---"
    done
}

cmd_test_medium() {
    echo "=== Testing on medium graphs ==="
    for graph in "$TEST_DIR"/medium_*.edgelist; do
        for binary_info in "$LINALG:linalg" "$GRAPHFIRST:graphfirst"; do
            binary="${binary_info%%:*}"
            name="${binary_info##*:}"
            if [ -x "$binary" ]; then
                run_bfs "$binary" "$name" "$graph" 0
            else
                echo "  [$name] not built, skipping"
            fi
        done
        echo "---"
    done
}

cmd_test_large() {
    echo "=== Testing on large graphs ==="
    for graph in "$TEST_DIR"/large_*.edgelist; do
        for binary_info in "$LINALG:linalg" "$GRAPHFIRST:graphfirst"; do
            binary="${binary_info%%:*}"
            name="${binary_info##*:}"
            if [ -x "$binary" ]; then
                run_bfs "$binary" "$name" "$graph" 0
            else
                echo "  [$name] not built, skipping"
            fi
        done
        echo "---"
    done
}

cmd_test_all() {
    cmd_test_tiny
    cmd_test_medium
    cmd_test_large
}

cmd_validate() {
    echo "=== Validating on tiny graphs (checking correctness) ==="
    for graph in "$TEST_DIR"/tiny_*.edgelist; do
        for binary_info in "$LINALG:linalg" "$GRAPHFIRST:graphfirst"; do
            binary="${binary_info%%:*}"
            name="${binary_info##*:}"
            if [ -x "$binary" ]; then
                run_and_validate "$binary" "$name" "$graph" 0
            fi
        done
        echo "---"
    done
}

cmd_bench() {
    echo "=== Benchmarking on large graphs ==="
    echo "(Add CUDA event timing to the implementations for proper benchmarks)"
    echo ""
    for graph in "$TEST_DIR"/large_*.edgelist; do
        echo "--- $(basename "$graph") ---"
        for binary_info in "$LINALG:linalg" "$GRAPHFIRST:graphfirst"; do
            binary="${binary_info%%:*}"
            name="${binary_info##*:}"
            if [ -x "$binary" ]; then
                echo "  [$name]"
                time "$binary" "$graph" 0 2>&1 | sed 's/^/    /'
            fi
        done
        echo ""
    done
}

# ---- Main ----

case "${1:-}" in
    build)       cmd_build ;;
    test-tiny)   cmd_test_tiny ;;
    test-medium) cmd_test_medium ;;
    test-large)  cmd_test_large ;;
    test-all)    cmd_test_all ;;
    validate)    cmd_validate ;;
    bench)       cmd_bench ;;
    *)
        echo "Usage: $0 {build|test-tiny|test-medium|test-large|test-all|validate|bench}"
        exit 1
        ;;
esac
