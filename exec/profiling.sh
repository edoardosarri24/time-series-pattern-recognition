#!/bin/bash

if [ -z "$1" ]; then
    echo "Error: Missing argument. Usage: $0 [seq|par]"
    exit 1
fi

MODE=$1
PPROF_PATH="/Users/edoardosarri/.asdf/installs/golang/1.25.6/bin/pprof"

if [[ "$MODE" != "seq" && "$MODE" != "par" ]]; then
    echo "Error: Invalid argument '$MODE'. Usage: $0 [seq|par]"
    exit 1
fi

echo "building (Profiling)..."
rm -rf build
cmake -S . -B build -DCMAKE_BUILD_TYPE=RelWithDebInfo -DENABLE_PROFILING=ON
cmake --build build

echo "executing ($MODE)..."
if [ "$MODE" == "par" ]; then
    TARGET="./build/parallel/parallel_pattern_recognition"
    OUT_DIR="parallel/result_profiling"
else
    TARGET="./build/sequential/sequential_pattern_recognition"
    OUT_DIR="sequential/result_profiling"
fi

mkdir -p "$OUT_DIR"
CPUPROFILE=whole.prof "$TARGET"
"$PPROF_PATH" -top "$TARGET" whole.prof > "$OUT_DIR/profile.txt"
"$PPROF_PATH" -pdf "$TARGET" whole.prof > "$OUT_DIR/profile.pdf"
rm whole.prof

echo "Profiling results saved to $OUT_DIR"