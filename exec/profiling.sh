#!/bin/bash

if [ -z "$1" ]; then
    echo "Error: Missing argument. Usage: $0 [seq|par]"
    exit 1
fi

MODE=$1
PPROF_PATH="$HOME/gperftools/bin/pprof"

if [[ "$MODE" != "seq" && "$MODE" != "par" ]]; then
    echo "Error: Invalid argument '$MODE'. Usage: $0 [seq|par]"
    exit 1
fi

echo "building (Profiling)..."
rm -rf build
cmake -S . -B build -DCMAKE_BUILD_TYPE=RelWithDebInfo -DENABLE_PROFILING=ON -DCMAKE_PREFIX_PATH="$HOME/gperftools"
cmake --build build

if [ "$MODE" == "par" ]; then
    TARGET="./build/parallel/parallel_pattern_recognition"
else
    TARGET="./build/sequential/sequential_pattern_recognition"
fi

echo "executing ($MODE)..."
if [ "$MODE" == "par" ]; then
    TARGET="./build/parallel/parallel_pattern_recognition"
    OUT_DIR="parallel"
else
    TARGET="./build/sequential/sequential_pattern_recognition"
    OUT_DIR="sequential"
fi

LD_LIBRARY_PATH="$HOME/gperftools/lib" CPUPROFILE=whole.prof "$TARGET" data/input.txt
"$PPROF_PATH" --text "$TARGET" whole.prof > "$OUT_DIR/profiling.txt"
rm whole.prof

echo "Profiling results saved to $OUT_DIR/profiling.txt"