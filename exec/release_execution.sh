#!/bin/bash

if [ -z "$1" ]; then
    echo "Error: Missing argument. Usage: $0 [seq|par] [ea (optional)]"
    exit 1
fi

MODE=$1
EARLY_ABANDONING=$2

if [[ "$MODE" != "seq" && "$MODE" != "par" ]]; then
    echo "Error: Invalid argument '$MODE'. Usage: $0 [seq|par] [ea (optional)]"
    exit 1
fi

CMAKE_FLAGS="-DCMAKE_BUILD_TYPE=Release"
if [ "$EARLY_ABANDONING" == "ea" ]; then
    echo "Early Abandoning ENABLED"
    CMAKE_FLAGS="$CMAKE_FLAGS -DENABLE_EARLY_ABANDONING=ON"
else
    echo "Early Abandoning DISABLED"
    CMAKE_FLAGS="$CMAKE_FLAGS -DENABLE_EARLY_ABANDONING=OFF"
fi

echo "building (Release)..."
rm -rf build
cmake -S . -B build $CMAKE_FLAGS
cmake --build build

echo "executing ($MODE)..."
if [ "$MODE" == "par" ]; then
    ./build/parallel/parallel_pattern_recognition data/input.txt
else
    ./build/sequential/sequential_pattern_recognition data/input.txt
fi
