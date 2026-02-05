#!/bin/bash

if [ -z "$1" ]; then
    echo "Error: Missing argument. Usage: $0 [seq|par]"
    exit 1
fi

MODE=$1

if [[ "$MODE" != "seq" && "$MODE" != "par" ]]; then
    echo "Error: Invalid argument '$MODE'. Usage: $0 [seq|par]"
    exit 1
fi

echo "building (Debug)..."
rm -rf build
cmake -S . -B build -DCMAKE_BUILD_TYPE=Debug
cmake --build build

echo "executing ($MODE)..."
if [ "$MODE" == "par" ]; then
    ./build/parallel/parallel_pattern_recognition data/input.txt
else
    ./build/sequential/sequential_pattern_recognition data/input.txt
fi