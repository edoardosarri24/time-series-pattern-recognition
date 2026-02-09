#!/bin/bash

if [ -z "$1" ]; then
    echo "Error: Missing argument. Usage: $0 [seq|par] [ea] [p] [q=<value>]"
    exit 1
fi

MODE=$1
shift # Remove the first argument (mode)

CMAKE_FLAGS="-DCMAKE_BUILD_TYPE=Release"
ENABLE_EA="OFF"
ENABLE_PADDING="OFF"
QUERY_LENGTH_VAL=64

# Iterate over remaining arguments
for arg in "$@"; do
    if [[ "$arg" == "ea" ]]; then
        ENABLE_EA="ON"
    elif [[ "$arg" == "p" ]]; then
        ENABLE_PADDING="ON"
    elif [[ "$arg" == q=* ]]; then
        QUERY_LENGTH_VAL="${arg#*=}"
    fi
done

# Apply settings
if [[ "$ENABLE_EA" == "ON" ]]; then
    echo "Early Abandoning ENABLED"
else
    echo "Early Abandoning DISABLED"
fi
CMAKE_FLAGS="$CMAKE_FLAGS -DENABLE_EARLY_ABANDONING=$ENABLE_EA"

if [[ "$ENABLE_PADDING" == "ON" ]]; then
    echo "Padding ENABLED"
else
    echo "Padding DISABLED"
fi
CMAKE_FLAGS="$CMAKE_FLAGS -DENABLE_PADDING=$ENABLE_PADDING"

echo "Query Length: $QUERY_LENGTH_VAL"
CMAKE_FLAGS="$CMAKE_FLAGS -DQUERY_LENGTH=$QUERY_LENGTH_VAL"

if [[ "$MODE" != "seq" && "$MODE" != "par" ]]; then
    echo "Error: Invalid argument '$MODE'. Usage: $0 [seq|par] [ea] [p] [q=<value>]"
    exit 1
fi

# Building.
echo "building (Release)..."
rm -rf build
cmake -S . -B build $CMAKE_FLAGS
cmake --build build

# Executing
echo "executing ($MODE)..."
if [ "$MODE" == "par" ]; then
    ./build/parallel/parallel_pattern_recognition data/input.txt
else
    ./build/sequential/sequential_pattern_recognition data/input.txt
fi