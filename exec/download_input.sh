#!/bin/bash

cd "$(dirname "$0")/script_download_input" || exit

if [ -f "../../data/input.txt" ]; then
    echo "input.txt alreaedy exists."
    exit 0
fi

uv run main.py "$@"