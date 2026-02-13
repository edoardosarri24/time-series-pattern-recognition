# Time Series Pattern Recognition
This project implements a Time Series Pattern Recognition algorithm in C++ using both a sequential approach and a parallel approach with CUDA. The goal is to find the best match for a given query pattern within a large time series dataset using the Sum of Absolute Differences (SAD) metric.

For a detailed analysis and explanation of the implementation see the [report.pdf](report.pdf) file.

### Dependencies
- CMake
- C++ Compiler (C++17 support required)
- CUDA Toolkit
- Python
- GPerftools (for profiling)

### Execution
The project execution is a two-step process:

##### Data Download
  Before running the application, download the input dataset with the following command:
```bash
./exec/download_input.sh [--multiplier <int>] [--noise <float>]
```
- `multiplier`: Optional. Multiplies the dataset size by generating noisy variations (default: 1).
- `noise`: Optional. Maximum amplitude of random noise added to augmented data (default: 0.01).

##### Build & Run
Use the scripts in the `exec/` directory to build and run the project automatically.
- Performance (Release):
    ```bash
    ./exec/release_execution.sh [seq|par] [ea] [p] [q=<value>] [bs=<value>]
    ```
    - Builds with `-O3 -march=native` for maximum performance. Use this for benchmarking.
    - `ea`: Optional. Enables "Early Abandoning" optimization ONLY for the sequential version.
    - `p`: Optional. Enables padding for cache alignment (default: OFF).
    - `q=<value>`: Optional. Sets the query length (default: 64).
    - `bs=<value>`: Optional. Sets the block size. Valid ONLY for the parallel version.
- Development (Debug): Builds with debug symbols and no optimizations.
    ```bash
    ./exec/debug_execution.sh [seq|par]
    ```
- Profiling: Builds in Release mode with profiling enabled.
    ```bash
    ./exec/profiling.sh [seq|par]
    ```
- Sanitizers (Debug):
    *   `./exec/AUBsanitizer.sh [seq|par]` - Runs with Address and Undefined Behavior Sanitizers.
    *   `./exec/Msanitizer.sh [seq|par]` - Runs with Memory Sanitizer (Linux only).

### Structures
.
├── CMakeLists.txt
├── README.md
├── agent.md - Agent specific documentation (gemini used)
├── exec - Helper scripts for building, running, and profiling
│   ├── AUBsanitizer.sh - Runs Address and Undefined Behavior Sanitizers
│   ├── Msanitizer.sh - Runs Memory Sanitizer
│   ├── debug_execution.sh - Builds and runs in Debug mode
│   ├── download_input.sh - Downloads and generates input datasets
│   ├── profiling.sh - Builds and runs with profiling enabled
│   ├── release_execution.sh - Builds and runs in Release mode for performance
│   └── script_download_input - Python source code for data download
├── parallel/ - Source code for the parallel CUDA implementation
├── report/ - Source code for the project report (LaTeX)
├── report.pdf - Project report
├── sequential/ - Source code for the sequential C++ implementation
├── slides/ - Source code for the slides (LaTeX)
└── slides.pdf - Presentation slides