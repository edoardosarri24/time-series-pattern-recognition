# Time Series Pattern Recognition

### Execution
The project execution is a two-step process:

##### Data Download
  Before running the application, download the input dataset with the following command:
```bash
./exec/download_input.sh [--multiplier <int>] [--noise <float>]
```
- `multiplier`: Optional. Multiplies the dataset size by generating noisy variations (default: 1).
- `noise`: Optional. Maximum amplitude of random noise added to augmented data (default: 0.01).
- *Note:* The script checks if `data/input.txt` already exists to avoid redundant downloads.

##### Build & Run
Use the scripts in the `exec/` directory to build and run the project automatically.
- Performance (Release):
    ```bash
    ./exec/release_execution.sh [seq|par] [ea] [p] [q=<value>]
    ```
    - Builds with `-O3 -march=native` for maximum performance. Use this for benchmarking.
    - `ea`: Optional. Enables "Early Abandoning" optimization ONLY for the sequential version.
    - `p`: Optional. Enables padding for cache alignment (default: OFF).
    - `q=<value>`: Optional. Sets the query length (default: 64).
- Development (Debug):
    ```bash
    ./exec/debug_execution.sh [seq|par]
    ```
    - Builds with debug symbols and no optimizations. Best for debugging logic.
- Profiling:
    ```bash
    ./exec/profiling.sh [seq|par]
    ```
    - Builds in Release mode with profiling enabled. Generates CPU profile reports (PDF/Text) in `[sequential|parallel]/result_profiling/`.
- Sanitizers (Debug):
    *   `./exec/AUBsanitizer.sh [seq|par]` - Runs with Address and Undefined Behavior Sanitizers.
    *   `./exec/Msanitizer.sh [seq|par]` - Runs with Memory Sanitizer (Linux only, requires Clang).
