#ifndef KERNEL_CUH
#define KERNEL_CUH

#include <vector>
#include "common.hpp"

namespace kernel {
    // Copy the query from the cpu memory to the device constant memory.
    void upload_query_to_constant_memory(const float* host_query_soa, size_t size_bytes);

    // Copy the input data from cpu memory to device global memory.
    void upload_input_to_global_memory(const float* host_input_soa, float* device_input, size_t size_bytes);

    // Copy the reduced results (sad values and indices) from device global memory to cpu memory.
    void download_results_to_host_memory(
        std::vector<float>& host_values,
        const float* device_values,
        std::vector<int>& host_indexes,
        const int* device_indexes,
        size_t grid_size);

    // Launch the kernel with block reduction support
    void launch_sad_pattern_matching(
        const float* device_input,
        float* device_output_values,
        int* device_output_indexes,
        size_t data_lenght);
}

#endif // KERNEL_CUH
