#ifndef PATTERN_MATCHING_CUH
#define PATTERN_MATCHING_CUH

#include <vector>
#include "common.hpp"

namespace kernel {
    // Copy the query from the cpu memory to the device constant memory.
    void upload_query_to_constant_memory(const float* host_query_soa, size_t size_bytes);
    // Copy the input data from cpu memory to device global memory.
    void upload_input_to_global_memory(const float* host_input_soa, float* device_input, size_t size_bytes);
    // Copy the output data from device global memory to cpu memory.
    void download_output_from_global_memory(std::vector<float>& host_output, const float* device_output, size_t size_bytes);
    // Lanch the kernel
    void launch_sad_pattern_matching(const float* device_input, float* device_output, size_t data_lenght);
} // namespace kernel

#endif // PATTERN_MATCHING_CUH
