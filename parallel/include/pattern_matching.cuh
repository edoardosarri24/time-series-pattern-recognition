#ifndef PATTERN_MATCHING_CUH
#define PATTERN_MATCHING_CUH

#include <vector>
#include "common.hpp"

// Copy the query from the cpu memory to the device constant memory.
void upload_query_to_constant_memory(const float* host_query_soa, size_t size_bytes);
// Lanch the kernel
void launch_sad_pattern_matching(const float* d_input, float* d_output, size_t N);

#endif // PATTERN_MATCHING_CUH
