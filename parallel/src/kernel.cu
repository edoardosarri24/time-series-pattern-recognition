#include "kernel.cuh"
#include "cuda_utils.cuh"
#include <cstdio>

// Contant memory allocation on device.
__constant__ float query[constants::DIM * constants::QUERY_LENGTH];

namespace kernel {
    // Il Kernel originale
    __global__ void sad_pattern_matching_kernel(const float* __restrict__ input, float* __restrict__ output, size_t data_lenght) {
        size_t tid = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
        if (tid > data_lenght - constants::QUERY_LENGTH)
            return;
        float sad = 0.0f;
        for (size_t d=0; d < constants::DIM; ++d) {
            size_t query_base = d * constants::QUERY_LENGTH; // Query is SoA: d * data_lenght + m
            size_t input_base = d * data_lenght + tid; // Input is SoA: d * data_lenght + (tid + m)
            for (size_t m=0; m < constants::QUERY_LENGTH; ++m) {
                float q_val = query[query_base + m];
                float in_val = input[input_base + m];
                sad += fabsf(q_val - in_val); // Single precision abs in cuda.
            }
        }
        output[tid] = sad;
    }

    void upload_query_to_constant_memory(const float* host_query_soa, size_t size_bytes) {
        CHECK_CUDA(cudaMemcpyToSymbol(query, host_query_soa, size_bytes));
        CHECK_CUDA(cudaDeviceSynchronize());
    }

    void upload_input_to_global_memory(const float* host_input_soa, float* device_input, size_t size_bytes) {
        CHECK_CUDA(cudaMemcpy(device_input, host_input_soa, size_bytes, cudaMemcpyHostToDevice));
        CHECK_CUDA(cudaDeviceSynchronize());
    }

    void download_output_from_global_memory(std::vector<float>& host_output, const float* device_output, size_t size_bytes) {
        CHECK_CUDA(cudaMemcpy(host_output.data(), device_output, size_bytes, cudaMemcpyDeviceToHost));
        CHECK_CUDA(cudaDeviceSynchronize());
    }

    void launch_sad_pattern_matching(const float* device_input, float* device_output, size_t data_lenght) {
        int block_size = constants::BLOCK_SIZE;
        int grid_size = (int)((data_lenght + block_size - 1) / block_size);
        printf("Launching Kernel with Grid=%d, Block=%d\n", grid_size, block_size);
        sad_pattern_matching_kernel<<<grid_size, block_size>>>(device_input, device_output, data_lenght);
        CHECK_CUDA(cudaGetLastError());
        CHECK_CUDA(cudaDeviceSynchronize());
    }
} // namespace kernel