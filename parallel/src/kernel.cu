#include "kernel.cuh"
#include "cuda_utils.cuh"
#include <cstdio>
#include <cfloat> // For FLT_MAX

// Constant memory allocation on device.
__constant__ float query[constants::DIM * constants::QUERY_LENGTH];

namespace kernel {

    __global__ void sad_pattern_matching_kernel(
            const float* __restrict__ input,
            float* __restrict__ output_values,
            int* __restrict__ output_idexes,
            size_t data_lenght) {
        // Shared memory mllocation.
        __shared__ float shared_values[constants::BLOCK_SIZE];
        __shared__ int shared_indexes[constants::BLOCK_SIZE];
        // Local accumulation.
        size_t tid = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
        unsigned int tid_local = threadIdx.x;
        float sad = FLT_MAX; // For the thread out of boundaries
        if (tid <= data_lenght - constants::QUERY_LENGTH) {
            sad = 0.0f;
            for (size_t d=0; d < constants::DIM; ++d) {
                size_t query_base = d * constants::QUERY_LENGTH;
                size_t input_base = d * data_lenght + tid;
                for (size_t m=0; m < constants::QUERY_LENGTH; ++m) {
                    float q_val = query[query_base + m];
                    float in_val = input[input_base + m];
                    sad += fabsf(q_val - in_val);
                }
            }
        }
        // Load each value into shared memory.
        shared_values[tid_local] = sad;
        shared_indexes[tid_local] = (int)tid;
        __syncthreads();
        // Block reduction. Follow the Fold concept.
        for (unsigned int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
            if (tid_local < stride) {
                if (shared_values[tid_local + stride] < shared_values[tid_local]) {
                    shared_values[tid_local] = shared_values[tid_local + stride];
                    shared_indexes[tid_local] = shared_indexes[tid_local + stride];
                }
            }
            __syncthreads();
        }
        // Result write-back.
        if (tid_local == 0) { // The best value is only in the thread 0.
            output_values[tid_local] = shared_values[0];
            output_idexes[tid_local] = shared_indexes[0];
        }
    }

    void upload_query_to_constant_memory(const float* host_query_soa, size_t size_bytes) {
        CHECK_CUDA(cudaMemcpyToSymbol(query, host_query_soa, size_bytes));
        CHECK_CUDA(cudaDeviceSynchronize());
    }

    void upload_input_to_global_memory(const float* host_input_soa, float* device_input, size_t size_bytes) {
        CHECK_CUDA(cudaMemcpy(device_input, host_input_soa, size_bytes, cudaMemcpyHostToDevice));
        CHECK_CUDA(cudaDeviceSynchronize());
    }

    void download_results_to_host_memory(
            std::vector<float>& host_values,
            const float* device_values,
            std::vector<int>& host_indexes,
            const int* device_indexes,
            size_t grid_size) {
        CHECK_CUDA(cudaMemcpy(host_values.data(), device_values, grid_size * sizeof(float), cudaMemcpyDeviceToHost));
        CHECK_CUDA(cudaMemcpy(host_indexes.data(), device_indexes, grid_size * sizeof(int), cudaMemcpyDeviceToHost));
        CHECK_CUDA(cudaDeviceSynchronize());
    }

    void launch_sad_pattern_matching(
            const float* device_input,
            float* device_output_values,
            int* device_output_indexes,
            size_t data_lenght) {
        int block_size = constants::BLOCK_SIZE;
        // block_size multiplier that cover whole data.
        int grid_size = (int)((data_lenght + block_size - 1) / block_size);
        printf("Launching Kernel with Grid=%d, Block=%d\n", grid_size, block_size);
        sad_pattern_matching_kernel<<<grid_size, block_size>>>(device_input, device_output_values, device_output_indexes, data_lenght);
        CHECK_CUDA(cudaGetLastError());
        CHECK_CUDA(cudaDeviceSynchronize());
    }

}
