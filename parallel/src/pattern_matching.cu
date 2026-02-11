#include "pattern_matching.cuh"
#include "cuda_utils.cuh"
#include <cstdio>

// Memoria costante definita staticamente in questo modulo
__constant__ float d_query[constants::DIM * constants::QUERY_LENGTH];

// Il Kernel originale
__global__ void sad_pattern_matching_kernel(const float* __restrict__ input, float* __restrict__ output, size_t N) {
    size_t tid = (size_t)blockIdx.x * blockDim.x + threadIdx.x;

    if (tid > N - constants::QUERY_LENGTH) return;

    float sad = 0.0f;

    for (size_t d = 0; d < constants::DIM; ++d) {
        // Query is SoA: d * M + m
        size_t query_base = d * constants::QUERY_LENGTH;
        
        // Input is SoA: d * N + (tid + m)
        size_t input_base = d * N + tid;

        for (size_t m = 0; m < constants::QUERY_LENGTH; ++m) {
            float q_val = d_query[query_base + m];
            float in_val = input[input_base + m];
            sad += fabsf(q_val - in_val);
        }
    }

    output[tid] = sad;
}

// Wrapper per caricare la memoria costante
void upload_query_to_constant_memory(const float* host_query_soa, size_t size_bytes) {
    // Nota: d_query è visibile solo in questo file, quindi usiamo cudaMemcpyToSymbol qui
    CHECK_CUDA(cudaMemcpyToSymbol(d_query, host_query_soa, size_bytes));
}

// Wrapper per calcolare la grid e lanciare il kernel
void launch_sad_pattern_matching(const float* d_input, float* d_output, size_t N) {
    int block_size = constants::BLOCK_SIZE;
    // Calcolo della grid size per coprire N
    int grid_size = (int)((N + block_size - 1) / block_size);

    printf("Launching Kernel with Grid=%d, Block=%d\n", grid_size, block_size);

    sad_pattern_matching_kernel<<<grid_size, block_size>>>(d_input, d_output, N);
    
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaDeviceSynchronize());
}