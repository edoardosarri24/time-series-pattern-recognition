#include "common.hpp"
#include "data_loader.hpp"
#include "query_generator.hpp"
#include "cuda_utils.cuh"
#include <iostream>
#include <limits>
#include <cmath>
#include <vector>

// Constant memory for Query (SoA)
// Size: DIM * QUERY_LENGTH floats.
__constant__ float d_query[constants::DIM * constants::QUERY_LENGTH];

__global__ void sad_pattern_matching(const float* __restrict__ input, float* __restrict__ output, size_t N) {
    size_t tid = (size_t)blockIdx.x * blockDim.x + threadIdx.x;

    if (tid > N - constants::QUERY_LENGTH) return;

    float sad = 0.0f;

    for (size_t d = 0; d < constants::DIM; ++d) {
        // Query is SoA: d * M + m
        // We know M is constants::QUERY_LENGTH
        size_t query_base = d * constants::QUERY_LENGTH;
        
        // Input is SoA: d * N + (tid + m)
        // Base for dimension d is d * N
        size_t input_base = d * N + tid;

        for (size_t m = 0; m < constants::QUERY_LENGTH; ++m) {
            float q_val = d_query[query_base + m];
            float in_val = input[input_base + m];
            sad += fabsf(q_val - in_val);
        }
    }

    output[tid] = sad;
}

int main(int argc, char** argv) {
    if (argc != 2) {
        std::cerr << "Usage: " << argv[0] << " <input_file>" << std::endl;
        return 1;
    }

    std::string filename = argv[1];

    // 1. Data Loading (Host)
    // Uses mmap and transposes AoS to SoA in Pinned Memory
    DataLoader loader(filename);
    try {
        loader.load();
    } catch (const std::exception& e) {
        std::cerr << "Error loading data: " << e.what() << std::endl;
        return 1;
    }

    size_t N = loader.get_num_timestamps();
    if (N < constants::QUERY_LENGTH) {
        std::cerr << "Input data too small." << std::endl;
        return 1;
    }

    std::cout << "Data loaded. N=" << N << std::endl;

    // 2. Query Generation (Host)
    // Uses seed 78 for reproducibility (matches sequential version)
    auto [query_soa, ground_truth_idx] = query_generator::generate(loader.get_aos_data(), 78);
    
    std::cout << "Query generated. Ground Truth Index: " << ground_truth_idx << std::endl;

    // 3. Device Allocation
    float* d_input = nullptr;
    float* d_output = nullptr;
    
    size_t input_size = N * constants::DIM * sizeof(float);
    size_t output_size = N * sizeof(float); 

    CHECK_CUDA(cudaMalloc(&d_input, input_size));
    CHECK_CUDA(cudaMalloc(&d_output, output_size));

    // 4. Data Transfer
    // Input SoA (Pinned -> Device)
    CHECK_CUDA(cudaMemcpy(d_input, loader.get_soa_pinned_data(), input_size, cudaMemcpyHostToDevice));

    // Query SoA (Vector -> Constant)
    CHECK_CUDA(cudaMemcpyToSymbol(d_query, query_soa.data(), query_soa.size() * sizeof(float)));

    // 5. Kernel Execution
    int block_size = constants::BLOCK_SIZE;
    // Calculate grid size. Ensure we cover all N (or at least up to N - QUERY_LENGTH)
    // Using (N + block_size - 1) / block_size covers all N.
    // The kernel returns early if tid > N - QUERY_LENGTH.
    int grid_size = (int)((N + block_size - 1) / block_size);

    std::cout << "Launching Kernel with Grid=" << grid_size << ", Block=" << block_size << std::endl;

    sad_pattern_matching<<<grid_size, block_size>>>(d_input, d_output, N);
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaDeviceSynchronize());

    // 6. Result Transfer
    std::vector<float> h_output(N);
    CHECK_CUDA(cudaMemcpy(h_output.data(), d_output, output_size, cudaMemcpyDeviceToHost));

    // 7. Host Reduction
    float min_sad = std::numeric_limits<float>::max();
    size_t best_index = 0;

    size_t valid_range = N - constants::QUERY_LENGTH;
    for (size_t i = 0; i <= valid_range; ++i) {
        if (h_output[i] < min_sad) {
            min_sad = h_output[i];
            best_index = i;
        }
    }

    // 8. Reporting
    std::cout << "Best Match Index: " << best_index << std::endl;
    std::cout << "SAD Value: " << min_sad << std::endl;
    std::cout << "Ground Truth Index: " << ground_truth_idx << std::endl;

    // Cleanup
    CHECK_CUDA(cudaFree(d_input));
    CHECK_CUDA(cudaFree(d_output));

    return 0;
}