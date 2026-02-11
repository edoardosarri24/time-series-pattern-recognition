#include "common.hpp"
#include "data_loader.hpp"
#include "query_generator.hpp"
#include "cuda_utils.cuh"
#include "pattern_matching.cuh"
#include <iostream>
#include <limits>
#include <vector>

int main(int argc, char** argv) {
    
    // Argoument check.
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <input_file>" << std::endl;
        return EXIT_FAILURE;
    }
    std::string filename = argv[1];

    // Data loading (host).
    DataLoader loader(filename);
    try {
        loader.load();
    } catch (const std::exception& e) {
        std::cerr << "Error loading data: " << e.what() << std::endl;
        return EXIT_FAILURE;
    }
    size_t N = loader.get_num_timestamps();
    if (N < constants::QUERY_LENGTH) {
        std::cerr << "Input data too small." << std::endl;
        return EXIT_FAILURE;
    }
    std::cout << "Data loaded. N=" << N << std::endl;

    // Query generation (host).
    auto [query_soa, ground_truth_idx] = query_generator::generate(loader.get_aos_data(), 78);
    std::cout << "Query generated. Ground Truth Index: " << ground_truth_idx << std::endl;



    // Device Allocation
    float* d_input = nullptr;
    float* d_output = nullptr;
    
    size_t input_size = N * constants::DIM * sizeof(float);
    size_t output_size = N * sizeof(float); 

    CHECK_CUDA(cudaMalloc(&d_input, input_size));
    CHECK_CUDA(cudaMalloc(&d_output, output_size));

    // Data transfer.
        // Input SoA.
    CHECK_CUDA(cudaMemcpy(d_input, loader.get_soa_pinned_data(), input_size, cudaMemcpyHostToDevice));
        // Query SoA
    upload_query_to_constant_memory(query_soa.data(), query_soa.size() * sizeof(float));

    // Kernel execution.
    launch_sad_pattern_matching(d_input, d_output, N);

    // Result transfer.
    std::vector<float> h_output(N);
    CHECK_CUDA(cudaMemcpy(h_output.data(), d_output, output_size, cudaMemcpyDeviceToHost));

    // Host reduction.
    float min_sad = std::numeric_limits<float>::max();
    size_t best_index = 0;
    size_t valid_range = N - constants::QUERY_LENGTH;
    for (size_t i = 0; i <= valid_range; ++i) {
        if (h_output[i] < min_sad) {
            min_sad = h_output[i];
            best_index = i;
        }
    }

    // Reporting
    std::cout << "Best Match Index: " << best_index << std::endl;
    std::cout << "SAD Value: " << min_sad << std::endl;
    std::cout << "Ground Truth Index: " << ground_truth_idx << std::endl;

    // Cleanup
    CHECK_CUDA(cudaFree(d_input));
    CHECK_CUDA(cudaFree(d_output));
    return EXIT_SUCCESS;
}
