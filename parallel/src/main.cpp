#include "common.hpp"
#include "data_loader.hpp"
#include "query_generator.hpp"
#include "cuda_utils.cuh"
#include "kernel.cuh"
#include <iostream>
#include <limits>
#include <vector>
#include <chrono> // For timing
#include <iomanip> // For std::fixed and std::setprecision

int main(int argc, char** argv) {

    auto start_total = std::chrono::high_resolution_clock::now(); // Start total timer

    // Argoument check.
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <input_file>" << std::endl;
        return EXIT_FAILURE;
    }
    std::string filename = argv[1];

    // Data loading.
    auto start_load = std::chrono::high_resolution_clock::now();
    DataLoader loader(filename);
    try {
        loader.load();
    } catch (const std::exception& e) {
        std::cerr << "Error loading data: " << e.what() << std::endl;
        return EXIT_FAILURE;
    }
    size_t data_lenght = loader.get_num_timestamps();
    if (data_lenght < constants::QUERY_LENGTH) {
        std::cerr << "Input data too small." << std::endl;
        return EXIT_FAILURE;
    }
    auto end_load = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed_load = end_load - start_load;
    std::cout << "Data loaded. data_lenght=" << data_lenght << std::endl;

    // Query generation.
    auto start_query_gen = std::chrono::high_resolution_clock::now();
    auto [query_soa, ground_truth_idx] = query_generator::generate(loader.get_aos_data(), 78);
    auto end_query_gen = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed_query_gen = end_query_gen - start_query_gen;
    std::cout << "Query generated. Ground Truth Index: " << ground_truth_idx << std::endl;

    // Device memory allocation.
    auto start_dev_alloc = std::chrono::high_resolution_clock::now();
    float* device_input = nullptr; // Input GPU memory.
    float* device_output = nullptr; // Output GPU memory.
    size_t input_size = data_lenght * constants::DIM * sizeof(float);
    size_t output_size = data_lenght * sizeof(float);
    CHECK_CUDA(cudaMalloc(&device_input, input_size));
    CHECK_CUDA(cudaMalloc(&device_output, output_size));
    auto end_dev_alloc = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed_dev_alloc = end_dev_alloc - start_dev_alloc;

    // Data transfer (Host to Device).
    auto start_h2d_transfer = std::chrono::high_resolution_clock::now();
    kernel::upload_input_to_global_memory(loader.get_soa_pinned_data(), device_input, input_size);
    kernel::upload_query_to_constant_memory(query_soa.data(), query_soa.size() * sizeof(float));
    auto end_h2d_transfer = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed_h2d_transfer = end_h2d_transfer - start_h2d_transfer;

    // Kernel execution.
    auto start_kernel_exec = std::chrono::high_resolution_clock::now();
    kernel::launch_sad_pattern_matching(device_input, device_output, data_lenght);
    auto end_kernel_exec = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed_kernel_exec = end_kernel_exec - start_kernel_exec;

    // Result transfer (Device to Host).
    auto start_d2h_transfer = std::chrono::high_resolution_clock::now();
    std::vector<float> host_result(data_lenght);
    kernel::download_output_from_global_memory(host_result, device_output, output_size);
    auto end_d2h_transfer = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed_d2h_transfer = end_d2h_transfer - start_d2h_transfer;

    // Host reduction.
    auto start_host_reduction = std::chrono::high_resolution_clock::now();
    float min_sad = std::numeric_limits<float>::max();
    size_t best_index = 0;
    size_t valid_range = data_lenght - constants::QUERY_LENGTH;
    for (size_t i=0; i <= valid_range; ++i) {
        if (host_result[i] < min_sad) {
            min_sad = host_result[i];
            best_index = i;
        }
    }
    auto end_host_reduction = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed_host_reduction = end_host_reduction - start_host_reduction;

    // Reporting
    std::cout << "\n--- Matching results ---\n";
    std::cout << "Best Match Index: " << best_index << std::endl;
    std::cout << "SAD Value: " << min_sad << std::endl;
    std::cout << "Ground Truth Index: " << ground_truth_idx << std::endl;
    std::cout << "\n--- Timing sesults ---\n";
    std::cout << std::fixed << std::setprecision(4);
    std::cout << "Data Loading Time:          " << elapsed_load.count() << " s\n";
    std::cout << "Query Generation Time:      " << elapsed_query_gen.count() << " s\n";
    std::cout << "Device Allocation Time:     " << elapsed_dev_alloc.count() << " s\n";
    std::cout << "H2D Transfer Time:          " << elapsed_h2d_transfer.count() << " s\n";
    std::cout << "Kernel Execution Time:      " << elapsed_kernel_exec.count() << " s\n";
    std::cout << "D2H Transfer Time:          " << elapsed_d2h_transfer.count() << " s\n";
    std::cout << "Host Reduction Time:        " << elapsed_host_reduction.count() << " s\n";
    auto end_total = std::chrono::high_resolution_clock::now(); // Stop total timer
    std::chrono::duration<double> elapsed_total = end_total - start_total;
    std::cout << "Total Execution Time:       " << elapsed_total.count() << " s\n";

    // Cleanup
    CHECK_CUDA(cudaFree(device_input));
    CHECK_CUDA(cudaFree(device_output));
    return EXIT_SUCCESS;
}
