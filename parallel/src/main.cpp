#include <iostream>
#include <cstdlib>
#include <cuda_runtime.h>

// CUDA Kernel
__global__ void hello_cuda() {
    printf("Hello World from GPU thread %d (Block %d)!\n", threadIdx.x, blockIdx.x);
}

// Host code
int main() {
    std::cout << "Starting GPU Connectivity Test..." << std::endl;

    // Launch kernel with 1 block and 1 thread
    hello_cuda<<<1, 1>>>();

    // Check for launch errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "CUDA Kernel Launch Error: " << cudaGetErrorString(err) << std::endl;
        return EXIT_FAILURE;
    }

    // Wait for GPU to finish
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        std::cerr << "CUDA Synchronize Error: " << cudaGetErrorString(err) << std::endl;
        return EXIT_FAILURE;
    }

    std::cout << "Test Complete." << std::endl;
    return EXIT_SUCCESS;
}
