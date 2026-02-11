#ifndef CUDA_UTILS_CUH
#define CUDA_UTILS_CUH

#include <cuda_runtime.h>
#include <iostream>
#include <cstdlib>

#define CHECK_CUDA(call) { 
    const cudaError_t error = call; 
    if (error != cudaSuccess) { 
        std::cerr << "CUDA Error: " << __FILE__ << ":" << __LINE__ << " " 
                  << cudaGetErrorString(error) << std::endl; 
        exit(EXIT_FAILURE); 
    } 
}

#endif // CUDA_UTILS_CUH
