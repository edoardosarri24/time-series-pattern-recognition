#include "DataLoader.hpp"
#include "common.hpp"
#include <fstream>
#include <stdexcept>
#include <iostream>

std::vector<float> DataLoader::load(const std::string& filename) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        throw std::runtime_error("Could not open file: " + filename);
    }

    std::vector<float> data;
    
    // Reserve memory to avoid reallocations (heuristic estimation)
    // Assuming file size / average chars per float roughly, but simple vector growth is fine for now
    // or we could check file size. 
    
    float temp_buffer[constants::ORIGINAL_DIM];

    while (file >> temp_buffer[0]) {
        // Read the remaining dimensions for the current timestamp
        for (size_t i = 1; i < constants::ORIGINAL_DIM; ++i) {
            if (!(file >> temp_buffer[i])) {
                throw std::runtime_error("File format error: incomplete timestamp data.");
            }
        }

        // Push data with padding
        for (size_t i = 0; i < constants::ORIGINAL_DIM; ++i) {
            data.push_back(temp_buffer[i]);
        }
        // Padding
        for (size_t i = constants::ORIGINAL_DIM; i < constants::PADDED_DIM; ++i) {
            data.push_back(0.0f);
        }
    }

    return data;
}
