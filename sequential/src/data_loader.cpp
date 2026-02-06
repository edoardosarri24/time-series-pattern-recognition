#include "data_loader.hpp"
#include "common.hpp"
#include <fstream>
#include <stdexcept>
#include <iostream>

std::vector<float> data_loader::load(const std::string& filename) {
    std::ifstream file(filename);
    if (!file.is_open())
        throw std::runtime_error("Could not open file: " + filename);

    // Define the output and estimate the number of element. This reduce the reallocation of the vector when is full.
    std::vector<float> data;
    file.seekg(0, std::ios::end); // Move the cursor to the end.
    std::streampos fileSize = file.tellg(); // Return the size.
    file.seekg(0, std::ios::beg); // Move the cursor at the benning.
    if (fileSize > 0) {
        constexpr size_t estimated_bytes_per_row = 15 * 6; // 15 charaters for float (estimate), 6 flow for line.
        size_t estimated_timestamps = static_cast<size_t>(fileSize) / estimated_bytes_per_row;
        size_t estimated_total_floats = estimated_timestamps * constants::PADDED_DIM;
        // Riserviamo la memoria. Questo evita copie costose durante i push_back.
        data.reserve(estimated_total_floats);
    }

    // Iterate through timestamp.
    float dimension_value;
    while (file >> dimension_value) { // Read and push the first dimension of the current timestamp.
        data.push_back(dimension_value);
        for (size_t i=1; i < constants::ORIGINAL_DIM; ++i) { // Read and push the remaining dimensions of the current timestamp.
            if (!(file >> dimension_value))
                throw std::runtime_error("File format error: incomplete timestamp data.");
            data.push_back(dimension_value);
        }
        // Push the padding.
        for (size_t i = constants::ORIGINAL_DIM; i < constants::PADDED_DIM; ++i)
            data.push_back(0.0f);
    }

    return data;
}
