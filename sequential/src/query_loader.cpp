#include "query_loader.hpp"
#include "common.hpp"
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <iostream>

std::vector<float> query_loader::load(const std::string& filename) {
    // Init.
    std::ifstream file(filename);
    if (!file.is_open())
        throw std::runtime_error("Could not open query file: " + filename);
    std::vector<float> query;
    query.reserve(constants::QUERY_LENGTH * constants::PADDED_DIM);
    std::string line;
    size_t lines_read = 0;
    // Query construction.
    while (lines_read < constants::QUERY_LENGTH && std::getline(file, line)) {
        // Skip empty lines.
        if (line.empty()) continue;
        // Add values.
        std::istringstream iss(line);
        float val;
        for (size_t d=0; d < constants::ORIGINAL_DIM; ++d) {
            if (!(iss >> val))
                throw std::runtime_error("Error parsing query file at line " + std::to_string(lines_read + 1));
            query.push_back(val);
        }
        // Add padding if required.
        for (size_t d = constants::ORIGINAL_DIM; d < constants::PADDED_DIM; ++d)
            query.push_back(0.0f);
        lines_read++;
    }
    if (lines_read < constants::QUERY_LENGTH)
        throw std::runtime_error("Query file too short. Expected "
            + std::to_string(constants::QUERY_LENGTH) + " lines, got " + std::to_string(lines_read));
    return query;
}
