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

    // Load into AoS struct.
    std::vector<float> query_aos;
    query_aos.reserve(constants::QUERY_LENGTH * constants::DIM);
    std::string line;
    size_t lines_read = 0;
    while (lines_read < constants::QUERY_LENGTH && std::getline(file, line)) {
        if (line.empty()) continue;
        std::istringstream iss(line);
        float val;
        for (size_t d = 0; d < constants::DIM; ++d) {
            if (!(iss >> val)) {
                throw std::runtime_error("Error parsing query file at line " + std::to_string(lines_read + 1));
            }
            query_aos.push_back(val);
        }
        lines_read++;
    }
    if (lines_read < constants::QUERY_LENGTH)
        throw std::runtime_error("Query file too short.");
    // Convert to SoA
    std::vector<float> query_soa;
    query_soa.reserve(constants::QUERY_LENGTH * constants::DIM);
    for (size_t d = 0; d < constants::DIM; ++d) {
        for (size_t m = 0; m < constants::QUERY_LENGTH; ++m)
            query_soa.push_back(query_aos[m * constants::DIM + d]);
    }
    return query_soa;
}
