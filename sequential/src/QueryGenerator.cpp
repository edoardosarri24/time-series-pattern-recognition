#include "QueryGenerator.hpp"
#include "common.hpp"
#include <random>
#include <algorithm>
#include <stdexcept>

std::vector<float> QueryGenerator::generate(const std::vector<float>& data, int seed) {
    size_t num_timestamps = data.size() / constants::PADDED_DIM;
    if (num_timestamps < constants::QUERY_LENGTH) {
        throw std::runtime_error("Data is smaller than query length.");
    }

    std::mt19937 gen(seed == -1 ? std::random_device{}() : static_cast<unsigned int>(seed));
    // Valid start range: [0, N - M]
    std::uniform_int_distribution<size_t> dist_idx(0, num_timestamps - constants::QUERY_LENGTH);
    size_t start_idx = dist_idx(gen);

    std::normal_distribution<float> dist_noise(0.0f, 0.1f);

    std::vector<float> query;
    query.reserve(constants::QUERY_LENGTH * constants::PADDED_DIM);

    for (size_t i = 0; i < constants::QUERY_LENGTH; ++i) {
        size_t current_timestamp_idx = start_idx + i;
        size_t base_offset = current_timestamp_idx * constants::PADDED_DIM;

        for (size_t d = 0; d < constants::ORIGINAL_DIM; ++d) {
             float val = data[base_offset + d];
             val += dist_noise(gen);
             query.push_back(val);
        }
        // Padding for query as well (kept 0)
        for (size_t d = constants::ORIGINAL_DIM; d < constants::PADDED_DIM; ++d) {
            query.push_back(0.0f);
        }
    }

    return query;
}
