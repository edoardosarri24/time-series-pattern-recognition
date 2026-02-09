#include "SAD_distance.hpp"
#include "common.hpp"
#include <cmath>
#include <algorithm>
#include <stdexcept>

SADResult SAD_distance::find_best_match(const std::vector<float>& data, const std::vector<float>& query) {
    // Define and check the dimensions of input and query.
    size_t n_timestamps = data.size() / constants::PADDED_DIM;
    if (n_timestamps < constants::QUERY_LENGTH)
        throw std::invalid_argument("Data size is smaller than query size");

    // Promise to compiler of aliasing.
    const float* __restrict__ data_ptr = data.data();
    const float* __restrict__ query_ptr = query.data();
    
    // Iterate through all timestamps.
    size_t best_index = 0;
    float min_dist = std::numeric_limits<float>::max();
    for (size_t t=0; t <= n_timestamps - constants::QUERY_LENGTH; ++t) {
        float current_dist = 0.0f;
        for (size_t i=0; i < constants::QUERY_LENGTH; ++i) { // Compute SAD for the current window.
            size_t data_offset = (t+i) * constants::PADDED_DIM;
            size_t query_offset = i * constants::PADDED_DIM;
            for (size_t d=0; d < constants::PADDED_DIM; ++d) { // Iterate through dimensions.
                float diff = data_ptr[data_offset + d] - query_ptr[query_offset + d];
                current_dist += std::abs(diff);
            }
#ifdef ENABLE_EARLY_ABANDONING
            // Early abandoning.
            if (current_dist >= min_dist)
                break;
#endif
        }
        // Update the best index and the best distance.
        if (current_dist < min_dist) {
            min_dist = current_dist;
            best_index = t;
        }
    }

    return {best_index, min_dist};
}

float SAD_distance::compute_SAD(const std::vector<float>& data, const std::vector<float>& query, std::size_t start_index) {
    size_t n_timestamps = data.size() / constants::PADDED_DIM;
    if (start_index > n_timestamps - constants::QUERY_LENGTH)
        throw std::invalid_argument("Start index out of bounds");

    // Promise to compiler of aliasing.
    const float* __restrict__ data_ptr = data.data();
    const float* __restrict__ query_ptr = query.data();

    // Iterate through timestamps.
    float current_dist = 0.0f;
    for (size_t i=0; i < constants::QUERY_LENGTH; ++i) {
        size_t data_offset = (start_index+i) * constants::PADDED_DIM;
        size_t query_offset = i * constants::PADDED_DIM;
        for (size_t d=0; d < constants::PADDED_DIM; ++d) {
            float diff = data_ptr[data_offset + d] - query_ptr[query_offset + d];
            current_dist += std::abs(diff);
        }
    }
    return current_dist;
}
