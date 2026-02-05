#include "SAD_distance.hpp"
#include "common.hpp"
#include <cmath>
#include <algorithm>
#include <stdexcept>

SADResult SAD_distance::find_best_match(const std::vector<float>& data, const std::vector<float>& query) {
    // define and check the dimensions of input and query.
    size_t n_timestamps = data.size() / constants::PADDED_DIM;
    if (n_timestamps < constants::QUERY_LENGTH)
        throw std::invalid_argument("Data size is smaller than query size");

    size_t best_index = 0;
    float min_dist = std::numeric_limits<float>::max();

    // Iterate through all timestamps.
    for (size_t t=0; t <= n_timestamps - constants::QUERY_LENGTH; ++t) {
        float current_dist = 0.0f;


        for (size_t i=0; i < constants::QUERY_LENGTH; ++i) { // Compute SAD for the current window.
            size_t data_offset = (t+i) * constants::PADDED_DIM;
            size_t query_offset = i * constants::PADDED_DIM;
            for (size_t d=0; d < constants::PADDED_DIM; ++d) { // Iterate through dimensions.
                float diff = data[data_offset + d] - query[query_offset + d];
                current_dist += std::abs(diff);
            }
            if (current_dist >= min_dist) // Early Abandoning
                break;
        }

        if (current_dist < min_dist) {
            min_dist = current_dist;
            best_index = t;
        }
    }

    return {best_index, min_dist};
}
