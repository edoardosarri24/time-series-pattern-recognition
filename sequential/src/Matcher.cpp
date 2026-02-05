#include "Matcher.hpp"
#include "common.hpp"
#include <cmath>
#include <algorithm>

MatchResult Matcher::find_best_match(const std::vector<float>& data, const std::vector<float>& query) {
    size_t n_timestamps = data.size() / constants::PADDED_DIM;
    size_t m_query = constants::QUERY_LENGTH;

    if (n_timestamps < m_query) {
        return {0, std::numeric_limits<float>::max()};
    }

    size_t best_index = 0;
    float min_dist = std::numeric_limits<float>::max();

    // Iterate through all possible start positions
    for (size_t i = 0; i <= n_timestamps - m_query; ++i) {
        float current_dist = 0.0f;
        
        // Compute SAD for the current window
        for (size_t j = 0; j < m_query; ++j) {
            size_t data_offset = (i + j) * constants::PADDED_DIM;
            size_t query_offset = j * constants::PADDED_DIM;

            // Inner loop over dimensions
            // Note: Since padding is 0 in both query and data, 
            // iterating up to PADDED_DIM adds 0 to distance, which is correct.
            // This might help auto-vectorization.
            for (size_t d = 0; d < constants::PADDED_DIM; ++d) {
                float diff = data[data_offset + d] - query[query_offset + d];
                current_dist += std::abs(diff);
            }

            // Early Abandoning
            if (current_dist >= min_dist) {
                break; 
            }
        }

        if (current_dist < min_dist) {
            min_dist = current_dist;
            best_index = i;
        }
    }

    return {best_index, min_dist};
}
