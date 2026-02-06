#ifndef QUERY_GENERATOR_HPP
#define QUERY_GENERATOR_HPP

#include <vector>
#include <cstddef>

namespace query_generator {

    struct QueryResult {
        std::vector<float> query;
        size_t start_index;
    };

    /**
     * Generates a query by extracting a subsequence of length constants::QUERY_LENGTH
     * starting at a random index from the data.
     * Adds Gaussian noise (mu=0, sigma=0.1).
     */
    QueryResult generate(const std::vector<float>& data, int seed = -1);
}

#endif // QUERY_GENERATOR_HPP
