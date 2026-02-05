#ifndef QUERY_GENERATOR_HPP
#define QUERY_GENERATOR_HPP

#include <vector>

namespace QueryGenerator {
    /**
     * Generates a query by extracting a subsequence of length constants::QUERY_LENGTH
     * starting at a random index from the data.
     * Adds Gaussian noise (mu=0, sigma=0.1).
     */
    std::vector<float> generate(const std::vector<float>& data, int seed = -1);
}

#endif // QUERY_GENERATOR_HPP
