#ifndef QUERY_GENERATOR_HPP
#define QUERY_GENERATOR_HPP

#include <vector>
#include <cstddef>

namespace query_generator {

    struct QueryResult {
        std::vector<float> query_soa; // SoA layout: [d0m0..d0mM, d1m0..d1mM, ...]
        size_t start_index;
    };

    // Generate query from AoS input data.
    QueryResult generate(const std::vector<float>& aos_data, int seed = -1);
}

#endif // QUERY_GENERATOR_HPP
