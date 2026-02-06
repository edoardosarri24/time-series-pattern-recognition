#ifndef QUERY_GENERATOR_HPP
#define QUERY_GENERATOR_HPP

#include <vector>
#include <cstddef>

namespace query_generator {

    struct QueryResult {
        std::vector<float> query;
        size_t start_index;
    };

    QueryResult generate(const std::vector<float>& data, int seed = -1);
}

#endif // QUERY_GENERATOR_HPP
