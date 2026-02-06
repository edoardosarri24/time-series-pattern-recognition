#include "query_generator.hpp"
#include "common.hpp"
#include <random>
#include <algorithm>
#include <stdexcept>

query_generator::QueryResult query_generator::generate(const std::vector<float>& data, int seed) {
    // Define and check the dimensions of input and query.
    size_t num_timestamps = data.size() / constants::PADDED_DIM;
    if (num_timestamps < constants::QUERY_LENGTH)
        throw std::runtime_error("Data is smaller than query length.");

    // Define random generator for query index and noise.
    std::normal_distribution<float> noise_distribution(0.0f, 0.01f);
    std::uniform_int_distribution<size_t> index_distribution(0, num_timestamps - constants::QUERY_LENGTH);
    std::mt19937 gen(seed == -1 ? std::random_device{}() : static_cast<unsigned int>(seed));

    // Query costruction
    std::vector<float> query;
    size_t start_index = index_distribution(gen);
    query.reserve(constants::QUERY_LENGTH * constants::PADDED_DIM); // Avoid copy/move if query will full.
    for (size_t i=0; i < constants::QUERY_LENGTH; ++i) {
        size_t current_timestamp = start_index + i;
        size_t start_offset = current_timestamp * constants::PADDED_DIM;
        // Push the real data plus noise.
        for (size_t d=0; d < constants::ORIGINAL_DIM; ++d) {
            float value = data[start_offset + d];
            value += noise_distribution(gen);
            query.push_back(value);
        }
        // Push the padding.
        for (size_t d = constants::ORIGINAL_DIM; d < constants::PADDED_DIM; ++d)
            query.push_back(0.0f);
    }

    return {query, start_index};
}
