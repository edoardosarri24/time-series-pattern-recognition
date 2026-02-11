#include "query_generator.hpp"
#include "common.hpp"
#include <random>
#include <algorithm>
#include <stdexcept>

query_generator::QueryResult query_generator::generate(const std::vector<float>& aos_data, int seed) {
    // Check timestamps vs data dimension.
    size_t num_timestamps = aos_data.size() / constants::DIM;
    if (num_timestamps < constants::QUERY_LENGTH)
        throw std::runtime_error("Data is smaller than query length.");

    // Define random generator for query index and noise.
    std::normal_distribution<float> noise_distribution(0.0f, 0.01f);
    std::uniform_int_distribution<size_t> index_distribution(0, num_timestamps - constants::QUERY_LENGTH);
    std::mt19937 gen(seed == -1 ? std::random_device{}() : static_cast<unsigned int>(seed));

    // Query costruction
    std::vector<float> query_soa;
    size_t start_index = index_distribution(gen);
    query_soa.reserve(constants::QUERY_LENGTH * constants::DIM);
    for (size_t d=0; d < constants::DIM; ++d) {
        for (size_t m=0; m < constants::QUERY_LENGTH; ++m) {
            size_t current_timestamp = start_index + m;
            float value = aos_data[current_timestamp * constants::DIM + d];
            value += noise_distribution(gen);
            query_soa.push_back(value);
        }
    }

    return {query_soa, start_index};
}
