#ifndef SAD_DISTANCE_HPP
#define SAD_DISTANCE_HPP

#include <vector>
#include <limits>
#include <cstddef>

struct SADResult {
    std::size_t index;
    float value;
};

namespace SAD_distance {
    SADResult find_best_match(const std::vector<float>& data, const std::vector<float>& query);
}

#endif // SAD_DISTANCE_HPP
