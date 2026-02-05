#ifndef MATCHER_HPP
#define MATCHER_HPP

#include <vector>
#include <limits>
#include <cstddef>

struct MatchResult {
    std::size_t index;
    float value;
};

namespace Matcher {
    /**
     * Finds the best match for the query in the data using SAD distance
     * and Early Abandoning optimization.
     */
    MatchResult find_best_match(const std::vector<float>& data, const std::vector<float>& query);
}

#endif // MATCHER_HPP
