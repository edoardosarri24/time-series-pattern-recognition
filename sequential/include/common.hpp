#ifndef COMMON_HPP
#define COMMON_HPP

#include <cstddef>

namespace constants {
    constexpr std::size_t ORIGINAL_DIM = 6;
    constexpr std::size_t PADDED_DIM = 8; // Aligned to 32 bytes (8 * 4 bytes)
    constexpr std::size_t QUERY_LENGTH = 64; // 1/2 of the whole human movement.
}

#endif // COMMON_HPP
