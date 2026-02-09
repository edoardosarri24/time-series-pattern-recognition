#ifndef COMMON_HPP
#define COMMON_HPP

#include <cstddef>
#include <bit>

namespace constants {
    constexpr std::size_t ORIGINAL_DIM = 6;
    constexpr std::size_t PADDED_DIM = std::bit_ceil(ORIGINAL_DIM); // Aligned to next power of 2.
    constexpr std::size_t QUERY_LENGTH = 64; // 1/2 of the whole human movement.
}

#endif // COMMON_HPP
