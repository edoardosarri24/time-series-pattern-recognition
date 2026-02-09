#ifndef COMMON_HPP
#define COMMON_HPP

#include <cstddef>
#include <bit>

namespace constants {
    constexpr std::size_t ORIGINAL_DIM = 6;
#ifdef ENABLE_PADDING
    constexpr std::size_t PADDED_DIM = std::bit_ceil(ORIGINAL_DIM); // Aligned to next power of 2.
#else
    constexpr std::size_t PADDED_DIM = ORIGINAL_DIM; // No padding
#endif
#ifndef QUERY_LENGTH_VAL
    #define QUERY_LENGTH_VAL 64
#endif
    constexpr std::size_t QUERY_LENGTH = QUERY_LENGTH_VAL; // 1/2 of the whole human movement.
}

#endif // COMMON_HPP
