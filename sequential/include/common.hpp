#ifndef COMMON_HPP
#define COMMON_HPP

#include <cstddef>

namespace constants {
    constexpr std::size_t ORIGINAL_DIM = 6;

    constexpr std::size_t next_power_of_two(std::size_t n) {
        if (n <= 1) return 1;
        std::size_t power = 1;
        while (power < n)
            power <<= 1;
        return power;
    }

#ifdef ENABLE_PADDING
    constexpr std::size_t PADDED_DIM = next_power_of_two(ORIGINAL_DIM); // Aligned to next power of 2.
#else
    constexpr std::size_t PADDED_DIM = ORIGINAL_DIM; // No padding
#endif
#ifndef QUERY_LENGTH_VAL
    #define QUERY_LENGTH_VAL 64
#endif
    constexpr std::size_t QUERY_LENGTH = QUERY_LENGTH_VAL; // 1/2 of the whole human movement.
}

#endif // COMMON_HPP
