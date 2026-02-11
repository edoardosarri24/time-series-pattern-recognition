#ifndef COMMON_HPP
#define COMMON_HPP

#include <cstddef>

namespace constants {
    constexpr std::size_t DIM = 6;

#ifndef QUERY_LENGTH_VAL
    #define QUERY_LENGTH_VAL 64
#endif
    constexpr std::size_t QUERY_LENGTH = QUERY_LENGTH_VAL;

#ifndef BLOCK_SIZE_VAL
    #define BLOCK_SIZE_VAL 256
#endif
    constexpr int BLOCK_SIZE = BLOCK_SIZE_VAL;
}

#endif // COMMON_HPP
