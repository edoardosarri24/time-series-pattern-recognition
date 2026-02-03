project(sequential_pattern_recognition CXX)
add_executable(sequential_pattern_recognition)

# sources
target_sources(sequential_pattern_recognition PRIVATE
    src/main.cpp
)

# includes
target_include_directories(sequential_pattern_recognition PRIVATE
    include
    ${CMAKE_CURRENT_BINARY_DIR}/include
)

# libraries
# target_link_libraries(sequential_pattern_recognition PRIVATE OpenMP::OpenMP_C)

# hyperparameters
#target_compile_definitions(sequential_pattern_recognition PRIVATE
#    DATA_DIR="${CMAKE_SOURCE_DIR}/data"
#    N_GRAM_SIZE=${N_GRAM_SIZE}
#    TOP_K=${TOP_K}
#)
