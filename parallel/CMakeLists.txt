add_executable(parallel_pattern_recognition)

# Compiler flags
set(QUERY_LENGTH 64 CACHE STRING "Length of the query")
target_compile_definitions(parallel_pattern_recognition PRIVATE QUERY_LENGTH_VAL=${QUERY_LENGTH})

set(BLOCK_SIZE 256 CACHE STRING "Dimension of the block")
target_compile_definitions(parallel_pattern_recognition PRIVATE BLOCK_SIZE_VAL=${BLOCK_SIZE})

# Sources
target_sources(parallel_pattern_recognition PRIVATE
    src/main.cpp
    src/data_loader.cpp
    src/query_generator.cpp
    src/kernel.cu
)

# Includes
target_include_directories(parallel_pattern_recognition PRIVATE
    include
    ${CMAKE_CURRENT_BINARY_DIR}/include
    ${CUDA_INCLUDE_DIRS}
)

if(ENABLE_PROFILING)
    target_link_libraries(parallel_pattern_recognition PRIVATE ${PROFILING_LIBS})
endif()

# Link CUDA runtime
target_link_libraries(parallel_pattern_recognition PRIVATE ${CUDA_LIBRARIES})