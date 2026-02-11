project(parallel_pattern_recognition LANGUAGES CXX CUDA)

add_executable(parallel_pattern_recognition)

# Compiler flags
set(QUERY_LENGTH 64 CACHE STRING "Length of the query")
target_compile_definitions(parallel_pattern_recognition PRIVATE QUERY_LENGTH_VAL=${QUERY_LENGTH})

set(BLOCK_SIZE 256 CACHE STRING "Dimension of the block")
target_compile_definitions(parallel_pattern_recognition PRIVATE BLOCK_SIZE_VAL=${BLOCK_SIZE})

# sources
target_sources(parallel_pattern_recognition PRIVATE
    src/main.cpp
    src/data_loader.cpp
    src/query_generator.cpp
)

set_source_files_properties(src/main.cpp PROPERTIES LANGUAGE CUDA)

# includes
target_include_directories(parallel_pattern_recognition PRIVATE
    include
    ${CMAKE_CURRENT_BINARY_DIR}/include
)