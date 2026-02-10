project(parallel_pattern_recognition LANGUAGES CXX CUDA)

add_executable(parallel_pattern_recognition)

# sources
target_sources(parallel_pattern_recognition PRIVATE
    src/main.cpp
)

set_source_files_properties(src/main.cpp PROPERTIES LANGUAGE CUDA)

# includes
target_include_directories(parallel_pattern_recognition PRIVATE
    include
    ${CMAKE_CURRENT_BINARY_DIR}/include
)