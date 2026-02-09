project(sequential_pattern_recognition CXX)


add_executable(sequential_pattern_recognition)

# Compiler flags
set(QUERY_LENGTH 64 CACHE STRING "Length of the query")
target_compile_definitions(sequential_pattern_recognition PRIVATE QUERY_LENGTH_VAL=${QUERY_LENGTH})

option(ENABLE_EARLY_ABANDONING "Enable early abandoning optimization" OFF)
if(ENABLE_EARLY_ABANDONING)
    target_compile_definitions(sequential_pattern_recognition PRIVATE ENABLE_EARLY_ABANDONING)
endif()

option(ENABLE_PADDING "Enable automatic padding for cache alignment" OFF)
if(ENABLE_PADDING)
    target_compile_definitions(sequential_pattern_recognition PRIVATE ENABLE_PADDING)
endif()

# sources
target_sources(sequential_pattern_recognition PRIVATE
    src/main.cpp
    src/data_loader.cpp
    src/query_generator.cpp
    src/SAD_distance.cpp
)

# includes
target_include_directories(sequential_pattern_recognition PRIVATE
    include
    ${CMAKE_CURRENT_BINARY_DIR}/include
)