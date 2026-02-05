project(sequential_pattern_recognition CXX)
add_executable(sequential_pattern_recognition)

# sources
target_sources(sequential_pattern_recognition PRIVATE
    src/main.cpp
    src/DataLoader.cpp
    src/QueryGenerator.cpp
    src/SAD_distance.cpp
)

# includes
target_include_directories(sequential_pattern_recognition PRIVATE
    include
    ${CMAKE_CURRENT_BINARY_DIR}/include
)