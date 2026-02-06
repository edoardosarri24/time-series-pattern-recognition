project(sequential_pattern_recognition CXX)


add_executable(sequential_pattern_recognition)

# Compiler flags
option(ENABLE_EARLY_ABANDONING "Enable early abandoning optimization" OFF)
if(ENABLE_EARLY_ABANDONING)
    target_compile_definitions(sequential_pattern_recognition PRIVATE ENABLE_EARLY_ABANDONING)
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