#ifndef QUERY_LOADER_HPP
#define QUERY_LOADER_HPP

#include <vector>
#include <string>

namespace query_loader {
    // Loads query from file and returns it in SoA format (flattened).
    std::vector<float> load(const std::string& filename);
}

#endif // QUERY_LOADER_HPP
