#ifndef DATA_LOADER_HPP
#define DATA_LOADER_HPP

#include <vector>
#include <string>

namespace DataLoader {
    /**
     * Loads data from the specified file.
     * Expects 6 floats per line.
     * Stores them in a flat vector with padding (8 floats per timestamp).
     */
    std::vector<float> load(const std::string& filename);
}

#endif // DATA_LOADER_HPP
