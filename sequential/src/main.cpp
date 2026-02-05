#include <iostream>
#include <cstdlib>
#include <chrono>
#include <iomanip>

#include "DataLoader.hpp"
#include "QueryGenerator.hpp"
#include "Matcher.hpp"
#include "common.hpp"

int main(int argc, char** argv) {
    // Argoument check
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <input_file>" << std::endl;
        return EXIT_FAILURE;
    }
    std::string input_file = argv[1];

    try {
        // Data loading
        auto start_load = std::chrono::high_resolution_clock::now();
        std::vector<float> data = DataLoader::load(input_file);
        auto end_load = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> elapsed_load = end_load - start_load;
        std::cout << "Data loaded. Size: " << data.size() / constants::PADDED_DIM
            << " timestamps. Time: " << elapsed_load.count() << "seconds\n";

        // Query generation
        std::vector<float> query = QueryGenerator::generate(data, 78);
        std::cout << "Query generated.\n";

        // Pattern matching
        auto start_match = std::chrono::high_resolution_clock::now();
        MatchResult result = Matcher::find_best_match(data, query);
        auto end_match = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> elapsed_match = end_match - start_match;

        // Reporting
        std::cout << "Best Match Found:\n";
        std::cout << "- Index: " << result.index << "\n";
        std::cout << "- SAD: " << result.value << "\n";
        std::cout << "- Matching Time: " << elapsed_match.count() << "s\n";

    } catch (const std::exception& exception) {
        std::cerr << "Error: " << exception.what() << std::endl;
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}
