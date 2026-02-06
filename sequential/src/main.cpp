#include <iostream>
#include <cstdlib>
#include <chrono>
#include <iomanip>

#include "data_loader.hpp"
#include "query_generator.hpp"
#include "SAD_distance.hpp"
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
        std::vector<float> data = data_loader::load(input_file);
        auto end_load = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> elapsed_load = end_load - start_load;
        std::cout << "Data loaded. Size: " << data.size() / constants::PADDED_DIM
            << " timestamps. Time: " << elapsed_load.count() << "seconds\n";

        // Query generation
        query_generator::QueryResult query_result = query_generator::generate(data, 78);
        std::vector<float> query = query_result.query;

        // Pattern matching
        auto start_match = std::chrono::high_resolution_clock::now();
        SADResult result = SAD_distance::find_best_match(data, query);
        auto end_match = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> elapsed_match = end_match - start_match;

        // Reporting
        std::cout << "Best Match Found:\n";
        std::cout << "- True index: " << query_result.start_index << "\n";
        std::cout << "- Found index: " << result.index << "\n";
        std::cout << "- SAD value: " << result.value << "\n";
        std::cout << "- Matching Time: " << elapsed_match.count() << "seconds\n";

    } catch (const std::exception& exception) {
        std::cerr << "Error: " << exception.what() << std::endl;
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}
