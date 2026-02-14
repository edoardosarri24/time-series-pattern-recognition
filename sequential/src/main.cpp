#include <iostream>
#include <cstdlib>
#include <chrono>
#include <iomanip>

#include "data_loader.hpp"
#include "query_loader.hpp"
#include "SAD_distance.hpp"
#include "common.hpp"

int main(int argc, char** argv) {

    // Argoument check
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <input_file> [query_file]" << std::endl;
        return EXIT_FAILURE;
    }
    std::string input_file = argv[1];
    std::string query_file = (argc > 2) ? argv[2] : "data/query.txt";

    try {
        auto start_total = std::chrono::high_resolution_clock::now();

        // Data loading
        auto start_load = std::chrono::high_resolution_clock::now();
        DataLoader loader(input_file);
        std::vector<float> data = loader.load();
        auto end_load = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> elapsed_load = end_load - start_load;
        std::cout << "Data loaded. Size: " << data.size() / constants::PADDED_DIM
            << " timestamps. Time: " << elapsed_load.count() << " seconds\n";

        // Query Loading
        auto start_query = std::chrono::high_resolution_clock::now();
        std::vector<float> query = query_loader::load(query_file);
        auto end_query = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> elapsed_query = end_query - start_query;
        std::cout << "Query loaded. Time: " << elapsed_query.count() << " seconds\n";

        // Pattern matching
        auto start_match = std::chrono::high_resolution_clock::now();
        SADResult result = SAD_distance::find_best_match(data, query);
        auto end_match = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> elapsed_match = end_match - start_match;
        auto end_total = std::chrono::high_resolution_clock::now();

        // Reporting
        std::cout << "Best Match Found:\n";
        std::cout << "- Index founded: " << result.index << "\n";
        std::cout << "- SAD founded: " << result.value << "\n";
        std::cout << "- Matching Time: " << elapsed_match.count() << " seconds\n";
        std::chrono::duration<double> elapsed_total = end_total - start_total;
        std::cout << "- Total Time: " << elapsed_total.count() << " seconds\n";

    } catch (const std::exception& exception) {
        std::cerr << "Error: " << exception.what() << std::endl;
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}
