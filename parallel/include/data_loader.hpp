#ifndef DATA_LOADER_HPP
#define DATA_LOADER_HPP

#include <string>
#include <vector>
#include <cstddef>

class DataLoader {
public:
    explicit DataLoader(const std::string& filename);
    ~DataLoader(); // Frees pinned memory

    // Loads data from file (AoS), then transposes to SoA in Pinned Memory.
    void load();

    // Returns the raw AoS data (useful for query generation)
    const std::vector<float>& get_aos_data() const;

    // Returns the pointer to the SoA data in Pinned Memory
    float* get_soa_pinned_data() const;

    // Returns the number of timestamps (N)
    size_t get_num_timestamps() const;

private:
    std::string filename_;
    std::vector<float> aos_data_;
    float* soa_pinned_data_ = nullptr;
    size_t num_timestamps_ = 0;

    const char* skip_whitespace(const char* ptr, const char* end) const;
};

#endif // DATA_LOADER_HPP
