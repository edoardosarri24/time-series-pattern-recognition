#ifndef DATA_LOADER_HPP
#define DATA_LOADER_HPP

#include <vector>
#include <string>

class DataLoader {
public:
    explicit DataLoader(const std::string& filename);
    std::vector<float> load();

private:
    std::string filename_;
    const char* skip_whitespace(const char* ptr, const char* end) const;
};

#endif // DATA_LOADER_HPP
