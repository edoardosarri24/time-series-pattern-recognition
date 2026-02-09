#include "data_loader.hpp"
#include "common.hpp"
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#include <charconv>
#include <stdexcept>
#include <vector>
#include <system_error>

namespace {
    // Helper class for RAII handling of mmap
    class MemoryMappedFile {
    public:
        MemoryMappedFile(const std::string& filename) {
            fd_ = open(filename.c_str(), O_RDONLY);
            if (fd_ == -1)
                throw std::runtime_error("Could not open file: " + filename);
            struct stat sb;
            if (fstat(fd_, &sb) == -1) {
                close(fd_);
                throw std::runtime_error("Could not get file size: " + filename);
            }
            size_ = static_cast<size_t>(sb.st_size);
            if (size_ == 0) {
                data_ = nullptr;
                return;
            }
            data_ = static_cast<char*>(mmap(nullptr, size_, PROT_READ, MAP_PRIVATE, fd_, 0));
            if (data_ == MAP_FAILED) {
                close(fd_);
                throw std::runtime_error("Could not mmap file: " + filename);
            }
            madvise(data_, size_, MADV_SEQUENTIAL);
        }

        ~MemoryMappedFile() {
            if (data_ && (data_ != MAP_FAILED))
                munmap(data_, size_);
            if (fd_ != -1)
                close(fd_);
        }

        const char* begin() const { return data_; }
        const char* end() const { return data_ + size_; }
        size_t size() const { return size_; }

    private:
        int fd_ = -1;
        char* data_ = nullptr;
        size_t size_ = 0;
    };
}

DataLoader::DataLoader(const std::string& filename) : filename_(filename) {}

const char* DataLoader::skip_whitespace(const char* ptr, const char* end) const {
    while (ptr < end && std::isspace(static_cast<unsigned char>(*ptr))) {
        ++ptr;
    }
    return ptr;
}

std::vector<float> DataLoader::load() {
    MemoryMappedFile mapped_file(filename_);
    if (mapped_file.size() == 0)
        return {};

    std::vector<float> data;
    // Heuristic reservation
    constexpr size_t estimated_bytes_per_row = 15 * constants::ORIGINAL_DIM; // 15 Byte per float.
    size_t estimated_timestamps = mapped_file.size() / estimated_bytes_per_row;
    data.reserve(estimated_timestamps * constants::PADDED_DIM);

    const char* ptr = mapped_file.begin();
    const char* end = mapped_file.end();

    while (ptr < end) {
        // Skip whitespace.
        ptr = skip_whitespace(ptr, end);
        if (ptr >= end)
            break;
        // Parse first dimension.
        float val;
        auto [next_ptr, error_code] = std::from_chars(ptr, end, val);
        if (error_code != std::errc()) // Handle a different default value for the error.
            throw std::runtime_error("File format error: failed to parse float.");
        data.push_back(val);
        ptr = next_ptr; // next_prt is the ptr to the next float string start.
        // Parse remaining dimensions
        for (size_t i=1; i < constants::ORIGINAL_DIM; ++i) {
            ptr = skip_whitespace(ptr, end);
            if (ptr >= end)
                throw std::runtime_error("File format error: incomplete timestamp data.");

            auto [next_dim_ptr, error_code_dim] = std::from_chars(ptr, end, val);
            if (error_code_dim != std::errc())
                throw std::runtime_error("File format error: failed to parse float.");
            data.push_back(val);
            ptr = next_dim_ptr;
        }
        // Add padding
        for (size_t i = constants::ORIGINAL_DIM; i < constants::PADDED_DIM; ++i)
            data.push_back(0.0f);
    }

    return data;
}
