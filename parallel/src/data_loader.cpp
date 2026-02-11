#include "data_loader.hpp"
#include "common.hpp"
#include "cuda_utils.cuh"
#include <cuda_runtime.h>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#include <charconv>
#include <stdexcept>
#include <system_error>
#include <cstring> // for strerror

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

DataLoader::~DataLoader() {
    if (soa_pinned_data_)
        cudaFreeHost(soa_pinned_data_);
}

const char* DataLoader::skip_whitespace(const char* ptr, const char* end) const {
    while (ptr < end && std::isspace(static_cast<unsigned char>(*ptr)))
        ++ptr;
    return ptr;
}

void DataLoader::load() {
    MemoryMappedFile mapped_file(filename_);
    if (mapped_file.size() == 0)
        throw std::runtime_error("Input file is empty.");

    // Heuristic reservation
    constexpr size_t estimated_bytes_per_row = 15 * constants::DIM; // 15 Byte per float.
    size_t estimated_timestamps = mapped_file.size() / estimated_bytes_per_row;
    aos_data_.reserve(estimated_timestamps * constants::DIM);

    // Data in AoS.
    const char* ptr = mapped_file.begin();
    const char* end = mapped_file.end();
    while (ptr < end) {
        // Skip whitespace.
        ptr = skip_whitespace(ptr, end);
        if (ptr >= end) break;
        // Parse all dimensions.
        for (size_t d=0; d < constants::DIM; ++d) {
            float val;
            auto [next_ptr, error_code] = std::from_chars(ptr, end, val);
            if (error_code != std::errc()) // Handle a different default value for the error.
                throw std::runtime_error("File format error: failed to parse float.");
            aos_data_.push_back(val);
            ptr = next_ptr; // next_prt is the ptr to the next float string start.
            // Skip space/newline to get to next number
            ptr = skip_whitespace(ptr, end);
        }
    }

    // From AoS to SoA
    size_t total_elements = aos_data_.size();
    CHECK_CUDA(cudaMallocHost(&soa_pinned_data_, total_elements * sizeof(float)));
    size_t N = aos_data_.size() / constants::DIM; // Number of timestamps.
    size_t D = constants::DIM;
    for (size_t t=0; t < N; ++t) {
        for (size_t d=0; d < D; ++d) {
            float val = aos_data_[t * D + d];
            soa_pinned_data_[d * N + t] = val;
        }
    }
}

const std::vector<float>& DataLoader::get_aos_data() const { return aos_data_; }

float* DataLoader::get_soa_pinned_data() const { return soa_pinned_data_; }

size_t DataLoader::get_num_timestamps() const {
    if (aos_data_.empty())
        return 0;
    return aos_data_.size() / constants::DIM;
}
