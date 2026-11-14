#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <string>
#include <vector>

namespace pcfg {

class Sha256 {
public:
    Sha256();
    void update(const std::uint8_t* data, std::size_t size);
    void update(const std::vector<std::uint8_t>& data);
    std::array<std::uint8_t, 32> finish();

    static std::array<std::uint8_t, 32> digest(const std::vector<std::uint8_t>& data);

private:
    void transform(const std::uint8_t block[64]);

    std::array<std::uint32_t, 8> state_{};
    std::array<std::uint8_t, 64> buffer_{};
    std::uint64_t total_bytes_ = 0;
    std::size_t buffer_size_ = 0;
    bool finished_ = false;
};

struct FileFingerprint {
    std::array<std::uint8_t, 32> sha256{};
    std::uint64_t bytes = 0;
    std::uint64_t lines = 0;
};

FileFingerprint fingerprint_file(const std::filesystem::path& path);
std::string hex_encode(const std::uint8_t* data, std::size_t size);
std::string hex_encode(const std::array<std::uint8_t, 32>& data);

}  // namespace pcfg
