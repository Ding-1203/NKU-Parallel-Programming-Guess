#include "pcfg/sha256.hpp"

#include <algorithm>
#include <fstream>
#include <stdexcept>

namespace pcfg {
namespace {

constexpr std::array<std::uint32_t, 64> kRoundConstants = {
    0x428a2f98u, 0x71374491u, 0xb5c0fbcfu, 0xe9b5dba5u, 0x3956c25bu, 0x59f111f1u, 0x923f82a4u, 0xab1c5ed5u,
    0xd807aa98u, 0x12835b01u, 0x243185beu, 0x550c7dc3u, 0x72be5d74u, 0x80deb1feu, 0x9bdc06a7u, 0xc19bf174u,
    0xe49b69c1u, 0xefbe4786u, 0x0fc19dc6u, 0x240ca1ccu, 0x2de92c6fu, 0x4a7484aau, 0x5cb0a9dcu, 0x76f988dau,
    0x983e5152u, 0xa831c66du, 0xb00327c8u, 0xbf597fc7u, 0xc6e00bf3u, 0xd5a79147u, 0x06ca6351u, 0x14292967u,
    0x27b70a85u, 0x2e1b2138u, 0x4d2c6dfcu, 0x53380d13u, 0x650a7354u, 0x766a0abbu, 0x81c2c92eu, 0x92722c85u,
    0xa2bfe8a1u, 0xa81a664bu, 0xc24b8b70u, 0xc76c51a3u, 0xd192e819u, 0xd6990624u, 0xf40e3585u, 0x106aa070u,
    0x19a4c116u, 0x1e376c08u, 0x2748774cu, 0x34b0bcb5u, 0x391c0cb3u, 0x4ed8aa4au, 0x5b9cca4fu, 0x682e6ff3u,
    0x748f82eeu, 0x78a5636fu, 0x84c87814u, 0x8cc70208u, 0x90befffau, 0xa4506cebu, 0xbef9a3f7u, 0xc67178f2u,
};

std::uint32_t rotate_right(std::uint32_t value, std::uint32_t amount) {
    return (value >> amount) | (value << (32u - amount));
}

}  // namespace

Sha256::Sha256()
    : state_{0x6a09e667u, 0xbb67ae85u, 0x3c6ef372u, 0xa54ff53au,
             0x510e527fu, 0x9b05688cu, 0x1f83d9abu, 0x5be0cd19u} {}

void Sha256::update(const std::uint8_t* data, std::size_t size) {
    if (finished_) {
        throw std::logic_error("SHA-256 update after finish");
    }
    total_bytes_ += static_cast<std::uint64_t>(size);
    while (size > 0) {
        const std::size_t take = std::min(size, buffer_.size() - buffer_size_);
        std::copy_n(data, take, buffer_.begin() + static_cast<std::ptrdiff_t>(buffer_size_));
        buffer_size_ += take;
        data += take;
        size -= take;
        if (buffer_size_ == buffer_.size()) {
            transform(buffer_.data());
            buffer_size_ = 0;
        }
    }
}

void Sha256::update(const std::vector<std::uint8_t>& data) {
    update(data.data(), data.size());
}

std::array<std::uint8_t, 32> Sha256::finish() {
    if (finished_) {
        throw std::logic_error("SHA-256 finish called twice");
    }
    finished_ = true;
    const std::uint64_t bit_length = total_bytes_ * 8u;

    buffer_[buffer_size_++] = 0x80u;
    if (buffer_size_ > 56) {
        std::fill(buffer_.begin() + static_cast<std::ptrdiff_t>(buffer_size_), buffer_.end(), 0u);
        transform(buffer_.data());
        buffer_size_ = 0;
    }
    std::fill(buffer_.begin() + static_cast<std::ptrdiff_t>(buffer_size_), buffer_.begin() + 56, 0u);
    for (int i = 0; i < 8; ++i) {
        buffer_[63 - i] = static_cast<std::uint8_t>((bit_length >> (8 * i)) & 0xffu);
    }
    transform(buffer_.data());

    std::array<std::uint8_t, 32> output{};
    for (std::size_t i = 0; i < state_.size(); ++i) {
        output[i * 4] = static_cast<std::uint8_t>(state_[i] >> 24u);
        output[i * 4 + 1] = static_cast<std::uint8_t>(state_[i] >> 16u);
        output[i * 4 + 2] = static_cast<std::uint8_t>(state_[i] >> 8u);
        output[i * 4 + 3] = static_cast<std::uint8_t>(state_[i]);
    }
    return output;
}

void Sha256::transform(const std::uint8_t block[64]) {
    std::array<std::uint32_t, 64> words{};
    for (std::size_t i = 0; i < 16; ++i) {
        words[i] = (static_cast<std::uint32_t>(block[i * 4]) << 24u) |
                   (static_cast<std::uint32_t>(block[i * 4 + 1]) << 16u) |
                   (static_cast<std::uint32_t>(block[i * 4 + 2]) << 8u) |
                   static_cast<std::uint32_t>(block[i * 4 + 3]);
    }
    for (std::size_t i = 16; i < words.size(); ++i) {
        const std::uint32_t s0 = rotate_right(words[i - 15], 7) ^ rotate_right(words[i - 15], 18) ^ (words[i - 15] >> 3u);
        const std::uint32_t s1 = rotate_right(words[i - 2], 17) ^ rotate_right(words[i - 2], 19) ^ (words[i - 2] >> 10u);
        words[i] = words[i - 16] + s0 + words[i - 7] + s1;
    }

    auto working = state_;
    for (std::size_t i = 0; i < words.size(); ++i) {
        const std::uint32_t s1 = rotate_right(working[4], 6) ^ rotate_right(working[4], 11) ^ rotate_right(working[4], 25);
        const std::uint32_t choose = (working[4] & working[5]) ^ (~working[4] & working[6]);
        const std::uint32_t temp1 = working[7] + s1 + choose + kRoundConstants[i] + words[i];
        const std::uint32_t s0 = rotate_right(working[0], 2) ^ rotate_right(working[0], 13) ^ rotate_right(working[0], 22);
        const std::uint32_t majority = (working[0] & working[1]) ^ (working[0] & working[2]) ^ (working[1] & working[2]);
        const std::uint32_t temp2 = s0 + majority;

        working[7] = working[6];
        working[6] = working[5];
        working[5] = working[4];
        working[4] = working[3] + temp1;
        working[3] = working[2];
        working[2] = working[1];
        working[1] = working[0];
        working[0] = temp1 + temp2;
    }
    for (std::size_t i = 0; i < state_.size(); ++i) {
        state_[i] += working[i];
    }
}

std::array<std::uint8_t, 32> Sha256::digest(const std::vector<std::uint8_t>& data) {
    Sha256 hash;
    hash.update(data);
    return hash.finish();
}

FileFingerprint fingerprint_file(const std::filesystem::path& path) {
    std::ifstream input(path, std::ios::binary);
    if (!input) {
        throw std::runtime_error("Cannot open input file: " + path.string());
    }

    Sha256 hash;
    FileFingerprint result;
    std::array<char, 1 << 20> buffer{};
    bool saw_any = false;
    char last = 0;
    while (input) {
        input.read(buffer.data(), static_cast<std::streamsize>(buffer.size()));
        const std::streamsize count = input.gcount();
        if (count <= 0) {
            break;
        }
        hash.update(reinterpret_cast<const std::uint8_t*>(buffer.data()), static_cast<std::size_t>(count));
        result.bytes += static_cast<std::uint64_t>(count);
        for (std::streamsize i = 0; i < count; ++i) {
            if (buffer[static_cast<std::size_t>(i)] == '\n') {
                ++result.lines;
            }
        }
        saw_any = true;
        last = buffer[static_cast<std::size_t>(count - 1)];
    }
    if (!input.eof()) {
        throw std::runtime_error("Failed while reading input file: " + path.string());
    }
    if (saw_any && last != '\n') {
        ++result.lines;
    }
    result.sha256 = hash.finish();
    return result;
}

std::string hex_encode(const std::uint8_t* data, std::size_t size) {
    static constexpr char kHex[] = "0123456789abcdef";
    std::string output(size * 2, '0');
    for (std::size_t i = 0; i < size; ++i) {
        output[i * 2] = kHex[data[i] >> 4u];
        output[i * 2 + 1] = kHex[data[i] & 0x0fu];
    }
    return output;
}

std::string hex_encode(const std::array<std::uint8_t, 32>& data) {
    return hex_encode(data.data(), data.size());
}

}  // namespace pcfg
