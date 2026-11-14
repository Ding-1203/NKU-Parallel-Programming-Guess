#include "pcfg/md5.hpp"

#include "md5_internal.hpp"
#include "pcfg/sha256.hpp"

#include <array>
#include <cstdint>
#include <cstring>
#include <vector>

namespace pcfg {
namespace {

void compress_block(std::array<std::uint32_t, 4>& state, const std::uint8_t* block) {
    std::array<std::uint32_t, 16> words{};
    for (std::size_t i = 0; i < words.size(); ++i) {
        const std::size_t base = i * 4;
        words[i] = static_cast<std::uint32_t>(block[base]) |
                   (static_cast<std::uint32_t>(block[base + 1]) << 8u) |
                   (static_cast<std::uint32_t>(block[base + 2]) << 16u) |
                   (static_cast<std::uint32_t>(block[base + 3]) << 24u);
    }

    std::uint32_t a = state[0];
    std::uint32_t b = state[1];
    std::uint32_t c = state[2];
    std::uint32_t d = state[3];
    for (std::uint32_t i = 0; i < 64; ++i) {
        std::uint32_t f = 0;
        std::uint32_t g = 0;
        if (i < 16) {
            f = (b & c) | (~b & d);
            g = i;
        } else if (i < 32) {
            f = (d & b) | (~d & c);
            g = (5u * i + 1u) % 16u;
        } else if (i < 48) {
            f = b ^ c ^ d;
            g = (3u * i + 5u) % 16u;
        } else {
            f = c ^ (b | ~d);
            g = (7u * i) % 16u;
        }
        const std::uint32_t next_d = d;
        d = c;
        c = b;
        b += md5_internal::rotate_left(a + f + md5_internal::kConstant[i] + words[g],
                                       md5_internal::kShift[i]);
        a = next_d;
    }
    state[0] += a;
    state[1] += b;
    state[2] += c;
    state[3] += d;
}

}  // namespace

Md5Digest md5(std::string_view input) {
    std::array<std::uint32_t, 4> state = md5_internal::kInitialState;

    if (input.size() <= 55) {
        std::array<std::uint8_t, 64> block{};
        if (!input.empty()) std::memcpy(block.data(), input.data(), input.size());
        block[input.size()] = 0x80u;
        md5_internal::store_le64(block.data() + 56, static_cast<std::uint64_t>(input.size()) * 8u);
        compress_block(state, block.data());
        return md5_internal::state_to_digest(state);
    }

    std::vector<std::uint8_t> message(input.begin(), input.end());
    const std::uint64_t bit_length = static_cast<std::uint64_t>(message.size()) * 8u;
    message.push_back(0x80u);
    while (message.size() % 64 != 56) message.push_back(0u);
    const std::size_t length_offset = message.size();
    message.resize(length_offset + 8);
    md5_internal::store_le64(message.data() + length_offset, bit_length);

    for (std::size_t offset = 0; offset < message.size(); offset += 64) {
        compress_block(state, message.data() + offset);
    }
    return md5_internal::state_to_digest(state);
}

std::string md5_hex(std::string_view input) {
    const Md5Digest digest = md5(input);
    return hex_encode(digest.data(), digest.size());
}

}  // namespace pcfg
