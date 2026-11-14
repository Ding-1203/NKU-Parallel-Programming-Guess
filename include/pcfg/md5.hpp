#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <string>
#include <string_view>

namespace pcfg {

using Md5Digest = std::array<std::uint8_t, 16>;

inline constexpr std::size_t kMd5BatchLanes = 8;

Md5Digest md5(std::string_view input);
std::string md5_hex(std::string_view input);
bool md5_avx2_available() noexcept;
void md5_batch8(const std::array<std::string_view, kMd5BatchLanes>& inputs,
                std::array<Md5Digest, kMd5BatchLanes>& outputs);

}  // namespace pcfg
