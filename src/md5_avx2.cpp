#include "pcfg/md5.hpp"

#include "md5_internal.hpp"

#include <array>
#include <cstdint>
#include <cstring>

#if (defined(__x86_64__) || defined(__i386__)) && (defined(__GNUC__) || defined(__clang__))
#include <immintrin.h>
#define PCFG_MD5_AVX2_TARGET 1
#else
#define PCFG_MD5_AVX2_TARGET 0
#endif

namespace pcfg {
namespace {

#if PCFG_MD5_AVX2_TARGET

__attribute__((target("avx2"), noinline))
void md5_batch8_avx2(const std::array<std::string_view, kMd5BatchLanes>& inputs,
                     std::array<Md5Digest, kMd5BatchLanes>& outputs) {
    alignas(32) std::uint32_t words[16][kMd5BatchLanes] = {};
    for (std::size_t lane = 0; lane < kMd5BatchLanes; ++lane) {
        std::array<std::uint8_t, 64> block{};
        if (!inputs[lane].empty()) {
            std::memcpy(block.data(), inputs[lane].data(), inputs[lane].size());
        }
        block[inputs[lane].size()] = 0x80u;
        md5_internal::store_le64(block.data() + 56,
                                 static_cast<std::uint64_t>(inputs[lane].size()) * 8u);
        for (std::size_t word = 0; word < 16; ++word) {
            const std::size_t base = word * 4;
            words[word][lane] = static_cast<std::uint32_t>(block[base]) |
                                (static_cast<std::uint32_t>(block[base + 1]) << 8u) |
                                (static_cast<std::uint32_t>(block[base + 2]) << 16u) |
                                (static_cast<std::uint32_t>(block[base + 3]) << 24u);
        }
    }

    __m256i message[16];
    for (std::size_t word = 0; word < 16; ++word) {
        message[word] = _mm256_load_si256(reinterpret_cast<const __m256i*>(words[word]));
    }

    __m256i a = _mm256_set1_epi32(static_cast<int>(md5_internal::kInitialState[0]));
    __m256i b = _mm256_set1_epi32(static_cast<int>(md5_internal::kInitialState[1]));
    __m256i c = _mm256_set1_epi32(static_cast<int>(md5_internal::kInitialState[2]));
    __m256i d = _mm256_set1_epi32(static_cast<int>(md5_internal::kInitialState[3]));
    const __m256i initial_a = a;
    const __m256i initial_b = b;
    const __m256i initial_c = c;
    const __m256i initial_d = d;
    const __m256i all_ones = _mm256_set1_epi32(-1);

#define MD5_F(x, y, z) _mm256_or_si256(_mm256_and_si256((x), (y)), _mm256_andnot_si256((x), (z)))
#define MD5_G(x, y, z) _mm256_or_si256(_mm256_and_si256((z), (x)), _mm256_andnot_si256((z), (y)))
#define MD5_H(x, y, z) _mm256_xor_si256(_mm256_xor_si256((x), (y)), (z))
#define MD5_I(x, y, z) _mm256_xor_si256((y), _mm256_or_si256((x), _mm256_xor_si256((z), all_ones)))
#define MD5_STEP(function, word, constant_index, shift) do { \
    __m256i sum = _mm256_add_epi32(a, function(b, c, d)); \
    sum = _mm256_add_epi32(sum, message[word]); \
    sum = _mm256_add_epi32(sum, _mm256_set1_epi32(static_cast<int>(md5_internal::kConstant[constant_index]))); \
    const __m256i rotated = _mm256_or_si256(_mm256_slli_epi32(sum, shift), \
                                            _mm256_srli_epi32(sum, 32 - shift)); \
    const __m256i previous_d = d; \
    d = c; \
    c = b; \
    b = _mm256_add_epi32(b, rotated); \
    a = previous_d; \
} while (false)

    MD5_STEP(MD5_F, 0, 0, 7);
    MD5_STEP(MD5_F, 1, 1, 12);
    MD5_STEP(MD5_F, 2, 2, 17);
    MD5_STEP(MD5_F, 3, 3, 22);
    MD5_STEP(MD5_F, 4, 4, 7);
    MD5_STEP(MD5_F, 5, 5, 12);
    MD5_STEP(MD5_F, 6, 6, 17);
    MD5_STEP(MD5_F, 7, 7, 22);
    MD5_STEP(MD5_F, 8, 8, 7);
    MD5_STEP(MD5_F, 9, 9, 12);
    MD5_STEP(MD5_F, 10, 10, 17);
    MD5_STEP(MD5_F, 11, 11, 22);
    MD5_STEP(MD5_F, 12, 12, 7);
    MD5_STEP(MD5_F, 13, 13, 12);
    MD5_STEP(MD5_F, 14, 14, 17);
    MD5_STEP(MD5_F, 15, 15, 22);

    MD5_STEP(MD5_G, 1, 16, 5);
    MD5_STEP(MD5_G, 6, 17, 9);
    MD5_STEP(MD5_G, 11, 18, 14);
    MD5_STEP(MD5_G, 0, 19, 20);
    MD5_STEP(MD5_G, 5, 20, 5);
    MD5_STEP(MD5_G, 10, 21, 9);
    MD5_STEP(MD5_G, 15, 22, 14);
    MD5_STEP(MD5_G, 4, 23, 20);
    MD5_STEP(MD5_G, 9, 24, 5);
    MD5_STEP(MD5_G, 14, 25, 9);
    MD5_STEP(MD5_G, 3, 26, 14);
    MD5_STEP(MD5_G, 8, 27, 20);
    MD5_STEP(MD5_G, 13, 28, 5);
    MD5_STEP(MD5_G, 2, 29, 9);
    MD5_STEP(MD5_G, 7, 30, 14);
    MD5_STEP(MD5_G, 12, 31, 20);

    MD5_STEP(MD5_H, 5, 32, 4);
    MD5_STEP(MD5_H, 8, 33, 11);
    MD5_STEP(MD5_H, 11, 34, 16);
    MD5_STEP(MD5_H, 14, 35, 23);
    MD5_STEP(MD5_H, 1, 36, 4);
    MD5_STEP(MD5_H, 4, 37, 11);
    MD5_STEP(MD5_H, 7, 38, 16);
    MD5_STEP(MD5_H, 10, 39, 23);
    MD5_STEP(MD5_H, 13, 40, 4);
    MD5_STEP(MD5_H, 0, 41, 11);
    MD5_STEP(MD5_H, 3, 42, 16);
    MD5_STEP(MD5_H, 6, 43, 23);
    MD5_STEP(MD5_H, 9, 44, 4);
    MD5_STEP(MD5_H, 12, 45, 11);
    MD5_STEP(MD5_H, 15, 46, 16);
    MD5_STEP(MD5_H, 2, 47, 23);

    MD5_STEP(MD5_I, 0, 48, 6);
    MD5_STEP(MD5_I, 7, 49, 10);
    MD5_STEP(MD5_I, 14, 50, 15);
    MD5_STEP(MD5_I, 5, 51, 21);
    MD5_STEP(MD5_I, 12, 52, 6);
    MD5_STEP(MD5_I, 3, 53, 10);
    MD5_STEP(MD5_I, 10, 54, 15);
    MD5_STEP(MD5_I, 1, 55, 21);
    MD5_STEP(MD5_I, 8, 56, 6);
    MD5_STEP(MD5_I, 15, 57, 10);
    MD5_STEP(MD5_I, 6, 58, 15);
    MD5_STEP(MD5_I, 13, 59, 21);
    MD5_STEP(MD5_I, 4, 60, 6);
    MD5_STEP(MD5_I, 11, 61, 10);
    MD5_STEP(MD5_I, 2, 62, 15);
    MD5_STEP(MD5_I, 9, 63, 21);

#undef MD5_STEP
#undef MD5_I
#undef MD5_H
#undef MD5_G
#undef MD5_F

    a = _mm256_add_epi32(a, initial_a);
    b = _mm256_add_epi32(b, initial_b);
    c = _mm256_add_epi32(c, initial_c);
    d = _mm256_add_epi32(d, initial_d);

    alignas(32) std::uint32_t states[4][kMd5BatchLanes];
    _mm256_store_si256(reinterpret_cast<__m256i*>(states[0]), a);
    _mm256_store_si256(reinterpret_cast<__m256i*>(states[1]), b);
    _mm256_store_si256(reinterpret_cast<__m256i*>(states[2]), c);
    _mm256_store_si256(reinterpret_cast<__m256i*>(states[3]), d);
    _mm256_zeroupper();

    for (std::size_t lane = 0; lane < kMd5BatchLanes; ++lane) {
        outputs[lane] = md5_internal::state_to_digest(
            {states[0][lane], states[1][lane], states[2][lane], states[3][lane]});
    }
}

#endif

}  // namespace

bool md5_avx2_available() noexcept {
#if PCFG_MD5_AVX2_TARGET
    static const bool available = [] {
        __builtin_cpu_init();
        return __builtin_cpu_supports("avx2");
    }();
    return available;
#else
    return false;
#endif
}

void md5_batch8(const std::array<std::string_view, kMd5BatchLanes>& inputs,
                std::array<Md5Digest, kMd5BatchLanes>& outputs) {
#if PCFG_MD5_AVX2_TARGET
    bool all_single_block = true;
    for (const std::string_view input : inputs) {
        if (input.size() > 55) {
            all_single_block = false;
            break;
        }
    }
    if (all_single_block && md5_avx2_available()) {
        md5_batch8_avx2(inputs, outputs);
        return;
    }
#endif
    for (std::size_t lane = 0; lane < kMd5BatchLanes; ++lane) {
        outputs[lane] = md5(inputs[lane]);
    }
}

}  // namespace pcfg
