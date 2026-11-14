#include "pcfg/batch_plan_io.hpp"

#include <algorithm>
#include <array>
#include <limits>
#include <stdexcept>
#include <string>

namespace pcfg {
namespace {

constexpr std::array<std::uint8_t, 4> kMagic = {'P', 'B', 'P', '1'};
constexpr std::uint32_t kVersion = 1;

void append_u32(std::vector<std::uint8_t>& output, std::uint32_t value) {
    for (unsigned shift = 0; shift < 32; shift += 8) {
        output.push_back(static_cast<std::uint8_t>(value >> shift));
    }
}

void append_u64(std::vector<std::uint8_t>& output, std::uint64_t value) {
    for (unsigned shift = 0; shift < 64; shift += 8) {
        output.push_back(static_cast<std::uint8_t>(value >> shift));
    }
}

class Reader {
public:
    explicit Reader(const std::vector<std::uint8_t>& bytes) : bytes_(bytes) {}

    std::uint32_t u32() {
        require(4);
        std::uint32_t value = 0;
        for (unsigned index = 0; index < 4; ++index) {
            value |= static_cast<std::uint32_t>(bytes_[position_++]) << (8 * index);
        }
        return value;
    }

    std::uint64_t u64() {
        require(8);
        std::uint64_t value = 0;
        for (unsigned index = 0; index < 8; ++index) {
            value |= static_cast<std::uint64_t>(bytes_[position_++]) << (8 * index);
        }
        return value;
    }

    std::string string(std::size_t size) {
        require(size);
        std::string value(reinterpret_cast<const char*>(bytes_.data() + position_), size);
        position_ += size;
        return value;
    }

    bool finished() const { return position_ == bytes_.size(); }

private:
    void require(std::size_t size) const {
        if (position_ > bytes_.size() || size > bytes_.size() - position_) {
            throw std::runtime_error("Truncated BatchPlan payload");
        }
    }

    const std::vector<std::uint8_t>& bytes_;
    std::size_t position_ = 0;
};

std::size_t as_size(std::uint64_t value, const char* field) {
    if (value > static_cast<std::uint64_t>(std::numeric_limits<std::size_t>::max())) {
        throw std::runtime_error(std::string("BatchPlan field does not fit size_t: ") + field);
    }
    return static_cast<std::size_t>(value);
}

void validate_structure(const BatchPlan& plan) {
    std::size_t expected = 0;
    for (const CandidateRange& range : plan.ranges) {
        if (range.batch_begin != expected || range.suffix_begin > range.suffix_end) {
            throw std::runtime_error("BatchPlan ranges are not contiguous");
        }
        const std::size_t count = range.suffix_end - range.suffix_begin;
        if (count > std::numeric_limits<std::size_t>::max() - expected) {
            throw std::runtime_error("BatchPlan candidate count overflow");
        }
        expected += count;
    }
    if (expected != plan.candidate_count || (plan.candidate_count != 0 && plan.ranges.empty())) {
        throw std::runtime_error("BatchPlan candidate count mismatch");
    }
}

}  // namespace

std::vector<std::uint8_t> encode_batch_plan(const BatchPlan& plan) {
    validate_structure(plan);
    std::vector<std::uint8_t> output;
    output.reserve(24 + plan.ranges.size() * 48);
    output.insert(output.end(), kMagic.begin(), kMagic.end());
    append_u32(output, kVersion);
    append_u64(output, static_cast<std::uint64_t>(plan.candidate_count));
    append_u64(output, static_cast<std::uint64_t>(plan.ranges.size()));
    for (const CandidateRange& range : plan.ranges) {
        append_u64(output, static_cast<std::uint64_t>(range.prefix.size()));
        output.insert(output.end(), range.prefix.begin(), range.prefix.end());
        append_u64(output, static_cast<std::uint64_t>(range.suffix_segment_index));
        append_u64(output, static_cast<std::uint64_t>(range.suffix_begin));
        append_u64(output, static_cast<std::uint64_t>(range.suffix_end));
        append_u64(output, static_cast<std::uint64_t>(range.batch_begin));
    }
    return output;
}

BatchPlan decode_batch_plan(const std::vector<std::uint8_t>& bytes) {
    if (bytes.size() < 24 || !std::equal(kMagic.begin(), kMagic.end(), bytes.begin())) {
        throw std::runtime_error("Invalid BatchPlan magic");
    }
    Reader reader(bytes);
    static_cast<void>(reader.string(kMagic.size()));
    if (reader.u32() != kVersion) throw std::runtime_error("Unsupported BatchPlan version");

    BatchPlan plan;
    plan.candidate_count = as_size(reader.u64(), "candidate_count");
    const std::size_t range_count = as_size(reader.u64(), "range_count");
    if (range_count > bytes.size() / 40 + 1) throw std::runtime_error("Invalid BatchPlan range count");
    plan.ranges.reserve(range_count);
    for (std::size_t index = 0; index < range_count; ++index) {
        CandidateRange range;
        const std::size_t prefix_size = as_size(reader.u64(), "prefix_size");
        range.prefix = reader.string(prefix_size);
        range.suffix_segment_index = as_size(reader.u64(), "suffix_segment_index");
        range.suffix_begin = as_size(reader.u64(), "suffix_begin");
        range.suffix_end = as_size(reader.u64(), "suffix_end");
        range.batch_begin = as_size(reader.u64(), "batch_begin");
        plan.ranges.push_back(std::move(range));
    }
    if (!reader.finished()) throw std::runtime_error("Trailing bytes in BatchPlan payload");
    validate_structure(plan);
    return plan;
}

CandidateSlice partition_candidate_slice(std::size_t candidate_count,
                                         std::size_t rank,
                                         std::size_t world_size,
                                         std::size_t lanes) {
    if (world_size == 0 || rank >= world_size || lanes == 0) {
        throw std::runtime_error("Invalid candidate slice partition parameters");
    }
    const std::size_t groups = candidate_count / lanes;
    const std::size_t begin_group = groups * rank / world_size;
    const std::size_t end_group = groups * (rank + 1) / world_size;
    CandidateSlice slice{begin_group * lanes, end_group * lanes};
    if (rank + 1 == world_size) slice.end = candidate_count;
    return slice;
}

}  // namespace pcfg
