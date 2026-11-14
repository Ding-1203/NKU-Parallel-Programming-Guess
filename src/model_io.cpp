#include "pcfg/model_io.hpp"

#include "pcfg/sha256.hpp"

#include <algorithm>
#include <array>
#include <cstring>
#include <fstream>
#include <limits>
#include <stdexcept>
#include <vector>

namespace pcfg {
namespace {

constexpr std::array<std::uint8_t, 16> kMagic = {
    'N', 'K', 'U', 'P', 'C', 'F', 'G', 'B', 'A', 'S', 'E', 'V', '1', 0, 0, 0,
};
constexpr std::uint32_t kEndianMarker = 0x01020304u;
constexpr std::uint64_t kMaxCollectionSize = 1000000000ull;
constexpr std::uint32_t kMaxStringSize = 16u * 1024u * 1024u;

void put_u8(std::vector<std::uint8_t>& output, std::uint8_t value) {
    output.push_back(value);
}

void put_u32(std::vector<std::uint8_t>& output, std::uint32_t value) {
    for (int i = 0; i < 4; ++i) output.push_back(static_cast<std::uint8_t>(value >> (8 * i)));
}

void put_u64(std::vector<std::uint8_t>& output, std::uint64_t value) {
    for (int i = 0; i < 8; ++i) output.push_back(static_cast<std::uint8_t>(value >> (8 * i)));
}

void put_string(std::vector<std::uint8_t>& output, const std::string& value) {
    if (value.size() > std::numeric_limits<std::uint32_t>::max()) {
        throw std::runtime_error("String is too large for model format");
    }
    put_u32(output, static_cast<std::uint32_t>(value.size()));
    output.insert(output.end(), value.begin(), value.end());
}

void put_key(std::vector<std::uint8_t>& output, const SegmentKey& key) {
    put_u8(output, static_cast<std::uint8_t>(key.type));
    put_u32(output, key.length);
}

class Cursor {
public:
    explicit Cursor(const std::vector<std::uint8_t>& data) : data_(data) {}

    std::uint8_t u8() {
        require(1);
        return data_[position_++];
    }

    std::uint32_t u32() {
        require(4);
        std::uint32_t value = 0;
        for (int i = 0; i < 4; ++i) value |= static_cast<std::uint32_t>(data_[position_++]) << (8 * i);
        return value;
    }

    std::uint64_t u64() {
        require(8);
        std::uint64_t value = 0;
        for (int i = 0; i < 8; ++i) value |= static_cast<std::uint64_t>(data_[position_++]) << (8 * i);
        return value;
    }

    std::string string() {
        const std::uint32_t size = u32();
        if (size > kMaxStringSize) throw std::runtime_error("Model string exceeds safety limit");
        require(size);
        std::string value(reinterpret_cast<const char*>(data_.data() + position_), size);
        position_ += size;
        return value;
    }

    SegmentKey key() {
        const std::uint8_t raw_type = u8();
        if (raw_type < static_cast<std::uint8_t>(SegmentType::Letter) ||
            raw_type > static_cast<std::uint8_t>(SegmentType::Symbol)) {
            throw std::runtime_error("Invalid segment type in model");
        }
        SegmentKey result{static_cast<SegmentType>(raw_type), u32()};
        if (result.length == 0) throw std::runtime_error("Zero-length segment in model");
        return result;
    }

    bool at_end() const { return position_ == data_.size(); }

private:
    void require(std::size_t count) const {
        if (count > data_.size() - position_) throw std::runtime_error("Truncated model payload");
    }

    const std::vector<std::uint8_t>& data_;
    std::size_t position_ = 0;
};

std::vector<std::uint8_t> read_all(const std::filesystem::path& path) {
    std::ifstream input(path, std::ios::binary | std::ios::ate);
    if (!input) throw std::runtime_error("Cannot open model: " + path.string());
    const std::streamsize size = input.tellg();
    if (size < 0) throw std::runtime_error("Cannot determine model size: " + path.string());
    input.seekg(0);
    std::vector<std::uint8_t> data(static_cast<std::size_t>(size));
    if (size > 0) input.read(reinterpret_cast<char*>(data.data()), size);
    if (!input) throw std::runtime_error("Cannot read model: " + path.string());
    return data;
}

std::uint32_t read_header_u32(const std::vector<std::uint8_t>& bytes, std::size_t offset) {
    std::uint32_t value = 0;
    for (int i = 0; i < 4; ++i) value |= static_cast<std::uint32_t>(bytes[offset + i]) << (8 * i);
    return value;
}

std::uint64_t read_header_u64(const std::vector<std::uint8_t>& bytes, std::size_t offset) {
    std::uint64_t value = 0;
    for (int i = 0; i < 8; ++i) value |= static_cast<std::uint64_t>(bytes[offset + i]) << (8 * i);
    return value;
}

}  // namespace

void ModelStore::save(const Model& model, const std::filesystem::path& path) {
    std::vector<std::uint8_t> payload;
    payload.reserve(1024 * 1024);

    payload.insert(payload.end(), model.metadata.source_sha256.begin(), model.metadata.source_sha256.end());
    put_u64(payload, model.metadata.source_bytes);
    put_u64(payload, model.metadata.source_lines);
    put_u64(payload, model.metadata.processed_records);
    put_u64(payload, model.metadata.skipped_empty);
    put_u64(payload, model.metadata.train_limit);
    put_u32(payload, model.metadata.ordering_version);
    put_u32(payload, model.metadata.classifier_version);
    put_u64(payload, model.total_preterminal_count);

    put_u64(payload, model.preterminals.size());
    for (const Preterminal& pt : model.preterminals) {
        put_u64(payload, pt.count);
        put_u32(payload, static_cast<std::uint32_t>(pt.pattern.size()));
        for (const SegmentKey& key : pt.pattern) put_key(payload, key);
    }

    put_u64(payload, model.segments.size());
    for (const SegmentTable& table : model.segments) {
        put_key(payload, table.key);
        put_u64(payload, table.total_count);
        put_u64(payload, table.values.size());
        for (const SegmentValue& value : table.values) {
            put_string(payload, value.value);
            put_u64(payload, value.count);
        }
    }

    const auto checksum = Sha256::digest(payload);
    if (!path.parent_path().empty()) std::filesystem::create_directories(path.parent_path());
    std::ofstream output(path, std::ios::binary | std::ios::trunc);
    if (!output) throw std::runtime_error("Cannot create model: " + path.string());
    output.write(reinterpret_cast<const char*>(kMagic.data()), static_cast<std::streamsize>(kMagic.size()));
    std::vector<std::uint8_t> header;
    put_u32(header, kSchemaVersion);
    put_u32(header, kEndianMarker);
    put_u64(header, payload.size());
    output.write(reinterpret_cast<const char*>(header.data()), static_cast<std::streamsize>(header.size()));
    output.write(reinterpret_cast<const char*>(payload.data()), static_cast<std::streamsize>(payload.size()));
    output.write(reinterpret_cast<const char*>(checksum.data()), static_cast<std::streamsize>(checksum.size()));
    if (!output) throw std::runtime_error("Failed while writing model: " + path.string());
}

Model ModelStore::load(const std::filesystem::path& path) {
    const std::vector<std::uint8_t> bytes = read_all(path);
    constexpr std::size_t kHeaderSize = 16 + 4 + 4 + 8;
    constexpr std::size_t kChecksumSize = 32;
    if (bytes.size() < kHeaderSize + kChecksumSize) throw std::runtime_error("Model file is too small");
    if (!std::equal(kMagic.begin(), kMagic.end(), bytes.begin())) throw std::runtime_error("Model magic mismatch");
    if (read_header_u32(bytes, 16) != kSchemaVersion) throw std::runtime_error("Unsupported model schema version");
    if (read_header_u32(bytes, 20) != kEndianMarker) throw std::runtime_error("Model endian marker mismatch");
    const std::uint64_t payload_size = read_header_u64(bytes, 24);
    if (payload_size != bytes.size() - kHeaderSize - kChecksumSize) throw std::runtime_error("Model payload size mismatch");

    const auto payload_begin = bytes.begin() + static_cast<std::ptrdiff_t>(kHeaderSize);
    const auto payload_end = payload_begin + static_cast<std::ptrdiff_t>(payload_size);
    std::vector<std::uint8_t> payload(payload_begin, payload_end);
    const auto actual_checksum = Sha256::digest(payload);
    if (!std::equal(actual_checksum.begin(), actual_checksum.end(), payload_end)) {
        throw std::runtime_error("Model payload checksum mismatch");
    }

    Cursor cursor(payload);
    Model model;
    for (std::uint8_t& byte : model.metadata.source_sha256) byte = cursor.u8();
    model.metadata.source_bytes = cursor.u64();
    model.metadata.source_lines = cursor.u64();
    model.metadata.processed_records = cursor.u64();
    model.metadata.skipped_empty = cursor.u64();
    model.metadata.train_limit = cursor.u64();
    model.metadata.ordering_version = cursor.u32();
    model.metadata.classifier_version = cursor.u32();
    model.total_preterminal_count = cursor.u64();

    const std::uint64_t preterminal_count = cursor.u64();
    if (preterminal_count > kMaxCollectionSize) throw std::runtime_error("Too many preterminals in model");
    model.preterminals.reserve(static_cast<std::size_t>(preterminal_count));
    for (std::uint64_t i = 0; i < preterminal_count; ++i) {
        Preterminal pt;
        pt.count = cursor.u64();
        const std::uint32_t pattern_size = cursor.u32();
        if (pattern_size == 0 || pattern_size > 1000000u) throw std::runtime_error("Invalid preterminal length");
        pt.pattern.reserve(pattern_size);
        for (std::uint32_t j = 0; j < pattern_size; ++j) pt.pattern.push_back(cursor.key());
        model.preterminals.push_back(std::move(pt));
    }

    const std::uint64_t segment_count = cursor.u64();
    if (segment_count > kMaxCollectionSize) throw std::runtime_error("Too many segment tables in model");
    model.segments.reserve(static_cast<std::size_t>(segment_count));
    for (std::uint64_t i = 0; i < segment_count; ++i) {
        SegmentTable table;
        table.key = cursor.key();
        table.total_count = cursor.u64();
        const std::uint64_t value_count = cursor.u64();
        if (value_count == 0 || value_count > kMaxCollectionSize) throw std::runtime_error("Invalid segment value count");
        table.values.reserve(static_cast<std::size_t>(value_count));
        std::uint64_t computed_total = 0;
        for (std::uint64_t j = 0; j < value_count; ++j) {
            SegmentValue value{cursor.string(), cursor.u64()};
            if (value.value.size() != table.key.length || value.count == 0) throw std::runtime_error("Invalid segment value entry");
            computed_total += value.count;
            table.values.push_back(std::move(value));
        }
        if (computed_total != table.total_count) throw std::runtime_error("Segment frequency total mismatch");
        model.segments.push_back(std::move(table));
    }
    if (!cursor.at_end()) throw std::runtime_error("Unexpected trailing model payload data");
    if (model.total_preterminal_count != model.metadata.processed_records) throw std::runtime_error("Model record count mismatch");
    model.rebuild_index();
    for (const Preterminal& pt : model.preterminals) {
        for (const SegmentKey& key : pt.pattern) static_cast<void>(model.segment(key));
    }
    return model;
}

}  // namespace pcfg
