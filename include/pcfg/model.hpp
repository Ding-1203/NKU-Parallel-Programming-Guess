#pragma once

#include <array>
#include <cstdint>
#include <filesystem>
#include <map>
#include <string>
#include <unordered_map>
#include <vector>

namespace pcfg {

enum class SegmentType : std::uint8_t {
    Letter = 1,
    Digit = 2,
    Symbol = 3,
};

struct SegmentKey {
    SegmentType type = SegmentType::Symbol;
    std::uint32_t length = 0;

    bool operator==(const SegmentKey& other) const;
    bool operator<(const SegmentKey& other) const;
};

struct SegmentKeyHash {
    std::size_t operator()(const SegmentKey& key) const noexcept;
};

struct SegmentValue {
    std::string value;
    std::uint64_t count = 0;
};

struct SegmentTable {
    SegmentKey key;
    std::uint64_t total_count = 0;
    std::vector<SegmentValue> values;
};

struct Preterminal {
    std::vector<SegmentKey> pattern;
    std::uint64_t count = 0;
};

struct TrainingMetadata {
    std::array<std::uint8_t, 32> source_sha256{};
    std::uint64_t source_bytes = 0;
    std::uint64_t source_lines = 0;
    std::uint64_t processed_records = 0;
    std::uint64_t skipped_empty = 0;
    std::uint64_t train_limit = 0;
    std::uint32_t ordering_version = 1;
    std::uint32_t classifier_version = 1;
};

class Model {
public:
    TrainingMetadata metadata;
    std::vector<Preterminal> preterminals;
    std::vector<SegmentTable> segments;
    std::uint64_t total_preterminal_count = 0;

    void rebuild_index();
    const SegmentTable& segment(const SegmentKey& key) const;
    std::size_t segment_index(const SegmentKey& key) const;

private:
    std::unordered_map<SegmentKey, std::size_t, SegmentKeyHash> segment_index_;
};

struct TrainingOptions {
    std::filesystem::path input;
    std::uint64_t train_limit = 0;
};

class ModelTrainer {
public:
    Model train(const TrainingOptions& options) const;

private:
    static SegmentType classify(unsigned char value);
};

std::string segment_key_string(const SegmentKey& key);
std::string preterminal_string(const Preterminal& pt);

}  // namespace pcfg
