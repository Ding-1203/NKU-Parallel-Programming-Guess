#include "pcfg/model.hpp"

#include "pcfg/sha256.hpp"

#include <algorithm>
#include <fstream>
#include <numeric>
#include <stdexcept>

namespace pcfg {
namespace {

using Pattern = std::vector<SegmentKey>;

struct ParsedPassword {
    Pattern pattern;
    std::vector<std::pair<SegmentKey, std::string>> segments;
};

ParsedPassword parse_password(const std::string& password) {
    ParsedPassword result;
    if (password.empty()) {
        return result;
    }

    const auto classify = [](unsigned char c) {
        if ((c >= 'A' && c <= 'Z') || (c >= 'a' && c <= 'z')) return SegmentType::Letter;
        if (c >= '0' && c <= '9') return SegmentType::Digit;
        return SegmentType::Symbol;
    };

    std::size_t begin = 0;
    SegmentType current = classify(static_cast<unsigned char>(password[0]));
    for (std::size_t i = 1; i <= password.size(); ++i) {
        const bool boundary = i == password.size() || classify(static_cast<unsigned char>(password[i])) != current;
        if (!boundary) {
            continue;
        }
        SegmentKey key{current, static_cast<std::uint32_t>(i - begin)};
        result.pattern.push_back(key);
        result.segments.emplace_back(key, password.substr(begin, i - begin));
        if (i < password.size()) {
            begin = i;
            current = classify(static_cast<unsigned char>(password[i]));
        }
    }
    return result;
}

}  // namespace

bool SegmentKey::operator==(const SegmentKey& other) const {
    return type == other.type && length == other.length;
}

bool SegmentKey::operator<(const SegmentKey& other) const {
    if (type != other.type) return static_cast<std::uint8_t>(type) < static_cast<std::uint8_t>(other.type);
    return length < other.length;
}

std::size_t SegmentKeyHash::operator()(const SegmentKey& key) const noexcept {
    return (static_cast<std::size_t>(static_cast<std::uint8_t>(key.type)) << 32u) ^ key.length;
}

void Model::rebuild_index() {
    segment_index_.clear();
    segment_index_.reserve(segments.size() * 2);
    for (std::size_t i = 0; i < segments.size(); ++i) {
        if (!segment_index_.emplace(segments[i].key, i).second) {
            throw std::runtime_error("Model contains duplicate segment table: " + segment_key_string(segments[i].key));
        }
    }
}

const SegmentTable& Model::segment(const SegmentKey& key) const {
    return segments.at(segment_index(key));
}

std::size_t Model::segment_index(const SegmentKey& key) const {
    const auto found = segment_index_.find(key);
    if (found == segment_index_.end()) {
        throw std::runtime_error("Missing segment table: " + segment_key_string(key));
    }
    return found->second;
}

SegmentType ModelTrainer::classify(unsigned char value) {
    if ((value >= 'A' && value <= 'Z') || (value >= 'a' && value <= 'z')) return SegmentType::Letter;
    if (value >= '0' && value <= '9') return SegmentType::Digit;
    return SegmentType::Symbol;
}

Model ModelTrainer::train(const TrainingOptions& options) const {
    if (options.input.empty()) {
        throw std::runtime_error("Training input path is empty");
    }
    const FileFingerprint fingerprint = fingerprint_file(options.input);

    std::ifstream input(options.input, std::ios::binary);
    if (!input) {
        throw std::runtime_error("Cannot open training input: " + options.input.string());
    }

    std::map<Pattern, std::uint64_t> preterminal_counts;
    std::map<SegmentKey, std::map<std::string, std::uint64_t>> segment_counts;
    std::string line;
    std::uint64_t processed = 0;
    std::uint64_t skipped_empty = 0;

    while (std::getline(input, line)) {
        if (!line.empty() && line.back() == '\r') {
            line.pop_back();
        }
        if (line.empty()) {
            ++skipped_empty;
            continue;
        }
        if (options.train_limit != 0 && processed >= options.train_limit) {
            break;
        }

        ParsedPassword parsed = parse_password(line);
        if (parsed.pattern.empty()) {
            ++skipped_empty;
            continue;
        }
        ++preterminal_counts[parsed.pattern];
        for (const auto& item : parsed.segments) {
            ++segment_counts[item.first][item.second];
        }
        ++processed;
    }
    if (!input.eof() && input.fail()) {
        throw std::runtime_error("Failed while reading training input: " + options.input.string());
    }
    if (processed == 0) {
        throw std::runtime_error("Training input contains no usable passwords");
    }

    Model model;
    model.metadata.source_sha256 = fingerprint.sha256;
    model.metadata.source_bytes = fingerprint.bytes;
    model.metadata.source_lines = fingerprint.lines;
    model.metadata.processed_records = processed;
    model.metadata.skipped_empty = skipped_empty;
    model.metadata.train_limit = options.train_limit;
    model.total_preterminal_count = processed;

    model.preterminals.reserve(preterminal_counts.size());
    for (const auto& entry : preterminal_counts) {
        model.preterminals.push_back(Preterminal{entry.first, entry.second});
    }
    std::sort(model.preterminals.begin(), model.preterminals.end(), [](const Preterminal& left, const Preterminal& right) {
        if (left.count != right.count) return left.count > right.count;
        return left.pattern < right.pattern;
    });

    model.segments.reserve(segment_counts.size());
    for (const auto& entry : segment_counts) {
        SegmentTable table;
        table.key = entry.first;
        table.values.reserve(entry.second.size());
        for (const auto& value_count : entry.second) {
            table.values.push_back(SegmentValue{value_count.first, value_count.second});
            table.total_count += value_count.second;
        }
        std::sort(table.values.begin(), table.values.end(), [](const SegmentValue& left, const SegmentValue& right) {
            if (left.count != right.count) return left.count > right.count;
            return left.value < right.value;
        });
        model.segments.push_back(std::move(table));
    }
    model.rebuild_index();
    return model;
}

std::string segment_key_string(const SegmentKey& key) {
    char prefix = 'S';
    if (key.type == SegmentType::Letter) prefix = 'L';
    if (key.type == SegmentType::Digit) prefix = 'D';
    return std::string(1, prefix) + std::to_string(key.length);
}

std::string preterminal_string(const Preterminal& pt) {
    std::string result;
    for (const SegmentKey& key : pt.pattern) {
        result += segment_key_string(key);
    }
    return result;
}

}  // namespace pcfg
