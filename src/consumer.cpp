#include "pcfg/consumer.hpp"

#include "pcfg/md5.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <cstring>
#include <fstream>
#include <iterator>
#include <omp.h>
#include <stdexcept>
#include <string_view>

namespace pcfg {
namespace {

double elapsed_seconds(std::chrono::steady_clock::time_point start) {
    return std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
}

constexpr std::size_t kMinGroupsPerThread = 256;

std::uint64_t fold_digest64(const Md5Digest& digest) {
    std::uint64_t folded = 0;
    for (std::size_t i = 0; i < 8; ++i) {
        folded |= static_cast<std::uint64_t>(digest[i]) << (8 * i);
    }
    return folded;
}

void update_hash_guard(std::uint64_t folded, std::uint64_t& guard) {
    guard ^= folded + 0x9e3779b97f4a7c15ull + (guard << 6u) + (guard >> 2u);
}

}  // namespace

CandidateConsumer::CandidateConsumer(const ConsumerOptions& options, RunMetrics& metrics)
    : options_(options), metrics_(metrics) {
    if (options_.threads == 0) throw std::runtime_error("Consumer thread count must be positive");
    metrics_.threads_requested = options_.threads;
    metrics_.threads_used = 1;
    metrics_.signature_enabled = options_.signature;
    metrics_.verification_enabled = !options_.test_data.empty();
    if (metrics_.verification_enabled) load_test_data(options_.test_data, options_.test_limit);
}

void CandidateConsumer::consume(const std::vector<std::string>& candidates, std::uint64_t global_begin) {
    const bool detailed = options_.instrumentation == InstrumentationMode::Detailed;
    if (options_.signature || !targets_.empty()) {
        const auto start = detailed ? std::chrono::steady_clock::now()
                                    : std::chrono::steady_clock::time_point{};
        for (std::size_t i = 0; i < candidates.size(); ++i) {
            const std::string& candidate = candidates[i];
            if (options_.signature) {
                const std::uint64_t hash = hash_string64(candidate);
                metrics_.sig1 += hash;
                metrics_.sig2 += hash * (global_begin + i + 0x9e3779b97f4a7c15ull);
            }
            if (!targets_.empty() && targets_.find(candidate) != targets_.end()) {
                ++metrics_.candidate_hits;
                cracked_.insert(candidate);
            }
        }
        metrics_.cracked_unique = cracked_.size();
        if (detailed) metrics_.verify += elapsed_seconds(start);
    }

    const auto start_hash = detailed ? std::chrono::steady_clock::now()
                                     : std::chrono::steady_clock::time_point{};
    const std::size_t full_groups = candidates.size() / kMd5BatchLanes;
    const bool use_parallel = options_.threads > 1 &&
                              full_groups / options_.threads >= kMinGroupsPerThread;

    if (!use_parallel) {
        std::size_t candidate_index = 0;
        std::array<std::string_view, kMd5BatchLanes> inputs{};
        std::array<Md5Digest, kMd5BatchLanes> outputs{};
        for (; candidate_index + kMd5BatchLanes <= candidates.size(); candidate_index += kMd5BatchLanes) {
            for (std::size_t lane = 0; lane < kMd5BatchLanes; ++lane) {
                inputs[lane] = candidates[candidate_index + lane];
            }
            md5_batch8(inputs, outputs);
            for (std::size_t lane = 0; lane < kMd5BatchLanes; ++lane) {
                update_hash_guard(fold_digest64(outputs[lane]), metrics_.hash_guard);
            }
        }
        for (; candidate_index < candidates.size(); ++candidate_index) {
            update_hash_guard(fold_digest64(md5(candidates[candidate_index])), metrics_.hash_guard);
        }
        if (detailed) metrics_.hash_compute += elapsed_seconds(start_hash);
    } else {
        folded_scratch_.resize(candidates.size());
        int actual_threads = 1;
        const int requested_threads = static_cast<int>(options_.threads);
#pragma omp parallel num_threads(requested_threads) shared(actual_threads)
        {
#pragma omp single
            actual_threads = omp_get_num_threads();
#pragma omp for schedule(static)
            for (std::ptrdiff_t group = 0; group < static_cast<std::ptrdiff_t>(full_groups); ++group) {
                const std::size_t begin = static_cast<std::size_t>(group) * kMd5BatchLanes;
                std::array<std::string_view, kMd5BatchLanes> inputs{};
                std::array<Md5Digest, kMd5BatchLanes> outputs{};
                for (std::size_t lane = 0; lane < kMd5BatchLanes; ++lane) {
                    inputs[lane] = candidates[begin + lane];
                }
                md5_batch8(inputs, outputs);
                for (std::size_t lane = 0; lane < kMd5BatchLanes; ++lane) {
                    folded_scratch_[begin + lane] = fold_digest64(outputs[lane]);
                }
            }
        }
        if (actual_threads != requested_threads) {
            throw std::runtime_error("OpenMP created " + std::to_string(actual_threads) +
                                     " threads; requested " + std::to_string(requested_threads));
        }
        metrics_.threads_used = std::max<std::uint64_t>(metrics_.threads_used, actual_threads);
        for (std::size_t i = full_groups * kMd5BatchLanes; i < candidates.size(); ++i) {
            folded_scratch_[i] = fold_digest64(md5(candidates[i]));
        }
        if (detailed) metrics_.hash_compute += elapsed_seconds(start_hash);
        const auto merge_start = detailed ? std::chrono::steady_clock::now()
                                          : std::chrono::steady_clock::time_point{};
        for (std::size_t i = 0; i < candidates.size(); ++i) {
            update_hash_guard(folded_scratch_[i], metrics_.hash_guard);
        }
        if (detailed) metrics_.hash_merge += elapsed_seconds(merge_start);
    }
    if (detailed) metrics_.hash += elapsed_seconds(start_hash);
    metrics_.generated += candidates.size();
}
void CandidateConsumer::consume(const BatchPlan& plan,
                                const Model& model,
                                std::uint64_t global_begin) {
    merge_slice_result(compute_slice(plan, model, global_begin, 0, plan.candidate_count));
}

FusedSliceResult CandidateConsumer::compute_slice(const BatchPlan& plan,
                                                  const Model& model,
                                                  std::uint64_t global_begin,
                                                  std::size_t slice_begin,
                                                  std::size_t slice_end) {
    std::size_t expected_begin = 0;
    for (const CandidateRange& range : plan.ranges) {
        if (range.batch_begin != expected_begin || range.suffix_begin > range.suffix_end ||
            range.suffix_segment_index >= model.segments.size()) {
            throw std::runtime_error("Invalid or non-contiguous BatchPlan");
        }
        const SegmentTable& suffix = model.segments[range.suffix_segment_index];
        if (range.suffix_end > suffix.values.size()) {
            throw std::runtime_error("BatchPlan suffix range is out of bounds");
        }
        expected_begin += range.suffix_end - range.suffix_begin;
    }
    if (expected_begin != plan.candidate_count ||
        (plan.candidate_count != 0 && plan.ranges.empty())) {
        throw std::runtime_error("BatchPlan candidate count mismatch");
    }
    if (slice_begin > slice_end || slice_end > plan.candidate_count) {
        throw std::runtime_error("Fused slice is out of BatchPlan bounds");
    }

    FusedSliceResult result;
    result.slice_begin = slice_begin;
    result.slice_end = slice_end;
    result.folded.resize(slice_end - slice_begin);
    if (slice_begin == slice_end) return result;

    const bool detailed = options_.instrumentation == InstrumentationMode::Detailed;
    const auto worker_start = detailed ? std::chrono::steady_clock::now()
                                       : std::chrono::steady_clock::time_point{};

    struct ValidationLocal {
        std::uint64_t sig1 = 0;
        std::uint64_t sig2 = 0;
        std::uint64_t hits = 0;
        std::uint64_t long_fallbacks = 0;
        std::vector<std::string> matched;
    };

    const auto find_range = [&plan](std::size_t candidate_index) {
        std::size_t range_index = 0;
        while (range_index + 1 < plan.ranges.size() &&
               candidate_index >= plan.ranges[range_index + 1].batch_begin) {
            ++range_index;
        }
        return range_index;
    };
    const auto fill_candidate = [&plan, &model](
                                    std::size_t candidate_index,
                                    std::size_t& range_index,
                                    std::array<char, 55>& short_buffer,
                                    std::string& long_buffer,
                                    std::uint64_t& long_fallbacks) -> std::string_view {
        while (range_index + 1 < plan.ranges.size() &&
               candidate_index >= plan.ranges[range_index + 1].batch_begin) {
            ++range_index;
        }
        const CandidateRange& range = plan.ranges[range_index];
        const std::size_t offset = candidate_index - range.batch_begin;
        const SegmentTable& suffix_table = model.segments[range.suffix_segment_index];
        const std::string& suffix = suffix_table.values[range.suffix_begin + offset].value;
        const std::size_t length = range.prefix.size() + suffix.size();
        if (length <= short_buffer.size()) {
            std::memcpy(short_buffer.data(), range.prefix.data(), range.prefix.size());
            std::memcpy(short_buffer.data() + range.prefix.size(), suffix.data(), suffix.size());
            return std::string_view(short_buffer.data(), length);
        }
        long_buffer = range.prefix + suffix;
        ++long_fallbacks;
        return long_buffer;
    };
    const auto validate_candidate = [this, global_begin](
                                        std::string_view candidate,
                                        std::size_t batch_index,
                                        ValidationLocal& local) {
        if (options_.signature) {
            const std::uint64_t hash = hash_string64(candidate);
            local.sig1 += hash;
            local.sig2 += hash * (global_begin + batch_index + 0x9e3779b97f4a7c15ull);
        }
        if (!targets_.empty()) {
            std::string owned(candidate);
            if (targets_.find(owned) != targets_.end()) {
                ++local.hits;
                local.matched.push_back(std::move(owned));
            }
        }
    };
    const auto process_scalar = [&](std::size_t begin,
                                    std::size_t end,
                                    ValidationLocal& local) {
        if (begin == end) return;
        std::size_t range_index = find_range(begin);
        std::array<char, 55> short_buffer{};
        std::string long_buffer;
        for (std::size_t index = begin; index < end; ++index) {
            const std::string_view candidate =
                fill_candidate(index, range_index, short_buffer, long_buffer,
                               local.long_fallbacks);
            validate_candidate(candidate, index, local);
            result.folded[index - slice_begin] = fold_digest64(md5(candidate));
        }
    };
    const auto process_groups = [&](std::size_t group_begin,
                                    std::size_t group_end,
                                    ValidationLocal& local) {
        if (group_begin == group_end) return;
        std::size_t range_index = find_range(group_begin * kMd5BatchLanes);
        std::array<std::array<char, 55>, kMd5BatchLanes> short_buffers{};
        std::array<std::string, kMd5BatchLanes> long_buffers{};
        std::array<std::string_view, kMd5BatchLanes> inputs{};
        std::array<Md5Digest, kMd5BatchLanes> outputs{};
        for (std::size_t group = group_begin; group < group_end; ++group) {
            const std::size_t begin = group * kMd5BatchLanes;
            for (std::size_t lane = 0; lane < kMd5BatchLanes; ++lane) {
                const std::size_t index = begin + lane;
                inputs[lane] = fill_candidate(index, range_index, short_buffers[lane],
                                              long_buffers[lane], local.long_fallbacks);
                validate_candidate(inputs[lane], index, local);
            }
            md5_batch8(inputs, outputs);
            for (std::size_t lane = 0; lane < kMd5BatchLanes; ++lane) {
                result.folded[begin + lane - slice_begin] = fold_digest64(outputs[lane]);
            }
        }
    };

    const std::size_t aligned_begin =
        std::min(slice_end, ((slice_begin + kMd5BatchLanes - 1) / kMd5BatchLanes) *
                                kMd5BatchLanes);
    const std::size_t aligned_end = slice_end / kMd5BatchLanes * kMd5BatchLanes;
    const std::size_t full_groups =
        aligned_end > aligned_begin ? (aligned_end - aligned_begin) / kMd5BatchLanes : 0;
    const bool use_parallel = options_.threads > 1 &&
                              full_groups / options_.threads >= kMinGroupsPerThread;
    const std::size_t local_count = use_parallel ? options_.threads : 1;
    std::vector<ValidationLocal> locals(local_count);

    process_scalar(slice_begin, aligned_begin, locals[0]);
    int actual_threads = 1;
    if (use_parallel) {
        const int requested_threads = static_cast<int>(options_.threads);
        const std::size_t first_group = aligned_begin / kMd5BatchLanes;
#pragma omp parallel num_threads(requested_threads) shared(actual_threads)
        {
            const int thread_id = omp_get_thread_num();
            const int team_size = omp_get_num_threads();
#pragma omp single
            actual_threads = team_size;
            const std::size_t begin =
                first_group + full_groups * static_cast<std::size_t>(thread_id) /
                                  static_cast<std::size_t>(team_size);
            const std::size_t end =
                first_group + full_groups * static_cast<std::size_t>(thread_id + 1) /
                                  static_cast<std::size_t>(team_size);
            process_groups(begin, end, locals[static_cast<std::size_t>(thread_id)]);
        }
        if (actual_threads != requested_threads) {
            throw std::runtime_error("OpenMP created " + std::to_string(actual_threads) +
                                     " threads; requested " + std::to_string(requested_threads));
        }
    } else {
        process_groups(aligned_begin / kMd5BatchLanes,
                       aligned_begin / kMd5BatchLanes + full_groups, locals[0]);
    }
    process_scalar(aligned_end, slice_end, locals[0]);

    result.threads_used = static_cast<std::size_t>(actual_threads);
    result.fused_group_count = full_groups;
    result.scalar_tail_count = (aligned_begin - slice_begin) + (slice_end - aligned_end);
    if (detailed) result.worker_seconds = elapsed_seconds(worker_start);
    for (ValidationLocal& local : locals) {
        result.sig1 += local.sig1;
        result.sig2 += local.sig2;
        result.candidate_hits += local.hits;
        result.long_fallback_count += local.long_fallbacks;
        result.matched.insert(result.matched.end(),
                              std::make_move_iterator(local.matched.begin()),
                              std::make_move_iterator(local.matched.end()));
    }
    return result;
}

void CandidateConsumer::merge_slice_result(FusedSliceResult result) {
    metrics_.fused_pipeline = true;
    const bool detailed = options_.instrumentation == InstrumentationMode::Detailed;
    const auto merge_start = detailed ? std::chrono::steady_clock::now()
                                      : std::chrono::steady_clock::time_point{};
    for (const std::uint64_t folded : result.folded) {
        update_hash_guard(folded, metrics_.hash_guard);
    }
    const double merge_seconds = detailed ? elapsed_seconds(merge_start) : 0.0;
    const std::size_t generated = result.folded.size();

    metrics_.threads_used =
        std::max<std::uint64_t>(metrics_.threads_used, result.threads_used);
    metrics_.sig1 += result.sig1;
    metrics_.sig2 += result.sig2;
    metrics_.candidate_hits += result.candidate_hits;
    for (std::string& match : result.matched) cracked_.insert(std::move(match));
    metrics_.cracked_unique = cracked_.size();
    metrics_.generated += generated;

    if (detailed) {
        metrics_.fused_worker += result.worker_seconds;
        metrics_.hash_compute += result.worker_seconds;
        metrics_.ordered_guard_merge += merge_seconds;
        metrics_.hash_merge += merge_seconds;
        metrics_.hash += result.worker_seconds + merge_seconds;
        metrics_.sampler_counters.fused_group_count += result.fused_group_count;
        metrics_.sampler_counters.scalar_tail_count += result.scalar_tail_count;
        metrics_.sampler_counters.long_fallback_count += result.long_fallback_count;
        metrics_.sampler_counters.peak_live_candidate_slots =
            std::max<std::uint64_t>(metrics_.sampler_counters.peak_live_candidate_slots,
                                    result.threads_used * kMd5BatchLanes);
    }
}

std::uint64_t CandidateConsumer::hash_string64(std::string_view value) {
    std::uint64_t hash = 1469598103934665603ull;
    for (unsigned char byte : value) {
        hash ^= byte;
        hash *= 1099511628211ull;
    }
    return hash;
}

void CandidateConsumer::load_test_data(const std::filesystem::path& path, std::uint64_t limit) {
    std::ifstream input(path, std::ios::binary);
    if (!input) throw std::runtime_error("Cannot open test data: " + path.string());
    std::string line;
    std::uint64_t loaded = 0;
    while (std::getline(input, line)) {
        if (!line.empty() && line.back() == '\r') line.pop_back();
        if (line.empty()) continue;
        targets_.insert(line);
        ++loaded;
        if (limit != 0 && loaded >= limit) break;
    }
    if (!input.eof() && input.fail()) throw std::runtime_error("Failed while reading test data: " + path.string());
}

}  // namespace pcfg
