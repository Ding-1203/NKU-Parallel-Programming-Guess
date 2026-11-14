#pragma once

#include "pcfg/metrics.hpp"
#include "pcfg/model.hpp"
#include "pcfg/sampler.hpp"

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <string>
#include <string_view>
#include <unordered_set>
#include <vector>

namespace pcfg {

struct ConsumerOptions {
    bool signature = false;
    std::filesystem::path test_data;
    std::uint64_t test_limit = 0;
    InstrumentationMode instrumentation = InstrumentationMode::Off;
    std::size_t threads = 1;
};

struct FusedSliceResult {
    std::size_t slice_begin = 0;
    std::size_t slice_end = 0;
    std::size_t threads_used = 1;
    std::vector<std::uint64_t> folded;
    std::uint64_t sig1 = 0;
    std::uint64_t sig2 = 0;
    std::uint64_t candidate_hits = 0;
    std::uint64_t fused_group_count = 0;
    std::uint64_t scalar_tail_count = 0;
    std::uint64_t long_fallback_count = 0;
    double worker_seconds = 0.0;
    std::vector<std::string> matched;
};

class CandidateConsumer {
public:
    CandidateConsumer(const ConsumerOptions& options, RunMetrics& metrics);
    void consume(const std::vector<std::string>& candidates, std::uint64_t global_begin);
    void consume(const BatchPlan& plan, const Model& model, std::uint64_t global_begin);
    FusedSliceResult compute_slice(const BatchPlan& plan,
                                   const Model& model,
                                   std::uint64_t global_begin,
                                   std::size_t slice_begin,
                                   std::size_t slice_end);
    void merge_slice_result(FusedSliceResult result);

private:
    static std::uint64_t hash_string64(std::string_view value);
    void load_test_data(const std::filesystem::path& path, std::uint64_t limit);

    ConsumerOptions options_;
    RunMetrics& metrics_;
    std::unordered_set<std::string> targets_;
    std::unordered_set<std::string> cracked_;
    std::vector<std::uint64_t> folded_scratch_;
};

}  // namespace pcfg
