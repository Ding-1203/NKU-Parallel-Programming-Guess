#pragma once

#include "pcfg/model.hpp"
#include "pcfg/metrics.hpp"

#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>
#include <vector>

namespace pcfg {

struct SearchState {
    std::size_t preterminal_index = 0;
    std::vector<std::uint32_t> value_indices;
    std::uint32_t pivot = 0;
    double probability = 0.0;
    std::uint64_t sequence = 0;
};

struct SampleBatch {
    std::vector<std::string> candidates;
};

struct CandidateRange {
    std::string prefix;
    std::size_t suffix_segment_index = 0;
    std::size_t suffix_begin = 0;
    std::size_t suffix_end = 0;
    std::size_t batch_begin = 0;
};

struct BatchPlan {
    std::vector<CandidateRange> ranges;
    std::size_t candidate_count = 0;
};

struct SamplerProfile {
    double queue_pop = 0.0;
    double queue_expand_insert = 0.0;
    double generate = 0.0;
    double batch_plan = 0.0;
    SamplerCounters counters;
};

class SerialSampler {
public:
    explicit SerialSampler(const Model& model,
                           InstrumentationMode instrumentation = InstrumentationMode::Off);

    bool empty() const;
    SearchState pop();
    void expand(const SearchState& state);
    std::size_t last_segment_size(const SearchState& state) const;
    void generate_range(const SearchState& state,
                        std::size_t begin,
                        std::size_t end,
                        std::vector<std::string>& output) const;
    SampleBatch sample_batch(std::size_t max_candidates);
    BatchPlan plan_batch(std::size_t max_candidates);
    const SamplerProfile& profile() const;
    std::size_t queue_size() const;

private:
    double probability(const SearchState& state) const;
    void insert(SearchState state);
    bool higher_priority(const SearchState& left, const SearchState& right) const;

    bool lower_priority(const SearchState& left, const SearchState& right) const;
    const Model& model_;
    std::vector<SearchState> queue_;
    InstrumentationMode instrumentation_;
    SamplerProfile profile_;
    std::optional<SearchState> active_;
    std::size_t active_offset_ = 0;
    std::uint64_t next_sequence_ = 0;
};

}  // namespace pcfg
