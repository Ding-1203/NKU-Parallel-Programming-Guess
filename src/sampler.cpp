#include "pcfg/sampler.hpp"

#include <algorithm>
#include <chrono>
#include <stdexcept>

namespace pcfg {
namespace {

double elapsed_seconds(std::chrono::steady_clock::time_point start) {
    return std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
}

}  // namespace

SerialSampler::SerialSampler(const Model& model, InstrumentationMode instrumentation)
    : model_(model), instrumentation_(instrumentation) {
    if (model_.total_preterminal_count == 0) throw std::runtime_error("Cannot sample from an empty model");
    queue_.reserve(model_.preterminals.size());
    for (std::size_t i = 0; i < model_.preterminals.size(); ++i) {
        const Preterminal& pt = model_.preterminals[i];
        if (pt.pattern.empty()) throw std::runtime_error("Model contains empty preterminal");
        SearchState state;
        state.preterminal_index = i;
        state.value_indices.assign(pt.pattern.size(), 0);
        state.sequence = next_sequence_++;
        state.probability = probability(state);
        queue_.push_back(std::move(state));
        if (instrumentation_ == InstrumentationMode::Detailed) {
            ++profile_.counters.insert_count;
        }
    }
    std::make_heap(queue_.begin(), queue_.end(),
        [this](const SearchState& left, const SearchState& right) {
            return lower_priority(left, right);
        });
    if (instrumentation_ == InstrumentationMode::Detailed) {
        profile_.counters.initial_queue_size = queue_.size();
        profile_.counters.max_queue_size = queue_.size();
    }
}

bool SerialSampler::empty() const {
    return !active_.has_value() && queue_.empty();
}

SearchState SerialSampler::pop() {
    if (queue_.empty()) throw std::runtime_error("Pop from empty sampler queue");
    const auto start = instrumentation_ == InstrumentationMode::Detailed
                           ? std::chrono::steady_clock::now()
                           : std::chrono::steady_clock::time_point{};
    if (instrumentation_ == InstrumentationMode::Detailed) {
        ++profile_.counters.pop_count;
    }
    std::pop_heap(queue_.begin(), queue_.end(),
        [this](const SearchState& left, const SearchState& right) {
            return lower_priority(left, right);
        });
    SearchState state = std::move(queue_.back());
    queue_.pop_back();
    if (instrumentation_ == InstrumentationMode::Detailed) {
        profile_.queue_pop += elapsed_seconds(start);
    }
    return state;
}

void SerialSampler::expand(const SearchState& state) {
    const auto start = instrumentation_ == InstrumentationMode::Detailed
                           ? std::chrono::steady_clock::now()
                           : std::chrono::steady_clock::time_point{};
    if (instrumentation_ == InstrumentationMode::Detailed) ++profile_.counters.expand_count;
    const Preterminal& pt = model_.preterminals.at(state.preterminal_index);
    if (pt.pattern.size() <= 1) {
        if (instrumentation_ == InstrumentationMode::Detailed) profile_.queue_expand_insert += elapsed_seconds(start);
        return;
    }
    for (std::size_t i = state.pivot; i + 1 < pt.pattern.size(); ++i) {
        SearchState child = state;
        const SegmentTable& table = model_.segment(pt.pattern[i]);
        ++child.value_indices[i];
        if (child.value_indices[i] >= table.values.size()) continue;
        child.pivot = static_cast<std::uint32_t>(i);
        child.sequence = next_sequence_++;
        child.probability = probability(child);
        insert(std::move(child));
    }
    if (instrumentation_ == InstrumentationMode::Detailed) {
        profile_.queue_expand_insert += elapsed_seconds(start);
    }
}

std::size_t SerialSampler::last_segment_size(const SearchState& state) const {
    const Preterminal& pt = model_.preterminals.at(state.preterminal_index);
    return model_.segment(pt.pattern.back()).values.size();
}

void SerialSampler::generate_range(const SearchState& state,
                                   std::size_t begin,
                                   std::size_t end,
                                   std::vector<std::string>& output) const {
    const Preterminal& pt = model_.preterminals.at(state.preterminal_index);
    const SegmentTable& last = model_.segment(pt.pattern.back());
    if (begin > end || end > last.values.size()) throw std::out_of_range("Invalid generate range");

    std::string prefix;
    for (std::size_t i = 0; i + 1 < pt.pattern.size(); ++i) {
        const SegmentTable& table = model_.segment(pt.pattern[i]);
        const std::size_t index = state.value_indices.at(i);
        if (index >= table.values.size()) throw std::runtime_error("Search state index out of range");
        prefix += table.values[index].value;
    }
    output.reserve(output.size() + end - begin);
    for (std::size_t i = begin; i < end; ++i) output.push_back(prefix + last.values[i].value);
}

SampleBatch SerialSampler::sample_batch(std::size_t max_candidates) {
    if (max_candidates == 0) throw std::invalid_argument("Batch size must be positive");
    SampleBatch batch;
    batch.candidates.reserve(max_candidates);
    if (instrumentation_ == InstrumentationMode::Detailed) ++profile_.counters.batch_count;

    while (batch.candidates.size() < max_candidates && !empty()) {
        if (!active_) {
            active_ = pop();
            active_offset_ = 0;
        }
        const std::size_t available = last_segment_size(*active_) - active_offset_;
        const std::size_t take = std::min(available, max_candidates - batch.candidates.size());
        const auto start_generate = instrumentation_ == InstrumentationMode::Detailed
                                        ? std::chrono::steady_clock::now()
                                        : std::chrono::steady_clock::time_point{};
        const std::size_t old_size = batch.candidates.size();
        generate_range(*active_, active_offset_, active_offset_ + take, batch.candidates);
        if (instrumentation_ == InstrumentationMode::Detailed) {
            profile_.generate += elapsed_seconds(start_generate);
            ++profile_.counters.range_count;
            for (std::size_t i = old_size; i < batch.candidates.size(); ++i) {
                const std::uint64_t length = batch.candidates[i].size();
                profile_.counters.candidate_bytes += length;
                if (profile_.counters.min_candidate_length == 0 || length < profile_.counters.min_candidate_length) {
                    profile_.counters.min_candidate_length = length;
                }
                profile_.counters.max_candidate_length =
                    std::max(profile_.counters.max_candidate_length, length);
            }
        }
        active_offset_ += take;

        if (active_offset_ == last_segment_size(*active_)) {
            expand(*active_);
            active_.reset();
            active_offset_ = 0;
            if (instrumentation_ == InstrumentationMode::Detailed) ++profile_.counters.completed_pt_count;
        }
    }
    return batch;
}

BatchPlan SerialSampler::plan_batch(std::size_t max_candidates) {
    if (max_candidates == 0) throw std::invalid_argument("Batch size must be positive");
    BatchPlan plan;
    if (instrumentation_ == InstrumentationMode::Detailed) ++profile_.counters.batch_count;

    while (plan.candidate_count < max_candidates && !empty()) {
        if (!active_) {
            active_ = pop();
            active_offset_ = 0;
        }
        const std::size_t available = last_segment_size(*active_) - active_offset_;
        const std::size_t take = std::min(available, max_candidates - plan.candidate_count);
        const auto start_plan = instrumentation_ == InstrumentationMode::Detailed
                                    ? std::chrono::steady_clock::now()
                                    : std::chrono::steady_clock::time_point{};

        const Preterminal& pt = model_.preterminals.at(active_->preterminal_index);
        CandidateRange range;
        for (std::size_t i = 0; i + 1 < pt.pattern.size(); ++i) {
            const SegmentTable& table = model_.segment(pt.pattern[i]);
            const std::size_t index = active_->value_indices.at(i);
            if (index >= table.values.size()) throw std::runtime_error("Search state index out of range");
            range.prefix += table.values[index].value;
        }
        range.suffix_segment_index = model_.segment_index(pt.pattern.back());
        range.suffix_begin = active_offset_;
        range.suffix_end = active_offset_ + take;
        range.batch_begin = plan.candidate_count;
        plan.ranges.push_back(std::move(range));
        plan.candidate_count += take;

        if (instrumentation_ == InstrumentationMode::Detailed) {
            profile_.batch_plan += elapsed_seconds(start_plan);
            ++profile_.counters.range_count;
            const CandidateRange& recorded = plan.ranges.back();
            const SegmentTable& suffix = model_.segments.at(recorded.suffix_segment_index);
            for (std::size_t i = recorded.suffix_begin; i < recorded.suffix_end; ++i) {
                const std::uint64_t length = recorded.prefix.size() + suffix.values[i].value.size();
                profile_.counters.candidate_bytes += length;
                if (profile_.counters.min_candidate_length == 0 ||
                    length < profile_.counters.min_candidate_length) {
                    profile_.counters.min_candidate_length = length;
                }
                profile_.counters.max_candidate_length =
                    std::max(profile_.counters.max_candidate_length, length);
            }
        }

        active_offset_ += take;
        if (active_offset_ == last_segment_size(*active_)) {
            expand(*active_);
            active_.reset();
            active_offset_ = 0;
            if (instrumentation_ == InstrumentationMode::Detailed) ++profile_.counters.completed_pt_count;
        }
    }
    return plan;
}

const SamplerProfile& SerialSampler::profile() const {
    return profile_;
}

std::size_t SerialSampler::queue_size() const {
    return queue_.size() + (active_.has_value() ? 1u : 0u);
}

double SerialSampler::probability(const SearchState& state) const {
    const Preterminal& pt = model_.preterminals.at(state.preterminal_index);
    double result = static_cast<double>(pt.count) / static_cast<double>(model_.total_preterminal_count);
    for (std::size_t i = 0; i < pt.pattern.size(); ++i) {
        const SegmentTable& table = model_.segment(pt.pattern[i]);
        const std::size_t index = state.value_indices.at(i);
        if (index >= table.values.size() || table.total_count == 0) throw std::runtime_error("Invalid state probability input");
        result *= static_cast<double>(table.values[index].count) / static_cast<double>(table.total_count);
    }
    return result;
}

void SerialSampler::insert(SearchState state) {
    queue_.push_back(std::move(state));
    std::push_heap(queue_.begin(), queue_.end(),
        [this](const SearchState& left, const SearchState& right) {
            return lower_priority(left, right);
        });
    if (instrumentation_ == InstrumentationMode::Detailed) {
        ++profile_.counters.insert_count;
        profile_.counters.max_queue_size =
            std::max<std::uint64_t>(profile_.counters.max_queue_size, queue_.size());
    }
}

bool SerialSampler::lower_priority(const SearchState& left, const SearchState& right) const {
    return higher_priority(right, left);
}

bool SerialSampler::higher_priority(const SearchState& left, const SearchState& right) const {
    if (left.probability != right.probability) return left.probability > right.probability;
    const Preterminal& left_pt = model_.preterminals.at(left.preterminal_index);
    const Preterminal& right_pt = model_.preterminals.at(right.preterminal_index);
    if (left_pt.pattern != right_pt.pattern) return left_pt.pattern < right_pt.pattern;
    if (left.value_indices != right.value_indices) return left.value_indices < right.value_indices;
    if (left.pivot != right.pivot) return left.pivot < right.pivot;
    return left.sequence < right.sequence;
}

}  // namespace pcfg
