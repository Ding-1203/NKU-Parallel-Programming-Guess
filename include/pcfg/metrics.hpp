#pragma once

#include <cstdint>
#include <filesystem>
#include <string>

namespace pcfg {

enum class InstrumentationMode {
    Off,
    Detailed,
};

std::string instrumentation_name(InstrumentationMode mode);
InstrumentationMode parse_instrumentation(const std::string& value);

struct SamplerCounters {
    std::uint64_t initial_queue_size = 0;
    std::uint64_t max_queue_size = 0;
    std::uint64_t pop_count = 0;
    std::uint64_t insert_count = 0;
    std::uint64_t expand_count = 0;
    std::uint64_t completed_pt_count = 0;
    std::uint64_t batch_count = 0;
    std::uint64_t range_count = 0;
    std::uint64_t fused_group_count = 0;
    std::uint64_t scalar_tail_count = 0;
    std::uint64_t long_fallback_count = 0;
    std::uint64_t peak_live_candidate_slots = 0;
    std::uint64_t shifted_on_pop = 0;
    std::uint64_t shifted_on_insert = 0;
    std::uint64_t candidate_bytes = 0;
    std::uint64_t min_candidate_length = 0;
    std::uint64_t max_candidate_length = 0;
};

struct RunMetrics {
    std::uint64_t generated = 0;
    std::uint64_t candidate_hits = 0;
    std::uint64_t cracked_unique = 0;
    std::uint64_t sig1 = 0;
    std::uint64_t sig2 = 0;
    std::uint64_t hash_guard = 0;
    std::uint64_t threads_requested = 1;
    std::uint64_t threads_used = 1;
    std::uint64_t ranks_requested = 1;
    std::uint64_t ranks_used = 1;

    double load_model = 0.0;
    double sampler_init = 0.0;
    double queue_pop = 0.0;
    double queue_expand_insert = 0.0;
    double generate = 0.0;
    double hash = 0.0;
    double hash_compute = 0.0;
    double hash_merge = 0.0;
    double batch_plan = 0.0;
    double fused_worker = 0.0;
    double ordered_guard_merge = 0.0;
    double plan_broadcast = 0.0;
    double rank_compute_min = 0.0;
    double rank_compute_mean = 0.0;
    double rank_compute_max = 0.0;
    double rank_sync = 0.0;
    double digest_gather = 0.0;
    double verify = 0.0;
    double core_total = 0.0;
    double end_to_end = 0.0;

    std::uint64_t limit = 0;
    std::uint64_t batch_size = 0;
    std::uint64_t model_records = 0;
    bool signature_enabled = false;
    bool verification_enabled = false;
    bool fused_pipeline = false;
    bool mpi_pipeline = false;
    std::uint64_t mpi_batch_count = 0;
    std::uint64_t plan_broadcast_bytes = 0;
    std::uint64_t digest_gather_bytes = 0;
    std::uint64_t matched_gather_bytes = 0;
    std::uint64_t rank_rss_min_bytes = 0;
    std::uint64_t rank_rss_mean_bytes = 0;
    std::uint64_t rank_rss_max_bytes = 0;
    InstrumentationMode instrumentation = InstrumentationMode::Off;
    SamplerCounters sampler_counters;
    std::string model_path;
};

class MetricsCollector {
public:
    explicit MetricsCollector(RunMetrics metrics);
    RunMetrics& mutable_metrics();
    const RunMetrics& metrics() const;

    std::string human_summary() const;
    std::string result_line() const;
    std::string json() const;
    void write_json(const std::filesystem::path& path) const;

private:
    RunMetrics metrics_;
};

}  // namespace pcfg
