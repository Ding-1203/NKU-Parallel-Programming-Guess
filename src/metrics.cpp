#include "pcfg/metrics.hpp"

#include <filesystem>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <stdexcept>

namespace pcfg {
namespace {

std::string json_escape(const std::string& value) {
    std::ostringstream output;
    for (unsigned char c : value) {
        switch (c) {
            case '"': output << "\\\""; break;
            case '\\': output << "\\\\"; break;
            case '\b': output << "\\b"; break;
            case '\f': output << "\\f"; break;
            case '\n': output << "\\n"; break;
            case '\r': output << "\\r"; break;
            case '\t': output << "\\t"; break;
            default:
                if (c < 0x20u) {
                    output << "\\u" << std::hex << std::setw(4) << std::setfill('0')
                           << static_cast<int>(c) << std::dec;
                } else {
                    output << static_cast<char>(c);
                }
        }
    }
    return output.str();
}

double queue_total(const RunMetrics& metrics) {
    return metrics.sampler_init + metrics.queue_pop + metrics.queue_expand_insert;
}

const char* execution_mode(const RunMetrics& metrics) {
    if (metrics.mpi_pipeline) return "mpi-fused";
    if (metrics.fused_pipeline) return metrics.threads_requested > 1 ? "openmp-fused" : "serial-fused";
    return metrics.threads_requested > 1 ? "openmp-md5" : "serial";
}

}  // namespace

std::string instrumentation_name(InstrumentationMode mode) {
    switch (mode) {
        case InstrumentationMode::Off: return "off";
        case InstrumentationMode::Detailed: return "detailed";
    }
    throw std::runtime_error("Unknown instrumentation mode");
}

InstrumentationMode parse_instrumentation(const std::string& value) {
    if (value == "off") return InstrumentationMode::Off;
    if (value == "detailed") return InstrumentationMode::Detailed;
    throw std::runtime_error("Invalid instrumentation mode: " + value + " (expected off or detailed)");
}

MetricsCollector::MetricsCollector(RunMetrics metrics) : metrics_(std::move(metrics)) {}

RunMetrics& MetricsCollector::mutable_metrics() { return metrics_; }
const RunMetrics& MetricsCollector::metrics() const { return metrics_; }

std::string MetricsCollector::human_summary() const {
    std::ostringstream out;
    out << std::fixed << std::setprecision(6)
        << "========== PCFG Sampling ==========" << '\n'
        << "instrumentation:  " << instrumentation_name(metrics_.instrumentation) << '\n'
        << "threads:          " << metrics_.threads_used << " / " << metrics_.threads_requested
        << " (used / requested)" << '\n';
    if (metrics_.mpi_pipeline) {
        out << "ranks:            " << metrics_.ranks_used << " / " << metrics_.ranks_requested
            << " (used / requested)" << '\n';
    }
    out << "generated:        " << metrics_.generated << '\n'
        << "candidate_hits:   " << metrics_.candidate_hits << '\n'
        << "cracked_unique:   " << metrics_.cracked_unique << '\n'
        << "sig1/sig2:        " << metrics_.sig1 << " / " << metrics_.sig2 << '\n'
        << "load_model:       " << metrics_.load_model << " s" << '\n';
    if (metrics_.instrumentation == InstrumentationMode::Detailed) {
        out << "sampler_init:     " << metrics_.sampler_init << " s" << '\n'
            << "queue_pop:       " << metrics_.queue_pop << " s" << '\n'
            << "queue_expand:    " << metrics_.queue_expand_insert << " s" << '\n'
            << "queue_total:     " << queue_total(metrics_) << " s" << '\n'
            << "generate:        " << metrics_.generate << " s" << '\n'
            << "batch_plan:      " << metrics_.batch_plan << " s" << '\n'
            << "fused_worker:    " << metrics_.fused_worker << " s" << '\n'
            << "guard_merge:     " << metrics_.ordered_guard_merge << " s" << '\n';
        if (metrics_.mpi_pipeline) {
            out << "plan_broadcast:  " << metrics_.plan_broadcast << " s" << '\n'
                << "rank_compute:    " << metrics_.rank_compute_min << " / "
                << metrics_.rank_compute_mean << " / " << metrics_.rank_compute_max
                << " s (min / mean / max)" << '\n'
                << "rank_sync:       " << metrics_.rank_sync << " s" << '\n'
                << "digest_gather:   " << metrics_.digest_gather << " s" << '\n';
        }
        out << "hash:            " << metrics_.hash << " s" << '\n'
            << "verify:          " << metrics_.verify << " s" << '\n';
    }
    out << "core_total:       " << metrics_.core_total << " s" << '\n'
        << "end_to_end:       " << metrics_.end_to_end << " s";
    return out.str();
}

std::string MetricsCollector::result_line() const {
    std::ostringstream out;
    out << std::fixed << std::setprecision(9)
        << "RESULT mode=" << execution_mode(metrics_)
        << " instrumentation=" << instrumentation_name(metrics_.instrumentation)
        << " threads_requested=" << metrics_.threads_requested
        << " threads_used=" << metrics_.threads_used
        << " ranks_requested=" << metrics_.ranks_requested
        << " ranks_used=" << metrics_.ranks_used
        << " generated=" << metrics_.generated
        << " candidate_hits=" << metrics_.candidate_hits
        << " cracked_unique=" << metrics_.cracked_unique
        << " sig1=" << metrics_.sig1
        << " sig2=" << metrics_.sig2
        << " hash_guard=" << metrics_.hash_guard
        << " limit=" << metrics_.limit
        << " batch_size=" << metrics_.batch_size
        << " model_records=" << metrics_.model_records
        << " signature=" << (metrics_.signature_enabled ? 1 : 0)
        << " verification=" << (metrics_.verification_enabled ? 1 : 0)
        << " load_model=" << metrics_.load_model;
    if (metrics_.instrumentation == InstrumentationMode::Detailed) {
        out << " sampler_init=" << metrics_.sampler_init
            << " queue_pop=" << metrics_.queue_pop
            << " queue_expand_insert=" << metrics_.queue_expand_insert
            << " queue_total=" << queue_total(metrics_)
            << " generate=" << metrics_.generate
            << " batch_plan=" << metrics_.batch_plan
            << " fused_worker=" << metrics_.fused_worker
            << " ordered_guard_merge=" << metrics_.ordered_guard_merge;
        if (metrics_.mpi_pipeline) {
            out << " plan_broadcast=" << metrics_.plan_broadcast
                << " rank_compute_min=" << metrics_.rank_compute_min
                << " rank_compute_mean=" << metrics_.rank_compute_mean
                << " rank_compute_max=" << metrics_.rank_compute_max
                << " rank_sync=" << metrics_.rank_sync
                << " digest_gather=" << metrics_.digest_gather;
        }
        out << " hash=" << metrics_.hash
            << " verify=" << metrics_.verify;
    }
    out << " core_total=" << metrics_.core_total
        << " end_to_end=" << metrics_.end_to_end;
    return out.str();
}

std::string MetricsCollector::json() const {
    const bool detailed = metrics_.instrumentation == InstrumentationMode::Detailed;
    const SamplerCounters& c = metrics_.sampler_counters;
    std::ostringstream out;
    out << std::fixed << std::setprecision(9)
        << "{" << '\n'
        << "  \"schema\": \"pcfg-baseline-result-v2\"," << '\n'
        << "  \"mode\": \"" << execution_mode(metrics_) << "\"," << '\n'
        << "  \"instrumentation\": \"" << instrumentation_name(metrics_.instrumentation) << "\"," << '\n'
        << "  \"detailed_available\": " << (detailed ? "true" : "false") << "," << '\n'
        << "  \"model_path\": \"" << json_escape(metrics_.model_path) << "\"," << '\n'
        << "  \"model_records\": " << metrics_.model_records << "," << '\n'
        << "  \"limit\": " << metrics_.limit << "," << '\n'
        << "  \"batch_size\": " << metrics_.batch_size << "," << '\n'
        << "  \"threads_requested\": " << metrics_.threads_requested << "," << '\n'
        << "  \"threads_used\": " << metrics_.threads_used << "," << '\n'
        << "  \"ranks_requested\": " << metrics_.ranks_requested << "," << '\n'
        << "  \"ranks_used\": " << metrics_.ranks_used << "," << '\n'
        << "  \"signature_enabled\": " << (metrics_.signature_enabled ? "true" : "false") << "," << '\n'
        << "  \"verification_enabled\": " << (metrics_.verification_enabled ? "true" : "false") << "," << '\n'
        << "  \"generated\": " << metrics_.generated << "," << '\n'
        << "  \"candidate_hits\": " << metrics_.candidate_hits << "," << '\n'
        << "  \"cracked_unique\": " << metrics_.cracked_unique << "," << '\n'
        << "  \"sig1\": " << metrics_.sig1 << "," << '\n'
        << "  \"sig2\": " << metrics_.sig2 << "," << '\n'
        << "  \"hash_guard\": " << metrics_.hash_guard << "," << '\n'
        << "  \"timing_seconds\": {" << '\n'
        << "    \"load_model\": " << metrics_.load_model << "," << '\n'
        << "    \"core_total\": " << metrics_.core_total << "," << '\n'
        << "    \"end_to_end\": " << metrics_.end_to_end << "," << '\n'
        << "    \"detailed\": ";
    if (!detailed) {
        out << "null" << '\n';
    } else {
        out << "{" << '\n'
            << "      \"sampler_init\": " << metrics_.sampler_init << "," << '\n'
            << "      \"queue_pop\": " << metrics_.queue_pop << "," << '\n'
            << "      \"queue_expand_insert\": " << metrics_.queue_expand_insert << "," << '\n'
            << "      \"queue_total\": " << queue_total(metrics_) << "," << '\n'
            << "      \"generate\": " << metrics_.generate << "," << '\n'
            << "      \"batch_plan\": " << metrics_.batch_plan << "," << '\n'
            << "      \"fused_worker\": " << metrics_.fused_worker << "," << '\n'
            << "      \"ordered_guard_merge\": " << metrics_.ordered_guard_merge << "," << '\n'
            << "      \"plan_broadcast\": " << metrics_.plan_broadcast << "," << '\n'
            << "      \"rank_compute_min\": " << metrics_.rank_compute_min << "," << '\n'
            << "      \"rank_compute_mean\": " << metrics_.rank_compute_mean << "," << '\n'
            << "      \"rank_compute_max\": " << metrics_.rank_compute_max << "," << '\n'
            << "      \"rank_sync\": " << metrics_.rank_sync << "," << '\n'
            << "      \"digest_gather\": " << metrics_.digest_gather << "," << '\n'
            << "      \"hash\": " << metrics_.hash << "," << '\n'
            << "      \"hash_compute\": " << metrics_.hash_compute << "," << '\n'
            << "      \"hash_merge\": " << metrics_.hash_merge << "," << '\n'
            << "      \"verify\": " << metrics_.verify << '\n'
            << "    }" << '\n';
    }
    out << "  }," << '\n'
        << "  \"mpi\": ";
    if (!metrics_.mpi_pipeline) {
        out << "null," << '\n';
    } else {
        const double imbalance = metrics_.rank_compute_mean > 0.0
                                     ? metrics_.rank_compute_max / metrics_.rank_compute_mean
                                     : 0.0;
        out << "{" << '\n'
            << "    \"batch_count\": " << metrics_.mpi_batch_count << "," << '\n'
            << "    \"plan_broadcast_bytes\": " << metrics_.plan_broadcast_bytes << "," << '\n'
            << "    \"digest_gather_bytes\": " << metrics_.digest_gather_bytes << "," << '\n'
            << "    \"matched_gather_bytes\": " << metrics_.matched_gather_bytes << "," << '\n'
            << "    \"rank_rss_min_bytes\": " << metrics_.rank_rss_min_bytes << "," << '\n'
            << "    \"rank_rss_mean_bytes\": " << metrics_.rank_rss_mean_bytes << "," << '\n'
            << "    \"rank_rss_max_bytes\": " << metrics_.rank_rss_max_bytes << "," << '\n'
            << "    \"rank_compute_imbalance\": " << imbalance << '\n'
            << "  }," << '\n';
    }
    out << "  \"sampler_counters\": ";
    if (!detailed) {
        out << "null" << '\n';
    } else {
        out << "{" << '\n'
            << "    \"initial_queue_size\": " << c.initial_queue_size << "," << '\n'
            << "    \"max_queue_size\": " << c.max_queue_size << "," << '\n'
            << "    \"pop_count\": " << c.pop_count << "," << '\n'
            << "    \"insert_count\": " << c.insert_count << "," << '\n'
            << "    \"expand_count\": " << c.expand_count << "," << '\n'
            << "    \"completed_pt_count\": " << c.completed_pt_count << "," << '\n'
            << "    \"batch_count\": " << c.batch_count << "," << '\n'
            << "    \"range_count\": " << c.range_count << "," << '\n'
            << "    \"fused_group_count\": " << c.fused_group_count << "," << '\n'
            << "    \"scalar_tail_count\": " << c.scalar_tail_count << "," << '\n'
            << "    \"long_fallback_count\": " << c.long_fallback_count << "," << '\n'
            << "    \"peak_live_candidate_slots\": " << c.peak_live_candidate_slots << "," << '\n'
            << "    \"shifted_on_pop\": " << c.shifted_on_pop << "," << '\n'
            << "    \"shifted_on_insert\": " << c.shifted_on_insert << "," << '\n'
            << "    \"candidate_bytes\": " << c.candidate_bytes << "," << '\n'
            << "    \"min_candidate_length\": " << c.min_candidate_length << "," << '\n'
            << "    \"max_candidate_length\": " << c.max_candidate_length << '\n'
            << "  }" << '\n';
    }
    out << "}" << '\n';
    return out.str();
}

void MetricsCollector::write_json(const std::filesystem::path& path) const {
    if (!path.parent_path().empty()) std::filesystem::create_directories(path.parent_path());
    std::ofstream output(path, std::ios::trunc);
    if (!output) throw std::runtime_error("Cannot create metrics file: " + path.string());
    output << json();
    if (!output) throw std::runtime_error("Cannot write metrics file: " + path.string());
}

}  // namespace pcfg
