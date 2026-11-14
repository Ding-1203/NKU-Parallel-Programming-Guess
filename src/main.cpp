#include "pcfg/consumer.hpp"
#include "pcfg/metrics.hpp"
#include "pcfg/model.hpp"
#include "pcfg/model_io.hpp"
#include "pcfg/sampler.hpp"
#include "pcfg/sha256.hpp"

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>

namespace {

using Clock = std::chrono::steady_clock;

double seconds_since(Clock::time_point start) {
    return std::chrono::duration<double>(Clock::now() - start).count();
}

std::uint64_t parse_u64(const std::string& value, const std::string& name) {
    try {
        std::size_t position = 0;
        const std::uint64_t result = std::stoull(value, &position);
        if (position != value.size()) throw std::invalid_argument("suffix");
        return result;
    } catch (...) {
        throw std::runtime_error("Invalid integer for " + name + ": " + value);
    }
}

std::string require_value(int argc, char** argv, int& index, const std::string& option) {
    if (index >= argc) throw std::runtime_error(option + " requires a value");
    return argv[index++];
}

void print_usage(const char* program) {
    std::cerr
        << "Usage:" << '\n'
        << "  " << program << " train --input PATH --output PATH [--train-limit N] [--metrics-out PATH]" << '\n'
        << "  " << program << " sample --model PATH [--limit N] [--batch-size N]" << '\n'
        << "      [--threads N]" << '\n'
        << "      [--test-data PATH] [--test-limit N] [--signature] [--metrics-out PATH]" << '\n'
        << "      [--instrumentation off|detailed] [--trace-out PATH] [--trace-interval N]" << '\n';
}

void copy_sampler_profile(const pcfg::SerialSampler& sampler, pcfg::RunMetrics& metrics) {
    const pcfg::SamplerProfile& profile = sampler.profile();
    metrics.queue_pop = profile.queue_pop;
    metrics.queue_expand_insert = profile.queue_expand_insert;
    metrics.generate = profile.generate;
    metrics.batch_plan = profile.batch_plan;
    const pcfg::SamplerCounters fused = metrics.sampler_counters;
    metrics.sampler_counters = profile.counters;
    metrics.sampler_counters.fused_group_count = fused.fused_group_count;
    metrics.sampler_counters.scalar_tail_count = fused.scalar_tail_count;
    metrics.sampler_counters.long_fallback_count = fused.long_fallback_count;
    metrics.sampler_counters.peak_live_candidate_slots = fused.peak_live_candidate_slots;
}

void write_trace_record(std::ofstream& trace,
                        const pcfg::SerialSampler& sampler,
                        const pcfg::RunMetrics& metrics,
                        Clock::time_point core_start) {
    const pcfg::SamplerProfile& p = sampler.profile();
    const pcfg::SamplerCounters& c = p.counters;
    trace << std::fixed << std::setprecision(9)
          << "{"
          << "\"generated\":" << metrics.generated
          << ",\"elapsed_core\":" << seconds_since(core_start)
          << ",\"queue_size\":" << sampler.queue_size()
          << ",\"sampler_init\":" << metrics.sampler_init
          << ",\"queue_pop\":" << p.queue_pop
          << ",\"queue_expand_insert\":" << p.queue_expand_insert
          << ",\"generate\":" << p.generate
          << ",\"batch_plan\":" << p.batch_plan
          << ",\"fused_worker\":" << metrics.fused_worker
          << ",\"ordered_guard_merge\":" << metrics.ordered_guard_merge
          << ",\"hash\":" << metrics.hash
          << ",\"verify\":" << metrics.verify
          << ",\"pop_count\":" << c.pop_count
          << ",\"insert_count\":" << c.insert_count
          << ",\"expand_count\":" << c.expand_count
          << ",\"completed_pt_count\":" << c.completed_pt_count
          << ",\"shifted_on_pop\":" << c.shifted_on_pop
          << ",\"shifted_on_insert\":" << c.shifted_on_insert
          << "}" << '\n';
    if (!trace) throw std::runtime_error("Failed while writing trace");
}

int run_train(int argc, char** argv) {
    std::filesystem::path input;
    std::filesystem::path output;
    std::filesystem::path metrics_output;
    std::uint64_t train_limit = 0;
    int index = 2;
    while (index < argc) {
        const std::string option = argv[index++];
        if (option == "--input") input = require_value(argc, argv, index, option);
        else if (option == "--output") output = require_value(argc, argv, index, option);
        else if (option == "--train-limit") train_limit = parse_u64(require_value(argc, argv, index, option), option);
        else if (option == "--metrics-out") metrics_output = require_value(argc, argv, index, option);
        else throw std::runtime_error("Unknown train option: " + option);
    }
    if (input.empty() || output.empty()) throw std::runtime_error("train requires --input and --output");

    const auto start = Clock::now();
    pcfg::Model model = pcfg::ModelTrainer().train(pcfg::TrainingOptions{input, train_limit});
    const double train_seconds = seconds_since(start);
    const auto save_start = Clock::now();
    pcfg::ModelStore::save(model, output);
    const double save_seconds = seconds_since(save_start);
    const double total_seconds = seconds_since(start);

    std::cout << "========== PCFG Model Training ==========" << '\n'
              << "source:            " << input << '\n'
              << "output:            " << output << '\n'
              << "source_sha256:     " << pcfg::hex_encode(model.metadata.source_sha256) << '\n'
              << "source_lines:      " << model.metadata.source_lines << '\n'
              << "processed_records: " << model.metadata.processed_records << '\n'
              << "skipped_empty:     " << model.metadata.skipped_empty << '\n'
              << "preterminals:      " << model.preterminals.size() << '\n'
              << "segment_tables:    " << model.segments.size() << '\n'
              << std::fixed << std::setprecision(6)
              << "train:             " << train_seconds << " s" << '\n'
              << "save:              " << save_seconds << " s" << '\n'
              << "total:             " << total_seconds << " s" << '\n';
    std::cout << "RESULT mode=train records=" << model.metadata.processed_records
              << " source_lines=" << model.metadata.source_lines
              << " skipped_empty=" << model.metadata.skipped_empty
              << " preterminals=" << model.preterminals.size()
              << " segments=" << model.segments.size()
              << " train=" << train_seconds
              << " save=" << save_seconds
              << " total=" << total_seconds << '\n';

    if (!metrics_output.empty()) {
        if (!metrics_output.parent_path().empty()) std::filesystem::create_directories(metrics_output.parent_path());
        std::ofstream json(metrics_output);
        if (!json) throw std::runtime_error("Cannot create training metrics: " + metrics_output.string());
        json << "{" << '\n'
             << "  \"schema\": \"pcfg-baseline-training-v1\"," << '\n'
             << "  \"source\": \"" << input.string() << "\"," << '\n'
             << "  \"output\": \"" << output.string() << "\"," << '\n'
             << "  \"source_sha256\": \"" << pcfg::hex_encode(model.metadata.source_sha256) << "\"," << '\n'
             << "  \"source_lines\": " << model.metadata.source_lines << "," << '\n'
             << "  \"processed_records\": " << model.metadata.processed_records << "," << '\n'
             << "  \"skipped_empty\": " << model.metadata.skipped_empty << "," << '\n'
             << "  \"preterminals\": " << model.preterminals.size() << "," << '\n'
             << "  \"segment_tables\": " << model.segments.size() << "," << '\n'
             << "  \"train_seconds\": " << train_seconds << "," << '\n'
             << "  \"save_seconds\": " << save_seconds << "," << '\n'
             << "  \"total_seconds\": " << total_seconds << '\n'
             << "}" << '\n';
    }
    return 0;
}

int run_sample(int argc, char** argv) {
    std::filesystem::path model_path;
    std::filesystem::path test_data;
    std::filesystem::path metrics_output;
    std::filesystem::path trace_output;
    std::uint64_t limit = 10000000;
    std::uint64_t batch_size = 1000000;
    std::uint64_t test_limit = 0;
    std::uint64_t trace_interval = 100000;
    std::uint64_t threads = 1;
    bool signature = false;
    pcfg::InstrumentationMode instrumentation = pcfg::InstrumentationMode::Off;

    int index = 2;
    while (index < argc) {
        const std::string option = argv[index++];
        if (option == "--model") model_path = require_value(argc, argv, index, option);
        else if (option == "--limit") limit = parse_u64(require_value(argc, argv, index, option), option);
        else if (option == "--batch-size") batch_size = parse_u64(require_value(argc, argv, index, option), option);
        else if (option == "--test-data") test_data = require_value(argc, argv, index, option);
        else if (option == "--test-limit") test_limit = parse_u64(require_value(argc, argv, index, option), option);
        else if (option == "--threads") threads = parse_u64(require_value(argc, argv, index, option), option);
        else if (option == "--signature") signature = true;
        else if (option == "--metrics-out") metrics_output = require_value(argc, argv, index, option);
        else if (option == "--instrumentation") {
            instrumentation = pcfg::parse_instrumentation(require_value(argc, argv, index, option));
        } else if (option == "--trace-out") {
            trace_output = require_value(argc, argv, index, option);
        } else if (option == "--trace-interval") {
            trace_interval = parse_u64(require_value(argc, argv, index, option), option);
        } else {
            throw std::runtime_error("Unknown sample option: " + option);
        }
    }
    if (model_path.empty()) throw std::runtime_error("sample requires --model");
    if (limit == 0 || batch_size == 0 || trace_interval == 0) {
        throw std::runtime_error("--limit, --batch-size and --trace-interval must be positive");
    }
    if (threads == 0 || threads > static_cast<std::uint64_t>(std::numeric_limits<int>::max())) {
        throw std::runtime_error("--threads must be between 1 and INT_MAX");
    }
    if (!trace_output.empty() && instrumentation != pcfg::InstrumentationMode::Detailed) {
        throw std::runtime_error("--trace-out requires --instrumentation detailed");
    }
    if (batch_size > static_cast<std::uint64_t>(std::numeric_limits<std::size_t>::max())) {
        throw std::runtime_error("--batch-size is too large");
    }

    const auto start_end_to_end = Clock::now();
    const auto load_start = Clock::now();
    pcfg::Model model = pcfg::ModelStore::load(model_path);
    const double load_seconds = seconds_since(load_start);

    pcfg::RunMetrics initial;
    initial.limit = limit;
    initial.batch_size = batch_size;
    initial.model_records = model.metadata.processed_records;
    initial.model_path = model_path.string();
    initial.load_model = load_seconds;
    initial.instrumentation = instrumentation;
    initial.threads_requested = threads;
    initial.threads_used = 1;
    pcfg::MetricsCollector collector(std::move(initial));
    pcfg::RunMetrics& metrics = collector.mutable_metrics();

    const auto core_start = Clock::now();
    const auto sampler_start = instrumentation == pcfg::InstrumentationMode::Detailed
                                   ? Clock::now()
                                   : Clock::time_point{};
    pcfg::SerialSampler sampler(model, instrumentation);
    if (instrumentation == pcfg::InstrumentationMode::Detailed) {
        metrics.sampler_init = seconds_since(sampler_start);
    }
    pcfg::CandidateConsumer consumer(
        pcfg::ConsumerOptions{signature, test_data, test_limit, instrumentation,
                              static_cast<std::size_t>(threads)}, metrics);

    std::ofstream trace;
    if (!trace_output.empty()) {
        if (!trace_output.parent_path().empty()) std::filesystem::create_directories(trace_output.parent_path());
        trace.open(trace_output, std::ios::trunc);
        if (!trace) throw std::runtime_error("Cannot create trace file: " + trace_output.string());
        write_trace_record(trace, sampler, metrics, core_start);
    }

    while (metrics.generated < limit && !sampler.empty()) {
        const std::uint64_t remaining = limit - metrics.generated;
        std::uint64_t request64 = std::min<std::uint64_t>(remaining, batch_size);
        if (trace.is_open()) {
            const std::uint64_t until_trace = trace_interval - (metrics.generated % trace_interval);
            request64 = std::min(request64, until_trace);
        }
        pcfg::BatchPlan plan = sampler.plan_batch(static_cast<std::size_t>(request64));
        if (plan.candidate_count == 0) break;
        consumer.consume(plan, model, metrics.generated);
        if (trace.is_open() && (metrics.generated % trace_interval == 0 || metrics.generated == limit)) {
            write_trace_record(trace, sampler, metrics, core_start);
        }
    }
    if (instrumentation == pcfg::InstrumentationMode::Detailed) copy_sampler_profile(sampler, metrics);
    metrics.core_total = seconds_since(core_start);
    metrics.end_to_end = seconds_since(start_end_to_end);
    if (metrics.generated != limit) {
        throw std::runtime_error("Sampler exhausted after " + std::to_string(metrics.generated) +
                                 " candidates; requested " + std::to_string(limit));
    }

    std::cout << collector.human_summary() << '\n' << collector.result_line() << '\n';
    if (!metrics_output.empty()) collector.write_json(metrics_output);
    return 0;
}

}  // namespace

int main(int argc, char** argv) {
    try {
        if (argc < 2) {
            print_usage(argv[0]);
            return 2;
        }
        const std::string command = argv[1];
        if (command == "train") return run_train(argc, argv);
        if (command == "sample") return run_sample(argc, argv);
        throw std::runtime_error("Unknown command: " + command);
    } catch (const std::exception& error) {
        std::cerr << "ERROR: " << error.what() << '\n';
        print_usage(argv[0]);
        return 1;
    }
}
