#include "pcfg/batch_plan_io.hpp"
#include "pcfg/consumer.hpp"
#include "pcfg/md5.hpp"
#include "pcfg/metrics.hpp"
#include "pcfg/model_io.hpp"
#include "pcfg/sampler.hpp"

#include <mpi.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

#include <sys/resource.h>

namespace {

using Clock = std::chrono::steady_clock;

enum class Control : int {
    Stop = 0,
    Work = 1,
    Error = 2,
};

struct SampleOptions {
    std::filesystem::path model;
    std::filesystem::path test_data;
    std::filesystem::path metrics_out;
    std::filesystem::path trace_out;
    std::uint64_t limit = 10000000;
    std::uint64_t batch_size = 1000000;
    std::uint64_t test_limit = 0;
    std::uint64_t threads = 1;
    bool signature = false;
    pcfg::InstrumentationMode instrumentation = pcfg::InstrumentationMode::Off;
};

double seconds_since(Clock::time_point start) {
    return std::chrono::duration<double>(Clock::now() - start).count();
}

void mpi_check(int code, const char* operation) {
    if (code == MPI_SUCCESS) return;
    std::array<char, MPI_MAX_ERROR_STRING> buffer{};
    int length = 0;
    MPI_Error_string(code, buffer.data(), &length);
    throw std::runtime_error(std::string(operation) + ": " +
                             std::string(buffer.data(), static_cast<std::size_t>(length)));
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
        << "  mpirun -np R " << program << " sample --model PATH [--limit N] [--batch-size N]" << '\n'
        << "      [--threads N] [--test-data PATH] [--test-limit N] [--signature]" << '\n'
        << "      [--metrics-out PATH] [--instrumentation off|detailed] [--trace-out PATH]" << '\n'
        << "  mpirun -np 2 " << program << " self-test" << '\n';
}

SampleOptions parse_sample_options(int argc, char** argv) {
    SampleOptions options;
    int index = 2;
    while (index < argc) {
        const std::string option = argv[index++];
        if (option == "--model") options.model = require_value(argc, argv, index, option);
        else if (option == "--limit") options.limit = parse_u64(require_value(argc, argv, index, option), option);
        else if (option == "--batch-size") options.batch_size = parse_u64(require_value(argc, argv, index, option), option);
        else if (option == "--test-data") options.test_data = require_value(argc, argv, index, option);
        else if (option == "--test-limit") options.test_limit = parse_u64(require_value(argc, argv, index, option), option);
        else if (option == "--threads") options.threads = parse_u64(require_value(argc, argv, index, option), option);
        else if (option == "--signature") options.signature = true;
        else if (option == "--metrics-out") options.metrics_out = require_value(argc, argv, index, option);
        else if (option == "--instrumentation") {
            options.instrumentation =
                pcfg::parse_instrumentation(require_value(argc, argv, index, option));
        } else if (option == "--trace-out") {
            options.trace_out = require_value(argc, argv, index, option);
        } else if (option == "--trace-interval") {
            static_cast<void>(require_value(argc, argv, index, option));
        } else {
            throw std::runtime_error("Unknown sample option: " + option);
        }
    }
    if (options.model.empty()) throw std::runtime_error("sample requires --model");
    if (options.limit == 0 || options.batch_size == 0) {
        throw std::runtime_error("--limit and --batch-size must be positive");
    }
    if (options.threads == 0 ||
        options.threads > static_cast<std::uint64_t>(std::numeric_limits<int>::max())) {
        throw std::runtime_error("--threads must be between 1 and INT_MAX");
    }
    if (options.batch_size > static_cast<std::uint64_t>(std::numeric_limits<int>::max())) {
        throw std::runtime_error("--batch-size must fit MPI int counts");
    }
    if (!options.trace_out.empty() &&
        options.instrumentation != pcfg::InstrumentationMode::Detailed) {
        throw std::runtime_error("--trace-out requires --instrumentation detailed");
    }
    return options;
}

std::array<std::uint8_t, 32> read_model_payload_checksum(
    const std::filesystem::path& path) {
    std::ifstream input(path, std::ios::binary | std::ios::ate);
    if (!input) throw std::runtime_error("Cannot open model identity: " + path.string());
    const std::streamoff size = input.tellg();
    if (size < 32) throw std::runtime_error("Model is too small for checksum identity");
    input.seekg(size - 32);
    std::array<std::uint8_t, 32> checksum{};
    input.read(reinterpret_cast<char*>(checksum.data()),
               static_cast<std::streamsize>(checksum.size()));
    if (!input) throw std::runtime_error("Cannot read model payload checksum");
    return checksum;
}

void verify_model_identity(const pcfg::Model& model,
                           const std::array<std::uint8_t, 32>& checksum,
                           int rank) {
    std::array<std::uint8_t, 32> expected_checksum = checksum;
    std::array<std::uint8_t, 32> expected_source = model.metadata.source_sha256;
    std::array<std::uint64_t, 4> identity = {
        model.metadata.processed_records,
        static_cast<std::uint64_t>(model.preterminals.size()),
        static_cast<std::uint64_t>(model.segments.size()),
        model.total_preterminal_count,
    };
    if (rank != 0) {
        expected_checksum.fill(0);
        expected_source.fill(0);
        identity.fill(0);
    }
    mpi_check(MPI_Bcast(expected_checksum.data(), static_cast<int>(expected_checksum.size()),
                        MPI_BYTE, 0, MPI_COMM_WORLD),
              "MPI_Bcast(model checksum)");
    mpi_check(MPI_Bcast(expected_source.data(), static_cast<int>(expected_source.size()),
                        MPI_BYTE, 0, MPI_COMM_WORLD),
              "MPI_Bcast(source checksum)");
    mpi_check(MPI_Bcast(identity.data(), static_cast<int>(identity.size()), MPI_UINT64_T,
                        0, MPI_COMM_WORLD),
              "MPI_Bcast(model identity)");
    const bool matches = expected_checksum == checksum &&
                         expected_source == model.metadata.source_sha256 &&
                         identity[0] == model.metadata.processed_records &&
                         identity[1] == model.preterminals.size() &&
                         identity[2] == model.segments.size() &&
                         identity[3] == model.total_preterminal_count;
    int local = matches ? 1 : 0;
    int all_match = 0;
    mpi_check(MPI_Allreduce(&local, &all_match, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD),
              "MPI_Allreduce(model identity)");
    if (all_match == 0) throw std::runtime_error("Model identity differs between MPI ranks");
}

std::vector<std::uint8_t> pack_strings(const std::vector<std::string>& values) {
    std::vector<std::uint8_t> output;
    for (const std::string& value : values) {
        if (value.size() > std::numeric_limits<std::uint32_t>::max()) {
            throw std::runtime_error("Matched candidate is too long to serialize");
        }
        const std::uint32_t size = static_cast<std::uint32_t>(value.size());
        for (unsigned shift = 0; shift < 32; shift += 8) {
            output.push_back(static_cast<std::uint8_t>(size >> shift));
        }
        output.insert(output.end(), value.begin(), value.end());
    }
    return output;
}

std::vector<std::string> unpack_strings(const std::vector<std::uint8_t>& bytes) {
    std::vector<std::string> output;
    std::size_t position = 0;
    while (position < bytes.size()) {
        if (bytes.size() - position < 4) throw std::runtime_error("Truncated matched-string payload");
        std::uint32_t size = 0;
        for (unsigned index = 0; index < 4; ++index) {
            size |= static_cast<std::uint32_t>(bytes[position++]) << (8 * index);
        }
        if (size > bytes.size() - position) throw std::runtime_error("Invalid matched-string length");
        output.emplace_back(reinterpret_cast<const char*>(bytes.data() + position), size);
        position += size;
    }
    return output;
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

void gather_validation(const pcfg::FusedSliceResult& local,
                       int rank,
                       int world_size,
                       pcfg::FusedSliceResult& global) {
    const std::array<std::uint64_t, 7> send = {
        local.sig1, local.sig2, local.candidate_hits,
        local.fused_group_count, local.scalar_tail_count,
        local.long_fallback_count, static_cast<std::uint64_t>(local.threads_used),
    };
    std::vector<std::uint64_t> received;
    if (rank == 0) received.resize(static_cast<std::size_t>(world_size) * send.size());
    mpi_check(MPI_Gather(send.data(), static_cast<int>(send.size()), MPI_UINT64_T,
                         received.data(), static_cast<int>(send.size()), MPI_UINT64_T,
                         0, MPI_COMM_WORLD),
              "MPI_Gather(validation)");
    if (rank != 0) return;
    global.threads_used = 1;
    for (int source = 0; source < world_size; ++source) {
        const std::size_t offset = static_cast<std::size_t>(source) * send.size();
        global.sig1 += received[offset];
        global.sig2 += received[offset + 1];
        global.candidate_hits += received[offset + 2];
        global.fused_group_count += received[offset + 3];
        global.scalar_tail_count += received[offset + 4];
        global.long_fallback_count += received[offset + 5];
        global.threads_used =
            std::max<std::size_t>(global.threads_used,
                                  static_cast<std::size_t>(received[offset + 6]));
    }
}

void gather_matches(const std::vector<std::string>& local_matches,
                    int rank,
                    int world_size,
                    pcfg::RunMetrics& metrics,
                    pcfg::FusedSliceResult& global) {
    const std::vector<std::uint8_t> packed = pack_strings(local_matches);
    const std::uint64_t local_size = packed.size();
    std::vector<std::uint64_t> sizes;
    if (rank == 0) sizes.resize(static_cast<std::size_t>(world_size));
    mpi_check(MPI_Gather(&local_size, 1, MPI_UINT64_T, sizes.data(), 1, MPI_UINT64_T,
                         0, MPI_COMM_WORLD),
              "MPI_Gather(match sizes)");

    std::vector<int> counts;
    std::vector<int> displacements;
    std::vector<std::uint8_t> gathered;
    if (rank == 0) {
        counts.resize(static_cast<std::size_t>(world_size));
        displacements.resize(static_cast<std::size_t>(world_size));
        std::uint64_t total = 0;
        for (int source = 0; source < world_size; ++source) {
            if (sizes[static_cast<std::size_t>(source)] >
                static_cast<std::uint64_t>(std::numeric_limits<int>::max())) {
                throw std::runtime_error("Matched-string payload exceeds MPI int count");
            }
            counts[static_cast<std::size_t>(source)] =
                static_cast<int>(sizes[static_cast<std::size_t>(source)]);
            displacements[static_cast<std::size_t>(source)] = static_cast<int>(total);
            total += sizes[static_cast<std::size_t>(source)];
            if (total > static_cast<std::uint64_t>(std::numeric_limits<int>::max())) {
                throw std::runtime_error("Gathered matched strings exceed MPI int count");
            }
        }
        gathered.resize(static_cast<std::size_t>(total));
        metrics.matched_gather_bytes += total - sizes.front();
    }
    mpi_check(MPI_Gatherv(packed.data(), static_cast<int>(packed.size()), MPI_BYTE,
                          gathered.data(), counts.data(), displacements.data(), MPI_BYTE,
                          0, MPI_COMM_WORLD),
              "MPI_Gatherv(matches)");
    if (rank == 0) global.matched = unpack_strings(gathered);
}

void write_batch_trace(std::ofstream& trace,
                       std::uint64_t batch,
                       const pcfg::RunMetrics& metrics,
                       std::size_t plan_bytes,
                       double compute_min,
                       double compute_mean,
                       double compute_max,
                       double sync,
                       double gather) {
    trace << std::fixed << std::setprecision(9)
          << "{"
          << "\"batch\":" << batch
          << ",\"generated\":" << metrics.generated
          << ",\"plan_bytes\":" << plan_bytes
          << ",\"compute_min\":" << compute_min
          << ",\"compute_mean\":" << compute_mean
          << ",\"compute_max\":" << compute_max
          << ",\"rank_sync\":" << sync
          << ",\"digest_gather\":" << gather
          << "}" << '\n';
    if (!trace) throw std::runtime_error("Failed while writing MPI batch trace");
}

int run_self_test(int rank, int world_size) {
    pcfg::BatchPlan plan;
    if (rank == 0) {
        plan.candidate_count = 13;
        plan.ranges = {{"prefix", 2, 5, 10, 0}, {"x", 3, 1, 9, 5}};
    }
    std::vector<std::uint8_t> payload = rank == 0 ? pcfg::encode_batch_plan(plan)
                                                  : std::vector<std::uint8_t>{};
    std::uint64_t size = payload.size();
    mpi_check(MPI_Bcast(&size, 1, MPI_UINT64_T, 0, MPI_COMM_WORLD),
              "MPI_Bcast(self-test size)");
    if (rank != 0) payload.resize(static_cast<std::size_t>(size));
    mpi_check(MPI_Bcast(payload.data(), static_cast<int>(payload.size()), MPI_BYTE,
                        0, MPI_COMM_WORLD),
              "MPI_Bcast(self-test payload)");
    plan = pcfg::decode_batch_plan(payload);
    const pcfg::CandidateSlice slice =
        pcfg::partition_candidate_slice(plan.candidate_count,
                                        static_cast<std::size_t>(rank),
                                        static_cast<std::size_t>(world_size),
                                        pcfg::kMd5BatchLanes);
    const std::uint64_t local = slice.end - slice.begin;
    std::uint64_t total = 0;
    mpi_check(MPI_Reduce(&local, &total, 1, MPI_UINT64_T, MPI_SUM, 0, MPI_COMM_WORLD),
              "MPI_Reduce(self-test coverage)");
    if (rank == 0) {
        if (total != plan.candidate_count) throw std::runtime_error("MPI slice coverage mismatch");
        std::cout << "MPI self-test passed with " << world_size << " ranks" << '\n';
    }
    return 0;
}

int run_sample(int argc, char** argv, int rank, int world_size) {
    const SampleOptions options = parse_sample_options(argc, argv);
    const auto end_to_end_start = Clock::now();
    const auto load_start = Clock::now();
    pcfg::Model model = pcfg::ModelStore::load(options.model);
    const auto model_checksum = read_model_payload_checksum(options.model);
    const double local_load = seconds_since(load_start);
    verify_model_identity(model, model_checksum, rank);
    double max_load = 0.0;
    mpi_check(MPI_Allreduce(&local_load, &max_load, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD),
              "MPI_Allreduce(model load)");

    pcfg::RunMetrics initial;
    initial.limit = options.limit;
    initial.batch_size = options.batch_size;
    initial.model_records = model.metadata.processed_records;
    initial.model_path = options.model.string();
    initial.load_model = max_load;
    initial.instrumentation = options.instrumentation;
    initial.threads_requested = options.threads;
    initial.threads_used = 1;
    initial.ranks_requested = static_cast<std::uint64_t>(world_size);
    initial.ranks_used = static_cast<std::uint64_t>(world_size);
    initial.mpi_pipeline = true;
    pcfg::MetricsCollector collector(std::move(initial));
    pcfg::RunMetrics& metrics = collector.mutable_metrics();

    mpi_check(MPI_Barrier(MPI_COMM_WORLD), "MPI_Barrier(core start)");
    const auto core_start = Clock::now();
    std::unique_ptr<pcfg::SerialSampler> sampler;
    if (rank == 0) {
        const auto sampler_start = options.instrumentation == pcfg::InstrumentationMode::Detailed
                                       ? Clock::now()
                                       : Clock::time_point{};
        sampler = std::make_unique<pcfg::SerialSampler>(model, options.instrumentation);
        if (options.instrumentation == pcfg::InstrumentationMode::Detailed) {
            metrics.sampler_init = seconds_since(sampler_start);
        }
    }
    pcfg::CandidateConsumer consumer(
        pcfg::ConsumerOptions{options.signature, options.test_data, options.test_limit,
                              options.instrumentation,
                              static_cast<std::size_t>(options.threads)},
        metrics);

    std::ofstream trace;
    if (rank == 0 && !options.trace_out.empty()) {
        if (!options.trace_out.parent_path().empty()) {
            std::filesystem::create_directories(options.trace_out.parent_path());
        }
        trace.open(options.trace_out, std::ios::trunc);
        if (!trace) throw std::runtime_error("Cannot create MPI trace file");
    }

    std::uint64_t root_generated = 0;
    std::uint64_t batch_index = 0;
    while (true) {
        Control control = Control::Stop;
        pcfg::BatchPlan plan;
        std::uint64_t global_begin = root_generated;
        if (rank == 0) {
            if (root_generated < options.limit && !sampler->empty()) {
                const std::uint64_t request =
                    std::min<std::uint64_t>(options.limit - root_generated, options.batch_size);
                plan = sampler->plan_batch(static_cast<std::size_t>(request));
                control = plan.candidate_count == 0 ? Control::Error : Control::Work;
            } else if (root_generated != options.limit) {
                control = Control::Error;
            }
        }

        int control_value = static_cast<int>(control);
        const auto broadcast_start =
            options.instrumentation == pcfg::InstrumentationMode::Detailed
                ? Clock::now() : Clock::time_point{};
        mpi_check(MPI_Bcast(&control_value, 1, MPI_INT, 0, MPI_COMM_WORLD),
                  "MPI_Bcast(control)");
        control = static_cast<Control>(control_value);
        if (control == Control::Stop) break;
        if (control == Control::Error) throw std::runtime_error("Rank 0 sampler exhausted early");

        std::vector<std::uint8_t> plan_payload =
            rank == 0 ? pcfg::encode_batch_plan(plan) : std::vector<std::uint8_t>{};
        std::uint64_t payload_size = plan_payload.size();
        mpi_check(MPI_Bcast(&global_begin, 1, MPI_UINT64_T, 0, MPI_COMM_WORLD),
                  "MPI_Bcast(global begin)");
        mpi_check(MPI_Bcast(&payload_size, 1, MPI_UINT64_T, 0, MPI_COMM_WORLD),
                  "MPI_Bcast(plan size)");
        if (payload_size > static_cast<std::uint64_t>(std::numeric_limits<int>::max())) {
            throw std::runtime_error("BatchPlan payload exceeds MPI int count");
        }
        if (rank != 0) plan_payload.resize(static_cast<std::size_t>(payload_size));
        mpi_check(MPI_Bcast(plan_payload.data(), static_cast<int>(plan_payload.size()), MPI_BYTE,
                            0, MPI_COMM_WORLD),
                  "MPI_Bcast(plan payload)");
        if (rank != 0) plan = pcfg::decode_batch_plan(plan_payload);
        const double broadcast_seconds =
            options.instrumentation == pcfg::InstrumentationMode::Detailed
                ? seconds_since(broadcast_start) : 0.0;

        const pcfg::CandidateSlice slice =
            pcfg::partition_candidate_slice(plan.candidate_count,
                                            static_cast<std::size_t>(rank),
                                            static_cast<std::size_t>(world_size),
                                            pcfg::kMd5BatchLanes);
        pcfg::FusedSliceResult local =
            consumer.compute_slice(plan, model, global_begin, slice.begin, slice.end);

        double sync_seconds = 0.0;
        std::vector<double> compute_times;
        if (options.instrumentation == pcfg::InstrumentationMode::Detailed) {
            const auto sync_start = Clock::now();
            mpi_check(MPI_Barrier(MPI_COMM_WORLD), "MPI_Barrier(detailed compute)");
            sync_seconds = seconds_since(sync_start);
            if (rank == 0) compute_times.resize(static_cast<std::size_t>(world_size));
            mpi_check(MPI_Gather(&local.worker_seconds, 1, MPI_DOUBLE,
                                 compute_times.data(), 1, MPI_DOUBLE,
                                 0, MPI_COMM_WORLD),
                      "MPI_Gather(compute times)");
        }

        std::vector<int> digest_counts;
        std::vector<int> digest_displacements;
        pcfg::FusedSliceResult global;
        if (rank == 0) {
            digest_counts.resize(static_cast<std::size_t>(world_size));
            digest_displacements.resize(static_cast<std::size_t>(world_size));
            for (int source = 0; source < world_size; ++source) {
                const pcfg::CandidateSlice source_slice =
                    pcfg::partition_candidate_slice(plan.candidate_count,
                                                    static_cast<std::size_t>(source),
                                                    static_cast<std::size_t>(world_size),
                                                    pcfg::kMd5BatchLanes);
                digest_counts[static_cast<std::size_t>(source)] =
                    static_cast<int>(source_slice.end - source_slice.begin);
                digest_displacements[static_cast<std::size_t>(source)] =
                    static_cast<int>(source_slice.begin);
            }
            global.slice_begin = 0;
            global.slice_end = plan.candidate_count;
            global.folded.resize(plan.candidate_count);
        }

        const auto gather_start =
            options.instrumentation == pcfg::InstrumentationMode::Detailed
                ? Clock::now() : Clock::time_point{};
        mpi_check(MPI_Gatherv(local.folded.data(), static_cast<int>(local.folded.size()),
                              MPI_UINT64_T, global.folded.data(), digest_counts.data(),
                              digest_displacements.data(), MPI_UINT64_T, 0, MPI_COMM_WORLD),
                  "MPI_Gatherv(digests)");
        const double gather_seconds =
            options.instrumentation == pcfg::InstrumentationMode::Detailed
                ? seconds_since(gather_start) : 0.0;

        gather_validation(local, rank, world_size, global);
        if (!options.test_data.empty()) {
            gather_matches(local.matched, rank, world_size, metrics, global);
        }
        if (rank == 0) {
            double compute_min = 0.0;
            double compute_mean = 0.0;
            double compute_max = 0.0;
            if (!compute_times.empty()) {
                const auto bounds = std::minmax_element(compute_times.begin(), compute_times.end());
                compute_min = *bounds.first;
                compute_max = *bounds.second;
                compute_mean = std::accumulate(compute_times.begin(), compute_times.end(), 0.0) /
                               static_cast<double>(compute_times.size());
                global.worker_seconds = compute_max;
            }
            consumer.merge_slice_result(std::move(global));
            root_generated = metrics.generated;
            ++batch_index;
            if (options.instrumentation == pcfg::InstrumentationMode::Detailed) {
                metrics.plan_broadcast += broadcast_seconds;
                metrics.rank_compute_min += compute_min;
                metrics.rank_compute_mean += compute_mean;
                metrics.rank_compute_max += compute_max;
                metrics.rank_sync += sync_seconds;
                metrics.digest_gather += gather_seconds;
                metrics.mpi_batch_count += 1;
                metrics.plan_broadcast_bytes +=
                    payload_size * static_cast<std::uint64_t>(world_size - 1);
                metrics.digest_gather_bytes +=
                    static_cast<std::uint64_t>(plan.candidate_count -
                                              static_cast<std::size_t>(digest_counts.front())) *
                    sizeof(std::uint64_t);
                if (trace.is_open()) {
                    write_batch_trace(trace, batch_index, metrics, plan_payload.size(),
                                      compute_min, compute_mean, compute_max,
                                      sync_seconds, gather_seconds);
                }
            }
        }
    }

    if (rank == 0 && options.instrumentation == pcfg::InstrumentationMode::Detailed) {
        copy_sampler_profile(*sampler, metrics);
    }
    mpi_check(MPI_Barrier(MPI_COMM_WORLD), "MPI_Barrier(core end)");
    const double core_seconds = seconds_since(core_start);
    struct rusage usage {};
    if (getrusage(RUSAGE_SELF, &usage) != 0) {
        throw std::runtime_error("getrusage failed");
    }
    const std::uint64_t local_rss =
        static_cast<std::uint64_t>(usage.ru_maxrss) * 1024u;
    std::vector<std::uint64_t> rank_rss;
    if (rank == 0) rank_rss.resize(static_cast<std::size_t>(world_size));
    mpi_check(MPI_Gather(&local_rss, 1, MPI_UINT64_T, rank_rss.data(), 1,
                         MPI_UINT64_T, 0, MPI_COMM_WORLD),
              "MPI_Gather(rank RSS)");
    if (rank == 0) {
        const auto rss_bounds = std::minmax_element(rank_rss.begin(), rank_rss.end());
        metrics.rank_rss_min_bytes = *rss_bounds.first;
        metrics.rank_rss_max_bytes = *rss_bounds.second;
        metrics.rank_rss_mean_bytes =
            std::accumulate(rank_rss.begin(), rank_rss.end(), std::uint64_t{0}) /
            static_cast<std::uint64_t>(rank_rss.size());
        metrics.core_total = core_seconds;
        metrics.end_to_end = seconds_since(end_to_end_start);
        if (metrics.generated != options.limit) {
            throw std::runtime_error("MPI sampler generated an unexpected candidate count");
        }
        std::cout << collector.human_summary() << '\n' << collector.result_line() << '\n';
        if (!options.metrics_out.empty()) collector.write_json(options.metrics_out);
    }
    return 0;
}

}  // namespace

int main(int argc, char** argv) {
    int provided = MPI_THREAD_SINGLE;
    int rank = 0;
    bool initialized = false;
    try {
        mpi_check(MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided),
                  "MPI_Init_thread");
        initialized = true;
        mpi_check(MPI_Comm_rank(MPI_COMM_WORLD, &rank), "MPI_Comm_rank");
        int world_size = 1;
        mpi_check(MPI_Comm_size(MPI_COMM_WORLD, &world_size), "MPI_Comm_size");
        mpi_check(MPI_Comm_set_errhandler(MPI_COMM_WORLD, MPI_ERRORS_RETURN),
                  "MPI_Comm_set_errhandler");
        if (provided < MPI_THREAD_FUNNELED) {
            throw std::runtime_error("MPI implementation does not provide MPI_THREAD_FUNNELED");
        }
        if (argc < 2) {
            if (rank == 0) print_usage(argv[0]);
            MPI_Finalize();
            return 2;
        }
        const std::string command = argv[1];
        int result = 0;
        if (command == "self-test") result = run_self_test(rank, world_size);
        else if (command == "sample") result = run_sample(argc, argv, rank, world_size);
        else throw std::runtime_error("Unknown command: " + command);
        mpi_check(MPI_Finalize(), "MPI_Finalize");
        return result;
    } catch (const std::exception& error) {
        std::cerr << "MPI ERROR rank " << rank << ": " << error.what() << '\n';
        if (initialized) {
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        return 1;
    }
}
