#include "pcfg/batch_plan_io.hpp"
#include "pcfg/consumer.hpp"
#include "pcfg/md5.hpp"
#include "pcfg/model.hpp"
#include "pcfg/model_io.hpp"
#include "pcfg/sampler.hpp"
#include "pcfg/sha256.hpp"

#include <chrono>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <random>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace {

int failures = 0;

#define CHECK(condition) do { \
    if (!(condition)) { \
        std::cerr << "CHECK failed at " << __FILE__ << ':' << __LINE__ << ": " #condition "\n"; \
        ++failures; \
    } \
} while (false)

template <typename Callable>
void check_throws(Callable&& callable) {
    try {
        callable();
        CHECK(false);
    } catch (const std::exception&) {
    }
}

void write_binary(const std::filesystem::path& path, const std::string& data) {
    std::ofstream output(path, std::ios::binary | std::ios::trunc);
    if (!output) throw std::runtime_error("test file create failed");
    output.write(data.data(), static_cast<std::streamsize>(data.size()));
}

std::vector<std::uint8_t> read_binary(const std::filesystem::path& path) {
    std::ifstream input(path, std::ios::binary | std::ios::ate);
    const auto size = input.tellg();
    input.seekg(0);
    std::vector<std::uint8_t> bytes(static_cast<std::size_t>(size));
    input.read(reinterpret_cast<char*>(bytes.data()), size);
    return bytes;
}

std::vector<std::string> sample_all(const pcfg::Model& model, std::size_t count, std::size_t batch_size) {
    pcfg::SerialSampler sampler(model);
    std::vector<std::string> output;
    while (output.size() < count && !sampler.empty()) {
        pcfg::SampleBatch batch = sampler.sample_batch(std::min(batch_size, count - output.size()));
        output.insert(output.end(), batch.candidates.begin(), batch.candidates.end());
    }
    return output;
}

std::vector<std::string> materialize_plan(const pcfg::BatchPlan& plan,
                                          const pcfg::Model& model) {
    std::vector<std::string> output(plan.candidate_count);
    for (const pcfg::CandidateRange& range : plan.ranges) {
        const pcfg::SegmentTable& suffix = model.segments.at(range.suffix_segment_index);
        for (std::size_t i = range.suffix_begin; i < range.suffix_end; ++i) {
            output[range.batch_begin + i - range.suffix_begin] =
                range.prefix + suffix.values[i].value;
        }
    }
    return output;
}

std::uint64_t scalar_hash_guard(const std::vector<std::string>& candidates) {
    std::uint64_t guard = 0;
    for (const std::string& candidate : candidates) {
        const pcfg::Md5Digest digest = pcfg::md5(candidate);
        std::uint64_t folded = 0;
        for (std::size_t i = 0; i < 8; ++i) folded |= static_cast<std::uint64_t>(digest[i]) << (8 * i);
        guard ^= folded + 0x9e3779b97f4a7c15ull + (guard << 6u) + (guard >> 2u);
    }
    return guard;
}

void test_batch_plan_codec_and_partition() {
    pcfg::BatchPlan plan;
    plan.candidate_count = 17;
    plan.ranges = {
        {"prefix-a", 3, 2, 7, 0},
        {"", 5, 10, 22, 5},
    };
    const std::vector<std::uint8_t> first = pcfg::encode_batch_plan(plan);
    const std::vector<std::uint8_t> second = pcfg::encode_batch_plan(plan);
    CHECK(first == second);

    const pcfg::BatchPlan decoded = pcfg::decode_batch_plan(first);
    CHECK(decoded.candidate_count == plan.candidate_count);
    CHECK(decoded.ranges.size() == plan.ranges.size());
    for (std::size_t i = 0; i < plan.ranges.size(); ++i) {
        CHECK(decoded.ranges[i].prefix == plan.ranges[i].prefix);
        CHECK(decoded.ranges[i].suffix_segment_index == plan.ranges[i].suffix_segment_index);
        CHECK(decoded.ranges[i].suffix_begin == plan.ranges[i].suffix_begin);
        CHECK(decoded.ranges[i].suffix_end == plan.ranges[i].suffix_end);
        CHECK(decoded.ranges[i].batch_begin == plan.ranges[i].batch_begin);
    }

    std::vector<std::uint8_t> truncated(first.begin(), first.end() - 1);
    check_throws([&] { static_cast<void>(pcfg::decode_batch_plan(truncated)); });
    std::vector<std::uint8_t> corrupt = first;
    corrupt[0] ^= 1;
    check_throws([&] { static_cast<void>(pcfg::decode_batch_plan(corrupt)); });
    std::vector<std::uint8_t> trailing = first;
    trailing.push_back(0);
    check_throws([&] { static_cast<void>(pcfg::decode_batch_plan(trailing)); });

    for (const std::size_t ranks :
         {std::size_t{1}, std::size_t{2}, std::size_t{4}, std::size_t{8}}) {
        std::size_t expected_begin = 0;
        for (std::size_t rank = 0; rank < ranks; ++rank) {
            const pcfg::CandidateSlice slice =
                pcfg::partition_candidate_slice(plan.candidate_count, rank, ranks,
                                                pcfg::kMd5BatchLanes);
            CHECK(slice.begin == expected_begin);
            CHECK(slice.begin <= slice.end);
            if (rank + 1 != ranks) CHECK(slice.end % pcfg::kMd5BatchLanes == 0);
            expected_begin = slice.end;
        }
        CHECK(expected_begin == plan.candidate_count);
    }
    check_throws([&] {
        static_cast<void>(pcfg::partition_candidate_slice(10, 2, 2));
    });
}

void test_hashes() {
    CHECK(pcfg::md5_hex("") == "d41d8cd98f00b204e9800998ecf8427e");
    CHECK(pcfg::md5_hex("a") == "0cc175b9c0f1b6a831c399e269772661");
    CHECK(pcfg::md5_hex("abc") == "900150983cd24fb0d6963f7d28e17f72");
    CHECK(pcfg::md5_hex("message digest") == "f96b697d7cb7938d525a2f31aaf161d0");
    CHECK(pcfg::md5_hex(std::string(55, 'a')) == "ef1772b6dff9a122358552954ad0df65");
    CHECK(pcfg::md5_hex(std::string(56, 'a')) == "3b0c8ac703f828b04c6c197006d17218");
    CHECK(pcfg::md5_hex(std::string(63, 'a')) == "b06521f39153d618550606be297466d5");
    CHECK(pcfg::md5_hex(std::string(64, 'a')) == "014842d480b571495a4a0363793f7367");
    CHECK(pcfg::md5_hex(std::string(80, 'a')) == "b15af9cdabbaea0516866a33d8fd0f98");
    CHECK(pcfg::md5_hex(std::string(119, 'a')) == "8a7bd0732ed6a28ce75f6dabc90e1613");
    CHECK(pcfg::md5_hex(std::string(120, 'a')) == "5f61c0ccad4cac44c75ff505e1f1e537");

    const std::vector<std::uint8_t> abc = {'a', 'b', 'c'};
    CHECK(pcfg::hex_encode(pcfg::Sha256::digest(abc)) ==
          "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad");
}

void test_md5_batch8() {
    std::mt19937 random(4526);
    std::uniform_int_distribution<int> length_distribution(0, 55);
    std::uniform_int_distribution<int> byte_distribution(32, 126);
    for (int iteration = 0; iteration < 32; ++iteration) {
        std::array<std::string, pcfg::kMd5BatchLanes> storage;
        std::array<std::string_view, pcfg::kMd5BatchLanes> inputs{};
        std::array<pcfg::Md5Digest, pcfg::kMd5BatchLanes> outputs{};
        for (std::size_t lane = 0; lane < pcfg::kMd5BatchLanes; ++lane) {
            storage[lane].resize(static_cast<std::size_t>(length_distribution(random)));
            for (char& byte : storage[lane]) byte = static_cast<char>(byte_distribution(random));
            inputs[lane] = storage[lane];
        }
        pcfg::md5_batch8(inputs, outputs);
        for (std::size_t lane = 0; lane < pcfg::kMd5BatchLanes; ++lane) {
            CHECK(outputs[lane] == pcfg::md5(inputs[lane]));
        }
    }

    std::array<std::string, pcfg::kMd5BatchLanes> mixed = {
        "", "short", std::string(55, 'a'), std::string(56, 'b'),
        "lane4", std::string(1, 'x'), std::string(120, 'c'), "lane7",
    };
    std::array<std::string_view, pcfg::kMd5BatchLanes> mixed_inputs{};
    std::array<pcfg::Md5Digest, pcfg::kMd5BatchLanes> mixed_outputs{};
    for (std::size_t lane = 0; lane < pcfg::kMd5BatchLanes; ++lane) {
        mixed_inputs[lane] = mixed[lane];
    }
    pcfg::md5_batch8(mixed_inputs, mixed_outputs);
    for (std::size_t lane = 0; lane < pcfg::kMd5BatchLanes; ++lane) {
        CHECK(mixed_outputs[lane] == pcfg::md5(mixed_inputs[lane]));
    }
}

void test_training_model_and_sampling(const std::filesystem::path& directory) {
    const auto input = directory / "train.txt";
    write_binary(input, "aa1\r\nab1\r\naa2\r\n\r\n");

    pcfg::Model model = pcfg::ModelTrainer().train({input, 0});
    CHECK(model.metadata.source_lines == 4);
    CHECK(model.metadata.processed_records == 3);
    CHECK(model.metadata.skipped_empty == 1);
    CHECK(model.preterminals.size() == 1);
    CHECK(pcfg::preterminal_string(model.preterminals.front()) == "L2D1");
    CHECK(model.segment({pcfg::SegmentType::Letter, 2}).values.front().value == "aa");
    CHECK(model.segment({pcfg::SegmentType::Digit, 1}).values.front().value == "1");

    const auto model_a = directory / "model-a.pcfg";
    const auto model_b = directory / "model-b.pcfg";
    pcfg::ModelStore::save(model, model_a);
    pcfg::ModelStore::save(model, model_b);
    CHECK(read_binary(model_a) == read_binary(model_b));

    pcfg::Model loaded = pcfg::ModelStore::load(model_a);
    CHECK(loaded.metadata.processed_records == 3);
    CHECK(loaded.preterminals.size() == model.preterminals.size());

    const std::vector<std::string> expected = {"aa1", "aa2", "ab1", "ab2"};
    CHECK(sample_all(loaded, 4, 1) == expected);
    CHECK(sample_all(loaded, 4, 3) == expected);

    pcfg::SerialSampler planner(loaded, pcfg::InstrumentationMode::Detailed);
    pcfg::BatchPlan first_plan = planner.plan_batch(3);
    pcfg::BatchPlan second_plan = planner.plan_batch(1);
    std::vector<std::string> planned = materialize_plan(first_plan, loaded);
    const std::vector<std::string> planned_tail = materialize_plan(second_plan, loaded);
    planned.insert(planned.end(), planned_tail.begin(), planned_tail.end());
    CHECK(planned == expected);
    CHECK(first_plan.candidate_count == 3);
    CHECK(second_plan.candidate_count == 1);
    CHECK(planner.profile().counters.range_count >= 2);
    CHECK(planner.profile().batch_plan >= 0.0);

    pcfg::SerialSampler detailed(loaded, pcfg::InstrumentationMode::Detailed);
    pcfg::SampleBatch detailed_batch = detailed.sample_batch(4);
    CHECK(detailed_batch.candidates == expected);
    CHECK(detailed.profile().counters.initial_queue_size == 1);
    CHECK(detailed.profile().counters.max_queue_size == 1);
    CHECK(detailed.profile().counters.pop_count == 2);
    CHECK(detailed.profile().counters.insert_count == 2);
    CHECK(detailed.profile().counters.candidate_bytes == 12);
    CHECK(detailed.profile().generate >= 0.0);
    CHECK(detailed.queue_size() == 0);

    std::vector<std::uint8_t> corrupt = read_binary(model_a);
    corrupt.back() ^= 0x55u;
    const auto bad_model = directory / "bad.pcfg";
    std::ofstream bad(bad_model, std::ios::binary | std::ios::trunc);
    bad.write(reinterpret_cast<const char*>(corrupt.data()), static_cast<std::streamsize>(corrupt.size()));
    bad.close();
    check_throws([&] { static_cast<void>(pcfg::ModelStore::load(bad_model)); });
}

void test_heap_order_and_ties(const std::filesystem::path& directory) {
    const auto input = directory / "heap-order.txt";
    write_binary(input, "a1\nb1\na!\nb!\n");
    pcfg::Model model = pcfg::ModelTrainer().train({input, 0});

    const std::vector<std::string> expected = {"a1", "b1", "a!", "b!"};
    CHECK(sample_all(model, 4, 1) == expected);
    CHECK(sample_all(model, 4, 3) == expected);

    pcfg::SerialSampler detailed(model, pcfg::InstrumentationMode::Detailed);
    CHECK(detailed.sample_batch(4).candidates == expected);
    const pcfg::SamplerCounters& counters = detailed.profile().counters;
    CHECK(counters.initial_queue_size == 2);
    CHECK(counters.max_queue_size == 2);
    CHECK(counters.pop_count == 4);
    CHECK(counters.insert_count == 4);
    CHECK(counters.shifted_on_pop == 0);
    CHECK(counters.shifted_on_insert == 0);
    CHECK(detailed.queue_size() == 0);
}

void test_line_rules_and_limit(const std::filesystem::path& directory) {
    const auto input = directory / "line-rules.txt";
    write_binary(input, "ab 1\r\n\r\nXY9\r\nlast!\r\n");
    pcfg::Model model = pcfg::ModelTrainer().train({input, 2});
    CHECK(model.metadata.source_lines == 4);
    CHECK(model.metadata.processed_records == 2);
    CHECK(model.metadata.skipped_empty == 1);
    CHECK(model.metadata.train_limit == 2);

    bool saw_space = false;
    for (const auto& table : model.segments) {
        for (const auto& value : table.values) {
            if (value.value == " ") saw_space = true;
        }
    }
    CHECK(saw_space);
}

void test_consumer(const std::filesystem::path& directory) {
    const auto targets = directory / "targets.txt";
    write_binary(targets, "aa1\r\nab2\r\n");
    pcfg::RunMetrics metrics;
    pcfg::CandidateConsumer consumer({true, targets, 0, pcfg::InstrumentationMode::Detailed}, metrics);
    consumer.consume({"aa1", "aa2", "ab1", "ab2", "aa1"}, 0);
    CHECK(metrics.generated == 5);
    CHECK(metrics.candidate_hits == 3);
    CHECK(metrics.cracked_unique == 2);
    CHECK(metrics.sig1 != 0);
    CHECK(metrics.sig2 != 0);
    CHECK(metrics.hash_guard != 0);
    CHECK(metrics.hash > 0.0);
    CHECK(metrics.verify > 0.0);

    pcfg::RunMetrics off_metrics;
    pcfg::CandidateConsumer off_consumer({false, {}, 0, pcfg::InstrumentationMode::Off}, off_metrics);
    off_consumer.consume({"one", "two"}, 0);
    CHECK(off_metrics.generated == 2);
    CHECK(off_metrics.hash_guard != 0);
    CHECK(off_metrics.hash == 0.0);
    CHECK(off_metrics.verify == 0.0);

    std::vector<std::string> tail_candidates;
    for (std::size_t i = 0; i < 17; ++i) {
        tail_candidates.push_back("candidate-" + std::to_string(i));
    }
    for (std::size_t count = 1; count <= tail_candidates.size(); ++count) {
        const std::vector<std::string> prefix(tail_candidates.begin(), tail_candidates.begin() + count);
        pcfg::RunMetrics batch_metrics;
        pcfg::CandidateConsumer batch_consumer(
            {false, {}, 0, pcfg::InstrumentationMode::Off}, batch_metrics);
        batch_consumer.consume(prefix, 0);
        CHECK(batch_metrics.generated == count);
        CHECK(batch_metrics.hash_guard == scalar_hash_guard(prefix));
    }

    std::vector<std::string> parallel_candidates;
    parallel_candidates.reserve(16393);
    for (std::size_t i = 0; i < 16393; ++i) {
        parallel_candidates.push_back("parallel-candidate-" + std::to_string(i));
    }
    const std::uint64_t expected_guard = scalar_hash_guard(parallel_candidates);
    for (const std::size_t threads : {std::size_t{1}, std::size_t{2}, std::size_t{4}, std::size_t{8}}) {
        pcfg::RunMetrics parallel_metrics;
        pcfg::CandidateConsumer parallel_consumer(
            {false, {}, 0, pcfg::InstrumentationMode::Off, threads}, parallel_metrics);
        parallel_consumer.consume(parallel_candidates, 0);
        CHECK(parallel_metrics.generated == parallel_candidates.size());
        CHECK(parallel_metrics.hash_guard == expected_guard);
        CHECK(parallel_metrics.threads_requested == threads);
        CHECK(parallel_metrics.threads_used == threads);
    }

    const std::vector<std::string> below_threshold(
        parallel_candidates.begin(), parallel_candidates.begin() + 8191);
    pcfg::RunMetrics below_metrics;
    pcfg::CandidateConsumer below_consumer(
        {false, {}, 0, pcfg::InstrumentationMode::Off, 4}, below_metrics);
    below_consumer.consume(below_threshold, 0);
    CHECK(below_metrics.hash_guard == scalar_hash_guard(below_threshold));
    CHECK(below_metrics.threads_requested == 4);
    CHECK(below_metrics.threads_used == 1);

    const std::vector<std::string> first_batch(
        parallel_candidates.begin(), parallel_candidates.begin() + 5000);
    const std::vector<std::string> second_batch(
        parallel_candidates.begin() + 5000, parallel_candidates.end());
    pcfg::RunMetrics reused_metrics;
    pcfg::CandidateConsumer reused_consumer(
        {false, {}, 0, pcfg::InstrumentationMode::Off, 2}, reused_metrics);
    reused_consumer.consume(first_batch, 0);
    reused_consumer.consume(second_batch, first_batch.size());
    CHECK(reused_metrics.generated == parallel_candidates.size());
    CHECK(reused_metrics.hash_guard == expected_guard);
    CHECK(reused_metrics.threads_used == 2);

    pcfg::RunMetrics invalid_metrics;
    check_throws([&] {
        pcfg::CandidateConsumer invalid(
            {false, {}, 0, pcfg::InstrumentationMode::Off, 0}, invalid_metrics);
    });
}

void test_fused_consumer(const std::filesystem::path& directory) {
    pcfg::Model model;
    pcfg::SegmentTable table;
    table.key = {pcfg::SegmentType::Symbol, 1};
    std::vector<std::string> candidates;
    candidates.reserve(16393);
    for (std::size_t i = 0; i < 16393; ++i) {
        candidates.push_back("fused-" + std::to_string(i));
    }
    candidates[10] = std::string(55, 'a');
    candidates[11] = std::string(56, 'b');
    for (const std::string& candidate : candidates) {
        table.values.push_back({candidate, 1});
        ++table.total_count;
    }
    model.segments.push_back(std::move(table));

    pcfg::BatchPlan plan;
    plan.candidate_count = candidates.size();
    plan.ranges = {
        {"", 0, 0, 5, 0},
        {"", 0, 5, 8003, 5},
        {"", 0, 8003, candidates.size(), 8003},
    };
    const auto targets = directory / "fused-targets.txt";
    write_binary(targets, candidates[0] + "\n" + candidates[11] + "\n");

    pcfg::RunMetrics oracle_metrics;
    pcfg::CandidateConsumer oracle(
        {true, targets, 0, pcfg::InstrumentationMode::Detailed, 1}, oracle_metrics);
    oracle.consume(candidates, 7);

    for (const std::size_t threads :
         {std::size_t{1}, std::size_t{2}, std::size_t{4}, std::size_t{8}}) {
        pcfg::RunMetrics fused_metrics;
        pcfg::CandidateConsumer fused(
            {true, targets, 0, pcfg::InstrumentationMode::Detailed, threads}, fused_metrics);
        fused.consume(plan, model, 7);
        CHECK(fused_metrics.fused_pipeline);
        CHECK(fused_metrics.generated == oracle_metrics.generated);
        CHECK(fused_metrics.hash_guard == oracle_metrics.hash_guard);
        CHECK(fused_metrics.sig1 == oracle_metrics.sig1);
        CHECK(fused_metrics.sig2 == oracle_metrics.sig2);
        CHECK(fused_metrics.candidate_hits == oracle_metrics.candidate_hits);
        CHECK(fused_metrics.cracked_unique == oracle_metrics.cracked_unique);
        CHECK(fused_metrics.threads_used == threads);
        CHECK(fused_metrics.sampler_counters.fused_group_count ==
              candidates.size() / pcfg::kMd5BatchLanes);
        CHECK(fused_metrics.sampler_counters.scalar_tail_count ==
              candidates.size() % pcfg::kMd5BatchLanes);
        CHECK(fused_metrics.sampler_counters.long_fallback_count == 1);
        CHECK(fused_metrics.sampler_counters.peak_live_candidate_slots ==
              threads * pcfg::kMd5BatchLanes);
    }

    pcfg::BatchPlan below;
    below.candidate_count = 8191;
    below.ranges = {{"", 0, 0, below.candidate_count, 0}};
    pcfg::RunMetrics below_metrics;
    pcfg::CandidateConsumer below_consumer(
        {false, {}, 0, pcfg::InstrumentationMode::Detailed, 4}, below_metrics);
    below_consumer.consume(below, model, 0);
    const std::vector<std::string> below_candidates(
        candidates.begin(), candidates.begin() + below.candidate_count);
    CHECK(below_metrics.hash_guard == scalar_hash_guard(below_candidates));
    CHECK(below_metrics.threads_used == 1);
    CHECK(below_metrics.sampler_counters.peak_live_candidate_slots ==
          pcfg::kMd5BatchLanes);

    for (const std::size_t ranks :
         {std::size_t{1}, std::size_t{2}, std::size_t{4}, std::size_t{8}}) {
        pcfg::RunMetrics sliced_metrics;
        pcfg::CandidateConsumer sliced(
            {true, targets, 0, pcfg::InstrumentationMode::Detailed, 1}, sliced_metrics);
        pcfg::FusedSliceResult global;
        global.slice_begin = 0;
        global.slice_end = plan.candidate_count;
        global.threads_used = 1;
        for (std::size_t rank = 0; rank < ranks; ++rank) {
            const pcfg::CandidateSlice slice =
                pcfg::partition_candidate_slice(plan.candidate_count, rank, ranks,
                                                pcfg::kMd5BatchLanes);
            pcfg::FusedSliceResult local =
                sliced.compute_slice(plan, model, 7, slice.begin, slice.end);
            global.folded.insert(global.folded.end(), local.folded.begin(), local.folded.end());
            global.sig1 += local.sig1;
            global.sig2 += local.sig2;
            global.candidate_hits += local.candidate_hits;
            global.fused_group_count += local.fused_group_count;
            global.scalar_tail_count += local.scalar_tail_count;
            global.long_fallback_count += local.long_fallback_count;
            global.worker_seconds = std::max(global.worker_seconds, local.worker_seconds);
            global.matched.insert(global.matched.end(),
                                  local.matched.begin(), local.matched.end());
        }
        sliced.merge_slice_result(std::move(global));
        CHECK(sliced_metrics.generated == oracle_metrics.generated);
        CHECK(sliced_metrics.hash_guard == oracle_metrics.hash_guard);
        CHECK(sliced_metrics.sig1 == oracle_metrics.sig1);
        CHECK(sliced_metrics.sig2 == oracle_metrics.sig2);
        CHECK(sliced_metrics.candidate_hits == oracle_metrics.candidate_hits);
        CHECK(sliced_metrics.cracked_unique == oracle_metrics.cracked_unique);
    }

    pcfg::BatchPlan invalid = plan;
    invalid.ranges[1].batch_begin = 6;
    pcfg::RunMetrics invalid_metrics;
    pcfg::CandidateConsumer invalid_consumer(
        {false, {}, 0, pcfg::InstrumentationMode::Off, 1}, invalid_metrics);
    check_throws([&] { invalid_consumer.consume(invalid, model, 0); });
}

void test_metrics_schema() {
    CHECK(pcfg::instrumentation_name(pcfg::InstrumentationMode::Off) == "off");
    CHECK(pcfg::parse_instrumentation("detailed") == pcfg::InstrumentationMode::Detailed);
    check_throws([] { static_cast<void>(pcfg::parse_instrumentation("invalid")); });

    pcfg::RunMetrics off;
    off.instrumentation = pcfg::InstrumentationMode::Off;
    const std::string off_json = pcfg::MetricsCollector(off).json();
    CHECK(off_json.find("pcfg-baseline-result-v2") != std::string::npos);
    CHECK(off_json.find("\"threads_requested\": 1") != std::string::npos);
    CHECK(off_json.find("\"threads_used\": 1") != std::string::npos);
    CHECK(off_json.find("\"detailed\": null") != std::string::npos);
    CHECK(off_json.find("\"sampler_counters\": null") != std::string::npos);

    pcfg::RunMetrics detailed;
    detailed.instrumentation = pcfg::InstrumentationMode::Detailed;
    const std::string detailed_json = pcfg::MetricsCollector(detailed).json();
    CHECK(detailed_json.find("\"sampler_init\":") != std::string::npos);
    CHECK(detailed_json.find("\"initial_queue_size\":") != std::string::npos);

    detailed.mpi_pipeline = true;
    detailed.ranks_requested = 4;
    detailed.ranks_used = 4;
    detailed.rank_rss_max_bytes = 123456;
    const std::string mpi_json = pcfg::MetricsCollector(detailed).json();
    CHECK(mpi_json.find("\"mode\": \"mpi-fused\"") != std::string::npos);
    CHECK(mpi_json.find("\"ranks_used\": 4") != std::string::npos);
    CHECK(mpi_json.find("\"rank_rss_max_bytes\": 123456") != std::string::npos);
}

}  // namespace

int main() {
    const auto unique = std::to_string(std::chrono::steady_clock::now().time_since_epoch().count());
    const auto directory = std::filesystem::temp_directory_path() / ("pcfg-tests-" + unique);
    std::filesystem::create_directories(directory);
    try {
        test_batch_plan_codec_and_partition();
        test_hashes();
        test_md5_batch8();
        test_training_model_and_sampling(directory);
        test_heap_order_and_ties(directory);
        test_line_rules_and_limit(directory);
        test_consumer(directory);
        test_fused_consumer(directory);
        test_metrics_schema();
    } catch (const std::exception& error) {
        std::cerr << "Unexpected exception: " << error.what() << '\n';
        ++failures;
    }
    std::filesystem::remove_all(directory);
    if (failures != 0) {
        std::cerr << failures << " test(s) failed\n";
        return 1;
    }
    std::cout << "All PCFG baseline tests passed\n";
    return 0;
}
