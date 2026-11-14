#pragma once

#include "pcfg/sampler.hpp"

#include <cstddef>
#include <cstdint>
#include <vector>

namespace pcfg {

struct CandidateSlice {
    std::size_t begin = 0;
    std::size_t end = 0;
};

std::vector<std::uint8_t> encode_batch_plan(const BatchPlan& plan);
BatchPlan decode_batch_plan(const std::vector<std::uint8_t>& bytes);
CandidateSlice partition_candidate_slice(std::size_t candidate_count,
                                         std::size_t rank,
                                         std::size_t world_size,
                                         std::size_t lanes = 8);

}  // namespace pcfg
