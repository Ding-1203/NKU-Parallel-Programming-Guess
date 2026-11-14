#pragma once

#include "pcfg/model.hpp"

#include <filesystem>

namespace pcfg {

class ModelStore {
public:
    static constexpr std::uint32_t kSchemaVersion = 1;

    static void save(const Model& model, const std::filesystem::path& path);
    static Model load(const std::filesystem::path& path);
};

}  // namespace pcfg
