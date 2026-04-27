#pragma once

#include "bc_specs.h"

#include <string>
#include <vector>

namespace pipebc {

struct RuntimeBCConfig {
  std::vector<VelocityPatchBCSpec> velocityPatchSpecs;
  std::vector<PressurePatchBCSpec> pressurePatchSpecs;
};

RuntimeBCConfig load_runtime_bc_config(const std::string& path);

void validate_runtime_bc_config_against_patches(
    const RuntimeBCConfig& cfg,
    const std::vector<std::string>& patchNames);

} // namespace pipebc
