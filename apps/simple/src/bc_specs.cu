#include "bc_specs.h"

#include <unordered_map>

namespace pipebc {

VelocityPatchBCSpec make_wall_noslip_bc(const std::string& patchName) {
  VelocityPatchBCSpec out;
  out.patchName = patchName;
  out.type = VelocityBCType::WallNoSlip;
  return out;
}

VelocityPatchBCSpec make_fixed_uniform_vector_bc(
    const std::string& patchName,
    const std::array<double,3>& value) {
  VelocityPatchBCSpec out;
  out.patchName = patchName;
  out.type = VelocityBCType::FixedUniformVector;
  out.uniformVector = value;
  return out;
}

VelocityPatchBCSpec make_fixed_vector_function_bc(
    const std::string& patchName,
    VectorFunction fn) {
  VelocityPatchBCSpec out;
  out.patchName = patchName;
  out.type = VelocityBCType::FixedVectorFunction;
  out.vectorFunction = std::move(fn);
  return out;
}

VelocityPatchBCSpec make_zero_gradient_velocity_bc(const std::string& patchName) {
  VelocityPatchBCSpec out;
  out.patchName = patchName;
  out.type = VelocityBCType::ZeroGradient;
  return out;
}

VelocityPatchBCSpec make_fixed_normal_speed_bc(
    const std::string& patchName,
    double signedNormalSpeed,
    NormalVectorMode mode) {
  VelocityPatchBCSpec out;
  out.patchName = patchName;
  out.type = VelocityBCType::FixedNormalSpeed;
  out.normalSpeed = signedNormalSpeed;
  out.normalVectorMode = mode;
  return out;
}

VelocityPatchBCSpec make_fixed_flow_rate_bc(
    const std::string& patchName,
    double signedFlowRate,
    NormalVectorMode mode) {
  VelocityPatchBCSpec out;
  out.patchName = patchName;
  out.type = VelocityBCType::FixedFlowRate;
  out.flowRate = signedFlowRate;
  out.normalVectorMode = mode;
  return out;
}

PressurePatchBCSpec make_pressure_fixed_value_bc(
    const std::string& patchName,
    double value) {
  PressurePatchBCSpec out;
  out.patchName = patchName;
  out.type = PressureBCType::FixedValue;
  out.value = value;
  return out;
}

PressurePatchBCSpec make_pressure_fixed_value_function_bc(
    const std::string& patchName,
    ScalarFunction fn) {
  PressurePatchBCSpec out;
  out.patchName = patchName;
  out.type = PressureBCType::FixedValueFunction;
  out.scalarFunction = std::move(fn);
  return out;
}

PressurePatchBCSpec make_pressure_zero_gradient_bc(const std::string& patchName) {
  PressurePatchBCSpec out;
  out.patchName = patchName;
  out.type = PressureBCType::ZeroGradient;
  return out;
}

PressurePatchBCSpec make_pressure_open_bc(const std::string& patchName) {
  PressurePatchBCSpec out;
  out.patchName = patchName;
  out.type = PressureBCType::Open;
  return out;
}

std::vector<std::string> duplicate_velocity_bc_patches(
    const std::vector<VelocityPatchBCSpec>& specs) {
  std::unordered_map<std::string, int> counts;
  std::vector<std::string> dup;
  for (const auto& s : specs) {
    int c = ++counts[s.patchName];
    if (c == 2) dup.push_back(s.patchName);
  }
  return dup;
}

std::vector<std::string> duplicate_pressure_bc_patches(
    const std::vector<PressurePatchBCSpec>& specs) {
  std::unordered_map<std::string, int> counts;
  std::vector<std::string> dup;
  for (const auto& s : specs) {
    int c = ++counts[s.patchName];
    if (c == 2) dup.push_back(s.patchName);
  }
  return dup;
}

bool pressure_reference_required(
    const std::vector<PressurePatchBCSpec>& specs) {
  for (const auto& s : specs) {
    if (s.type == PressureBCType::FixedValue ||
        s.type == PressureBCType::FixedValueFunction) {
      return false;
    }
  }
  return true;
}

} // namespace pipebc
