#pragma once

#include <array>
#include <functional>
#include <string>
#include <vector>

namespace pipebc {

enum class VelocityBCType {
  WallNoSlip,
  FixedUniformVector,
  FixedVectorFunction,
  ZeroGradient,
  FixedNormalSpeed,
  FixedFlowRate
};

enum class PressureBCType {
  FixedValue,
  FixedValueFunction,
  ZeroGradient,
  Open
};

enum class NormalVectorMode {
  AveragePatchNormal,
  LocalFaceNormal
};

using VectorFunction = std::function<std::array<double,3>(const std::array<double,3>&, double)>;
using ScalarFunction = std::function<double(const std::array<double,3>&, double)>;

struct VelocityPatchBCSpec {
  std::string patchName;
  VelocityBCType type = VelocityBCType::ZeroGradient;

  std::array<double,3> uniformVector{{0.0, 0.0, 0.0}};
  VectorFunction vectorFunction{};

  // Signed scalar normal data.
  // Positive means along the outward mesh normal.
  double normalSpeed = 0.0;
  double flowRate = 0.0;

  NormalVectorMode normalVectorMode = NormalVectorMode::AveragePatchNormal;
};

struct PressurePatchBCSpec {
  std::string patchName;
  PressureBCType type = PressureBCType::ZeroGradient;

  double value = 0.0;
  ScalarFunction scalarFunction{};
};

VelocityPatchBCSpec make_wall_noslip_bc(const std::string& patchName);

VelocityPatchBCSpec make_fixed_uniform_vector_bc(
    const std::string& patchName,
    const std::array<double,3>& value);

VelocityPatchBCSpec make_fixed_vector_function_bc(
    const std::string& patchName,
    VectorFunction fn);

VelocityPatchBCSpec make_zero_gradient_velocity_bc(const std::string& patchName);

VelocityPatchBCSpec make_fixed_normal_speed_bc(
    const std::string& patchName,
    double signedNormalSpeed,
    NormalVectorMode mode = NormalVectorMode::AveragePatchNormal);

VelocityPatchBCSpec make_fixed_flow_rate_bc(
    const std::string& patchName,
    double signedFlowRate,
    NormalVectorMode mode = NormalVectorMode::AveragePatchNormal);

PressurePatchBCSpec make_pressure_fixed_value_bc(
    const std::string& patchName,
    double value);

PressurePatchBCSpec make_pressure_fixed_value_function_bc(
    const std::string& patchName,
    ScalarFunction fn);

PressurePatchBCSpec make_pressure_zero_gradient_bc(const std::string& patchName);

PressurePatchBCSpec make_pressure_open_bc(const std::string& patchName);

std::vector<std::string> duplicate_velocity_bc_patches(
    const std::vector<VelocityPatchBCSpec>& specs);

std::vector<std::string> duplicate_pressure_bc_patches(
    const std::vector<PressurePatchBCSpec>& specs);

bool pressure_reference_required(
    const std::vector<PressurePatchBCSpec>& specs);

} // namespace pipebc
