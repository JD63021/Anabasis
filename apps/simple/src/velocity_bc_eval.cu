#include "velocity_bc_eval.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <string>

namespace pipebc {
namespace {

static inline double dot3(const std::array<double,3>& a,
                          const std::array<double,3>& b) {
  return a[0]*b[0] + a[1]*b[1] + a[2]*b[2];
}

static inline double norm3(const std::array<double,3>& a) {
  return std::sqrt(std::max(dot3(a, a), 0.0));
}

static std::array<double,3> normalized_or_throw(
    const std::array<double,3>& a,
    const std::string& what) {
  const double n = norm3(a);
  if (n <= 1.0e-30) {
    throw std::runtime_error("Cannot normalize vector for " + what);
  }
  return {{a[0]/n, a[1]/n, a[2]/n}};
}

static int find_patch_index_or_throw(const LegacyBCMeshView& mesh,
                                     const std::string& patchName) {
  if (!mesh.patchNames) {
    throw std::runtime_error("LegacyBCMeshView.patchNames is null");
  }
  for (int i = 0; i < static_cast<int>(mesh.patchNames->size()); ++i) {
    if ((*mesh.patchNames)[i] == patchName) return i;
  }
  throw std::runtime_error("Patch '" + patchName + "' not found in LegacyBCMeshView");
}

static std::array<double,3>
choose_normal_direction(const VelocityPatchBCSpec& spec,
                        const PatchGeometrySummary& patch,
                        const std::array<double,3>& faceNormal) {
  switch (spec.normalVectorMode) {
    case NormalVectorMode::AveragePatchNormal:
      return normalized_or_throw(
          patch.averageNormal,
          "AveragePatchNormal on patch '" + patch.patchName + "'");

    case NormalVectorMode::LocalFaceNormal:
      return normalized_or_throw(
          faceNormal,
          "LocalFaceNormal on patch '" + patch.patchName + "'");
  }

  throw std::runtime_error("Unhandled NormalVectorMode");
}

static double
eval_signed_normal_speed(const VelocityPatchBCSpec& spec,
                         const PatchGeometrySummary& patch) {
  switch (spec.type) {
    case VelocityBCType::FixedNormalSpeed:
      return spec.normalSpeed;

    case VelocityBCType::FixedFlowRate:
      if (patch.totalArea <= 1.0e-30) {
        throw std::runtime_error("Patch '" + patch.patchName + "' has near-zero area for FixedFlowRate");
      }
      return spec.flowRate / patch.totalArea;

    default:
      break;
  }
  throw std::runtime_error("eval_signed_normal_speed called for wrong VelocityBCType");
}

static void
check_fixed_flow_rate_mode(const VelocityPatchBCSpec& spec,
                           const PatchGeometrySummary& patch) {
  if (spec.type != VelocityBCType::FixedFlowRate) return;

  if (spec.normalVectorMode == NormalVectorMode::AveragePatchNormal) {
    if (patch.planarity < 0.98) {
      throw std::runtime_error(
          "FixedFlowRate with AveragePatchNormal requires an approximately planar patch. "
          "Patch '" + patch.patchName + "' has planarity = " + std::to_string(patch.planarity) +
          ". Use LocalFaceNormal instead.");
    }
  }
}

static std::array<double,3>
eval_velocity_spec_on_face(const VelocityPatchBCSpec& spec,
                           const PatchGeometrySummary& patch,
                           const std::array<double,3>& x,
                           const std::array<double,3>& faceNormal,
                           double time) {
  switch (spec.type) {
    case VelocityBCType::WallNoSlip:
      return {0.0, 0.0, 0.0};

    case VelocityBCType::FixedUniformVector:
      return spec.uniformVector;

    case VelocityBCType::FixedVectorFunction:
      if (!spec.vectorFunction) {
        throw std::runtime_error("FixedVectorFunction BC missing vectorFunction for patch '" + spec.patchName + "'");
      }
      return spec.vectorFunction(x, time);

    case VelocityBCType::ZeroGradient:
      return {0.0, 0.0, 0.0};

    case VelocityBCType::FixedNormalSpeed:
    case VelocityBCType::FixedFlowRate: {
      check_fixed_flow_rate_mode(spec, patch);
      const double un = eval_signed_normal_speed(spec, patch);
      const auto n = choose_normal_direction(spec, patch, faceNormal);
      return {un * n[0], un * n[1], un * n[2]};
    }
  }

  throw std::runtime_error("Unhandled VelocityBCType");
}

static double
eval_pressure_spec_on_face(const PressurePatchBCSpec& spec,
                           const std::array<double,3>& x,
                           double time) {
  switch (spec.type) {
    case PressureBCType::FixedValue:
      return spec.value;

    case PressureBCType::FixedValueFunction:
      if (!spec.scalarFunction) {
        throw std::runtime_error("FixedValueFunction BC missing scalarFunction for patch '" + spec.patchName + "'");
      }
      return spec.scalarFunction(x, time);

    case PressureBCType::ZeroGradient:
    case PressureBCType::Open:
      return 0.0;
  }

  throw std::runtime_error("Unhandled PressureBCType");
}

} // namespace

void apply_bc_specs_to_legacy_face_arrays(
    const LegacyBCMeshView& mesh,
    const std::vector<PatchGeometrySummary>& patchGeometryTable,
    const std::vector<VelocityPatchBCSpec>& velocityPatchSpecs,
    const std::vector<PressurePatchBCSpec>& pressurePatchSpecs,
    double time,
    std::vector<std::string>& bcUType,
    std::vector<std::string>& bcVType,
    std::vector<std::string>& bcWType,
    std::vector<std::string>& bcPType,
    std::vector<double>& uFaceBC,
    std::vector<double>& vFaceBC,
    std::vector<double>& wFaceBC,
    std::vector<double>& pFaceBC) {

  if (!mesh.patchNames || !mesh.patchStartFace || !mesh.patchNFaces ||
      !mesh.xf || !mesh.nf) {
    throw std::runtime_error("LegacyBCMeshView is missing required pointers");
  }

  const int nPatches = static_cast<int>(mesh.patchNames->size());

  if (static_cast<int>(bcUType.size()) != nPatches ||
      static_cast<int>(bcVType.size()) != nPatches ||
      static_cast<int>(bcWType.size()) != nPatches ||
      static_cast<int>(bcPType.size()) != nPatches) {
    throw std::runtime_error("Legacy BC patch type arrays have wrong size");
  }

  if (static_cast<int>(uFaceBC.size()) != mesh.nFaces ||
      static_cast<int>(vFaceBC.size()) != mesh.nFaces ||
      static_cast<int>(wFaceBC.size()) != mesh.nFaces ||
      static_cast<int>(pFaceBC.size()) != mesh.nFaces) {
    throw std::runtime_error("Legacy BC face arrays have wrong size");
  }

  for (const auto& spec : velocityPatchSpecs) {
    const int pidx = find_patch_index_or_throw(mesh, spec.patchName);
    const auto& patch = get_patch_geometry_or_throw(patchGeometryTable, spec.patchName);

    const int start = (*mesh.patchStartFace)[pidx];
    const int count = (*mesh.patchNFaces)[pidx];
    const int end = start + count;

    switch (spec.type) {
      case VelocityBCType::ZeroGradient:
        bcUType[pidx] = "Neumann";
        bcVType[pidx] = "Neumann";
        bcWType[pidx] = "Neumann";
        break;

      case VelocityBCType::WallNoSlip:
      case VelocityBCType::FixedUniformVector:
      case VelocityBCType::FixedVectorFunction:
      case VelocityBCType::FixedNormalSpeed:
      case VelocityBCType::FixedFlowRate:
        bcUType[pidx] = "Dirichlet";
        bcVType[pidx] = "Dirichlet";
        bcWType[pidx] = "Dirichlet";

        for (int f = start; f < end; ++f) {
          const auto U = eval_velocity_spec_on_face(
              spec, patch, (*mesh.xf)[f], (*mesh.nf)[f], time);
          uFaceBC[f] = U[0];
          vFaceBC[f] = U[1];
          wFaceBC[f] = U[2];
        }
        break;
    }
  }

  for (const auto& spec : pressurePatchSpecs) {
    const int pidx = find_patch_index_or_throw(mesh, spec.patchName);
    const int start = (*mesh.patchStartFace)[pidx];
    const int count = (*mesh.patchNFaces)[pidx];
    const int end = start + count;

    switch (spec.type) {
      case PressureBCType::ZeroGradient:
      case PressureBCType::Open:
        bcPType[pidx] = "Neumann";
        break;

      case PressureBCType::FixedValue:
      case PressureBCType::FixedValueFunction:
        bcPType[pidx] = "Dirichlet";
        for (int f = start; f < end; ++f) {
          pFaceBC[f] = eval_pressure_spec_on_face(spec, (*mesh.xf)[f], time);
        }
        break;
    }
  }
}

} // namespace pipebc
