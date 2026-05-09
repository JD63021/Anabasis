#pragma once

#include "bc_specs.h"
#include "patch_geometry.h"

#include <array>
#include <string>
#include <vector>

namespace pipebc {

struct LegacyBCMeshView {
  int nFaces = 0;
  int nInternalFaces = 0;

  const std::vector<std::string>* patchNames = nullptr;
  const std::vector<int>* patchStartFace = nullptr;
  const std::vector<int>* patchNFaces = nullptr;

  const std::vector<std::array<double,3>>* xf = nullptr;
  const std::vector<std::array<double,3>>* nf = nullptr;
};

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
    std::vector<double>& pFaceBC);

} // namespace pipebc
