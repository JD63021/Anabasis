#pragma once

#include "mesh.h"

enum class ScalarBCType {
  Dirichlet,
  Neumann
};

using ScalarBCFunc = std::function<double(const std::array<double,3>& x,
                                          const std::array<double,3>& outwardNormal)>;

struct ScalarPatchBC {
  std::string patchName;
  ScalarBCType type = ScalarBCType::Dirichlet;
  ScalarBCFunc evaluator;
};

struct ScalarBCSet {
  std::vector<ScalarPatchBC> patches;
};

struct BoundaryFaceData {
  std::vector<ScalarBCType> type;
  std::vector<double> value;
};

inline ScalarPatchBC make_dirichlet_patch_bc(const std::string& patchName, ScalarBCFunc fn) {
  return ScalarPatchBC{patchName, ScalarBCType::Dirichlet, std::move(fn)};
}

inline ScalarPatchBC make_neumann_patch_bc(const std::string& patchName, ScalarBCFunc fn) {
  return ScalarPatchBC{patchName, ScalarBCType::Neumann, std::move(fn)};
}

inline ScalarPatchBC make_dirichlet_constant_bc(const std::string& patchName, double value) {
  return make_dirichlet_patch_bc(patchName, [value](const auto&, const auto&) { return value; });
}

inline ScalarPatchBC make_neumann_constant_bc(const std::string& patchName, double value) {
  return make_neumann_patch_bc(patchName, [value](const auto&, const auto&) { return value; });
}

BoundaryFaceData build_boundary_face_data(const Mesh& mesh, const ScalarBCSet& bcSet);
