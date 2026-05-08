#pragma once

#include <array>
#include <string>
#include <vector>

namespace pipebc {

struct PatchGeometryInput {
  int nInternalFaces = 0;
  int nFaces = 0;

  const std::vector<std::array<double,3>>* xf = nullptr;
  const std::vector<std::array<double,3>>* nf = nullptr;
  const std::vector<std::array<double,3>>* Sf = nullptr;
  const std::vector<double>* Af = nullptr;

  const std::vector<std::string>* patchNames = nullptr;
  const std::vector<int>* patchStartFace = nullptr;
  const std::vector<int>* patchNFaces = nullptr;
};

struct PatchGeometrySummary {
  int patchIndex = -1;
  std::string patchName;

  std::vector<int> faces;

  double totalArea = 0.0;
  std::array<double,3> centroid{{0.0, 0.0, 0.0}};
  std::array<double,3> sumSf{{0.0, 0.0, 0.0}};
  std::array<double,3> averageNormal{{0.0, 0.0, 0.0}};
  double planarity = 0.0;

  std::array<double,3> tangent1{{0.0, 0.0, 0.0}};
  std::array<double,3> tangent2{{0.0, 0.0, 0.0}};
};

std::vector<PatchGeometrySummary>
build_patch_geometry_table(const PatchGeometryInput& in);

int
find_patch_index_by_name(const std::vector<PatchGeometrySummary>& patches,
                         const std::string& patchName);

const PatchGeometrySummary&
get_patch_geometry_or_throw(const std::vector<PatchGeometrySummary>& patches,
                            const std::string& patchName);

bool
is_patch_nearly_planar(const PatchGeometrySummary& patch,
                       double threshold = 0.98);

} // namespace pipebc
