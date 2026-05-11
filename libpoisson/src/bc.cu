#include "bc.h"

BoundaryFaceData build_boundary_face_data(const Mesh& mesh, const ScalarBCSet& bcSet) {
  BoundaryFaceData out;
  out.type.assign(mesh.nFaces, ScalarBCType::Dirichlet);
  out.value.assign(mesh.nFaces, 0.0);

  std::map<std::string, const ScalarPatchBC*> byName;
  for (const auto& bc : bcSet.patches) {
    byName[bc.patchName] = &bc;
  }

  for (int f = mesh.nInternalFaces; f < mesh.nFaces; ++f) {
    int pidx1 = mesh.bPatch[f];
    if (pidx1 <= 0) {
      throw std::runtime_error("Boundary face missing patch index at face " + std::to_string(f));
    }
    int pidx = pidx1 - 1;
    if (pidx < 0 || pidx >= static_cast<int>(mesh.patchNames.size())) {
      throw std::runtime_error("Invalid patch index at face " + std::to_string(f));
    }
    const std::string& patchName = mesh.patchNames[pidx];
    auto it = byName.find(patchName);
    if (it == byName.end()) {
      throw std::runtime_error("No BC supplied for patch '" + patchName + "'");
    }
    const ScalarPatchBC& bc = *it->second;
    if (!bc.evaluator) {
      throw std::runtime_error("BC evaluator missing for patch '" + patchName + "'");
    }
    out.type[f] = bc.type;
    out.value[f] = bc.evaluator(mesh.xf[f], mesh.nf[f]);
  }
  return out;
}
