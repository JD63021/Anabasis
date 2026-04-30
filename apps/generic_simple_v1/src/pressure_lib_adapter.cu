#include "pressure_lib_adapter.h"

#include <map>
#include <stdexcept>
#include <algorithm>
#include <mpi.h>

#include "scalar_elliptic.h"

namespace {

Mesh to_lib_mesh(const PressureLibMeshData& in) {
  Mesh out;
  out.P = in.P;
  out.faces = in.faces;
  out.owner = in.owner;
  out.neigh = in.neigh;
  out.bPatch = in.bPatch;
  out.patchNames = in.patchNames;
  out.nFaces = in.nFaces;
  out.nInternalFaces = in.nInternalFaces;
  out.nCells = in.nCells;
  out.cc = in.cc;
  out.xf = in.xf;
  out.nf = in.nf;
  out.Sf = in.Sf;
  out.vol = in.vol;
  out.Af = in.Af;
  out.cellNbrs = in.cellNbrs;
  out.cellBFace = in.cellBFace;
  out.cellFaces = in.cellFaces;
  out.cellOrient = in.cellOrient;
  out.maxNonOrthDeg = in.maxNonOrthDeg;
  return out;
}

double face_interp_lambda_local(const Mesh& mesh, int f) {
  const int P = mesh.owner[f];
  const int N = mesh.neigh[f];
  const auto d  = sub3(mesh.cc[N], mesh.cc[P]);
  const auto dx = sub3(mesh.xf[f], mesh.cc[P]);
  const double lam = dot3(dx, d) / std::max(dot3(d, d), 1e-30);
  return std::min(1.0, std::max(0.0, lam));
}

std::vector<double> build_gamma_face_from_rAU(
    const Mesh& mesh,
    const std::vector<double>& rAUCell) {

  if (static_cast<int>(rAUCell.size()) != mesh.nCells) {
    throw std::runtime_error("rAUCell size must equal mesh.nCells");
  }

  std::vector<double> gammaFace(mesh.nFaces, 0.0);

  for (int f = 0; f < mesh.nInternalFaces; ++f) {
    const int P = mesh.owner[f];
    const int N = mesh.neigh[f];
    const double lam = face_interp_lambda_local(mesh, f);
    gammaFace[f] = (1.0 - lam) * rAUCell[P] + lam * rAUCell[N];
  }

  for (int f = mesh.nInternalFaces; f < mesh.nFaces; ++f) {
    const int P = mesh.owner[f];
    gammaFace[f] = rAUCell[P];
  }

  return gammaFace;
}

BoundaryFaceData build_bc_face_data_from_patch_types(
    const Mesh& mesh,
    const std::vector<std::string>& bcPatchType,
    const std::vector<double>& bcFaceValue) {

  if (static_cast<int>(bcFaceValue.size()) != mesh.nFaces) {
    throw std::runtime_error("bcFaceValue size must equal mesh.nFaces");
  }

  BoundaryFaceData out;
  out.type.assign(mesh.nFaces, ScalarBCType::Neumann);
  out.value.assign(mesh.nFaces, 0.0);

  std::map<int,int> rawToCompact;
  int nextCompact = 0;

  for (int f = mesh.nInternalFaces; f < mesh.nFaces; ++f) {
    const int rawPatch = mesh.bPatch[f];
    auto it = rawToCompact.find(rawPatch);
    if (it == rawToCompact.end()) {
      rawToCompact[rawPatch] = nextCompact;
      it = rawToCompact.find(rawPatch);
      ++nextCompact;
    }

    const int patch = it->second;
    if (patch < 0 || patch >= static_cast<int>(bcPatchType.size())) {
      throw std::runtime_error(
          "Boundary patch index out of range after remap. rawPatch=" +
          std::to_string(rawPatch) +
          " compactPatch=" + std::to_string(patch) +
          " bcPatchType.size=" + std::to_string(bcPatchType.size()));
    }

    out.type[f] = (bcPatchType[patch] == "Dirichlet")
                ? ScalarBCType::Dirichlet
                : ScalarBCType::Neumann;
    out.value[f] = bcFaceValue[f];
  }

  return out;
}

struct PressureLibAssemblyImpl {
  Mesh mesh;
  CSRPattern pat;
};

} // namespace

void init_pressure_lib_assembly_handle(
    PressureLibAssemblyHandle& handle,
    const PressureLibMeshData& meshData) {

  destroy_pressure_lib_assembly_handle(handle);

  auto* impl = new PressureLibAssemblyImpl;
  impl->mesh = to_lib_mesh(meshData);
  impl->pat = build_scalar_pattern(impl->mesh);

  handle.impl = impl;
  handle.nRows = impl->pat.nRows;
  handle.nnz = impl->pat.nnz;
}

PressureLibAssembly assemble_pressure_with_libpoisson(
    PressureLibAssemblyHandle& handle,
    const std::vector<double>& rAUCell,
    const std::vector<std::string>& bcPatchType,
    const std::vector<double>& bcFaceValue,
    const std::vector<double>& rhs,
    const PressureLibOptions& opt) {

  if (!handle.impl) {
    throw std::runtime_error("PressureLibAssemblyHandle not initialized");
  }

  auto* impl = static_cast<PressureLibAssemblyImpl*>(handle.impl);
  const Mesh& mesh = impl->mesh;

  if (static_cast<int>(rhs.size()) != mesh.nCells) {
    throw std::runtime_error("rhs size must equal mesh.nCells");
  }

  const double t0 = MPI_Wtime();

  const std::vector<double> gammaFace =
      build_gamma_face_from_rAU(mesh, rAUCell);

  const BoundaryFaceData bcFaceData =
      build_bc_face_data_from_patch_types(mesh, bcPatchType, bcFaceValue);

  std::vector<double> cellSource(mesh.nCells, 0.0);
  for (int c = 0; c < mesh.nCells; ++c) {
    cellSource[c] = rhs[c] / std::max(mesh.vol[c], 1e-30);
  }

  std::vector<std::array<double,3>> grad(mesh.nCells, {0.0, 0.0, 0.0});
  std::vector<HYPRE_Complex> valuesH, rhsH;

  assemble_scalar_elliptic_system(
      mesh,
      impl->pat,
      gammaFace,
      cellSource,
      bcFaceData,
      grad,
      valuesH,
      rhsH,
      false, // orth pressure operator only; non-orth handled outside in SIMPLE loop
      opt.useReferenceCell,
      opt.referenceCell,
      opt.referenceValue);

  PressureLibAssembly out;
  out.values.resize(valuesH.size());
  out.rhs.resize(rhsH.size());
  for (std::size_t i = 0; i < valuesH.size(); ++i) out.values[i] = static_cast<double>(valuesH[i]);
  for (std::size_t i = 0; i < rhsH.size();    ++i) out.rhs[i]    = static_cast<double>(rhsH[i]);
  out.wallTime = MPI_Wtime() - t0;
  return out;
}

void destroy_pressure_lib_assembly_handle(PressureLibAssemblyHandle& handle) {
  if (!handle.impl) return;
  auto* impl = static_cast<PressureLibAssemblyImpl*>(handle.impl);
  delete impl;
  handle.impl = nullptr;
  handle.nRows = 0;
  handle.nnz = 0;
}
