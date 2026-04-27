#include "scalar_elliptic.h"

namespace {

double face_interp_lambda(const Mesh& mesh, int f) {
  int P = mesh.owner[f];
  int N = mesh.neigh[f];
  auto d = sub3(mesh.cc[N], mesh.cc[P]);
  auto dx = sub3(mesh.xf[f], mesh.cc[P]);
  double lam = dot3(dx, d) / std::max(dot3(d, d), 1e-30);
  return std::min(1.0, std::max(0.0, lam));
}

} // namespace

CSRPattern build_scalar_pattern(const Mesh& mesh) {
  CSRPattern pat;
  pat.nRows = mesh.nCells;
  pat.rows.resize(mesh.nCells);
  pat.ncols.resize(mesh.nCells);
  pat.rowOffsets.resize(mesh.nCells + 1);
  pat.diagPos.resize(mesh.nCells);
  pat.facePP.resize(mesh.nInternalFaces);
  pat.facePN.resize(mesh.nInternalFaces);
  pat.faceNP.resize(mesh.nInternalFaces);
  pat.faceNN.resize(mesh.nInternalFaces);

  std::vector<std::map<int,int>> pos(mesh.nCells);
  pat.rowOffsets[0] = 0;
  for (int c = 0; c < mesh.nCells; ++c) {
    pat.rows[c] = static_cast<HYPRE_BigInt>(c);
    std::vector<int> cols = mesh.cellNbrs[c];
    cols.insert(cols.begin(), c);
    std::sort(cols.begin(), cols.end());
    cols.erase(std::unique(cols.begin(), cols.end()), cols.end());
    pat.ncols[c] = static_cast<HYPRE_Int>(cols.size());
    pat.rowOffsets[c+1] = pat.rowOffsets[c] + static_cast<int>(cols.size());
    for (int j = 0; j < static_cast<int>(cols.size()); ++j) pos[c][cols[j]] = pat.rowOffsets[c] + j;
  }

  pat.nnz = pat.rowOffsets.back();
  pat.cols.resize(pat.nnz);
  for (int c = 0; c < mesh.nCells; ++c) {
    for (const auto& kv : pos[c]) pat.cols[kv.second] = static_cast<HYPRE_BigInt>(kv.first);
    pat.diagPos[c] = pos[c][c];
  }

  for (int f = 0; f < mesh.nInternalFaces; ++f) {
    int P = mesh.owner[f];
    int N = mesh.neigh[f];
    pat.facePP[f] = pos[P][P];
    pat.facePN[f] = pos[P][N];
    pat.faceNP[f] = pos[N][P];
    pat.faceNN[f] = pos[N][N];
  }

  return pat;
}

void assemble_scalar_elliptic_system(
    const Mesh& mesh,
    const CSRPattern& pat,
    const std::vector<double>& gammaFace,
    const std::vector<double>& cellSource,
    const BoundaryFaceData& bcFaceData,
    const std::vector<std::array<double,3>>& grad,
    std::vector<HYPRE_Complex>& values,
    std::vector<HYPRE_Complex>& rhs,
    bool includeNonOrth,
    bool useReferenceCell,
    int refCell,
    double refValue) {

  if (static_cast<int>(gammaFace.size()) != mesh.nFaces) {
    throw std::runtime_error("gammaFace must have size mesh.nFaces");
  }
  values.assign(pat.nnz, 0.0);
  rhs.assign(mesh.nCells, 0.0);

  for (int c = 0; c < mesh.nCells; ++c) {
    rhs[c] = static_cast<HYPRE_Complex>(cellSource[c] * mesh.vol[c]);
  }

  for (int f = 0; f < mesh.nInternalFaces; ++f) {
    int P = mesh.owner[f];
    int N = mesh.neigh[f];

    auto d = sub3(mesh.cc[N], mesh.cc[P]);
    double dDotS = dot3(d, mesh.Sf[f]);
    double D = gammaFace[f] * dot3(mesh.Sf[f], mesh.Sf[f]) / std::max(dDotS, 1e-30);

    values[pat.facePP[f]] += D;
    values[pat.facePN[f]] -= D;
    values[pat.faceNP[f]] -= D;
    values[pat.faceNN[f]] += D;

    if (includeNonOrth) {
      auto T = sub3(mesh.Sf[f], mul3(D / std::max(gammaFace[f], 1e-30), d));
      double lam = face_interp_lambda(mesh, f);
      auto gradF = add3(mul3(1.0 - lam, grad[P]), mul3(lam, grad[N]));
      double corr = gammaFace[f] * dot3(T, gradF);
      rhs[P] += static_cast<HYPRE_Complex>(corr);
      rhs[N] -= static_cast<HYPRE_Complex>(corr);
    }
  }

  for (int f = mesh.nInternalFaces; f < mesh.nFaces; ++f) {
    int P = mesh.owner[f];
    auto d = sub3(mesh.xf[f], mesh.cc[P]);
    double dDotS = dot3(d, mesh.Sf[f]);
    double D = gammaFace[f] * dot3(mesh.Sf[f], mesh.Sf[f]) / std::max(dDotS, 1e-30);

    if (bcFaceData.type[f] == ScalarBCType::Dirichlet) {
      values[pat.diagPos[P]] += D;
      rhs[P] += static_cast<HYPRE_Complex>(D * bcFaceData.value[f]);
      if (includeNonOrth) {
        auto T = sub3(mesh.Sf[f], mul3(D / std::max(gammaFace[f], 1e-30), d));
        double corr = gammaFace[f] * dot3(T, grad[P]);
        rhs[P] += static_cast<HYPRE_Complex>(corr);
      }
    } else {
      rhs[P] += static_cast<HYPRE_Complex>(-gammaFace[f] * bcFaceData.value[f] * mesh.Af[f]);
    }
  }

  if (useReferenceCell) {
    if (refCell < 0 || refCell >= mesh.nCells) {
      throw std::runtime_error("Invalid reference cell");
    }
    int rowStart = pat.rowOffsets[refCell];
    int rowEnd = pat.rowOffsets[refCell + 1];
    for (int k = rowStart; k < rowEnd; ++k) values[k] = 0.0;
    values[pat.diagPos[refCell]] = 1.0;
    rhs[refCell] = refValue;
  }
}
