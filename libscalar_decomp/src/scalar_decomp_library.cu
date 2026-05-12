#include "scalar_decomp_library.h"

#include <algorithm>
#include <array>
#include <cctype>
#include <cmath>
#include <map>
#include <stdexcept>
#include <string>
#include <vector>

#include <mpi.h>

extern "C" {
#include "HYPRE.h"
#include "HYPRE_IJ_mv.h"
#include "HYPRE_parcsr_ls.h"
#include "HYPRE_krylov.h"
}

#define HYPRE_CALL_SCALAR_DECOMP(stmt) do {                                      \
  HYPRE_Int _ierr = (stmt);                                                      \
  if (_ierr) {                                                                   \
    int _rank = 0; MPI_Comm_rank(MPI_COMM_WORLD, &_rank);                        \
    std::fprintf(stderr, "[%d] HYPRE ERROR %s:%d code=%d\n",                    \
                 _rank, __FILE__, __LINE__, (int)_ierr);                         \
    MPI_Abort(MPI_COMM_WORLD, (int)_ierr);                                        \
  }                                                                              \
} while (0)

namespace libscalar_decomp {

DistConvectionScheme convection_scheme_from_string(const std::string& name) {
  std::string v = name;
  for (char& c : v) c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));

  if (v == "upwind" || v == "first-order-upwind" || v == "firstorderupwind") {
    return DistConvectionScheme::Upwind;
  }

  if (v == "central" || v == "linear" || v == "gauss-linear" || v == "gausslinear") {
    return DistConvectionScheme::Central;
  }

  throw std::runtime_error("Unknown convection scheme '" + name + "'. Use upwind or central.");
}

const char* convection_scheme_name(DistConvectionScheme scheme) {
  return scheme == DistConvectionScheme::Upwind ? "upwind" : "central";
}

DistDiffusionScheme diffusion_scheme_from_string(const std::string& name) {
  std::string v = name;
  for (char& c : v) c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));

  if (v == "orth" || v == "orthogonal") return DistDiffusionScheme::Orth;
  if (v == "nonorth" || v == "nonorthogonal" || v == "corrected") return DistDiffusionScheme::NonOrth;

  throw std::runtime_error("Unknown diffusion scheme '" + name + "'. Use orth or nonorth.");
}

const char* diffusion_scheme_name(DistDiffusionScheme scheme) {
  return scheme == DistDiffusionScheme::NonOrth ? "nonorth" : "orth";
}

namespace {

static inline std::array<double,3> add3s(const std::array<double,3>& a, const std::array<double,3>& b) {
  return {a[0]+b[0], a[1]+b[1], a[2]+b[2]};
}
static inline std::array<double,3> sub3s(const std::array<double,3>& a, const std::array<double,3>& b) {
  return {a[0]-b[0], a[1]-b[1], a[2]-b[2]};
}
static inline std::array<double,3> mul3s(double s, const std::array<double,3>& a) {
  return {s*a[0], s*a[1], s*a[2]};
}
static inline double dot3s(const std::array<double,3>& a, const std::array<double,3>& b) {
  return a[0]*b[0] + a[1]*b[1] + a[2]*b[2];
}

struct LocalBoundaryFaceData {
  std::vector<ScalarBCType> type;
  std::vector<double> value;
};

LocalBoundaryFaceData build_physical_boundary_face_data_scalar(
    const DecompMesh& dm,
    const ScalarBCSet& bcSet) {
  const Mesh& mesh = dm.mesh;

  LocalBoundaryFaceData out;
  out.type.assign(mesh.nFaces, ScalarBCType::Neumann);
  out.value.assign(mesh.nFaces, 0.0);

  std::map<std::string, const ScalarPatchBC*> byName;
  for (const auto& bc : bcSet.patches) byName[bc.patchName] = &bc;

  for (int f = mesh.nInternalFaces; f < mesh.nFaces; ++f) {
    if (dm.isProcFace[f]) continue;

    const int pidx = mesh.bPatch[f] - 1;
    if (pidx < 0 || pidx >= static_cast<int>(mesh.patchNames.size())) {
      throw std::runtime_error("Boundary face has invalid patch index at face " + std::to_string(f));
    }

    const std::string& patchName = mesh.patchNames[pidx];
    auto it = byName.find(patchName);
    if (it == byName.end()) {
      throw std::runtime_error("No scalar BC supplied for physical patch '" + patchName + "'");
    }

    const ScalarPatchBC& bc = *it->second;
    if (!bc.evaluator) {
      throw std::runtime_error("BC evaluator missing for physical patch '" + patchName + "'");
    }

    out.type[f] = bc.type;
    out.value[f] = bc.evaluator(mesh.xf[f], mesh.nf[f]);
  }

  return out;
}

double face_interp_lambda_local_scalar(const Mesh& mesh, int f) {
  const int P = mesh.owner[f];
  const int N = mesh.neigh[f];
  const auto d = sub3s(mesh.cc[N], mesh.cc[P]);
  const auto dx = sub3s(mesh.xf[f], mesh.cc[P]);
  const double lam = dot3s(dx, d) / std::max(dot3s(d, d), 1.0e-30);
  return std::min(1.0, std::max(0.0, lam));
}

double face_interp_lambda_proc_scalar(const DecompMesh& dm, int f) {
  const Mesh& mesh = dm.mesh;
  const int P = mesh.owner[f];
  const auto d = sub3s(dm.remoteCCForFace[f], mesh.cc[P]);
  const auto dx = sub3s(mesh.xf[f], mesh.cc[P]);
  const double lam = dot3s(dx, d) / std::max(dot3s(d, d), 1.0e-30);
  return std::min(1.0, std::max(0.0, lam));
}

double boundary_equivalent_face_value_scalar(
    const Mesh& mesh,
    const LocalBoundaryFaceData& bcFaceData,
    const std::vector<double>& phi,
    int P,
    int f) {
  if (bcFaceData.type[f] == ScalarBCType::Dirichlet) {
    return bcFaceData.value[f];
  }

  const auto r = sub3s(mesh.xf[f], mesh.cc[P]);
  const double dn = std::max(dot3s(r, mesh.nf[f]), 1.0e-30);
  return phi[P] + bcFaceData.value[f] * dn;
}

void compute_lsq_gradient_scalar_decomp(
    const DecompMesh& dm,
    const std::vector<double>& phi,
    const std::vector<double>& remotePhiForFace,
    const LocalBoundaryFaceData& bcFaceData,
    std::vector<std::array<double,3>>& grad) {
  const Mesh& mesh = dm.mesh;
  grad.assign(mesh.nCells, {0.0, 0.0, 0.0});

  for (int P = 0; P < mesh.nCells; ++P) {
    const auto xP = mesh.cc[P];
    const double phiP = phi[P];

    double M[3][3] = {{0,0,0},{0,0,0},{0,0,0}};
    double rhs[3] = {0,0,0};

    auto add_constraint = [&](const std::array<double,3>& r, double dphi) {
      const double w = 1.0 / std::max(dot3s(r, r), 1.0e-30);
      for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) M[i][j] += w * r[i] * r[j];
        rhs[i] += w * r[i] * dphi;
      }
    };

    for (int N : mesh.cellNbrs[P]) {
      add_constraint(sub3s(mesh.cc[N], xP), phi[N] - phiP);
    }

    for (int f : mesh.cellBFace[P]) {
      if (dm.isProcFace[f]) {
        add_constraint(sub3s(dm.remoteCCForFace[f], xP), remotePhiForFace[f] - phiP);
      } else {
        const auto rcf = sub3s(mesh.xf[f], xP);

        if (bcFaceData.type[f] == ScalarBCType::Dirichlet) {
          add_constraint(rcf, bcFaceData.value[f] - phiP);
        } else {
          const double dn = std::max(dot3s(rcf, mesh.nf[f]), 1.0e-30);
          add_constraint(mul3s(dn, mesh.nf[f]), bcFaceData.value[f] * dn);
        }
      }
    }

    const double a=M[0][0], b=M[0][1], c=M[0][2];
    const double d=M[1][0], e=M[1][1], f=M[1][2];
    const double g=M[2][0], h=M[2][1], k=M[2][2];

    const double det = a*(e*k-f*h) - b*(d*k-f*g) + c*(d*h-e*g);

    if (std::fabs(det) > 1.0e-20) {
      double inv[3][3];
      inv[0][0]=(e*k-f*h)/det; inv[0][1]=(c*h-b*k)/det; inv[0][2]=(b*f-c*e)/det;
      inv[1][0]=(f*g-d*k)/det; inv[1][1]=(a*k-c*g)/det; inv[1][2]=(c*d-a*f)/det;
      inv[2][0]=(d*h-e*g)/det; inv[2][1]=(b*g-a*h)/det; inv[2][2]=(a*e-b*d)/det;

      grad[P] = {
        inv[0][0]*rhs[0] + inv[0][1]*rhs[1] + inv[0][2]*rhs[2],
        inv[1][0]*rhs[0] + inv[1][1]*rhs[1] + inv[1][2]*rhs[2],
        inv[2][0]*rhs[0] + inv[2][1]*rhs[1] + inv[2][2]*rhs[2]
      };
    }
  }
}

void compute_gauss_gradient_scalar_decomp(
    const DecompMesh& dm,
    const std::vector<double>& phi,
    const std::vector<double>& remotePhiForFace,
    const LocalBoundaryFaceData& bcFaceData,
    std::vector<std::array<double,3>>& grad) {
  const Mesh& mesh = dm.mesh;

  grad.assign(mesh.nCells, {0.0, 0.0, 0.0});

  for (int f = 0; f < mesh.nInternalFaces; ++f) {
    const int P = mesh.owner[f];
    const int N = mesh.neigh[f];

    const double lam = face_interp_lambda_local_scalar(mesh, f);
    const double phiF = (1.0 - lam) * phi[P] + lam * phi[N];

    for (int d = 0; d < 3; ++d) {
      const double contrib = phiF * mesh.Sf[f][d];
      grad[P][d] += contrib;
      grad[N][d] -= contrib;
    }
  }

  for (int f = mesh.nInternalFaces; f < mesh.nFaces; ++f) {
    const int P = mesh.owner[f];

    double phiF = 0.0;
    if (dm.isProcFace[f]) {
      const double lam = face_interp_lambda_proc_scalar(dm, f);
      phiF = (1.0 - lam) * phi[P] + lam * remotePhiForFace[f];
    } else {
      phiF = boundary_equivalent_face_value_scalar(mesh, bcFaceData, phi, P, f);
    }

    for (int d = 0; d < 3; ++d) {
      grad[P][d] += phiF * mesh.Sf[f][d];
    }
  }

  for (int c = 0; c < mesh.nCells; ++c) {
    const double invVol = 1.0 / std::max(mesh.vol[c], 1.0e-300);
    grad[c][0] *= invVol;
    grad[c][1] *= invVol;
    grad[c][2] *= invVol;
  }
}

void compute_gradient_scalar_decomp(
    const DecompMesh& dm,
    const std::vector<double>& phi,
    const LocalBoundaryFaceData& bcFaceData,
    const std::string& schemeName,
    std::vector<std::array<double,3>>& grad) {
  std::string v = schemeName;
  for (char& c : v) c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));

  const auto remotePhi = exchange_proc_face_scalar_owner_values(dm, phi);

  if (v == "gauss" || v == "green-gauss" || v == "greengauss") {
    compute_gauss_gradient_scalar_decomp(dm, phi, remotePhi, bcFaceData, grad);
  } else if (v == "lsq" || v == "least-squares" || v == "leastsquares") {
    compute_lsq_gradient_scalar_decomp(dm, phi, remotePhi, bcFaceData, grad);
  } else {
    throw std::runtime_error("Unknown grad scheme '" + schemeName + "'. Use lsq or gauss.");
  }
}

void validate_inputs(
    const DecompMesh& dm,
    const DistScalarTransportInputs& in) {
  const Mesh& mesh = dm.mesh;

  if (static_cast<int>(in.faceFlux.size()) != mesh.nFaces) {
    throw std::runtime_error("faceFlux must have size local mesh.nFaces");
  }
  if (static_cast<int>(in.gammaFace.size()) != mesh.nFaces) {
    throw std::runtime_error("gammaFace must have size local mesh.nFaces");
  }
  if (static_cast<int>(in.Su.size()) != mesh.nCells) {
    throw std::runtime_error("Su must have size local mesh.nCells");
  }
  if (static_cast<int>(in.Sp.size()) != mesh.nCells) {
    throw std::runtime_error("Sp must have size local mesh.nCells");
  }
}

void assemble_scalar_transport_decomp(
    const DecompMesh& dm,
    const DistCSRPattern& pat,
    const DistScalarTransportInputs& in,
    const LocalBoundaryFaceData& bcFaceData,
    const std::vector<std::array<double,3>>& grad,
    const std::vector<std::array<double,3>>& remoteGradForFace,
    const DistScalarTransportOptions& opt,
    std::vector<HYPRE_Complex>& values,
    std::vector<HYPRE_Complex>& rhs) {
  const Mesh& mesh = dm.mesh;

  validate_inputs(dm, in);

  values.assign(pat.nnz, 0.0);
  rhs.assign(mesh.nCells, 0.0);

  for (int c = 0; c < mesh.nCells; ++c) {
    rhs[c] = static_cast<HYPRE_Complex>(in.Su[c] * mesh.vol[c]);
    values[pat.diagPos[c]] += static_cast<HYPRE_Complex>(-in.Sp[c] * mesh.vol[c]);
  }

  const bool includeNonOrth = (opt.diffusionScheme == DistDiffusionScheme::NonOrth);

  for (int f = 0; f < mesh.nInternalFaces; ++f) {
    const int P = mesh.owner[f];
    const int N = mesh.neigh[f];

    const auto d = sub3s(mesh.cc[N], mesh.cc[P]);
    const double dDotS = dot3s(d, mesh.Sf[f]);
    const double gamma = in.gammaFace[f];
    const double D = gamma * dot3s(mesh.Sf[f], mesh.Sf[f]) / std::max(dDotS, 1.0e-30);

    values[pat.facePP[f]] += D;
    values[pat.facePN[f]] -= D;
    values[pat.faceNP[f]] -= D;
    values[pat.faceNN[f]] += D;

    if (includeNonOrth && std::abs(gamma) > 1.0e-300) {
      const auto T = sub3s(mesh.Sf[f], mul3s(D / gamma, d));
      const double lam = face_interp_lambda_local_scalar(mesh, f);
      const auto gradF = add3s(mul3s(1.0 - lam, grad[P]), mul3s(lam, grad[N]));
      const double corr = gamma * dot3s(T, gradF);
      rhs[P] += static_cast<HYPRE_Complex>(corr);
      rhs[N] -= static_cast<HYPRE_Complex>(corr);
    }

    const double F = in.faceFlux[f];

    if (opt.convectionScheme == DistConvectionScheme::Central) {
      const double lam = face_interp_lambda_local_scalar(mesh, f);
      const double aP = F * (1.0 - lam);
      const double aN = F * lam;

      values[pat.facePP[f]] += aP;
      values[pat.facePN[f]] += aN;
      values[pat.faceNP[f]] -= aP;
      values[pat.faceNN[f]] -= aN;
    } else {
      if (F >= 0.0) {
        values[pat.facePP[f]] += F;
        values[pat.faceNP[f]] -= F;
      } else {
        values[pat.facePN[f]] += F;
        values[pat.faceNN[f]] -= F;
      }
    }
  }

  for (size_t i = 0; i < pat.procFace.size(); ++i) {
    const int f = pat.procFace[i];
    const int P = pat.procOwner[i];

    const auto d = sub3s(dm.remoteCCForFace[f], mesh.cc[P]);
    const double dDotS = dot3s(d, mesh.Sf[f]);
    const double gamma = in.gammaFace[f];
    const double D = gamma * dot3s(mesh.Sf[f], mesh.Sf[f]) / std::max(dDotS, 1.0e-30);

    values[pat.procDiag[i]] += D;
    values[pat.procOff[i]]  -= D;

    if (includeNonOrth && std::abs(gamma) > 1.0e-300) {
      const auto T = sub3s(mesh.Sf[f], mul3s(D / gamma, d));
      const double lam = face_interp_lambda_proc_scalar(dm, f);
      const auto gradF = add3s(mul3s(1.0 - lam, grad[P]), mul3s(lam, remoteGradForFace[f]));
      const double corr = gamma * dot3s(T, gradF);
      rhs[P] += static_cast<HYPRE_Complex>(corr);
    }

    const double F = in.faceFlux[f];

    if (opt.convectionScheme == DistConvectionScheme::Central) {
      const double lam = face_interp_lambda_proc_scalar(dm, f);
      values[pat.procDiag[i]] += static_cast<HYPRE_Complex>(F * (1.0 - lam));
      values[pat.procOff[i]]  += static_cast<HYPRE_Complex>(F * lam);
    } else {
      if (F >= 0.0) {
        values[pat.procDiag[i]] += static_cast<HYPRE_Complex>(F);
      } else {
        values[pat.procOff[i]] += static_cast<HYPRE_Complex>(F);
      }
    }
  }

  for (int f = mesh.nInternalFaces; f < mesh.nFaces; ++f) {
    if (dm.isProcFace[f]) continue;

    const int P = mesh.owner[f];

    const auto d = sub3s(mesh.xf[f], mesh.cc[P]);
    const double dDotS = dot3s(d, mesh.Sf[f]);
    const double gamma = in.gammaFace[f];
    const double D = gamma * dot3s(mesh.Sf[f], mesh.Sf[f]) / std::max(dDotS, 1.0e-30);
    const double F = in.faceFlux[f];

    if (bcFaceData.type[f] == ScalarBCType::Dirichlet) {
      const double phiB = bcFaceData.value[f];

      values[pat.diagPos[P]] += static_cast<HYPRE_Complex>(D);
      rhs[P] += static_cast<HYPRE_Complex>(D * phiB);

      if (includeNonOrth && std::abs(gamma) > 1.0e-300) {
        const auto T = sub3s(mesh.Sf[f], mul3s(D / gamma, d));
        const double corr = gamma * dot3s(T, grad[P]);
        rhs[P] += static_cast<HYPRE_Complex>(corr);
      }

      // Strong Dirichlet face value for convective boundary contribution.
      rhs[P] += static_cast<HYPRE_Complex>(-F * phiB);
    } else {
      // Neumann value is prescribed outward normal gradient dphi/dn.
      const double gradn = bcFaceData.value[f];
      rhs[P] += static_cast<HYPRE_Complex>(gamma * gradn * mesh.Af[f]);

      // Zero-gradient/open style convective treatment: phi_f = phi_P.
      values[pat.diagPos[P]] += static_cast<HYPRE_Complex>(F);
    }
  }
}

struct DistSolverInfoLocal {
  std::vector<double> x;
  int iterations = 0;
  double relRes = 0.0;
};



static std::string lower_solver_name_scalar_decomp(std::string v) {
  for (char& c : v) c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
  return v;
}

static bool is_local_global_col_scalar_decomp(
    const DecompMesh& dm,
    HYPRE_BigInt g,
    int& jLocal) {
  if (g >= dm.ilower && g <= dm.iupper) {
    jLocal = static_cast<int>(g - dm.ilower);
    return true;
  }
  jLocal = -1;
  return false;
}

static std::map<HYPRE_BigInt, double> build_remote_value_map_scalar_decomp(
    const DecompMesh& dm,
    const std::vector<double>& x) {
  std::map<HYPRE_BigInt, double> remote;

  if (dm.procPatches.empty()) return remote;

  const auto remoteFaceValues = exchange_proc_face_scalar_owner_values(dm, x);

  for (const auto& pp : dm.procPatches) {
    for (int i = 0; i < pp.nFaces; ++i) {
      const int f = pp.startFace + i;
      if (f < 0 || f >= dm.mesh.nFaces) continue;
      const HYPRE_BigInt rg = dm.remoteRowForFace[f];
      if (rg >= 0) remote[rg] = remoteFaceValues[f];
    }
  }

  return remote;
}

static double value_for_global_col_scalar_decomp(
    const DecompMesh& dm,
    const std::vector<double>& x,
    const std::map<HYPRE_BigInt, double>& remote,
    HYPRE_BigInt g) {
  int j = -1;
  if (is_local_global_col_scalar_decomp(dm, g, j)) {
    return x[j];
  }

  auto it = remote.find(g);
  if (it != remote.end()) return it->second;

  // Should not happen for a correct processor-face pattern. Keep safe.
  return 0.0;
}

static std::vector<int> build_local_mcgs_colors_scalar_decomp(
    const DecompMesh& dm,
    const DistCSRPattern& pat,
    std::vector<int>& colorOffsets,
    std::vector<int>& colorCells) {
  const int n = pat.nRows;
  std::vector<int> color(n, -1);
  std::vector<int> mark(64, -1);

  int nColors = 0;
  int tag = 1;

  for (int i = 0; i < n; ++i) {
    ++tag;
    if (tag == 0x3fffffff) {
      std::fill(mark.begin(), mark.end(), -1);
      tag = 1;
    }

    for (int p = pat.rowOffsets[i]; p < pat.rowOffsets[i + 1]; ++p) {
      int j = -1;
      if (!is_local_global_col_scalar_decomp(dm, pat.cols[p], j)) continue;
      if (j == i) continue;

      const int cj = color[j];
      if (cj >= 0) {
        if (cj >= static_cast<int>(mark.size())) mark.resize(cj + 64, -1);
        mark[cj] = tag;
      }
    }

    int c = 0;
    while (c < nColors) {
      if (c >= static_cast<int>(mark.size())) mark.resize(c + 64, -1);
      if (mark[c] != tag) break;
      ++c;
    }

    color[i] = c;
    if (c == nColors) ++nColors;
  }

  long long localConflicts = 0;
  long long localEdges = 0;
  int localMaxDegree = 0;

  for (int i = 0; i < n; ++i) {
    int deg = 0;

    for (int p = pat.rowOffsets[i]; p < pat.rowOffsets[i + 1]; ++p) {
      int j = -1;
      if (!is_local_global_col_scalar_decomp(dm, pat.cols[p], j)) continue;
      if (j == i) continue;

      ++deg;
      if (color[i] == color[j]) ++localConflicts;
    }

    localMaxDegree = std::max(localMaxDegree, deg);
    localEdges += deg;
  }

  long long globalConflicts = 0;
  long long globalEdges = 0;
  int globalMaxDegree = 0;

  MPI_Allreduce(&localConflicts, &globalConflicts, 1, MPI_LONG_LONG, MPI_SUM, dm.comm);
  MPI_Allreduce(&localEdges, &globalEdges, 1, MPI_LONG_LONG, MPI_SUM, dm.comm);
  MPI_Allreduce(&localMaxDegree, &globalMaxDegree, 1, MPI_INT, MPI_MAX, dm.comm);

  if (globalConflicts != 0) {
    int r = 0;
    MPI_Comm_rank(dm.comm, &r);
    if (r == 0) {
      std::fprintf(stderr, "ERROR: host MCGS local coloring has %lld conflicts.\n", globalConflicts);
    }
    MPI_Abort(dm.comm, 3);
  }

  std::vector<int> counts(nColors, 0);
  for (int i = 0; i < n; ++i) counts[color[i]]++;

  colorOffsets.assign(nColors + 1, 0);
  for (int c = 0; c < nColors; ++c) colorOffsets[c + 1] = colorOffsets[c] + counts[c];

  std::vector<int> cursor = colorOffsets;
  colorCells.assign(n, 0);

  for (int i = 0; i < n; ++i) {
    const int c = color[i];
    colorCells[cursor[c]++] = i;
  }

  int localMinColorCount = n > 0 ? n : 0;
  int localMaxColorCount = 0;
  for (int c = 0; c < nColors; ++c) {
    localMinColorCount = std::min(localMinColorCount, counts[c]);
    localMaxColorCount = std::max(localMaxColorCount, counts[c]);
  }

  int globalMaxColors = 0;
  MPI_Allreduce(&nColors, &globalMaxColors, 1, MPI_INT, MPI_MAX, dm.comm);

  int rank = 0;
  MPI_Comm_rank(dm.comm, &rank);
  if (rank == 0) {
    std::printf("host MPI MCGS coloring: maxColorsAcrossRanks=%d localRank0Colors=%d "
                "rank0MinCells/color=%d rank0MaxCells/color=%d maxDegree=%d globalLocalEdges=%lld\n",
                globalMaxColors, nColors, localMinColorCount, localMaxColorCount,
                globalMaxDegree, globalEdges);
  }

  return color;
}

static double compute_residual_rel_host_mcgs_scalar_decomp(
    const DecompMesh& dm,
    const DistCSRPattern& pat,
    const std::vector<HYPRE_Complex>& values,
    const std::vector<HYPRE_Complex>& rhs,
    const std::vector<double>& x) {
  const auto remote = build_remote_value_map_scalar_decomp(dm, x);

  double localR2 = 0.0;
  double localB2 = 0.0;

  for (int i = 0; i < pat.nRows; ++i) {
    double Ax = 0.0;

    for (int p = pat.rowOffsets[i]; p < pat.rowOffsets[i + 1]; ++p) {
      const double a = static_cast<double>(values[p]);
      const HYPRE_BigInt g = pat.cols[p];
      Ax += a * value_for_global_col_scalar_decomp(dm, x, remote, g);
    }

    const double bi = static_cast<double>(rhs[i]);
    const double ri = bi - Ax;

    localR2 += ri * ri;
    localB2 += bi * bi;
  }

  double globalR2 = 0.0;
  double globalB2 = 0.0;

  MPI_Allreduce(&localR2, &globalR2, 1, MPI_DOUBLE, MPI_SUM, dm.comm);
  MPI_Allreduce(&localB2, &globalB2, 1, MPI_DOUBLE, MPI_SUM, dm.comm);

  return std::sqrt(globalR2 / std::max(globalB2, 1.0e-300));
}

static DistSolverInfoLocal solve_mcgs_host_decomp(
    const DecompMesh& dm,
    const DistCSRPattern& pat,
    const std::vector<HYPRE_Complex>& values,
    const std::vector<HYPRE_Complex>& rhs,
    const std::vector<double>& x0,
    const DistBiCGSTABOptions& opt) {
  if (static_cast<int>(rhs.size()) != pat.nRows) {
    throw std::runtime_error("solve_mcgs_host_decomp: rhs size mismatch");
  }

  std::vector<double> x(pat.nRows, 0.0);
  if (!x0.empty()) {
    if (static_cast<int>(x0.size()) != pat.nRows) {
      throw std::runtime_error("solve_mcgs_host_decomp: x0 size mismatch");
    }
    x = x0;
  }

  const int sweeps = std::max(opt.smootherSweeps, 0);
  const double omega = opt.smootherOmega;

  if (!(omega > 0.0)) {
    throw std::runtime_error("solve_mcgs_host_decomp: smootherOmega must be > 0");
  }

  std::vector<int> colorOffsets;
  std::vector<int> colorCells;
  build_local_mcgs_colors_scalar_decomp(dm, pat, colorOffsets, colorCells);

  const int nColors = static_cast<int>(colorOffsets.size()) - 1;

  double rel0 = -1.0;
  if (opt.smootherMonitor) {
    rel0 = compute_residual_rel_host_mcgs_scalar_decomp(dm, pat, values, rhs, x);
  }

  for (int sweep = 0; sweep < sweeps; ++sweep) {
    auto remote = build_remote_value_map_scalar_decomp(dm, x);

    for (int c = 0; c < nColors; ++c) {
      for (int kk = colorOffsets[c]; kk < colorOffsets[c + 1]; ++kk) {
        const int i = colorCells[kk];

        double diag = 0.0;
        double sumOff = 0.0;

        for (int p = pat.rowOffsets[i]; p < pat.rowOffsets[i + 1]; ++p) {
          const double a = static_cast<double>(values[p]);
          const HYPRE_BigInt g = pat.cols[p];

          int j = -1;
          if (is_local_global_col_scalar_decomp(dm, g, j)) {
            if (j == i) {
              diag = a;
            } else {
              sumOff += a * x[j];
            }
          } else {
            sumOff += a * value_for_global_col_scalar_decomp(dm, x, remote, g);
          }
        }

        if (std::abs(diag) > 1.0e-300) {
          const double gs = (static_cast<double>(rhs[i]) - sumOff) / diag;
          x[i] = (1.0 - omega) * x[i] + omega * gs;
        }
      }
    }

    if (opt.smootherSymmetric) {
      remote = build_remote_value_map_scalar_decomp(dm, x);

      for (int c = nColors - 1; c >= 0; --c) {
        for (int kk = colorOffsets[c]; kk < colorOffsets[c + 1]; ++kk) {
          const int i = colorCells[kk];

          double diag = 0.0;
          double sumOff = 0.0;

          for (int p = pat.rowOffsets[i]; p < pat.rowOffsets[i + 1]; ++p) {
            const double a = static_cast<double>(values[p]);
            const HYPRE_BigInt g = pat.cols[p];

            int j = -1;
            if (is_local_global_col_scalar_decomp(dm, g, j)) {
              if (j == i) {
                diag = a;
              } else {
                sumOff += a * x[j];
              }
            } else {
              sumOff += a * value_for_global_col_scalar_decomp(dm, x, remote, g);
            }
          }

          if (std::abs(diag) > 1.0e-300) {
            const double gs = (static_cast<double>(rhs[i]) - sumOff) / diag;
            x[i] = (1.0 - omega) * x[i] + omega * gs;
          }
        }
      }
    }
  }

  double rel = -1.0;
  if (opt.smootherMonitor) {
    rel = compute_residual_rel_host_mcgs_scalar_decomp(dm, pat, values, rhs, x);

    int rank = 0;
    MPI_Comm_rank(dm.comm, &rank);
    if (rank == 0) {
      std::printf("host MPI MCGS smoother: sweeps=%d omega=%.6g symmetric=%d rel0=%.6e rel=%.6e\n",
                  sweeps, omega, opt.smootherSymmetric, rel0, rel);
    }
  }

  DistSolverInfoLocal out;
  out.x = std::move(x);
  out.iterations = sweeps;
  out.relRes = rel;
  return out;
}


DistSolverInfoLocal solve_bicgstab_jacobi_decomp(
    const DecompMesh& dm,
    const DistCSRPattern& pat,
    const std::vector<HYPRE_Complex>& values,
    const std::vector<HYPRE_Complex>& rhs,
    const std::vector<double>& x0,
    const DistBiCGSTABOptions& opt) {
  {
    const std::string solverType = lower_solver_name_scalar_decomp(opt.solverType);
    if (solverType == "mcgs" ||
        solverType == "colored-gs" ||
        solverType == "multicolor-gs" ||
        solverType == "multi-color-gs" ||
        solverType == "mcgs-host" ||
        solverType == "host-mcgs") {
      return solve_mcgs_host_decomp(dm, pat, values, rhs, x0, opt);
    }
  }

  static bool hypreInitialized = false;

  if (!hypreInitialized) {
    HYPRE_CALL_SCALAR_DECOMP(HYPRE_Initialize());
#if defined(HYPRE_USING_GPU)
    HYPRE_CALL_SCALAR_DECOMP(HYPRE_DeviceInitialize());
    HYPRE_CALL_SCALAR_DECOMP(HYPRE_SetMemoryLocation(HYPRE_MEMORY_DEVICE));
    HYPRE_CALL_SCALAR_DECOMP(HYPRE_SetExecutionPolicy(HYPRE_EXEC_DEVICE));
#endif
    hypreInitialized = true;
  }

  HYPRE_IJMatrix Aij = nullptr;
  HYPRE_ParCSRMatrix Apar = nullptr;
  HYPRE_IJVector bij = nullptr, xij = nullptr;
  HYPRE_ParVector bpar = nullptr, xpar = nullptr;
  HYPRE_Solver solver = nullptr;

  HYPRE_CALL_SCALAR_DECOMP(HYPRE_IJMatrixCreate(dm.comm, dm.ilower, dm.iupper, dm.ilower, dm.iupper, &Aij));
  HYPRE_CALL_SCALAR_DECOMP(HYPRE_IJMatrixSetObjectType(Aij, HYPRE_PARCSR));
  HYPRE_CALL_SCALAR_DECOMP(HYPRE_IJMatrixSetRowSizes(Aij, const_cast<HYPRE_Int*>(pat.ncols.data())));
  HYPRE_CALL_SCALAR_DECOMP(HYPRE_IJMatrixInitialize_v2(Aij, HYPRE_MEMORY_HOST));
  HYPRE_CALL_SCALAR_DECOMP(HYPRE_IJMatrixSetValues(
      Aij,
      pat.nRows,
      const_cast<HYPRE_Int*>(pat.ncols.data()),
      const_cast<HYPRE_BigInt*>(pat.rows.data()),
      const_cast<HYPRE_BigInt*>(pat.cols.data()),
      const_cast<HYPRE_Complex*>(values.data())));
  HYPRE_CALL_SCALAR_DECOMP(HYPRE_IJMatrixAssemble(Aij));
#if defined(HYPRE_USING_GPU)
  HYPRE_CALL_SCALAR_DECOMP(HYPRE_IJMatrixMigrate(Aij, HYPRE_MEMORY_DEVICE));
#endif
  HYPRE_CALL_SCALAR_DECOMP(HYPRE_IJMatrixGetObject(Aij, reinterpret_cast<void**>(&Apar)));

  int hybridPreSweeps = 0;
  double hybridPreRel = -1.0;
  std::vector<double> krylovX0 = x0;

  {
    const std::string solverTypeHybrid = lower_solver_name_scalar_decomp(opt.solverType);
    const bool useMcgsThenKrylov =
        (solverTypeHybrid == "mcgs-bicgstab" ||
         solverTypeHybrid == "mcgs+bicgstab" ||
         solverTypeHybrid == "mcgs_then_bicgstab" ||
         solverTypeHybrid == "hybrid" ||
         solverTypeHybrid == "hybrid-bicgstab");

    if (useMcgsThenKrylov) {
      DistSolverInfoLocal pre =
          solve_mcgs_host_decomp(dm, pat, values, rhs, x0, opt);

      krylovX0 = std::move(pre.x);
      hybridPreSweeps = pre.iterations;
      hybridPreRel = pre.relRes;

      int rankPrint = 0;
      MPI_Comm_rank(dm.comm, &rankPrint);
      if (rankPrint == 0) {
        std::printf("hybrid momentum pre-smoother: MCGS sweeps=%d rel=%.6e, then BiCGSTAB+DiagScale\n",
                    hybridPreSweeps, hybridPreRel);
      }
    }
  }

  std::vector<HYPRE_Complex> xinit(pat.nRows, 0.0);
  if (!krylovX0.empty()) {
    if (static_cast<int>(krylovX0.size()) != pat.nRows) {
      throw std::runtime_error("krylovX0 must have size local nRows");
    }
    for (int i = 0; i < pat.nRows; ++i) xinit[i] = static_cast<HYPRE_Complex>(krylovX0[i]);
  }

  HYPRE_CALL_SCALAR_DECOMP(HYPRE_IJVectorCreate(dm.comm, dm.ilower, dm.iupper, &bij));
  HYPRE_CALL_SCALAR_DECOMP(HYPRE_IJVectorSetObjectType(bij, HYPRE_PARCSR));
  HYPRE_CALL_SCALAR_DECOMP(HYPRE_IJVectorInitialize_v2(bij, HYPRE_MEMORY_HOST));
  HYPRE_CALL_SCALAR_DECOMP(HYPRE_IJVectorSetValues(
      bij,
      pat.nRows,
      const_cast<HYPRE_BigInt*>(pat.rows.data()),
      const_cast<HYPRE_Complex*>(rhs.data())));
  HYPRE_CALL_SCALAR_DECOMP(HYPRE_IJVectorAssemble(bij));
#if defined(HYPRE_USING_GPU)
  HYPRE_CALL_SCALAR_DECOMP(HYPRE_IJVectorMigrate(bij, HYPRE_MEMORY_DEVICE));
#endif
  HYPRE_CALL_SCALAR_DECOMP(HYPRE_IJVectorGetObject(bij, reinterpret_cast<void**>(&bpar)));

  HYPRE_CALL_SCALAR_DECOMP(HYPRE_IJVectorCreate(dm.comm, dm.ilower, dm.iupper, &xij));
  HYPRE_CALL_SCALAR_DECOMP(HYPRE_IJVectorSetObjectType(xij, HYPRE_PARCSR));
  HYPRE_CALL_SCALAR_DECOMP(HYPRE_IJVectorInitialize_v2(xij, HYPRE_MEMORY_HOST));
  HYPRE_CALL_SCALAR_DECOMP(HYPRE_IJVectorSetValues(xij, pat.nRows, const_cast<HYPRE_BigInt*>(pat.rows.data()), xinit.data()));
  HYPRE_CALL_SCALAR_DECOMP(HYPRE_IJVectorAssemble(xij));
#if defined(HYPRE_USING_GPU)
  HYPRE_CALL_SCALAR_DECOMP(HYPRE_IJVectorMigrate(xij, HYPRE_MEMORY_DEVICE));
#endif
  HYPRE_CALL_SCALAR_DECOMP(HYPRE_IJVectorGetObject(xij, reinterpret_cast<void**>(&xpar)));

  std::string solverType = opt.solverType;
  for (char& c : solverType) c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));

  const bool useGMRES =
      (solverType == "gmres" || solverType == "fgmres" || solverType == "flexgmres");

  if (useGMRES) {
    HYPRE_CALL_SCALAR_DECOMP(HYPRE_ParCSRGMRESCreate(dm.comm, &solver));
    HYPRE_CALL_SCALAR_DECOMP(HYPRE_ParCSRGMRESSetTol(solver, std::max(opt.relTol, 0.0)));
    HYPRE_CALL_SCALAR_DECOMP(HYPRE_ParCSRGMRESSetAbsoluteTol(solver, std::max(opt.absTol, 0.0)));
    HYPRE_CALL_SCALAR_DECOMP(HYPRE_ParCSRGMRESSetMaxIter(solver, opt.maxIter));
    HYPRE_CALL_SCALAR_DECOMP(HYPRE_ParCSRGMRESSetKDim(solver, std::max(opt.gmresRestart, 2)));
    HYPRE_CALL_SCALAR_DECOMP(HYPRE_ParCSRGMRESSetPrintLevel(solver, opt.monitor ? 2 : 0));
    HYPRE_CALL_SCALAR_DECOMP(HYPRE_ParCSRGMRESSetLogging(solver, 1));

    HYPRE_CALL_SCALAR_DECOMP(HYPRE_ParCSRGMRESSetPrecond(
        solver,
        reinterpret_cast<HYPRE_PtrToParSolverFcn>(HYPRE_ParCSRDiagScale),
        reinterpret_cast<HYPRE_PtrToParSolverFcn>(HYPRE_ParCSRDiagScaleSetup),
        nullptr));

    HYPRE_CALL_SCALAR_DECOMP(HYPRE_ParCSRGMRESSetup(solver, Apar, bpar, xpar));
  } else {
    HYPRE_CALL_SCALAR_DECOMP(HYPRE_ParCSRBiCGSTABCreate(dm.comm, &solver));
    HYPRE_CALL_SCALAR_DECOMP(HYPRE_ParCSRBiCGSTABSetTol(solver, std::max(opt.relTol, 0.0)));
    HYPRE_CALL_SCALAR_DECOMP(HYPRE_ParCSRBiCGSTABSetAbsoluteTol(solver, std::max(opt.absTol, 0.0)));
    HYPRE_CALL_SCALAR_DECOMP(HYPRE_ParCSRBiCGSTABSetMaxIter(solver, opt.maxIter));
    HYPRE_CALL_SCALAR_DECOMP(HYPRE_ParCSRBiCGSTABSetPrintLevel(solver, opt.monitor ? 2 : 0));
    HYPRE_CALL_SCALAR_DECOMP(HYPRE_ParCSRBiCGSTABSetLogging(solver, 1));

    HYPRE_CALL_SCALAR_DECOMP(HYPRE_ParCSRBiCGSTABSetPrecond(
        solver,
        reinterpret_cast<HYPRE_PtrToParSolverFcn>(HYPRE_ParCSRDiagScale),
        reinterpret_cast<HYPRE_PtrToParSolverFcn>(HYPRE_ParCSRDiagScaleSetup),
        nullptr));

    HYPRE_CALL_SCALAR_DECOMP(HYPRE_ParCSRBiCGSTABSetup(solver, Apar, bpar, xpar));
  }

  HYPRE_Int solveIerr = useGMRES
      ? HYPRE_ParCSRGMRESSolve(solver, Apar, bpar, xpar)
      : HYPRE_ParCSRBiCGSTABSolve(solver, Apar, bpar, xpar);

  if (solveIerr) {
    int wrank = 0;
    MPI_Comm_rank(dm.comm, &wrank);
    std::fprintf(stderr,
                 "[%d] WARNING: HYPRE_ParCSR%sSolve returned code=%d; "
                 "continuing to inspect iterations/residual and current iterate.\n",
                 wrank, useGMRES ? "GMRES" : "BiCGSTAB", (int)solveIerr);
    HYPRE_ClearAllErrors();
  }

  HYPRE_Int its = 0;
  HYPRE_Real rel = 0.0;

  HYPRE_Int itsIerr = useGMRES
      ? HYPRE_ParCSRGMRESGetNumIterations(solver, &its)
      : HYPRE_ParCSRBiCGSTABGetNumIterations(solver, &its);

  if (itsIerr) {
    int wrank = 0;
    MPI_Comm_rank(dm.comm, &wrank);
    std::fprintf(stderr,
                 "[%d] WARNING: HYPRE_ParCSR%sGetNumIterations returned code=%d; "
                 "setting iterations=-1 and continuing.\n",
                 wrank, useGMRES ? "GMRES" : "BiCGSTAB", (int)itsIerr);
    its = -1;
    HYPRE_ClearAllErrors();
  }

  HYPRE_Int relIerr = useGMRES
      ? HYPRE_ParCSRGMRESGetFinalRelativeResidualNorm(solver, &rel)
      : HYPRE_ParCSRBiCGSTABGetFinalRelativeResidualNorm(solver, &rel);

  if (relIerr) {
    int wrank = 0;
    MPI_Comm_rank(dm.comm, &wrank);
    std::fprintf(stderr,
                 "[%d] WARNING: HYPRE_ParCSR%sGetFinalRelativeResidualNorm returned code=%d; "
                 "setting rel=1e300 and continuing.\n",
                 wrank, useGMRES ? "GMRES" : "BiCGSTAB", (int)relIerr);
    rel = 1.0e300;
    HYPRE_ClearAllErrors();
  }

#if defined(HYPRE_USING_GPU)
  HYPRE_CALL_SCALAR_DECOMP(HYPRE_IJVectorMigrate(xij, HYPRE_MEMORY_HOST));
#endif

  std::vector<HYPRE_Complex> xhost(pat.nRows, 0.0);
  HYPRE_CALL_SCALAR_DECOMP(HYPRE_IJVectorGetValues(
      xij,
      pat.nRows,
      const_cast<HYPRE_BigInt*>(pat.rows.data()),
      xhost.data()));

  DistSolverInfoLocal out;
  out.x.assign(pat.nRows, 0.0);
  for (int i = 0; i < pat.nRows; ++i) out.x[i] = static_cast<double>(xhost[i]);
  out.iterations = hybridPreSweeps + static_cast<int>(its);
  out.relRes = static_cast<double>(rel);

  if (solver) {
    if (useGMRES) {
      HYPRE_CALL_SCALAR_DECOMP(HYPRE_ParCSRGMRESDestroy(solver));
    } else {
      HYPRE_CALL_SCALAR_DECOMP(HYPRE_ParCSRBiCGSTABDestroy(solver));
    }
  }
  if (xij)    HYPRE_CALL_SCALAR_DECOMP(HYPRE_IJVectorDestroy(xij));
  if (bij)    HYPRE_CALL_SCALAR_DECOMP(HYPRE_IJVectorDestroy(bij));
  if (Aij)    HYPRE_CALL_SCALAR_DECOMP(HYPRE_IJMatrixDestroy(Aij));

  return out;
}


std::vector<double> apply_under_relaxation_and_extract_rAU_scalar_decomp(
    const DecompMesh& dm,
    const DistCSRPattern& pat,
    std::vector<HYPRE_Complex>& values,
    std::vector<HYPRE_Complex>& rhs,
    const std::vector<double>& phiOld,
    double underRelax,
    int rAUMode,
    double rAUScale) {
  const Mesh& mesh = dm.mesh;

  if (static_cast<int>(phiOld.size()) != mesh.nCells) {
    throw std::runtime_error("apply_under_relaxation_and_extract_rAU: phiOld size mismatch");
  }

  std::vector<double> rAU(mesh.nCells, 0.0);

  const double ur = std::max(underRelax, 1.0e-30);
  const double invRelax = 1.0 / ur;

  for (int c = 0; c < mesh.nCells; ++c) {
    const int diag = pat.diagPos[c];

    const double aPraw = static_cast<double>(values[diag]);

    if (std::abs(ur - 1.0) > 1.0e-15) {
      values[diag] = static_cast<HYPRE_Complex>(aPraw * invRelax);
      rhs[c] += static_cast<HYPRE_Complex>((invRelax - 1.0) * aPraw * phiOld[c]);
    }

    const double aPrelaxed = static_cast<double>(values[diag]);

    // v1.1b-compatible:
    // raw     rAU = V/aP_raw
    // relaxed rAU = V/aP_relaxed
    const double aForRAU = (rAUMode == 0) ? aPraw : aPrelaxed;

    rAU[c] = (std::abs(aForRAU) > 1.0e-300)
           ? rAUScale * mesh.vol[c] / aForRAU
           : 0.0;
  }

  return rAU;
}


} // namespace

DistScalarTransportResult solve_scalar_transport_decomp(
    const DecompMesh& dm,
    const DistScalarTransportInputs& in,
    const ScalarBCSet& bcSet,
    const DistScalarTransportOptions& opt,
    const std::vector<double>& x0) {
  const Mesh& mesh = dm.mesh;

  validate_inputs(dm, in);

  const LocalBoundaryFaceData bcFaceData = build_physical_boundary_face_data_scalar(dm, bcSet);
  const DistCSRPattern pat = build_decomp_scalar_pattern(dm);

  const bool includeNonOrth = (opt.diffusionScheme == DistDiffusionScheme::NonOrth);
  const int nOuter = includeNonOrth ? std::max(opt.nNonOrthCorr, 0) + 1 : 1;

  std::vector<double> phi(mesh.nCells, 0.0);
  if (!x0.empty()) {
    if (static_cast<int>(x0.size()) != mesh.nCells) {
      throw std::runtime_error("x0 must have size local mesh.nCells");
    }
    phi = x0;
  }

  std::vector<std::array<double,3>> grad(mesh.nCells, {0.0, 0.0, 0.0});
  std::vector<std::array<double,3>> remoteGrad(mesh.nFaces, {0.0, 0.0, 0.0});
  std::vector<HYPRE_Complex> values, rhs;
  std::vector<double> latestRAU(mesh.nCells, 0.0);
  DistSolverInfoLocal last;

  for (int outer = 0; outer < nOuter; ++outer) {
    compute_gradient_scalar_decomp(dm, phi, bcFaceData, opt.gradScheme, grad);
    remoteGrad = exchange_proc_face_vector_owner_values(dm, grad);

    assemble_scalar_transport_decomp(dm, pat, in, bcFaceData, grad, remoteGrad, opt, values, rhs);

    // Match serial simple_gpu momentum semantics:
    // apply equation under-relaxation to the assembled matrix and extract
    // rAU from the actual relaxed diagonal before the solve.
    latestRAU = apply_under_relaxation_and_extract_rAU_scalar_decomp(
        dm, pat, values, rhs, phi, opt.underRelax, opt.rAUMode, opt.rAUScale);

    last = solve_bicgstab_jacobi_decomp(dm, pat, values, rhs, phi, opt.solver);
    phi = last.x;
  }

  long long localNnz = static_cast<long long>(pat.nnz);
  long long globalNnz = 0;
  MPI_Allreduce(&localNnz, &globalNnz, 1, MPI_LONG_LONG, MPI_SUM, dm.comm);

  DistScalarTransportResult out;
  out.phi = std::move(phi);
  out.rAU = std::move(latestRAU);
  out.iterations = last.iterations;
  out.finalRelRes = last.relRes;
  out.nOuter = nOuter;
  out.globalNnz = globalNnz;
  return out;
}

} // namespace libscalar_decomp
