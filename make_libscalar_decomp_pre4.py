#!/usr/bin/env python3
from pathlib import Path

root = Path.cwd()

(root / "libscalar_decomp/include").mkdir(parents=True, exist_ok=True)
(root / "libscalar_decomp/src").mkdir(parents=True, exist_ok=True)
(root / "apps/scalar_transport_mms_decomp_gpu_mpi/src").mkdir(parents=True, exist_ok=True)

(root / "libscalar_decomp/include/scalar_decomp_library.h").write_text(r'''#pragma once

#include "poisson_decomp_library.h"

#include <string>
#include <vector>

namespace libscalar_decomp {

enum class DistConvectionScheme {
  Upwind,
  Central
};

DistConvectionScheme convection_scheme_from_string(const std::string& name);
const char* convection_scheme_name(DistConvectionScheme scheme);

enum class DistDiffusionScheme {
  Orth,
  NonOrth
};

DistDiffusionScheme diffusion_scheme_from_string(const std::string& name);
const char* diffusion_scheme_name(DistDiffusionScheme scheme);

struct DistBiCGSTABOptions {
  int maxIter = 1000;
  double relTol = 0.0;
  double absTol = 1e-8;
  int monitor = 0;
};

struct DistScalarTransportInputs {
  std::vector<double> faceFlux;   // size = local mesh.nFaces, oriented owner->neighbour/outward
  std::vector<double> gammaFace;  // size = local mesh.nFaces
  std::vector<double> Su;         // size = local mesh.nCells, source per volume
  std::vector<double> Sp;         // size = local mesh.nCells, implicit source per volume
};

struct DistScalarTransportOptions {
  DistConvectionScheme convectionScheme = DistConvectionScheme::Upwind;
  DistDiffusionScheme diffusionScheme = DistDiffusionScheme::NonOrth;

  std::string gradScheme = "lsq";
  int nNonOrthCorr = 2;

  DistBiCGSTABOptions solver;
};

struct DistScalarTransportResult {
  std::vector<double> phi;
  int iterations = 0;
  double finalRelRes = 0.0;
  int nOuter = 0;
  long long globalNnz = 0;
};

DistScalarTransportResult solve_scalar_transport_decomp(
    const DecompMesh& dm,
    const DistScalarTransportInputs& in,
    const ScalarBCSet& bcSet,
    const DistScalarTransportOptions& opt,
    const std::vector<double>& x0 = {});

} // namespace libscalar_decomp
''')

(root / "libscalar_decomp/src/scalar_decomp_library.cu").write_text(r'''#include "scalar_decomp_library.h"

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

DistSolverInfoLocal solve_bicgstab_jacobi_decomp(
    const DecompMesh& dm,
    const DistCSRPattern& pat,
    const std::vector<HYPRE_Complex>& values,
    const std::vector<HYPRE_Complex>& rhs,
    const std::vector<double>& x0,
    const DistBiCGSTABOptions& opt) {
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

  std::vector<HYPRE_Complex> xinit(pat.nRows, 0.0);
  if (!x0.empty()) {
    if (static_cast<int>(x0.size()) != pat.nRows) {
      throw std::runtime_error("x0 must have size local nRows");
    }
    for (int i = 0; i < pat.nRows; ++i) xinit[i] = static_cast<HYPRE_Complex>(x0[i]);
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
  HYPRE_CALL_SCALAR_DECOMP(HYPRE_ParCSRBiCGSTABSolve(solver, Apar, bpar, xpar));

  HYPRE_Int its = 0;
  HYPRE_Real rel = 0.0;
  HYPRE_CALL_SCALAR_DECOMP(HYPRE_ParCSRBiCGSTABGetNumIterations(solver, &its));
  HYPRE_CALL_SCALAR_DECOMP(HYPRE_ParCSRBiCGSTABGetFinalRelativeResidualNorm(solver, &rel));

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
  out.iterations = static_cast<int>(its);
  out.relRes = static_cast<double>(rel);

  if (solver) HYPRE_CALL_SCALAR_DECOMP(HYPRE_ParCSRBiCGSTABDestroy(solver));
  if (xij)    HYPRE_CALL_SCALAR_DECOMP(HYPRE_IJVectorDestroy(xij));
  if (bij)    HYPRE_CALL_SCALAR_DECOMP(HYPRE_IJVectorDestroy(bij));
  if (Aij)    HYPRE_CALL_SCALAR_DECOMP(HYPRE_IJMatrixDestroy(Aij));

  return out;
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
  DistSolverInfoLocal last;

  for (int outer = 0; outer < nOuter; ++outer) {
    compute_gradient_scalar_decomp(dm, phi, bcFaceData, opt.gradScheme, grad);
    remoteGrad = exchange_proc_face_vector_owner_values(dm, grad);

    assemble_scalar_transport_decomp(dm, pat, in, bcFaceData, grad, remoteGrad, opt, values, rhs);

    last = solve_bicgstab_jacobi_decomp(dm, pat, values, rhs, phi, opt.solver);
    phi = last.x;
  }

  long long localNnz = static_cast<long long>(pat.nnz);
  long long globalNnz = 0;
  MPI_Allreduce(&localNnz, &globalNnz, 1, MPI_LONG_LONG, MPI_SUM, dm.comm);

  DistScalarTransportResult out;
  out.phi = std::move(phi);
  out.iterations = last.iterations;
  out.finalRelRes = last.relRes;
  out.nOuter = nOuter;
  out.globalNnz = globalNnz;
  return out;
}

} // namespace libscalar_decomp
''')

(root / "libscalar_decomp/CMakeLists.txt").write_text(r'''add_library(libscalar_decomp STATIC
  src/scalar_decomp_library.cu
)

target_include_directories(libscalar_decomp PUBLIC
  ${CMAKE_CURRENT_SOURCE_DIR}/include
  ${CMAKE_SOURCE_DIR}/libpoisson/include
  ${CMAKE_SOURCE_DIR}/libpoisson_decomp/include
)

target_link_libraries(libscalar_decomp PUBLIC
  libpoisson_decomp
  libpoisson
  MPI::MPI_CXX
)

set_target_properties(libscalar_decomp PROPERTIES
  CUDA_SEPARABLE_COMPILATION ON
)
''')

(root / "apps/scalar_transport_mms_decomp_gpu_mpi/src/main.cu").write_text(r'''#include <mpi.h>
#include <cuda_runtime.h>

#include <array>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <set>
#include <stdexcept>
#include <string>
#include <vector>

#include "scalar_decomp_library.h"
#include "bc_runtime_config.h"

namespace {

constexpr double PI = 3.141592653589793238462643383279502884;

double phi_exact_scalar(const std::array<double,3>& x) {
  return std::sin(PI*x[0]) * std::sin(PI*x[1]) * std::sin(PI*x[2]);
}

std::array<double,3> grad_phi_exact_scalar(const std::array<double,3>& x) {
  return {
    PI * std::cos(PI*x[0]) * std::sin(PI*x[1]) * std::sin(PI*x[2]),
    PI * std::sin(PI*x[0]) * std::cos(PI*x[1]) * std::sin(PI*x[2]),
    PI * std::sin(PI*x[0]) * std::sin(PI*x[1]) * std::cos(PI*x[2])
  };
}

double source_scalar_mms(
    const std::array<double,3>& x,
    const std::array<double,3>& u,
    double gamma) {
  const double phi = phi_exact_scalar(x);
  const auto g = grad_phi_exact_scalar(x);

  const double adv = u[0]*g[0] + u[1]*g[1] + u[2]*g[2];
  const double diff = 3.0 * gamma * PI * PI * phi;

  return adv + diff;
}

ScalarBCSet make_exact_mms_dirichlet_bcs_for_physical_patches(const DecompMesh& dm) {
  std::set<std::string> procPatchNames;
  for (const auto& pp : dm.procPatches) procPatchNames.insert(pp.name);

  ScalarBCSet bc;
  for (const auto& name : dm.mesh.patchNames) {
    if (procPatchNames.count(name)) continue;

    bc.patches.push_back(make_dirichlet_patch_bc(
        name,
        [](const std::array<double,3>& x, const std::array<double,3>&) {
          return phi_exact_scalar(x);
        }));
  }

  return bc;
}

ScalarBCSet make_scalar_bcs_from_pressure_config(
    const DecompMesh& dm,
    const std::string& bcConfigPath) {
  auto cfg = pipebc::load_runtime_bc_config(bcConfigPath);
  pipebc::validate_runtime_bc_config_against_patches(cfg, dm.mesh.patchNames);

  const auto dups = pipebc::duplicate_pressure_bc_patches(cfg.pressurePatchSpecs);
  if (!dups.empty()) {
    throw std::runtime_error("Duplicate pressure/scalar BC for patch '" + dups.front() + "'");
  }

  ScalarBCSet bc;
  for (const auto& spec : cfg.pressurePatchSpecs) {
    if (spec.type == pipebc::PressureBCType::FixedValue) {
      bc.patches.push_back(make_dirichlet_constant_bc(spec.patchName, spec.value));
    } else if (spec.type == pipebc::PressureBCType::FixedValueFunction) {
      auto fn = spec.scalarFunction;
      bc.patches.push_back(make_dirichlet_patch_bc(
          spec.patchName,
          [fn](const std::array<double,3>& x, const std::array<double,3>&) {
            return fn ? fn(x, 0.0) : 0.0;
          }));
    } else if (spec.type == pipebc::PressureBCType::ZeroGradient ||
               spec.type == pipebc::PressureBCType::Open) {
      bc.patches.push_back(make_neumann_constant_bc(spec.patchName, 0.0));
    } else {
      throw std::runtime_error("Unsupported pressure BC type in scalar MMS config");
    }
  }

  return bc;
}

void cuda_check(cudaError_t e, const char* what) {
  if (e != cudaSuccess) {
    throw std::runtime_error(std::string(what) + ": " + cudaGetErrorString(e));
  }
}

} // namespace

int main(int argc, char** argv) {
  MPI_Init(&argc, &argv);

  int rank = 0, size = 1;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  try {
    std::string caseRoot = "/tmp/case_cube_0p03mm";
    std::string bcConfigPath;

    int device = rank;
    double tol = 1e-8;
    int maxit = 1000;

    std::string gradScheme = "lsq";
    std::string diffusionScheme = "nonorth";
    std::string convectionScheme = "upwind";
    int nNonOrthCorr = 4;

    double gamma = 0.01;
    std::array<double,3> u = {1.0, 0.0, 0.0};

    for (int i = 1; i < argc; ++i) {
      std::string a = argv[i];

      auto need = [&](const char* key) {
        if (i + 1 >= argc) {
          std::fprintf(stderr, "Missing value after %s\n", key);
          MPI_Abort(MPI_COMM_WORLD, 1);
        }
      };

      if (a == "-case-root") {
        need("-case-root");
        caseRoot = argv[++i];
      } else if (a == "-bc-config" || a == "-case-config") {
        need(a.c_str());
        bcConfigPath = argv[++i];
      } else if (a == "-device") {
        need("-device");
        device = std::atoi(argv[++i]);
      } else if (a == "-tol" || a == "-absTol" || a == "-scalar-tol") {
        need(a.c_str());
        tol = std::atof(argv[++i]);
      } else if (a == "-maxit" || a == "-scalar-maxit") {
        need(a.c_str());
        maxit = std::atoi(argv[++i]);
      } else if (a == "-grad-scheme") {
        need("-grad-scheme");
        gradScheme = argv[++i];
      } else if (a == "-diffusion-scheme" || a == "-laplacian-scheme") {
        need(a.c_str());
        diffusionScheme = argv[++i];
      } else if (a == "-convection-scheme") {
        need("-convection-scheme");
        convectionScheme = argv[++i];
      } else if (a == "-nNonOrthCorr") {
        need("-nNonOrthCorr");
        nNonOrthCorr = std::atoi(argv[++i]);
      } else if (a == "-gamma") {
        need("-gamma");
        gamma = std::atof(argv[++i]);
      } else if (a == "-ux") {
        need("-ux");
        u[0] = std::atof(argv[++i]);
      } else if (a == "-uy") {
        need("-uy");
        u[1] = std::atof(argv[++i]);
      } else if (a == "-uz") {
        need("-uz");
        u[2] = std::atof(argv[++i]);
      }
    }

    int devCount = 0;
    cuda_check(cudaGetDeviceCount(&devCount), "cudaGetDeviceCount");
    if (devCount > 0) cuda_check(cudaSetDevice(device % devCount), "cudaSetDevice");

    DecompMesh dm = read_decomposed_openfoam_case(caseRoot, MPI_COMM_WORLD);

    ScalarBCSet bcSet;
    if (!bcConfigPath.empty()) {
      bcSet = make_scalar_bcs_from_pressure_config(dm, bcConfigPath);
    } else {
      bcSet = make_exact_mms_dirichlet_bcs_for_physical_patches(dm);
    }

    libscalar_decomp::DistScalarTransportInputs in;
    in.faceFlux.assign(dm.mesh.nFaces, 0.0);
    in.gammaFace.assign(dm.mesh.nFaces, gamma);
    in.Su.assign(dm.mesh.nCells, 0.0);
    in.Sp.assign(dm.mesh.nCells, 0.0);

    for (int f = 0; f < dm.mesh.nFaces; ++f) {
      in.faceFlux[f] =
          u[0] * dm.mesh.Sf[f][0] +
          u[1] * dm.mesh.Sf[f][1] +
          u[2] * dm.mesh.Sf[f][2];
    }

    for (int c = 0; c < dm.mesh.nCells; ++c) {
      in.Su[c] = source_scalar_mms(dm.mesh.cc[c], u, gamma);
    }

    libscalar_decomp::DistScalarTransportOptions opt;
    opt.convectionScheme = libscalar_decomp::convection_scheme_from_string(convectionScheme);
    opt.diffusionScheme = libscalar_decomp::diffusion_scheme_from_string(diffusionScheme);
    opt.gradScheme = gradScheme;
    opt.nNonOrthCorr = nNonOrthCorr;
    opt.solver.maxIter = maxit;
    opt.solver.absTol = tol;
    opt.solver.relTol = 0.0;
    opt.solver.monitor = 0;

    MPI_Barrier(MPI_COMM_WORLD);
    const double t0 = MPI_Wtime();

    auto result = libscalar_decomp::solve_scalar_transport_decomp(dm, in, bcSet, opt);

    MPI_Barrier(MPI_COMM_WORLD);
    const double t1 = MPI_Wtime();

    double localL2 = 0.0;
    double localInf = 0.0;
    double localVol = 0.0;

    for (int c = 0; c < dm.mesh.nCells; ++c) {
      const double exact = phi_exact_scalar(dm.mesh.cc[c]);
      const double err = std::abs(result.phi[c] - exact);
      localL2 += err * err * dm.mesh.vol[c];
      localInf = std::max(localInf, err);
      localVol += dm.mesh.vol[c];
    }

    double globalL2 = 0.0;
    double globalInf = 0.0;
    double globalVol = 0.0;

    MPI_Allreduce(&localL2, &globalL2, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    MPI_Allreduce(&localInf, &globalInf, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
    MPI_Allreduce(&localVol, &globalVol, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

    globalL2 = std::sqrt(globalL2 / std::max(globalVol, 1.0e-300));

    double wall = t1 - t0;
    double maxWall = 0.0;
    MPI_Reduce(&wall, &maxWall, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    DistCSRPattern pat = build_decomp_scalar_pattern(dm);

    std::printf("rank %d/%d rows=[%lld,%lld] nLocal=%d nnz=%d internalFaces=%d procFaces=%zu boundaryFaces=%d\n",
                rank, size,
                (long long)dm.ilower, (long long)dm.iupper,
                dm.nLocal,
                pat.nnz,
                dm.mesh.nInternalFaces,
                pat.procFace.size(),
                dm.mesh.nFaces - dm.mesh.nInternalFaces - (int)pat.procFace.size());
    std::fflush(stdout);

    if (rank == 0) {
      std::printf("SCALAR_DECOMP_MMS setup: ranks=%d globalRows=%lld globalNnz=%lld bcConfig=%s grad=%s diffusion=%s convection=%s nNonOrthCorr=%d nOuter=%d gamma=%.6e U=(%.6e %.6e %.6e)\n",
                  size,
                  (long long)dm.globalN,
                  result.globalNnz,
                  bcConfigPath.empty() ? "<exact-mms-default>" : bcConfigPath.c_str(),
                  gradScheme.c_str(),
                  diffusionScheme.c_str(),
                  convectionScheme.c_str(),
                  nNonOrthCorr,
                  result.nOuter,
                  gamma,
                  u[0], u[1], u[2]);

      std::printf("SCALAR_DECOMP_MMS RESULT: its=%d finalRel=%.12e L2=%.12e Linf=%.12e wall=%.6e s\n",
                  result.iterations,
                  result.finalRelRes,
                  globalL2,
                  globalInf,
                  maxWall);

      if (std::isfinite(globalL2) && std::isfinite(globalInf)) {
        std::printf("SCALAR_DECOMP_MMS PASS_RAN\n");
      } else {
        std::printf("SCALAR_DECOMP_MMS FAIL_NAN_INF\n");
      }
    }

    MPI_Finalize();
    return 0;
  } catch (const std::exception& e) {
    std::fprintf(stderr, "rank %d ERROR: %s\n", rank, e.what());
    MPI_Abort(MPI_COMM_WORLD, 1);
  }

  MPI_Finalize();
  return 1;
}
''')

(root / "apps/scalar_transport_mms_decomp_gpu_mpi/CMakeLists.txt").write_text(r'''add_executable(scalar_transport_mms_decomp_gpu_mpi
  src/main.cu
  ${CMAKE_SOURCE_DIR}/apps/simple_gpu/src/bc_specs.cu
  ${CMAKE_SOURCE_DIR}/apps/simple_gpu/src/bc_runtime_config.cu
)

target_include_directories(scalar_transport_mms_decomp_gpu_mpi PRIVATE
  ${CMAKE_SOURCE_DIR}/libpoisson/include
  ${CMAKE_SOURCE_DIR}/libpoisson_decomp/include
  ${CMAKE_SOURCE_DIR}/libscalar_decomp/include
  ${CMAKE_SOURCE_DIR}/apps/simple_gpu/src
)

target_link_libraries(scalar_transport_mms_decomp_gpu_mpi PRIVATE
  libscalar_decomp
  libpoisson_decomp
  libpoisson
  MPI::MPI_CXX
)

set_target_properties(scalar_transport_mms_decomp_gpu_mpi PROPERTIES
  CUDA_SEPARABLE_COMPILATION ON
)
''')

cmake = root / "CMakeLists.txt"
txt = cmake.read_text()

if "add_subdirectory(libscalar_decomp)" not in txt:
    if "add_subdirectory(libpoisson_decomp)" in txt:
      txt = txt.replace("add_subdirectory(libpoisson_decomp)\n", "add_subdirectory(libpoisson_decomp)\nadd_subdirectory(libscalar_decomp)\n")
    else:
      txt += "\nadd_subdirectory(libscalar_decomp)\n"

if "add_subdirectory(apps/scalar_transport_mms_decomp_gpu_mpi)" not in txt:
    txt += "\nadd_subdirectory(apps/scalar_transport_mms_decomp_gpu_mpi)\n"

cmake.write_text(txt)

print("OK: created libscalar_decomp and scalar_transport_mms_decomp_gpu_mpi")
