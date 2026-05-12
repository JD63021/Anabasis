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
