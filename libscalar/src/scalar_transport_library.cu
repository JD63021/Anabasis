#include "scalar_transport_library.h"

#include <algorithm>
#include <array>
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

#define HYPRE_CALL(stmt) do { \
  HYPRE_Int _ierr = (stmt); \
  if (_ierr) { \
    int _rank = 0; MPI_Comm_rank(MPI_COMM_WORLD, &_rank); \
    std::fprintf(stderr, "[%d] HYPRE ERROR %s:%d code=%d\n", _rank, __FILE__, __LINE__, (int)_ierr); \
    MPI_Abort(MPI_COMM_WORLD, (int)_ierr); \
  } \
} while (0)

namespace libscalar {
namespace {

static inline std::array<double,3> add3(const std::array<double,3>& a, const std::array<double,3>& b) {
  return {a[0]+b[0], a[1]+b[1], a[2]+b[2]};
}
static inline std::array<double,3> sub3(const std::array<double,3>& a, const std::array<double,3>& b) {
  return {a[0]-b[0], a[1]-b[1], a[2]-b[2]};
}
static inline std::array<double,3> mul3(double s, const std::array<double,3>& a) {
  return {s*a[0], s*a[1], s*a[2]};
}
static inline double dot3(const std::array<double,3>& a, const std::array<double,3>& b) {
  return a[0]*b[0] + a[1]*b[1] + a[2]*b[2];
}
static inline double norm3(const std::array<double,3>& a) {
  return std::sqrt(dot3(a,a));
}

struct BoundaryFaceData {
  std::vector<ScalarBCType> type; // size nFaces
  std::vector<double> value;      // Dirichlet value or prescribed normal gradient
};

struct CSRPattern {
  int nRows = 0;
  int nnz = 0;
  std::vector<HYPRE_Int> ncols;
  std::vector<int> rowOffsets;
  std::vector<int> diagPos;
  std::vector<HYPRE_BigInt> rows;
  std::vector<HYPRE_BigInt> cols;
  std::vector<int> facePP, facePN, faceNP, faceNN;
};

struct SolverInfo {
  std::vector<double> x;
  int iterations = 0;
  double relRes = 0.0;
};

BoundaryFaceData build_boundary_face_data(const Mesh& mesh, const ScalarBCSet& bcSet) {
  BoundaryFaceData out;
  out.type.assign(mesh.nFaces, ScalarBCType::ZeroGradient);
  out.value.assign(mesh.nFaces, 0.0);

  std::map<std::string, ScalarPatchBC> byName;
  for (const auto& bc : bcSet.patches) byName[bc.patchName] = bc;

  const int nPatches = static_cast<int>(mesh.patchNames.size());
  int minRaw =  999999999;
  int maxRaw = -999999999;

  for (int f = mesh.nInternalFaces; f < mesh.nFaces; ++f) {
    if (f < 0 || f >= static_cast<int>(mesh.bPatch.size())) continue;
    minRaw = std::min(minRaw, mesh.bPatch[f]);
    maxRaw = std::max(maxRaw, mesh.bPatch[f]);
  }

  int patchBase = 0;
  if (nPatches > 0) {
    if (minRaw == 0 && maxRaw == nPatches - 1) {
      patchBase = 0; // zero-based
    } else if (minRaw == 1 && maxRaw == nPatches) {
      patchBase = 1; // one-based
    } else {
      // fallback: if everything is shifted positive by one, assume one-based
      if (minRaw >= 1 && maxRaw <= nPatches) patchBase = 1;
    }
  }

  for (int f = mesh.nInternalFaces; f < mesh.nFaces; ++f) {
    if (f < 0 || f >= static_cast<int>(mesh.bPatch.size())) {
      throw std::runtime_error("Boundary face patch array too small");
    }

    const int rawPatch = mesh.bPatch[f];
    const int patch = rawPatch - patchBase;

    if (patch < 0 || patch >= nPatches) {
      throw std::runtime_error(
          "Boundary face has invalid patch index: raw=" + std::to_string(rawPatch) +
          ", adjusted=" + std::to_string(patch) +
          ", nPatches=" + std::to_string(nPatches));
    }

    const std::string& patchName = mesh.patchNames[patch];
    auto it = byName.find(patchName);
    if (it == byName.end()) {
      throw std::runtime_error("No scalar BC provided for patch '" + patchName + "'");
    }

    out.type[f] = it->second.type;
    out.value[f] = it->second.evaluator(mesh.xf[f], mesh.nf[f]);
  }

  return out;
}

double face_interp_lambda(const Mesh& mesh, int f) {
  const int P = mesh.owner[f];
  const int N = mesh.neigh[f];
  const auto d  = sub3(mesh.cc[N], mesh.cc[P]);
  const auto dx = sub3(mesh.xf[f], mesh.cc[P]);
  const double lam = dot3(dx, d) / std::max(dot3(d, d), 1.0e-30);
  return std::min(1.0, std::max(0.0, lam));
}

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
    for (int j = 0; j < static_cast<int>(cols.size()); ++j) {
      pos[c][cols[j]] = pat.rowOffsets[c] + j;
    }
  }

  pat.nnz = pat.rowOffsets.back();
  pat.cols.resize(pat.nnz);

  for (int c = 0; c < mesh.nCells; ++c) {
    for (const auto& kv : pos[c]) pat.cols[kv.second] = static_cast<HYPRE_BigInt>(kv.first);
    pat.diagPos[c] = pos[c][c];
  }

  for (int f = 0; f < mesh.nInternalFaces; ++f) {
    const int P = mesh.owner[f];
    const int N = mesh.neigh[f];
    pat.facePP[f] = pos[P][P];
    pat.facePN[f] = pos[P][N];
    pat.faceNP[f] = pos[N][P];
    pat.faceNN[f] = pos[N][N];
  }

  return pat;
}

void compute_lsq_gradient(
    const Mesh& mesh,
    const std::vector<double>& phi,
    const BoundaryFaceData& bcFaceData,
    std::vector<std::array<double,3>>& grad) {

  grad.assign(mesh.nCells, {0.0, 0.0, 0.0});

  for (int P = 0; P < mesh.nCells; ++P) {
    const auto xP = mesh.cc[P];
    const double phiP = phi[P];

    double M[3][3] = {{0,0,0},{0,0,0},{0,0,0}};
    double rhs[3]  = {0,0,0};

    for (int N : mesh.cellNbrs[P]) {
      const auto r = sub3(mesh.cc[N], xP);
      const double dphi = phi[N] - phiP;
      const double w = 1.0 / std::max(dot3(r,r), 1.0e-30);
      for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) M[i][j] += w * r[i] * r[j];
        rhs[i] += w * r[i] * dphi;
      }
    }

    for (int f : mesh.cellBFace[P]) {
      const auto rcf = sub3(mesh.xf[f], xP);
      std::array<double,3> r{0.0, 0.0, 0.0};
      double dphi = 0.0;

      if (bcFaceData.type[f] == ScalarBCType::Dirichlet) {
        r = rcf;
        dphi = bcFaceData.value[f] - phiP;
      } else {
        const double dn = std::max(dot3(rcf, mesh.nf[f]), 1.0e-30);
        r = mul3(dn, mesh.nf[f]);
        dphi = bcFaceData.value[f] * dn; // prescribed dphi/dn * dn
      }

      const double w = 1.0 / std::max(dot3(r,r), 1.0e-30);
      for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) M[i][j] += w * r[i] * r[j];
        rhs[i] += w * r[i] * dphi;
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

void assemble_scalar_transport_system(
    const Mesh& mesh,
    const CSRPattern& pat,
    const ScalarTransportInputs& in,
    const BoundaryFaceData& bcFaceData,
    const std::vector<std::array<double,3>>& grad,
    const ScalarTransportOptions& opt,
    std::vector<HYPRE_Complex>& values,
    std::vector<HYPRE_Complex>& rhs) {

  if (static_cast<int>(in.faceFlux.size()) != mesh.nFaces) {
    throw std::runtime_error("faceFlux must have size mesh.nFaces");
  }
  if (static_cast<int>(in.gammaFace.size()) != mesh.nFaces) {
    throw std::runtime_error("gammaFace must have size mesh.nFaces");
  }
  if (static_cast<int>(in.Su.size()) != mesh.nCells) {
    throw std::runtime_error("Su must have size mesh.nCells");
  }
  if (static_cast<int>(in.Sp.size()) != mesh.nCells) {
    throw std::runtime_error("Sp must have size mesh.nCells");
  }

  values.assign(pat.nnz, 0.0);
  rhs.assign(mesh.nCells, 0.0);

  for (int c = 0; c < mesh.nCells; ++c) {
    rhs[c] = static_cast<HYPRE_Complex>(in.Su[c] * mesh.vol[c]);
    values[pat.diagPos[c]] += static_cast<HYPRE_Complex>(-in.Sp[c] * mesh.vol[c]);
  }

  const bool includeNonOrth = (opt.diffusionScheme == DiffusionScheme::NonOrth);

  for (int f = 0; f < mesh.nInternalFaces; ++f) {
    const int P = mesh.owner[f];
    const int N = mesh.neigh[f];

    const auto d = sub3(mesh.cc[N], mesh.cc[P]);
    const double dDotS = dot3(d, mesh.Sf[f]);
    const double gamma = in.gammaFace[f];
    const double D = gamma * dot3(mesh.Sf[f], mesh.Sf[f]) / std::max(dDotS, 1.0e-30);

    values[pat.facePP[f]] += D;
    values[pat.facePN[f]] -= D;
    values[pat.faceNP[f]] -= D;
    values[pat.faceNN[f]] += D;

    if (includeNonOrth) {
      const auto T = sub3(mesh.Sf[f], mul3(D / std::max(gamma, 1.0e-30), d));
      const double lam = face_interp_lambda(mesh, f);
      const auto gradF = add3(mul3(1.0 - lam, grad[P]), mul3(lam, grad[N]));
      const double corr = gamma * dot3(T, gradF);
      rhs[P] += static_cast<HYPRE_Complex>(corr);
      rhs[N] -= static_cast<HYPRE_Complex>(corr);
    }

    const double F = in.faceFlux[f];

    if (opt.convectionScheme == ConvectionScheme::Central) {
      const double lam = face_interp_lambda(mesh, f);
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

  for (int f = mesh.nInternalFaces; f < mesh.nFaces; ++f) {
    const int P = mesh.owner[f];
    const auto d = sub3(mesh.xf[f], mesh.cc[P]);
    const double dDotS = dot3(d, mesh.Sf[f]);
    const double gamma = in.gammaFace[f];
    const double D = gamma * dot3(mesh.Sf[f], mesh.Sf[f]) / std::max(dDotS, 1.0e-30);
    const double F = in.faceFlux[f];

    if (bcFaceData.type[f] == ScalarBCType::Dirichlet) {
      const double phiB = bcFaceData.value[f];

      values[pat.diagPos[P]] += D;
      rhs[P] += static_cast<HYPRE_Complex>(D * phiB);

      if (includeNonOrth) {
        const auto T = sub3(mesh.Sf[f], mul3(D / std::max(gamma, 1.0e-30), d));
        const double corr = gamma * dot3(T, grad[P]);
        rhs[P] += static_cast<HYPRE_Complex>(corr);
      }

      rhs[P] += static_cast<HYPRE_Complex>(-F * phiB);
    } else {
      const double gradn = (bcFaceData.type[f] == ScalarBCType::ZeroGradient) ? 0.0 : bcFaceData.value[f];
      rhs[P] += static_cast<HYPRE_Complex>( gamma * gradn * mesh.Af[f]);

      // convective outlet-like treatment: phi_f = phi_P
      values[pat.diagPos[P]] += static_cast<HYPRE_Complex>(F);
    }
  }
}

SolverInfo solve_bicgstab_diagscale_gpu(
    const CSRPattern& pat,
    const std::vector<HYPRE_Complex>& values,
    const std::vector<HYPRE_Complex>& rhs,
    const std::vector<double>& x0,
    const ScalarTransportOptions& opt) {

  if (opt.linearSolver != LinearSolverType::BiCGSTAB) {
    throw std::runtime_error("Only BiCGSTAB is implemented in this libscalar build.");
  }
  if (opt.preconditioner != PreconditionerType::Jacobi) {
    throw std::runtime_error("Only Jacobi/DiagScale is implemented in this libscalar build.");
  }

  // Temporary robust path:
  // force host-memory Hypre for the standalone scalar library solve.
  // This avoids the GPU IJVector get-values crash in the current handwritten path.
  HYPRE_SetMemoryLocation(HYPRE_MEMORY_HOST);
  HYPRE_SetExecutionPolicy(HYPRE_EXEC_HOST);

  HYPRE_IJMatrix Aij;
  HYPRE_ParCSRMatrix Apar;
  HYPRE_IJVector bij, xij;
  HYPRE_ParVector bpar, xpar;
  HYPRE_Solver solver;

  const HYPRE_BigInt ilower = 0;
  const HYPRE_BigInt iupper = static_cast<HYPRE_BigInt>(pat.nRows - 1);

  HYPRE_CALL(HYPRE_IJMatrixCreate(MPI_COMM_WORLD, ilower, iupper, ilower, iupper, &Aij));
  HYPRE_CALL(HYPRE_IJMatrixSetObjectType(Aij, HYPRE_PARCSR));
  HYPRE_CALL(HYPRE_IJMatrixInitialize(Aij));
  HYPRE_CALL(HYPRE_IJMatrixSetValues(
      Aij,
      pat.nRows,
      const_cast<HYPRE_Int*>(pat.ncols.data()),
      const_cast<HYPRE_BigInt*>(pat.rows.data()),
      const_cast<HYPRE_BigInt*>(pat.cols.data()),
      const_cast<HYPRE_Complex*>(values.data())));
  HYPRE_CALL(HYPRE_IJMatrixAssemble(Aij));
  HYPRE_CALL(HYPRE_IJMatrixGetObject(Aij, reinterpret_cast<void**>(&Apar)));

  HYPRE_CALL(HYPRE_IJVectorCreate(MPI_COMM_WORLD, ilower, iupper, &bij));
  HYPRE_CALL(HYPRE_IJVectorSetObjectType(bij, HYPRE_PARCSR));
  HYPRE_CALL(HYPRE_IJVectorInitialize(bij));

  std::vector<HYPRE_BigInt> rows(pat.nRows);
  for (int i = 0; i < pat.nRows; ++i) rows[i] = static_cast<HYPRE_BigInt>(i);

  HYPRE_CALL(HYPRE_IJVectorSetValues(
      bij,
      pat.nRows,
      rows.data(),
      const_cast<HYPRE_Complex*>(rhs.data())));
  HYPRE_CALL(HYPRE_IJVectorAssemble(bij));
  HYPRE_CALL(HYPRE_IJVectorGetObject(bij, reinterpret_cast<void**>(&bpar)));

  HYPRE_CALL(HYPRE_IJVectorCreate(MPI_COMM_WORLD, ilower, iupper, &xij));
  HYPRE_CALL(HYPRE_IJVectorSetObjectType(xij, HYPRE_PARCSR));
  HYPRE_CALL(HYPRE_IJVectorInitialize(xij));

  std::vector<HYPRE_Complex> xinit(pat.nRows, 0.0);
  if (!x0.empty()) {
    if (static_cast<int>(x0.size()) != pat.nRows) {
      throw std::runtime_error("x0 must have size nRows");
    }
    for (int i = 0; i < pat.nRows; ++i) xinit[i] = static_cast<HYPRE_Complex>(x0[i]);
  }

  HYPRE_CALL(HYPRE_IJVectorSetValues(xij, pat.nRows, rows.data(), xinit.data()));
  HYPRE_CALL(HYPRE_IJVectorAssemble(xij));
  HYPRE_CALL(HYPRE_IJVectorGetObject(xij, reinterpret_cast<void**>(&xpar)));

  HYPRE_CALL(HYPRE_ParCSRBiCGSTABCreate(MPI_COMM_WORLD, &solver));
  HYPRE_CALL(HYPRE_ParCSRBiCGSTABSetTol(solver, std::max(opt.relTol, 0.0)));
  HYPRE_CALL(HYPRE_ParCSRBiCGSTABSetAbsoluteTol(solver, std::max(opt.absTol, 0.0)));
  HYPRE_CALL(HYPRE_ParCSRBiCGSTABSetMaxIter(solver, opt.maxIter));
  HYPRE_CALL(HYPRE_ParCSRBiCGSTABSetPrintLevel(solver, opt.monitor ? 2 : 0));
  HYPRE_CALL(HYPRE_ParCSRBiCGSTABSetLogging(solver, 1));
  HYPRE_CALL(HYPRE_ParCSRBiCGSTABSetPrecond(
      solver,
      (HYPRE_PtrToParSolverFcn)HYPRE_ParCSRDiagScale,
      (HYPRE_PtrToParSolverFcn)HYPRE_ParCSRDiagScaleSetup,
      nullptr));

  HYPRE_CALL(HYPRE_ParCSRBiCGSTABSetup(solver, Apar, bpar, xpar));
  HYPRE_CALL(HYPRE_ParCSRBiCGSTABSolve(solver, Apar, bpar, xpar));

  HYPRE_Int its = 0;
  HYPRE_Real relres = 0.0;
  HYPRE_CALL(HYPRE_ParCSRBiCGSTABGetNumIterations(solver, &its));
  HYPRE_CALL(HYPRE_ParCSRBiCGSTABGetFinalRelativeResidualNorm(solver, &relres));

  std::vector<HYPRE_Complex> xvals(pat.nRows, 0.0);
  HYPRE_CALL(HYPRE_IJVectorGetValues(xij, pat.nRows, rows.data(), xvals.data()));

  SolverInfo out;
  out.x.resize(pat.nRows);
  for (int i = 0; i < pat.nRows; ++i) out.x[i] = static_cast<double>(xvals[i]);
  out.iterations = static_cast<int>(its);
  out.relRes = static_cast<double>(relres);

  HYPRE_CALL(HYPRE_ParCSRBiCGSTABDestroy(solver));
  HYPRE_CALL(HYPRE_IJVectorDestroy(xij));
  HYPRE_CALL(HYPRE_IJVectorDestroy(bij));
  HYPRE_CALL(HYPRE_IJMatrixDestroy(Aij));

  // Restore device mode for the rest of the codebase.
  HYPRE_SetExecutionPolicy(HYPRE_EXEC_DEVICE);
  HYPRE_SetMemoryLocation(HYPRE_MEMORY_DEVICE);

  return out;
}

} // namespace

ScalarTransportResult solve_steady_scalar_transport(
    const Mesh& mesh,
    const ScalarTransportInputs& in,
    const ScalarBCSet& bcSet,
    const ScalarTransportOptions& opt,
    const std::vector<double>& x0) {

  const auto bcFaceData = build_boundary_face_data(mesh, bcSet);
  const CSRPattern pat = build_scalar_pattern(mesh);

  const bool includeNonOrth = (opt.diffusionScheme == DiffusionScheme::NonOrth);
  const int nOuter = includeNonOrth ? (std::max(opt.nNonOrthCorr, 0) + 1) : 1;

  std::vector<double> phi(mesh.nCells, 0.0);
  if (!x0.empty()) {
    if (static_cast<int>(x0.size()) != mesh.nCells) {
      throw std::runtime_error("x0 must have size mesh.nCells");
    }
    phi = x0;
  }

  std::vector<std::array<double,3>> grad(mesh.nCells, {0.0, 0.0, 0.0});
  std::vector<HYPRE_Complex> values, rhs;
  SolverInfo last;

  for (int outer = 0; outer < nOuter; ++outer) {
    compute_lsq_gradient(mesh, phi, bcFaceData, grad);
    assemble_scalar_transport_system(mesh, pat, in, bcFaceData, grad, opt, values, rhs);
    last = solve_bicgstab_diagscale_gpu(pat, values, rhs, phi, opt);
    phi = last.x;
  }

  ScalarTransportResult out;
  out.phi = std::move(phi);
  out.iterations = last.iterations;
  out.finalRelRes = last.relRes;
  return out;
}

} // namespace libscalar
