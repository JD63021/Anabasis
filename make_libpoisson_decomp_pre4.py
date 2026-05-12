#!/usr/bin/env python3
from pathlib import Path

root = Path.cwd()

(root / "libpoisson_decomp/include").mkdir(parents=True, exist_ok=True)
(root / "libpoisson_decomp/src").mkdir(parents=True, exist_ok=True)
(root / "apps/poisson_mms_decomp_gpu_mpi/src").mkdir(parents=True, exist_ok=True)

(root / "libpoisson_decomp/include/poisson_decomp_library.h").write_text(r'''#pragma once

#include "poisson_library.h"

#include <mpi.h>

struct ProcPatchDecomp {
  std::string name;
  int nFaces = 0;
  int startFace = 0;
  int myProcNo = -1;
  int neighbProcNo = -1;
};

struct DecompMesh {
  Mesh mesh;
  MPI_Comm comm = MPI_COMM_WORLD;

  int rank = 0;
  int size = 1;
  int nLocal = 0;

  std::vector<int> counts;
  std::vector<HYPRE_BigInt> offsets;

  HYPRE_BigInt ilower = 0;
  HYPRE_BigInt iupper = -1;
  HYPRE_BigInt globalN = 0;

  std::vector<ProcPatchDecomp> procPatches;
  std::vector<char> isProcFace;

  std::vector<HYPRE_BigInt> remoteRowForFace;
  std::vector<std::array<double,3>> remoteCCForFace;
};

struct DistCSRPattern {
  int nRows = 0;
  int nnz = 0;

  std::vector<HYPRE_Int> ncols;
  std::vector<int> rowOffsets;
  std::vector<int> diagPos;

  std::vector<HYPRE_BigInt> rows;
  std::vector<HYPRE_BigInt> cols;

  std::vector<int> facePP, facePN, faceNP, faceNN;

  std::vector<int> procFace;
  std::vector<int> procOwner;
  std::vector<int> procDiag;
  std::vector<int> procOff;
};

struct DistEllipticOptions {
  int nNonOrthCorr = 2;
  std::string gradScheme = "lsq";
  std::string laplacianScheme = "nonorth";

  bool useReferenceCell = false;
  HYPRE_BigInt referenceGlobalCell = 0;
  double referenceValue = 0.0;

  HypreOptions hypre;
};

struct DistEllipticResult {
  std::vector<double> phi;
  HypreSolveInfo lastSolveInfo;

  int nOuter = 0;
  long long globalNnz = 0;

  double l2 = 0.0;
  double linf = 0.0;
};

DecompMesh read_decomposed_openfoam_case(
    const std::string& caseRoot,
    MPI_Comm comm = MPI_COMM_WORLD);

DistCSRPattern build_decomp_scalar_pattern(const DecompMesh& dm);

DistEllipticResult solve_scalar_elliptic_decomp(
    const DecompMesh& dm,
    const std::vector<double>& gammaFace,
    const std::vector<double>& cellSource,
    const ScalarBCSet& bcSet,
    const DistEllipticOptions& opts);

inline DistEllipticResult solve_poisson_decomp(
    const DecompMesh& dm,
    const std::vector<double>& cellSource,
    const ScalarBCSet& bcSet,
    const DistEllipticOptions& opts) {
  std::vector<double> gammaFace(dm.mesh.nFaces, 1.0);
  return solve_scalar_elliptic_decomp(dm, gammaFace, cellSource, bcSet, opts);
}

std::vector<double> exchange_proc_face_scalar_owner_values(
    const DecompMesh& dm,
    const std::vector<double>& phi);

std::vector<std::array<double,3>> exchange_proc_face_vector_owner_values(
    const DecompMesh& dm,
    const std::vector<std::array<double,3>>& vec);
''')

(root / "libpoisson_decomp/src/poisson_decomp_library.cu").write_text(r'''#include "poisson_decomp_library.h"

#include <fstream>
#include <regex>
#include <sstream>

namespace {

std::string read_text_file(const std::string& path) {
  std::ifstream in(path);
  if (!in) throw std::runtime_error("Could not open " + path);
  std::ostringstream ss;
  ss << in.rdbuf();
  return ss.str();
}

int find_int_entry(const std::string& body, const std::string& key, int def = -1) {
  std::regex re("\\b" + key + R"(\s+([-+]?[0-9]+)\s*;)");
  std::smatch m;
  if (std::regex_search(body, m, re)) return std::stoi(m[1].str());
  return def;
}

std::string find_word_entry(const std::string& body, const std::string& key) {
  std::regex re("\\b" + key + R"(\s+([A-Za-z0-9_]+)\s*;)");
  std::smatch m;
  if (std::regex_search(body, m, re)) return m[1].str();
  return "";
}

std::vector<ProcPatchDecomp> read_processor_patches_decomp(const std::string& boundaryPath) {
  const std::string txt = read_text_file(boundaryPath);
  std::vector<ProcPatchDecomp> out;

  std::regex blockRe(R"(([A-Za-z0-9_]+)\s*\{([^{}]*)\})");
  auto begin = std::sregex_iterator(txt.begin(), txt.end(), blockRe);
  auto end = std::sregex_iterator();

  for (auto it = begin; it != end; ++it) {
    const std::string name = (*it)[1].str();
    const std::string body = (*it)[2].str();

    if (find_word_entry(body, "type") != "processor") continue;

    ProcPatchDecomp p;
    p.name = name;
    p.nFaces = find_int_entry(body, "nFaces");
    p.startFace = find_int_entry(body, "startFace");
    p.myProcNo = find_int_entry(body, "myProcNo");
    p.neighbProcNo = find_int_entry(body, "neighbProcNo");

    if (p.nFaces < 0 || p.startFace < 0 || p.myProcNo < 0 || p.neighbProcNo < 0) {
      throw std::runtime_error("Incomplete processor patch in " + boundaryPath);
    }

    out.push_back(p);
  }

  return out;
}

void add_col(std::vector<std::map<HYPRE_BigInt,int>>& pos,
             int localRow,
             HYPRE_BigInt col) {
  if (pos[localRow].find(col) == pos[localRow].end()) {
    pos[localRow][col] = -1;
  }
}

double face_interp_lambda_local(const Mesh& mesh, int f) {
  const int P = mesh.owner[f];
  const int N = mesh.neigh[f];
  const auto d = sub3(mesh.cc[N], mesh.cc[P]);
  const auto dx = sub3(mesh.xf[f], mesh.cc[P]);
  const double lam = dot3(dx, d) / std::max(dot3(d, d), 1.0e-30);
  return std::min(1.0, std::max(0.0, lam));
}

double face_interp_lambda_proc(const DecompMesh& dm, int f) {
  const Mesh& mesh = dm.mesh;
  const int P = mesh.owner[f];
  const auto d = sub3(dm.remoteCCForFace[f], mesh.cc[P]);
  const auto dx = sub3(mesh.xf[f], mesh.cc[P]);
  const double lam = dot3(dx, d) / std::max(dot3(d, d), 1.0e-30);
  return std::min(1.0, std::max(0.0, lam));
}

BoundaryFaceData build_physical_boundary_face_data_decomp(
    const DecompMesh& dm,
    const ScalarBCSet& bcSet) {
  const Mesh& mesh = dm.mesh;

  BoundaryFaceData out;
  out.type.assign(mesh.nFaces, ScalarBCType::Neumann);
  out.value.assign(mesh.nFaces, 0.0);

  std::map<std::string, const ScalarPatchBC*> byName;
  for (const auto& bc : bcSet.patches) byName[bc.patchName] = &bc;

  for (int f = mesh.nInternalFaces; f < mesh.nFaces; ++f) {
    if (dm.isProcFace[f]) continue;

    int pidx1 = mesh.bPatch[f];
    if (pidx1 <= 0) {
      throw std::runtime_error("Physical boundary face missing patch index at face " + std::to_string(f));
    }

    int pidx = pidx1 - 1;
    if (pidx < 0 || pidx >= static_cast<int>(mesh.patchNames.size())) {
      throw std::runtime_error("Invalid patch index at face " + std::to_string(f));
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

double boundary_equivalent_face_value_decomp(
    const Mesh& mesh,
    const BoundaryFaceData& bcFaceData,
    const std::vector<double>& phi,
    int P,
    int f) {
  if (bcFaceData.type[f] == ScalarBCType::Dirichlet) {
    return bcFaceData.value[f];
  }

  const auto r = sub3(mesh.xf[f], mesh.cc[P]);
  const double dn = std::max(dot3(r, mesh.nf[f]), 1e-30);
  return phi[P] + bcFaceData.value[f] * dn;
}

void compute_lsq_gradient_decomp(
    const DecompMesh& dm,
    const std::vector<double>& phi,
    const std::vector<double>& remotePhiForFace,
    const BoundaryFaceData& bcFaceData,
    std::vector<std::array<double,3>>& grad) {
  const Mesh& mesh = dm.mesh;
  grad.assign(mesh.nCells, {0.0, 0.0, 0.0});

  for (int P = 0; P < mesh.nCells; ++P) {
    const auto xP = mesh.cc[P];
    const double phiP = phi[P];

    double M[3][3] = {{0,0,0},{0,0,0},{0,0,0}};
    double rhs[3] = {0,0,0};

    auto add_constraint = [&](const std::array<double,3>& r, double dphi) {
      const double w = 1.0 / std::max(dot3(r, r), 1e-30);
      for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) M[i][j] += w * r[i] * r[j];
        rhs[i] += w * r[i] * dphi;
      }
    };

    for (int N : mesh.cellNbrs[P]) {
      auto r = sub3(mesh.cc[N], xP);
      add_constraint(r, phi[N] - phiP);
    }

    for (int f : mesh.cellBFace[P]) {
      if (dm.isProcFace[f]) {
        auto r = sub3(dm.remoteCCForFace[f], xP);
        add_constraint(r, remotePhiForFace[f] - phiP);
      } else {
        auto rcf = sub3(mesh.xf[f], xP);
        if (bcFaceData.type[f] == ScalarBCType::Dirichlet) {
          add_constraint(rcf, bcFaceData.value[f] - phiP);
        } else {
          const double dn = std::max(dot3(rcf, mesh.nf[f]), 1e-30);
          auto r = mul3(dn, mesh.nf[f]);
          add_constraint(r, bcFaceData.value[f] * dn);
        }
      }
    }

    double a=M[0][0], b=M[0][1], c=M[0][2];
    double d=M[1][0], e=M[1][1], f=M[1][2];
    double g=M[2][0], h=M[2][1], k=M[2][2];
    double det = a*(e*k-f*h) - b*(d*k-f*g) + c*(d*h-e*g);

    if (std::fabs(det) > 1e-20) {
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

void compute_gauss_gradient_decomp(
    const DecompMesh& dm,
    const std::vector<double>& phi,
    const std::vector<double>& remotePhiForFace,
    const BoundaryFaceData& bcFaceData,
    std::vector<std::array<double,3>>& grad) {
  const Mesh& mesh = dm.mesh;

  grad.assign(mesh.nCells, {0.0, 0.0, 0.0});

  for (int f = 0; f < mesh.nInternalFaces; ++f) {
    const int P = mesh.owner[f];
    const int N = mesh.neigh[f];
    const double lam = face_interp_lambda_local(mesh, f);
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
      const double lam = face_interp_lambda_proc(dm, f);
      phiF = (1.0 - lam) * phi[P] + lam * remotePhiForFace[f];
    } else {
      phiF = boundary_equivalent_face_value_decomp(mesh, bcFaceData, phi, P, f);
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

void compute_gradient_decomp(
    const DecompMesh& dm,
    const std::vector<double>& phi,
    const BoundaryFaceData& bcFaceData,
    GradientScheme scheme,
    std::vector<std::array<double,3>>& grad) {
  const auto remotePhi = exchange_proc_face_scalar_owner_values(dm, phi);

  if (scheme == GradientScheme::Gauss) {
    compute_gauss_gradient_decomp(dm, phi, remotePhi, bcFaceData, grad);
  } else {
    compute_lsq_gradient_decomp(dm, phi, remotePhi, bcFaceData, grad);
  }
}

void assemble_scalar_elliptic_decomp(
    const DecompMesh& dm,
    const DistCSRPattern& pat,
    const std::vector<double>& gammaFace,
    const std::vector<double>& cellSource,
    const BoundaryFaceData& bcFaceData,
    const std::vector<std::array<double,3>>& grad,
    const std::vector<std::array<double,3>>& remoteGradForFace,
    std::vector<HYPRE_Complex>& values,
    std::vector<HYPRE_Complex>& rhs,
    bool includeNonOrth,
    bool useReferenceCell,
    HYPRE_BigInt refGlobalCell,
    double refValue) {
  const Mesh& mesh = dm.mesh;

  if (static_cast<int>(gammaFace.size()) != mesh.nFaces) {
    throw std::runtime_error("gammaFace must have size local mesh.nFaces");
  }

  if (static_cast<int>(cellSource.size()) != mesh.nCells) {
    throw std::runtime_error("cellSource must have size local mesh.nCells");
  }

  values.assign(pat.nnz, 0.0);
  rhs.assign(mesh.nCells, 0.0);

  for (int c = 0; c < mesh.nCells; ++c) {
    rhs[c] = static_cast<HYPRE_Complex>(cellSource[c] * mesh.vol[c]);
  }

  for (int f = 0; f < mesh.nInternalFaces; ++f) {
    const int P = mesh.owner[f];
    const int N = mesh.neigh[f];

    auto d = sub3(mesh.cc[N], mesh.cc[P]);
    const double dDotS = dot3(d, mesh.Sf[f]);
    const double gamma = gammaFace[f];
    const double D = gamma * dot3(mesh.Sf[f], mesh.Sf[f]) / std::max(dDotS, 1e-30);

    values[pat.facePP[f]] += D;
    values[pat.facePN[f]] -= D;
    values[pat.faceNP[f]] -= D;
    values[pat.faceNN[f]] += D;

    if (includeNonOrth) {
      auto T = sub3(mesh.Sf[f], mul3(D / std::max(gamma, 1e-30), d));
      const double lam = face_interp_lambda_local(mesh, f);
      auto gradF = add3(mul3(1.0 - lam, grad[P]), mul3(lam, grad[N]));
      const double corr = gamma * dot3(T, gradF);
      rhs[P] += static_cast<HYPRE_Complex>(corr);
      rhs[N] -= static_cast<HYPRE_Complex>(corr);
    }
  }

  for (size_t i = 0; i < pat.procFace.size(); ++i) {
    const int f = pat.procFace[i];
    const int P = pat.procOwner[i];

    auto d = sub3(dm.remoteCCForFace[f], mesh.cc[P]);
    const double dDotS = dot3(d, mesh.Sf[f]);
    const double gamma = gammaFace[f];
    const double D = gamma * dot3(mesh.Sf[f], mesh.Sf[f]) / std::max(dDotS, 1e-30);

    values[pat.procDiag[i]] += D;
    values[pat.procOff[i]]  -= D;

    if (includeNonOrth) {
      auto T = sub3(mesh.Sf[f], mul3(D / std::max(gamma, 1e-30), d));
      const double lam = face_interp_lambda_proc(dm, f);
      auto gradF = add3(mul3(1.0 - lam, grad[P]), mul3(lam, remoteGradForFace[f]));
      const double corr = gamma * dot3(T, gradF);
      rhs[P] += static_cast<HYPRE_Complex>(corr);
    }
  }

  for (int f = mesh.nInternalFaces; f < mesh.nFaces; ++f) {
    if (dm.isProcFace[f]) continue;

    const int P = mesh.owner[f];
    auto d = sub3(mesh.xf[f], mesh.cc[P]);
    const double dDotS = dot3(d, mesh.Sf[f]);
    const double gamma = gammaFace[f];
    const double D = gamma * dot3(mesh.Sf[f], mesh.Sf[f]) / std::max(dDotS, 1e-30);

    if (bcFaceData.type[f] == ScalarBCType::Dirichlet) {
      values[pat.diagPos[P]] += D;
      rhs[P] += static_cast<HYPRE_Complex>(D * bcFaceData.value[f]);

      if (includeNonOrth) {
        auto T = sub3(mesh.Sf[f], mul3(D / std::max(gamma, 1e-30), d));
        const double corr = gamma * dot3(T, grad[P]);
        rhs[P] += static_cast<HYPRE_Complex>(corr);
      }
    } else {
      rhs[P] += static_cast<HYPRE_Complex>(-gamma * bcFaceData.value[f] * mesh.Af[f]);
    }
  }

  if (useReferenceCell && refGlobalCell >= dm.ilower && refGlobalCell <= dm.iupper) {
    const int refCell = static_cast<int>(refGlobalCell - dm.ilower);
    const int rowStart = pat.rowOffsets[refCell];
    const int rowEnd = pat.rowOffsets[refCell + 1];

    for (int k = rowStart; k < rowEnd; ++k) values[k] = 0.0;
    values[pat.diagPos[refCell]] = 1.0;
    rhs[refCell] = static_cast<HYPRE_Complex>(refValue);
  }
}

HypreSolveInfo solve_distributed_hypre_gpu_hostij(
    const DecompMesh& dm,
    const DistCSRPattern& pat,
    const std::vector<HYPRE_Complex>& values,
    const std::vector<HYPRE_Complex>& rhs,
    std::vector<double>& x,
    const HypreOptions& opt) {
  static bool hypreInitialized = false;

  if (!hypreInitialized) {
    HYPRE_CALL(HYPRE_Initialize());
#if defined(HYPRE_USING_GPU)
    HYPRE_CALL(HYPRE_DeviceInitialize());
    HYPRE_CALL(HYPRE_SetMemoryLocation(HYPRE_MEMORY_DEVICE));
    HYPRE_CALL(HYPRE_SetExecutionPolicy(HYPRE_EXEC_DEVICE));
#endif
    hypreInitialized = true;
  }

  HYPRE_IJMatrix Aij = nullptr;
  HYPRE_ParCSRMatrix Apar = nullptr;
  HYPRE_IJVector bij = nullptr, xij = nullptr;
  HYPRE_ParVector bpar = nullptr, xpar = nullptr;
  HYPRE_Solver solver = nullptr, prec = nullptr;

  HYPRE_CALL(HYPRE_IJMatrixCreate(dm.comm, dm.ilower, dm.iupper, dm.ilower, dm.iupper, &Aij));
  HYPRE_CALL(HYPRE_IJMatrixSetObjectType(Aij, HYPRE_PARCSR));
  HYPRE_CALL(HYPRE_IJMatrixSetRowSizes(Aij, const_cast<HYPRE_Int*>(pat.ncols.data())));
  HYPRE_CALL(HYPRE_IJMatrixInitialize_v2(Aij, HYPRE_MEMORY_HOST));
  HYPRE_CALL(HYPRE_IJMatrixSetValues(
      Aij,
      pat.nRows,
      const_cast<HYPRE_Int*>(pat.ncols.data()),
      const_cast<HYPRE_BigInt*>(pat.rows.data()),
      const_cast<HYPRE_BigInt*>(pat.cols.data()),
      const_cast<HYPRE_Complex*>(values.data())));
  HYPRE_CALL(HYPRE_IJMatrixAssemble(Aij));
#if defined(HYPRE_USING_GPU)
  HYPRE_CALL(HYPRE_IJMatrixMigrate(Aij, HYPRE_MEMORY_DEVICE));
#endif
  HYPRE_CALL(HYPRE_IJMatrixGetObject(Aij, reinterpret_cast<void**>(&Apar)));

  std::vector<HYPRE_Complex> x0(pat.nRows, 0.0);
  if (static_cast<int>(x.size()) == pat.nRows) {
    for (int i = 0; i < pat.nRows; ++i) x0[i] = static_cast<HYPRE_Complex>(x[i]);
  }

  HYPRE_CALL(HYPRE_IJVectorCreate(dm.comm, dm.ilower, dm.iupper, &bij));
  HYPRE_CALL(HYPRE_IJVectorSetObjectType(bij, HYPRE_PARCSR));
  HYPRE_CALL(HYPRE_IJVectorInitialize_v2(bij, HYPRE_MEMORY_HOST));
  HYPRE_CALL(HYPRE_IJVectorSetValues(
      bij, pat.nRows,
      const_cast<HYPRE_BigInt*>(pat.rows.data()),
      const_cast<HYPRE_Complex*>(rhs.data())));
  HYPRE_CALL(HYPRE_IJVectorAssemble(bij));
#if defined(HYPRE_USING_GPU)
  HYPRE_CALL(HYPRE_IJVectorMigrate(bij, HYPRE_MEMORY_DEVICE));
#endif
  HYPRE_CALL(HYPRE_IJVectorGetObject(bij, reinterpret_cast<void**>(&bpar)));

  HYPRE_CALL(HYPRE_IJVectorCreate(dm.comm, dm.ilower, dm.iupper, &xij));
  HYPRE_CALL(HYPRE_IJVectorSetObjectType(xij, HYPRE_PARCSR));
  HYPRE_CALL(HYPRE_IJVectorInitialize_v2(xij, HYPRE_MEMORY_HOST));
  HYPRE_CALL(HYPRE_IJVectorSetValues(xij, pat.nRows, const_cast<HYPRE_BigInt*>(pat.rows.data()), x0.data()));
  HYPRE_CALL(HYPRE_IJVectorAssemble(xij));
#if defined(HYPRE_USING_GPU)
  HYPRE_CALL(HYPRE_IJVectorMigrate(xij, HYPRE_MEMORY_DEVICE));
#endif
  HYPRE_CALL(HYPRE_IJVectorGetObject(xij, reinterpret_cast<void**>(&xpar)));

  HYPRE_CALL(HYPRE_ParCSRPCGCreate(dm.comm, &solver));

  const double pcgRelTol = std::max(opt.relTol, 0.0);
  const double pcgAbsTol = (opt.absTol >= 0.0) ? opt.absTol
                           : ((opt.tol >= 0.0) ? opt.tol : 0.0);

  HYPRE_CALL(HYPRE_PCGSetMaxIter(solver, opt.maxIter));
  HYPRE_CALL(HYPRE_PCGSetTol(solver, pcgRelTol));
  HYPRE_CALL(HYPRE_PCGSetAbsoluteTol(solver, pcgAbsTol));
  HYPRE_CALL(HYPRE_PCGSetTwoNorm(solver, 1));
  HYPRE_CALL(HYPRE_PCGSetLogging(solver, 1));
  HYPRE_CALL(HYPRE_PCGSetPrintLevel(solver, opt.monitor ? 2 : 0));

  HYPRE_CALL(HYPRE_BoomerAMGCreate(&prec));
  HYPRE_CALL(HYPRE_BoomerAMGSetPrintLevel(prec, opt.monitor ? 1 : 0));
  HYPRE_CALL(HYPRE_BoomerAMGSetMaxIter(prec, opt.amgMaxIter));
  HYPRE_CALL(HYPRE_BoomerAMGSetTol(prec, 0.0));
  HYPRE_CALL(HYPRE_BoomerAMGSetRelaxType(prec, opt.amgRelaxType));
  HYPRE_CALL(HYPRE_BoomerAMGSetCoarsenType(prec, opt.amgCoarsenType));
  HYPRE_CALL(HYPRE_BoomerAMGSetInterpType(prec, opt.amgInterpType));
  HYPRE_CALL(HYPRE_BoomerAMGSetNumSweeps(prec, opt.amgNumSweeps));
  HYPRE_CALL(HYPRE_BoomerAMGSetPMaxElmts(prec, opt.amgPmax));
  HYPRE_CALL(HYPRE_BoomerAMGSetKeepTranspose(prec, opt.amgKeepTranspose));
  HYPRE_CALL(HYPRE_BoomerAMGSetTruncFactor(prec, opt.amgTruncFactor));
  HYPRE_CALL(HYPRE_BoomerAMGSetRAP2(prec, opt.amgRAP2));

  if (opt.amgAggLevels > 0) {
    HYPRE_CALL(HYPRE_BoomerAMGSetAggNumLevels(prec, opt.amgAggLevels));
    HYPRE_CALL(HYPRE_BoomerAMGSetAggInterpType(prec, opt.amgAggInterpType));
  }

  if (opt.amgStrongThreshold >= 0.0) {
    HYPRE_CALL(HYPRE_BoomerAMGSetStrongThreshold(prec, opt.amgStrongThreshold));
  }

  HYPRE_CALL(HYPRE_PCGSetPrecond(
      solver,
      reinterpret_cast<HYPRE_PtrToSolverFcn>(HYPRE_BoomerAMGSolve),
      reinterpret_cast<HYPRE_PtrToSolverFcn>(HYPRE_BoomerAMGSetup),
      prec));

  HYPRE_CALL(HYPRE_ParCSRPCGSetup(solver, Apar, bpar, xpar));
  HYPRE_CALL(HYPRE_ParCSRPCGSolve(solver, Apar, bpar, xpar));

  HypreSolveInfo info;
  HYPRE_CALL(HYPRE_PCGGetNumIterations(solver, &info.iterations));
  HYPRE_Real rel = 0.0;
  HYPRE_CALL(HYPRE_PCGGetFinalRelativeResidualNorm(solver, &rel));
  info.finalRelResNorm = static_cast<double>(rel);

#if defined(HYPRE_USING_GPU)
  HYPRE_CALL(HYPRE_IJVectorMigrate(xij, HYPRE_MEMORY_HOST));
#endif

  std::vector<HYPRE_Complex> xhost(pat.nRows, 0.0);
  HYPRE_CALL(HYPRE_IJVectorGetValues(
      xij, pat.nRows,
      const_cast<HYPRE_BigInt*>(pat.rows.data()),
      xhost.data()));

  x.assign(pat.nRows, 0.0);
  for (int i = 0; i < pat.nRows; ++i) x[i] = static_cast<double>(xhost[i]);

  if (prec)   HYPRE_CALL(HYPRE_BoomerAMGDestroy(prec));
  if (solver) HYPRE_CALL(HYPRE_ParCSRPCGDestroy(solver));
  if (bij)    HYPRE_CALL(HYPRE_IJVectorDestroy(bij));
  if (xij)    HYPRE_CALL(HYPRE_IJVectorDestroy(xij));
  if (Aij)    HYPRE_CALL(HYPRE_IJMatrixDestroy(Aij));

  return info;
}

} // namespace

DecompMesh read_decomposed_openfoam_case(
    const std::string& caseRoot,
    MPI_Comm comm) {
  DecompMesh dm;
  dm.comm = comm;
  MPI_Comm_rank(comm, &dm.rank);
  MPI_Comm_size(comm, &dm.size);

  const std::string polyMeshDir =
      caseRoot + "/processor" + std::to_string(dm.rank) + "/constant/polyMesh";

  dm.mesh = read_openfoam_polymesh(polyMeshDir);
  dm.procPatches = read_processor_patches_decomp(polyMeshDir + "/boundary");

  dm.nLocal = dm.mesh.nCells;

  std::vector<int> countsInt(dm.size, 0);
  MPI_Allgather(&dm.nLocal, 1, MPI_INT, countsInt.data(), 1, MPI_INT, comm);

  dm.counts = countsInt;
  dm.offsets.assign(dm.size + 1, 0);
  for (int r = 0; r < dm.size; ++r) {
    dm.offsets[r + 1] = dm.offsets[r] + static_cast<HYPRE_BigInt>(dm.counts[r]);
  }

  dm.ilower = dm.offsets[dm.rank];
  dm.iupper = dm.offsets[dm.rank] + dm.nLocal - 1;
  dm.globalN = dm.offsets[dm.size];

  auto local_row = [&](int c) -> HYPRE_BigInt {
    return dm.offsets[dm.rank] + static_cast<HYPRE_BigInt>(c);
  };

  dm.remoteRowForFace.assign(dm.mesh.nFaces, -1);
  dm.remoteCCForFace.assign(dm.mesh.nFaces, {0.0, 0.0, 0.0});

  for (size_t ipp = 0; ipp < dm.procPatches.size(); ++ipp) {
    const auto& pp = dm.procPatches[ipp];
    const int nbr = pp.neighbProcNo;
    const int tagBase = 1000 + static_cast<int>(ipp) * 10;

    std::vector<long long> sendRows(pp.nFaces, -1);
    std::vector<double> sendCC(3 * pp.nFaces, 0.0);

    for (int i = 0; i < pp.nFaces; ++i) {
      const int f = pp.startFace + i;
      const int P = dm.mesh.owner[f];

      sendRows[i] = static_cast<long long>(local_row(P));
      sendCC[3*i + 0] = dm.mesh.cc[P][0];
      sendCC[3*i + 1] = dm.mesh.cc[P][1];
      sendCC[3*i + 2] = dm.mesh.cc[P][2];
    }

    int recvN = 0;
    MPI_Sendrecv(&pp.nFaces, 1, MPI_INT, nbr, tagBase + 0,
                 &recvN, 1, MPI_INT, nbr, tagBase + 0,
                 comm, MPI_STATUS_IGNORE);

    if (recvN != pp.nFaces) {
      throw std::runtime_error("processor patch face-count mismatch on patch " + pp.name);
    }

    std::vector<long long> recvRows(recvN, -1);
    std::vector<double> recvCC(3 * recvN, 0.0);

    MPI_Sendrecv(sendRows.data(), pp.nFaces, MPI_LONG_LONG, nbr, tagBase + 1,
                 recvRows.data(), recvN, MPI_LONG_LONG, nbr, tagBase + 1,
                 comm, MPI_STATUS_IGNORE);

    MPI_Sendrecv(sendCC.data(), 3 * pp.nFaces, MPI_DOUBLE, nbr, tagBase + 2,
                 recvCC.data(), 3 * recvN, MPI_DOUBLE, nbr, tagBase + 2,
                 comm, MPI_STATUS_IGNORE);

    for (int i = 0; i < pp.nFaces; ++i) {
      const int f = pp.startFace + i;
      dm.remoteRowForFace[f] = static_cast<HYPRE_BigInt>(recvRows[i]);
      dm.remoteCCForFace[f] = {recvCC[3*i + 0], recvCC[3*i + 1], recvCC[3*i + 2]};
    }
  }

  dm.isProcFace.assign(dm.mesh.nFaces, 0);
  for (const auto& pp : dm.procPatches) {
    for (int i = 0; i < pp.nFaces; ++i) {
      dm.isProcFace[pp.startFace + i] = 1;
    }
  }

  return dm;
}

std::vector<double> exchange_proc_face_scalar_owner_values(
    const DecompMesh& dm,
    const std::vector<double>& phi) {
  if (static_cast<int>(phi.size()) != dm.mesh.nCells) {
    throw std::runtime_error("exchange_proc_face_scalar_owner_values: phi size mismatch");
  }

  std::vector<double> remote(dm.mesh.nFaces, 0.0);

  for (size_t ipp = 0; ipp < dm.procPatches.size(); ++ipp) {
    const auto& pp = dm.procPatches[ipp];
    const int nbr = pp.neighbProcNo;
    const int tag = 2000 + static_cast<int>(ipp);

    std::vector<double> send(pp.nFaces, 0.0);
    std::vector<double> recv(pp.nFaces, 0.0);

    for (int i = 0; i < pp.nFaces; ++i) {
      const int f = pp.startFace + i;
      send[i] = phi[dm.mesh.owner[f]];
    }

    MPI_Sendrecv(send.data(), pp.nFaces, MPI_DOUBLE, nbr, tag,
                 recv.data(), pp.nFaces, MPI_DOUBLE, nbr, tag,
                 dm.comm, MPI_STATUS_IGNORE);

    for (int i = 0; i < pp.nFaces; ++i) {
      const int f = pp.startFace + i;
      remote[f] = recv[i];
    }
  }

  return remote;
}

std::vector<std::array<double,3>> exchange_proc_face_vector_owner_values(
    const DecompMesh& dm,
    const std::vector<std::array<double,3>>& vec) {
  if (static_cast<int>(vec.size()) != dm.mesh.nCells) {
    throw std::runtime_error("exchange_proc_face_vector_owner_values: vector field size mismatch");
  }

  std::vector<std::array<double,3>> remote(dm.mesh.nFaces, {0.0, 0.0, 0.0});

  for (size_t ipp = 0; ipp < dm.procPatches.size(); ++ipp) {
    const auto& pp = dm.procPatches[ipp];
    const int nbr = pp.neighbProcNo;
    const int tag = 3000 + static_cast<int>(ipp);

    std::vector<double> send(3 * pp.nFaces, 0.0);
    std::vector<double> recv(3 * pp.nFaces, 0.0);

    for (int i = 0; i < pp.nFaces; ++i) {
      const int f = pp.startFace + i;
      const int P = dm.mesh.owner[f];
      send[3*i + 0] = vec[P][0];
      send[3*i + 1] = vec[P][1];
      send[3*i + 2] = vec[P][2];
    }

    MPI_Sendrecv(send.data(), 3 * pp.nFaces, MPI_DOUBLE, nbr, tag,
                 recv.data(), 3 * pp.nFaces, MPI_DOUBLE, nbr, tag,
                 dm.comm, MPI_STATUS_IGNORE);

    for (int i = 0; i < pp.nFaces; ++i) {
      const int f = pp.startFace + i;
      remote[f] = {recv[3*i + 0], recv[3*i + 1], recv[3*i + 2]};
    }
  }

  return remote;
}

DistCSRPattern build_decomp_scalar_pattern(const DecompMesh& dm) {
  const Mesh& mesh = dm.mesh;

  auto local_row = [&](int c) -> HYPRE_BigInt {
    return dm.offsets[dm.rank] + static_cast<HYPRE_BigInt>(c);
  };

  std::vector<std::map<HYPRE_BigInt,int>> pos(mesh.nCells);

  for (int c = 0; c < mesh.nCells; ++c) {
    add_col(pos, c, local_row(c));
  }

  for (int f = 0; f < mesh.nInternalFaces; ++f) {
    const int P = mesh.owner[f];
    const int N = mesh.neigh[f];
    add_col(pos, P, local_row(N));
    add_col(pos, N, local_row(P));
  }

  for (int f = mesh.nInternalFaces; f < mesh.nFaces; ++f) {
    if (dm.isProcFace[f]) {
      const int P = mesh.owner[f];
      add_col(pos, P, dm.remoteRowForFace[f]);
    }
  }

  DistCSRPattern pat;
  pat.nRows = mesh.nCells;
  pat.rows.resize(mesh.nCells);
  pat.ncols.resize(mesh.nCells);
  pat.rowOffsets.resize(mesh.nCells + 1);
  pat.diagPos.resize(mesh.nCells);
  pat.facePP.resize(mesh.nInternalFaces);
  pat.facePN.resize(mesh.nInternalFaces);
  pat.faceNP.resize(mesh.nInternalFaces);
  pat.faceNN.resize(mesh.nInternalFaces);

  pat.rowOffsets[0] = 0;
  for (int c = 0; c < mesh.nCells; ++c) {
    pat.rows[c] = local_row(c);
    int j = 0;
    for (auto& kv : pos[c]) {
      kv.second = pat.rowOffsets[c] + j;
      ++j;
    }
    pat.ncols[c] = static_cast<HYPRE_Int>(pos[c].size());
    pat.rowOffsets[c + 1] = pat.rowOffsets[c] + static_cast<int>(pos[c].size());
  }

  pat.nnz = pat.rowOffsets[mesh.nCells];
  pat.cols.resize(pat.nnz);

  for (int c = 0; c < mesh.nCells; ++c) {
    const HYPRE_BigInt diag = local_row(c);
    for (const auto& kv : pos[c]) {
      pat.cols[kv.second] = kv.first;
      if (kv.first == diag) pat.diagPos[c] = kv.second;
    }
  }

  for (int f = 0; f < mesh.nInternalFaces; ++f) {
    const int P = mesh.owner[f];
    const int N = mesh.neigh[f];
    pat.facePP[f] = pos[P][local_row(P)];
    pat.facePN[f] = pos[P][local_row(N)];
    pat.faceNP[f] = pos[N][local_row(P)];
    pat.faceNN[f] = pos[N][local_row(N)];
  }

  for (int f = mesh.nInternalFaces; f < mesh.nFaces; ++f) {
    if (!dm.isProcFace[f]) continue;
    const int P = mesh.owner[f];
    pat.procFace.push_back(f);
    pat.procOwner.push_back(P);
    pat.procDiag.push_back(pos[P][local_row(P)]);
    pat.procOff.push_back(pos[P][dm.remoteRowForFace[f]]);
  }

  return pat;
}

DistEllipticResult solve_scalar_elliptic_decomp(
    const DecompMesh& dm,
    const std::vector<double>& gammaFace,
    const std::vector<double>& cellSource,
    const ScalarBCSet& bcSet,
    const DistEllipticOptions& opts) {
  const Mesh& mesh = dm.mesh;

  const GradientScheme gradScheme = gradient_scheme_from_string(opts.gradScheme);

  if (opts.laplacianScheme != "orth" && opts.laplacianScheme != "nonorth") {
    throw std::runtime_error("Use laplacianScheme=orth or laplacianScheme=nonorth.");
  }

  const bool includeNonOrth = (opts.laplacianScheme == "nonorth");
  const int nOuter = includeNonOrth ? std::max(opts.nNonOrthCorr, 0) + 1 : 1;

  BoundaryFaceData bcFaceData = build_physical_boundary_face_data_decomp(dm, bcSet);

  int localAnyDirichlet = 0;
  for (int f = mesh.nInternalFaces; f < mesh.nFaces; ++f) {
    if (!dm.isProcFace[f] && bcFaceData.type[f] == ScalarBCType::Dirichlet) {
      localAnyDirichlet = 1;
      break;
    }
  }

  int globalAnyDirichlet = 0;
  MPI_Allreduce(&localAnyDirichlet, &globalAnyDirichlet, 1, MPI_INT, MPI_MAX, dm.comm);

  if (!globalAnyDirichlet && !opts.useReferenceCell) {
    throw std::runtime_error("Pure-Neumann distributed scalar elliptic problem needs useReferenceCell=true.");
  }

  DistCSRPattern pat = build_decomp_scalar_pattern(dm);

  long long localNnz = static_cast<long long>(pat.nnz);
  long long globalNnz = 0;
  MPI_Allreduce(&localNnz, &globalNnz, 1, MPI_LONG_LONG, MPI_SUM, dm.comm);

  std::vector<double> phi(mesh.nCells, 0.0);
  std::vector<std::array<double,3>> grad(mesh.nCells, {0.0, 0.0, 0.0});
  std::vector<std::array<double,3>> remoteGrad(mesh.nFaces, {0.0, 0.0, 0.0});
  std::vector<HYPRE_Complex> values;
  std::vector<HYPRE_Complex> rhs;
  HypreSolveInfo lastInfo{};

  for (int outer = 0; outer < nOuter; ++outer) {
    compute_gradient_decomp(dm, phi, bcFaceData, grad);
    remoteGrad = exchange_proc_face_vector_owner_values(dm, grad);

    assemble_scalar_elliptic_decomp(
        dm, pat, gammaFace, cellSource, bcFaceData, grad, remoteGrad,
        values, rhs, includeNonOrth,
        opts.useReferenceCell, opts.referenceGlobalCell, opts.referenceValue);

    lastInfo = solve_distributed_hypre_gpu_hostij(dm, pat, values, rhs, phi, opts.hypre);
  }

  DistEllipticResult out;
  out.phi = std::move(phi);
  out.lastSolveInfo = lastInfo;
  out.nOuter = nOuter;
  out.globalNnz = globalNnz;
  return out;
}
''')

(root / "libpoisson_decomp/CMakeLists.txt").write_text(r'''add_library(libpoisson_decomp STATIC
  src/poisson_decomp_library.cu
)

target_include_directories(libpoisson_decomp PUBLIC
  ${CMAKE_CURRENT_SOURCE_DIR}/include
  ${CMAKE_SOURCE_DIR}/libpoisson/include
)

target_link_libraries(libpoisson_decomp PUBLIC
  libpoisson
  MPI::MPI_CXX
)

set_target_properties(libpoisson_decomp PROPERTIES
  CUDA_SEPARABLE_COMPILATION ON
)
''')

(root / "apps/poisson_mms_decomp_gpu_mpi/src/main.cu").write_text(r'''#include <mpi.h>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <map>
#include <set>
#include <string>
#include <vector>

#include "poisson_decomp_library.h"
#include "mms.h"

#include "bc_runtime_config.h"

static ScalarBCSet make_exact_mms_dirichlet_bcs_for_physical_patches(const DecompMesh& dm) {
  std::set<std::string> procPatchNames;
  for (const auto& pp : dm.procPatches) procPatchNames.insert(pp.name);

  ScalarBCSet bc;
  for (const auto& name : dm.mesh.patchNames) {
    if (procPatchNames.count(name)) continue;
    bc.patches.push_back(make_dirichlet_patch_bc(
        name,
        [](const std::array<double,3>& x, const std::array<double,3>&) {
          return phi_exact(x);
        }));
  }
  return bc;
}

static ScalarBCSet make_scalar_bcs_from_pressure_config(
    const DecompMesh& dm,
    const std::string& bcConfigPath) {
  auto cfg = pipebc::load_runtime_bc_config(bcConfigPath);
  pipebc::validate_runtime_bc_config_against_patches(cfg, dm.mesh.patchNames);

  const auto dups = pipebc::duplicate_pressure_bc_patches(cfg.pressurePatchSpecs);
  if (!dups.empty()) {
    throw std::runtime_error("Duplicate pressure BC for patch '" + dups.front() + "' in " + bcConfigPath);
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
      throw std::runtime_error("Unsupported pressure BC type in " + bcConfigPath);
    }
  }

  return bc;
}

int main(int argc, char** argv) {
  MPI_Init(&argc, &argv);

  int rank = 0, size = 1;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  try {
    std::string caseRoot = "/tmp/case";
    std::string bcConfigPath;
    int device = rank;
    double tol = 1e-7;
    int maxit = 500;
    int nNonOrthCorr = 2;
    std::string gradScheme = "lsq";
    std::string laplacianScheme = "nonorth";

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
      } else if (a == "-tol" || a == "-p-tol") {
        need(a.c_str());
        tol = std::atof(argv[++i]);
      } else if (a == "-maxit" || a == "-p-maxit") {
        need(a.c_str());
        maxit = std::atoi(argv[++i]);
      } else if (a == "-nNonOrthCorr" || a == "-n-nonorth-corr") {
        need(a.c_str());
        nNonOrthCorr = std::atoi(argv[++i]);
      } else if (a == "-grad-scheme") {
        need("-grad-scheme");
        gradScheme = argv[++i];
      } else if (a == "-laplacian-scheme") {
        need("-laplacian-scheme");
        laplacianScheme = argv[++i];
      }
    }

    int devCount = 0;
    CUDA_CALL(cudaGetDeviceCount(&devCount));
    if (devCount > 0) CUDA_CALL(cudaSetDevice(device % devCount));

    DecompMesh dm = read_decomposed_openfoam_case(caseRoot, MPI_COMM_WORLD);

    ScalarBCSet bcSet;
    if (!bcConfigPath.empty()) {
      bcSet = make_scalar_bcs_from_pressure_config(dm, bcConfigPath);
    } else {
      bcSet = make_exact_mms_dirichlet_bcs_for_physical_patches(dm);
    }

    std::vector<double> cellSource(dm.mesh.nCells, 0.0);
    for (int c = 0; c < dm.mesh.nCells; ++c) {
      cellSource[c] = rhs_exact(dm.mesh.cc[c]);
    }

    DistEllipticOptions opts;
    opts.gradScheme = gradScheme;
    opts.laplacianScheme = laplacianScheme;
    opts.nNonOrthCorr = nNonOrthCorr;
    opts.hypre.maxIter = maxit;
    opts.hypre.absTol = tol;
    opts.hypre.relTol = 0.0;
    opts.hypre.tol = tol;
    opts.hypre.monitor = 0;
    opts.hypre.amgMaxIter = 1;
    opts.hypre.amgRelaxType = 18;
    opts.hypre.amgCoarsenType = 8;
    opts.hypre.amgInterpType = 6;
    opts.hypre.amgAggLevels = 1;
    opts.hypre.amgPmax = 4;
    opts.hypre.amgKeepTranspose = 1;

    MPI_Barrier(MPI_COMM_WORLD);
    const double t0 = MPI_Wtime();

    DistEllipticResult result = solve_poisson_decomp(dm, cellSource, bcSet, opts);

    MPI_Barrier(MPI_COMM_WORLD);
    const double t1 = MPI_Wtime();

    double localL2 = 0.0;
    double localInf = 0.0;
    double localVol = 0.0;

    for (int c = 0; c < dm.mesh.nCells; ++c) {
      const double ue = phi_exact(dm.mesh.cc[c]);
      const double e = std::abs(result.phi[c] - ue);
      localL2 += e * e * dm.mesh.vol[c];
      localInf = std::max(localInf, e);
      localVol += dm.mesh.vol[c];
    }

    double globalL2 = 0.0, globalInf = 0.0, globalVol = 0.0;
    MPI_Allreduce(&localL2, &globalL2, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    MPI_Allreduce(&localInf, &globalInf, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
    MPI_Allreduce(&localVol, &globalVol, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

    globalL2 = std::sqrt(globalL2 / std::max(globalVol, 1e-300));

    double wall = t1 - t0;
    double maxWall = 0.0;
    MPI_Reduce(&wall, &maxWall, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    long long localNnz = build_decomp_scalar_pattern(dm).nnz;
    long long globalNnz = 0;
    MPI_Allreduce(&localNnz, &globalNnz, 1, MPI_LONG_LONG, MPI_SUM, MPI_COMM_WORLD);

    std::printf("rank %d/%d rows=[%lld,%lld] nLocal=%d nnz=%lld internalFaces=%d procFaces=%zu boundaryFaces=%d\n",
                rank, size,
                (long long)dm.ilower, (long long)dm.iupper,
                dm.nLocal, localNnz,
                dm.mesh.nInternalFaces,
                build_decomp_scalar_pattern(dm).procFace.size(),
                dm.mesh.nFaces - dm.mesh.nInternalFaces - (int)build_decomp_scalar_pattern(dm).procFace.size());
    std::fflush(stdout);

    if (rank == 0) {
      std::printf("POISSON_DECOMP_MMS setup: ranks=%d globalRows=%lld globalNnz=%lld bcConfig=%s grad=%s laplacian=%s nNonOrthCorr=%d nOuter=%d\n",
                  size, (long long)dm.globalN, globalNnz,
                  bcConfigPath.empty() ? "<exact-mms-default>" : bcConfigPath.c_str(),
                  gradScheme.c_str(), laplacianScheme.c_str(),
                  nNonOrthCorr, result.nOuter);

      std::printf("POISSON_DECOMP_MMS RESULT: its=%d finalRel=%.12e L2=%.12e Linf=%.12e wall=%.6e s\n",
                  result.lastSolveInfo.iterations,
                  result.lastSolveInfo.finalRelResNorm,
                  globalL2, globalInf, maxWall);

      if (std::isfinite(globalL2) && std::isfinite(globalInf)) {
        std::printf("POISSON_DECOMP_MMS PASS_RAN\n");
      } else {
        std::printf("POISSON_DECOMP_MMS FAIL_NAN_INF\n");
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

(root / "apps/poisson_mms_decomp_gpu_mpi/CMakeLists.txt").write_text(r'''add_executable(poisson_mms_decomp_gpu_mpi
  src/main.cu
  ${CMAKE_SOURCE_DIR}/apps/simple_gpu/src/bc_specs.cu
  ${CMAKE_SOURCE_DIR}/apps/simple_gpu/src/bc_runtime_config.cu
)

target_include_directories(poisson_mms_decomp_gpu_mpi PRIVATE
  ${CMAKE_SOURCE_DIR}/libpoisson/include
  ${CMAKE_SOURCE_DIR}/libpoisson_decomp/include
  ${CMAKE_SOURCE_DIR}/apps/simple_gpu/src
)

target_link_libraries(poisson_mms_decomp_gpu_mpi PRIVATE
  libpoisson_decomp
  libpoisson
  MPI::MPI_CXX
)

set_target_properties(poisson_mms_decomp_gpu_mpi PROPERTIES
  CUDA_SEPARABLE_COMPILATION ON
)
''')

cmake = root / "CMakeLists.txt"
txt = cmake.read_text()

if "add_subdirectory(libpoisson_decomp)" not in txt:
    txt = txt.replace("add_subdirectory(libpoisson)\n", "add_subdirectory(libpoisson)\nadd_subdirectory(libpoisson_decomp)\n")

if "add_subdirectory(apps/poisson_mms_decomp_gpu_mpi)" not in txt:
    txt += "\nadd_subdirectory(apps/poisson_mms_decomp_gpu_mpi)\n"

cmake.write_text(txt)

print("OK: created libpoisson_decomp and apps/poisson_mms_decomp_gpu_mpi")
