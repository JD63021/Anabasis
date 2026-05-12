#include <mpi.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <set>
#include <stdexcept>
#include <string>
#include <vector>

#include "poisson_decomp_library.h"
#include "scalar_decomp_library.h"

namespace {

constexpr double PI = 3.141592653589793238462643383279502884;

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

void cuda_check(cudaError_t e, const char* what) {
  if (e != cudaSuccess) {
    throw std::runtime_error(std::string(what) + ": " + cudaGetErrorString(e));
  }
}

// Divergence-free MMS:
//   u =  sin(pi x) cos(pi y) cos(pi z)
//   v = -cos(pi x) sin(pi y) cos(pi z)
//   w = 0
//   p =  sin(pi x) sin(pi y) sin(pi z)
//
// div(U)=0 because du/dx + dv/dy = 0.
std::array<double,3> u_exact(const std::array<double,3>& x) {
  const double sx = std::sin(PI*x[0]);
  const double cx = std::cos(PI*x[0]);
  const double sy = std::sin(PI*x[1]);
  const double cy = std::cos(PI*x[1]);
  const double cz = std::cos(PI*x[2]);

  return {
     sx * cy * cz,
    -cx * sy * cz,
     0.0
  };
}

double p_exact(const std::array<double,3>& x) {
  return std::sin(PI*x[0]) * std::sin(PI*x[1]) * std::sin(PI*x[2]);
}

std::array<double,3> grad_p_exact(const std::array<double,3>& x) {
  return {
    PI * std::cos(PI*x[0]) * std::sin(PI*x[1]) * std::sin(PI*x[2]),
    PI * std::sin(PI*x[0]) * std::cos(PI*x[1]) * std::sin(PI*x[2]),
    PI * std::sin(PI*x[0]) * std::sin(PI*x[1]) * std::cos(PI*x[2])
  };
}

// For both u and v, laplacian(component) = -3*pi^2*component.
// Stokes equation:
//   -mu laplacian(U) + grad(p) = f
std::array<double,3> stokes_source_exact(const std::array<double,3>& x, double mu) {
  const auto U = u_exact(x);
  const auto gp = grad_p_exact(x);
  return {
    3.0 * mu * PI * PI * U[0] + gp[0],
    3.0 * mu * PI * PI * U[1] + gp[1],
    3.0 * mu * PI * PI * U[2] + gp[2]
  };
}

struct LocalBoundaryFaceData {
  std::vector<ScalarBCType> type;
  std::vector<double> value;
};

ScalarBCSet make_component_velocity_bc(const DecompMesh& dm, int comp) {
  std::set<std::string> procPatchNames;
  for (const auto& pp : dm.procPatches) procPatchNames.insert(pp.name);

  ScalarBCSet bc;

  for (const auto& name : dm.mesh.patchNames) {
    if (procPatchNames.count(name)) continue;

    bc.patches.push_back(make_dirichlet_patch_bc(
        name,
        [comp](const std::array<double,3>& x, const std::array<double,3>&) {
          return u_exact(x)[comp];
        }));
  }

  return bc;
}

ScalarBCSet make_pressure_exact_dirichlet_bc(const DecompMesh& dm) {
  std::set<std::string> procPatchNames;
  for (const auto& pp : dm.procPatches) procPatchNames.insert(pp.name);

  ScalarBCSet bc;

  for (const auto& name : dm.mesh.patchNames) {
    if (procPatchNames.count(name)) continue;

    bc.patches.push_back(make_dirichlet_patch_bc(
        name,
        [](const std::array<double,3>& x, const std::array<double,3>&) {
          return p_exact(x);
        }));
  }

  return bc;
}

ScalarBCSet make_pcorr_zero_neumann_bc(const DecompMesh& dm) {
  std::set<std::string> procPatchNames;
  for (const auto& pp : dm.procPatches) procPatchNames.insert(pp.name);

  ScalarBCSet bc;

  for (const auto& name : dm.mesh.patchNames) {
    if (procPatchNames.count(name)) continue;
    bc.patches.push_back(make_neumann_constant_bc(name, 0.0));
  }

  return bc;
}

LocalBoundaryFaceData build_boundary_face_data(
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
      throw std::runtime_error("bad boundary patch index");
    }

    const std::string& patchName = mesh.patchNames[pidx];
    auto it = byName.find(patchName);
    if (it == byName.end()) {
      throw std::runtime_error("No scalar BC supplied for physical patch '" + patchName + "'");
    }

    out.type[f] = it->second->type;
    out.value[f] = it->second->evaluator(mesh.xf[f], mesh.nf[f]);
  }

  return out;
}

double face_lambda_local(const Mesh& mesh, int f) {
  const int P = mesh.owner[f];
  const int N = mesh.neigh[f];
  const auto d = sub3(mesh.cc[N], mesh.cc[P]);
  const auto dx = sub3(mesh.xf[f], mesh.cc[P]);
  const double lam = dot3(dx, d) / std::max(dot3(d, d), 1e-30);
  return std::min(1.0, std::max(0.0, lam));
}

double face_lambda_proc(const DecompMesh& dm, int f) {
  const Mesh& mesh = dm.mesh;
  const int P = mesh.owner[f];
  const auto d = sub3(dm.remoteCCForFace[f], mesh.cc[P]);
  const auto dx = sub3(mesh.xf[f], mesh.cc[P]);
  const double lam = dot3(dx, d) / std::max(dot3(d, d), 1e-30);
  return std::min(1.0, std::max(0.0, lam));
}

void compute_lsq_gradient(
    const DecompMesh& dm,
    const std::vector<double>& phi,
    const LocalBoundaryFaceData& bcFaceData,
    std::vector<std::array<double,3>>& grad) {
  const Mesh& mesh = dm.mesh;
  const auto remotePhi = exchange_proc_face_scalar_owner_values(dm, phi);

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
      add_constraint(sub3(mesh.cc[N], xP), phi[N] - phiP);
    }

    for (int f : mesh.cellBFace[P]) {
      if (dm.isProcFace[f]) {
        add_constraint(sub3(dm.remoteCCForFace[f], xP), remotePhi[f] - phiP);
      } else {
        const auto rcf = sub3(mesh.xf[f], xP);

        if (bcFaceData.type[f] == ScalarBCType::Dirichlet) {
          add_constraint(rcf, bcFaceData.value[f] - phiP);
        } else {
          const double dn = std::max(dot3(rcf, mesh.nf[f]), 1e-30);
          add_constraint(mul3(dn, mesh.nf[f]), bcFaceData.value[f] * dn);
        }
      }
    }

    const double a=M[0][0], b=M[0][1], c=M[0][2];
    const double d=M[1][0], e=M[1][1], f=M[1][2];
    const double g=M[2][0], h=M[2][1], k=M[2][2];

    const double det = a*(e*k-f*h) - b*(d*k-f*g) + c*(d*h-e*g);

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

double compute_momentum_diag_cell(
    const DecompMesh& dm,
    double mu,
    int P) {
  const Mesh& mesh = dm.mesh;
  double diag = 0.0;

  for (int f = 0; f < mesh.nInternalFaces; ++f) {
    if (mesh.owner[f] != P && mesh.neigh[f] != P) continue;

    int C0 = mesh.owner[f];
    int C1 = mesh.neigh[f];
    auto d = sub3(mesh.cc[C1], mesh.cc[C0]);
    double dDotS = dot3(d, mesh.Sf[f]);
    double D = mu * dot3(mesh.Sf[f], mesh.Sf[f]) / std::max(dDotS, 1e-30);
    diag += D;
  }

  for (int f : mesh.cellBFace[P]) {
    auto d = dm.isProcFace[f] ? sub3(dm.remoteCCForFace[f], mesh.cc[P])
                              : sub3(mesh.xf[f], mesh.cc[P]);

    double dDotS = dot3(d, mesh.Sf[f]);
    double D = mu * dot3(mesh.Sf[f], mesh.Sf[f]) / std::max(dDotS, 1e-30);
    diag += D;
  }

  return std::max(diag, 1e-300);
}

std::vector<double> compute_rAU(const DecompMesh& dm, double mu) {
  std::vector<double> rAU(dm.mesh.nCells, 0.0);

  for (int c = 0; c < dm.mesh.nCells; ++c) {
    rAU[c] = 1.0 / compute_momentum_diag_cell(dm, mu, c);
  }

  return rAU;
}

std::vector<double> build_pressure_gamma_faces(
    const DecompMesh& dm,
    const std::vector<double>& rAU) {
  const Mesh& mesh = dm.mesh;
  const auto remoteRAU = exchange_proc_face_scalar_owner_values(dm, rAU);

  std::vector<double> gamma(mesh.nFaces, 0.0);

  for (int f = 0; f < mesh.nInternalFaces; ++f) {
    const int P = mesh.owner[f];
    const int N = mesh.neigh[f];
    const double lam = face_lambda_local(mesh, f);
    gamma[f] = (1.0 - lam) * rAU[P] + lam * rAU[N];
  }

  for (int f = mesh.nInternalFaces; f < mesh.nFaces; ++f) {
    const int P = mesh.owner[f];

    if (dm.isProcFace[f]) {
      const double lam = face_lambda_proc(dm, f);
      gamma[f] = (1.0 - lam) * rAU[P] + lam * remoteRAU[f];
    } else {
      gamma[f] = rAU[P];
    }
  }

  return gamma;
}

std::vector<double> build_flux_from_velocity(
    const DecompMesh& dm,
    const std::vector<double>& U,
    const std::vector<double>& V,
    const std::vector<double>& W) {
  const Mesh& mesh = dm.mesh;

  const auto rU = exchange_proc_face_scalar_owner_values(dm, U);
  const auto rV = exchange_proc_face_scalar_owner_values(dm, V);
  const auto rW = exchange_proc_face_scalar_owner_values(dm, W);

  std::vector<double> phi(mesh.nFaces, 0.0);

  for (int f = 0; f < mesh.nInternalFaces; ++f) {
    const int P = mesh.owner[f];
    const int N = mesh.neigh[f];
    const double lam = face_lambda_local(mesh, f);

    const double uf = (1.0-lam)*U[P] + lam*U[N];
    const double vf = (1.0-lam)*V[P] + lam*V[N];
    const double wf = (1.0-lam)*W[P] + lam*W[N];

    phi[f] = uf*mesh.Sf[f][0] + vf*mesh.Sf[f][1] + wf*mesh.Sf[f][2];
  }

  for (int f = mesh.nInternalFaces; f < mesh.nFaces; ++f) {
    const int P = mesh.owner[f];

    if (dm.isProcFace[f]) {
      const double lam = face_lambda_proc(dm, f);

      const double uf = (1.0-lam)*U[P] + lam*rU[f];
      const double vf = (1.0-lam)*V[P] + lam*rV[f];
      const double wf = (1.0-lam)*W[P] + lam*rW[f];

      phi[f] = uf*mesh.Sf[f][0] + vf*mesh.Sf[f][1] + wf*mesh.Sf[f][2];
    } else {
      const auto ub = u_exact(mesh.xf[f]);
      phi[f] = ub[0]*mesh.Sf[f][0] + ub[1]*mesh.Sf[f][1] + ub[2]*mesh.Sf[f][2];
    }
  }

  return phi;
}

std::vector<double> compute_divergence_sum(
    const DecompMesh& dm,
    const std::vector<double>& phi) {
  const Mesh& mesh = dm.mesh;
  std::vector<double> div(mesh.nCells, 0.0);

  for (int f = 0; f < mesh.nInternalFaces; ++f) {
    const int P = mesh.owner[f];
    const int N = mesh.neigh[f];
    div[P] += phi[f];
    div[N] -= phi[f];
  }

  for (int f = mesh.nInternalFaces; f < mesh.nFaces; ++f) {
    const int P = mesh.owner[f];
    div[P] += phi[f];
  }

  return div;
}

double global_max_abs(const DecompMesh& dm, const std::vector<double>& a) {
  double loc = 0.0;
  for (double v : a) loc = std::max(loc, std::abs(v));

  double glob = 0.0;
  MPI_Allreduce(&loc, &glob, 1, MPI_DOUBLE, MPI_MAX, dm.comm);
  return glob;
}

std::vector<double> correct_flux_with_pcorr(
    const DecompMesh& dm,
    const std::vector<double>& phiStar,
    const std::vector<double>& pCorr,
    const std::vector<double>& gammaFace,
    const std::vector<std::array<double,3>>& gradPCorr) {
  const Mesh& mesh = dm.mesh;

  const auto remotePCorr = exchange_proc_face_scalar_owner_values(dm, pCorr);
  const auto remoteGrad = exchange_proc_face_vector_owner_values(dm, gradPCorr);

  std::vector<double> phi = phiStar;

  for (int f = 0; f < mesh.nInternalFaces; ++f) {
    const int P = mesh.owner[f];
    const int N = mesh.neigh[f];

    const auto d = sub3(mesh.cc[N], mesh.cc[P]);
    const double dDotS = dot3(d, mesh.Sf[f]);
    const double D = gammaFace[f] * dot3(mesh.Sf[f], mesh.Sf[f]) / std::max(dDotS, 1e-30);

    const double lam = face_lambda_local(mesh, f);
    const auto gradF = add3(mul3(1.0-lam, gradPCorr[P]), mul3(lam, gradPCorr[N]));
    const auto T = sub3(mesh.Sf[f], mul3(D / std::max(gammaFace[f], 1e-300), d));

    const double q = D * (pCorr[N] - pCorr[P]) + gammaFace[f] * dot3(T, gradF);
    phi[f] -= q;
  }

  for (int f = mesh.nInternalFaces; f < mesh.nFaces; ++f) {
    if (!dm.isProcFace[f]) continue;

    const int P = mesh.owner[f];

    const auto d = sub3(dm.remoteCCForFace[f], mesh.cc[P]);
    const double dDotS = dot3(d, mesh.Sf[f]);
    const double D = gammaFace[f] * dot3(mesh.Sf[f], mesh.Sf[f]) / std::max(dDotS, 1e-30);

    const double lam = face_lambda_proc(dm, f);
    const auto gradF = add3(mul3(1.0-lam, gradPCorr[P]), mul3(lam, remoteGrad[f]));
    const auto T = sub3(mesh.Sf[f], mul3(D / std::max(gammaFace[f], 1e-300), d));

    const double q = D * (remotePCorr[f] - pCorr[P]) + gammaFace[f] * dot3(T, gradF);
    phi[f] -= q;
  }

  // Physical boundaries use zero-gradient pCorr: no pressure flux correction there.
  return phi;
}

} // namespace

int main(int argc, char** argv) {
  MPI_Init(&argc, &argv);

  int rank = 0, size = 1;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  try {
    std::string caseRoot = "/tmp/case_cube_0p03mm";
    int device = rank;

    double mu = 0.01;
    double tolMass = 1e-6;
    double tolVel = 1e-6;
    int nsteps = 100;
    int minSteps = 5;
    int maxit = 1000;

    double pRelax = 1.0;

    std::string gradScheme = "lsq";

    // SIMPLE-style defaults:
    // momentum predictors are intentionally inexact and can be orthogonal-only;
    // pressure correction can use several non-orthogonal correction sweeps.
    int momNonOrthCorr = 0;
    int pNonOrthCorr = 4;
    std::string momDiffusionScheme = "orth";
    std::string pLaplacianScheme = "nonorth";

    double momAbsTol = 1e-7;
    double momRelTol = 1e-5;
    double pAbsTol = 1e-10;

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
      } else if (a == "-device") {
        need("-device");
        device = std::atoi(argv[++i]);
      } else if (a == "-mu") {
        need("-mu");
        mu = std::atof(argv[++i]);
      } else if (a == "-tolMass") {
        need("-tolMass");
        tolMass = std::atof(argv[++i]);
      } else if (a == "-tolVel") {
        need("-tolVel");
        tolVel = std::atof(argv[++i]);
      } else if (a == "-nsteps") {
        need("-nsteps");
        nsteps = std::atoi(argv[++i]);
      } else if (a == "-min-steps") {
        need("-min-steps");
        minSteps = std::atoi(argv[++i]);
      } else if (a == "-maxit") {
        need("-maxit");
        maxit = std::atoi(argv[++i]);
      } else if (a == "-p-relax") {
        need("-p-relax");
        pRelax = std::atof(argv[++i]);
      } else if (a == "-grad-scheme") {
        need("-grad-scheme");
        gradScheme = argv[++i];
      } else if (a == "-nNonOrthCorr") {
        // Backward-compatible: set pressure correctors. Momentum stays with its own default.
        need("-nNonOrthCorr");
        pNonOrthCorr = std::atoi(argv[++i]);
      } else if (a == "-mom-nNonOrthCorr") {
        need("-mom-nNonOrthCorr");
        momNonOrthCorr = std::atoi(argv[++i]);
      } else if (a == "-p-nNonOrthCorr") {
        need("-p-nNonOrthCorr");
        pNonOrthCorr = std::atoi(argv[++i]);
      } else if (a == "-mom-diffusion-scheme") {
        need("-mom-diffusion-scheme");
        momDiffusionScheme = argv[++i];
      } else if (a == "-p-laplacian-scheme") {
        need("-p-laplacian-scheme");
        pLaplacianScheme = argv[++i];
      } else if (a == "-mom-absTol") {
        need("-mom-absTol");
        momAbsTol = std::atof(argv[++i]);
      } else if (a == "-mom-relTol") {
        need("-mom-relTol");
        momRelTol = std::atof(argv[++i]);
      } else if (a == "-p-absTol") {
        need("-p-absTol");
        pAbsTol = std::atof(argv[++i]);
      }
    }

    int devCount = 0;
    cuda_check(cudaGetDeviceCount(&devCount), "cudaGetDeviceCount");
    if (devCount > 0) cuda_check(cudaSetDevice(device % devCount), "cudaSetDevice");

    DecompMesh dm = read_decomposed_openfoam_case(caseRoot, MPI_COMM_WORLD);

    const Mesh& mesh = dm.mesh;

    ScalarBCSet ubc[3] = {
      make_component_velocity_bc(dm, 0),
      make_component_velocity_bc(dm, 1),
      make_component_velocity_bc(dm, 2)
    };

    ScalarBCSet pressureGradBC = make_pressure_exact_dirichlet_bc(dm);
    ScalarBCSet pcorrBC = make_pcorr_zero_neumann_bc(dm);

    LocalBoundaryFaceData pressureGradFaceBC = build_boundary_face_data(dm, pressureGradBC);
    LocalBoundaryFaceData pcorrFaceBC = build_boundary_face_data(dm, pcorrBC);

    std::vector<double> U(mesh.nCells, 0.0), V(mesh.nCells, 0.0), W(mesh.nCells, 0.0);
    std::vector<double> p(mesh.nCells, 0.0);

    std::vector<double> rAU = compute_rAU(dm, mu);
    std::vector<double> pGammaFace = build_pressure_gamma_faces(dm, rAU);

    libscalar_decomp::DistScalarTransportOptions momOpt;
    momOpt.convectionScheme = libscalar_decomp::DistConvectionScheme::Central;
    momOpt.diffusionScheme = libscalar_decomp::diffusion_scheme_from_string(momDiffusionScheme);
    momOpt.gradScheme = gradScheme;
    momOpt.nNonOrthCorr = momNonOrthCorr;
    momOpt.solver.maxIter = maxit;
    // SIMPLE momentum predictors are intentionally inexact.
    momOpt.solver.absTol = momAbsTol;
    momOpt.solver.relTol = momRelTol;
    momOpt.solver.monitor = 0;

    DistEllipticOptions pOpt;
    pOpt.gradScheme = gradScheme;
    pOpt.laplacianScheme = pLaplacianScheme;
    pOpt.nNonOrthCorr = pNonOrthCorr;
    pOpt.useReferenceCell = true;
    pOpt.referenceGlobalCell = 0;
    pOpt.referenceValue = 0.0;
    pOpt.hypre.maxIter = maxit;
    pOpt.hypre.absTol = pAbsTol;
    pOpt.hypre.relTol = 0.0;
    pOpt.hypre.tol = pAbsTol;
    pOpt.hypre.monitor = 0;
    pOpt.hypre.amgMaxIter = 1;
    pOpt.hypre.amgRelaxType = 18;
    pOpt.hypre.amgCoarsenType = 8;
    pOpt.hypre.amgInterpType = 6;
    pOpt.hypre.amgAggLevels = 1;
    pOpt.hypre.amgPmax = 4;
    pOpt.hypre.amgKeepTranspose = 1;

    if (rank == 0) {
      std::printf("STOKES_SIMPLE_DECOMP_MMS setup: ranks=%d globalRows=%lld case=%s mu=%.6e tolMass=%.3e tolVel=%.3e nsteps=%d minSteps=%d pRelax=%.3f grad=%s momScheme=%s momNonOrth=%d pScheme=%s pNonOrth=%d momTol(abs/rel)=%.1e/%.1e pAbsTol=%.1e\n",
                  size, (long long)dm.globalN, caseRoot.c_str(), mu, tolMass, tolVel, nsteps, minSteps, pRelax,
                  gradScheme.c_str(),
                  momDiffusionScheme.c_str(), momNonOrthCorr,
                  pLaplacianScheme.c_str(), pNonOrthCorr,
                  momAbsTol, momRelTol, pAbsTol);
      std::fflush(stdout);
    }

    double lastMass = 1e300;

    for (int step = 1; step <= nsteps; ++step) {
      std::vector<double> Uold = U;
      std::vector<double> Vold = V;
      std::vector<double> Wold = W;
      std::vector<double> pold = p;

      std::vector<std::array<double,3>> gradP;
      compute_lsq_gradient(dm, p, pressureGradFaceBC, gradP);

      libscalar_decomp::DistScalarTransportInputs in[3];

      for (int comp = 0; comp < 3; ++comp) {
        in[comp].faceFlux.assign(mesh.nFaces, 0.0);
        in[comp].gammaFace.assign(mesh.nFaces, mu);
        in[comp].Su.assign(mesh.nCells, 0.0);
        in[comp].Sp.assign(mesh.nCells, 0.0);

        for (int c = 0; c < mesh.nCells; ++c) {
          const auto f = stokes_source_exact(mesh.cc[c], mu);
          in[comp].Su[c] = f[comp] - gradP[c][comp];
        }
      }

      auto rx = libscalar_decomp::solve_scalar_transport_decomp(dm, in[0], ubc[0], momOpt, U);
      auto ry = libscalar_decomp::solve_scalar_transport_decomp(dm, in[1], ubc[1], momOpt, V);
      auto rz = libscalar_decomp::solve_scalar_transport_decomp(dm, in[2], ubc[2], momOpt, W);

      U = std::move(rx.phi);
      V = std::move(ry.phi);
      W = std::move(rz.phi);

      auto phiStar = build_flux_from_velocity(dm, U, V, W);
      auto divStar = compute_divergence_sum(dm, phiStar);
      const double massStar = global_max_abs(dm, divStar);

      std::vector<double> pCorrSource(mesh.nCells, 0.0);
      for (int c = 0; c < mesh.nCells; ++c) {
        pCorrSource[c] = -divStar[c] / std::max(mesh.vol[c], 1e-300);
      }

      auto pCorrResult = solve_scalar_elliptic_decomp(dm, pGammaFace, pCorrSource, pcorrBC, pOpt);
      std::vector<double> pCorr = std::move(pCorrResult.phi);

      std::vector<std::array<double,3>> gradPCorr;
      compute_lsq_gradient(dm, pCorr, pcorrFaceBC, gradPCorr);

      for (int c = 0; c < mesh.nCells; ++c) {
        U[c] -= rAU[c] * gradPCorr[c][0];
        V[c] -= rAU[c] * gradPCorr[c][1];
        W[c] -= rAU[c] * gradPCorr[c][2];
        p[c] += pRelax * pCorr[c];
      }

      auto phi = correct_flux_with_pcorr(dm, phiStar, pCorr, pGammaFace, gradPCorr);
      auto divAfter = compute_divergence_sum(dm, phi);
      const double massAfter = global_max_abs(dm, divAfter);

      lastMass = massAfter;

      double localDU = 0.0;
      double localUScale = 0.0;
      double localDP = 0.0;
      double localPScale = 0.0;

      for (int c = 0; c < mesh.nCells; ++c) {
        const double du0 = U[c] - Uold[c];
        const double du1 = V[c] - Vold[c];
        const double du2 = W[c] - Wold[c];

        const double duMag = std::sqrt(du0*du0 + du1*du1 + du2*du2);
        const double uMag  = std::sqrt(U[c]*U[c] + V[c]*V[c] + W[c]*W[c]);

        localDU = std::max(localDU, duMag);
        localUScale = std::max(localUScale, uMag);

        localDP = std::max(localDP, std::abs(p[c] - pold[c]));
        localPScale = std::max(localPScale, std::abs(p[c]));
      }

      double gDU = 0.0, gUScale = 0.0, gDP = 0.0, gPScale = 0.0;
      MPI_Allreduce(&localDU, &gDU, 1, MPI_DOUBLE, MPI_MAX, dm.comm);
      MPI_Allreduce(&localUScale, &gUScale, 1, MPI_DOUBLE, MPI_MAX, dm.comm);
      MPI_Allreduce(&localDP, &gDP, 1, MPI_DOUBLE, MPI_MAX, dm.comm);
      MPI_Allreduce(&localPScale, &gPScale, 1, MPI_DOUBLE, MPI_MAX, dm.comm);

      const double velRel = gDU / std::max(gUScale, 1.0e-30);
      const double pRel = gDP / std::max(gPScale, 1.0e-30);

      double localL2U = 0.0, localInfU = 0.0, localVol = 0.0;
      double localL2P = 0.0, localInfP = 0.0;

      for (int c = 0; c < mesh.nCells; ++c) {
        const auto ue = u_exact(mesh.cc[c]);
        const double eu0 = U[c] - ue[0];
        const double eu1 = V[c] - ue[1];
        const double eu2 = W[c] - ue[2];
        const double eu = std::sqrt(eu0*eu0 + eu1*eu1 + eu2*eu2);

        localL2U += eu * eu * mesh.vol[c];
        localInfU = std::max(localInfU, eu);

        const double ep = p[c] - p_exact(mesh.cc[c]);
        localL2P += ep * ep * mesh.vol[c];
        localInfP = std::max(localInfP, std::abs(ep));

        localVol += mesh.vol[c];
      }

      double gL2U=0, gInfU=0, gL2P=0, gInfP=0, gVol=0;
      MPI_Allreduce(&localL2U, &gL2U, 1, MPI_DOUBLE, MPI_SUM, dm.comm);
      MPI_Allreduce(&localInfU, &gInfU, 1, MPI_DOUBLE, MPI_MAX, dm.comm);
      MPI_Allreduce(&localL2P, &gL2P, 1, MPI_DOUBLE, MPI_SUM, dm.comm);
      MPI_Allreduce(&localInfP, &gInfP, 1, MPI_DOUBLE, MPI_MAX, dm.comm);
      MPI_Allreduce(&localVol, &gVol, 1, MPI_DOUBLE, MPI_SUM, dm.comm);

      gL2U = std::sqrt(gL2U / std::max(gVol, 1e-300));
      gL2P = std::sqrt(gL2P / std::max(gVol, 1e-300));

      if (rank == 0) {
        std::printf("iter %4d : massStar=%.12e massRes=%.12e velRel=%.12e pChangeRel=%.12e "
                    "UL2=%.12e Uinf=%.12e pL2=%.12e pInf=%.12e "
                    "velIts=[%d %d %d] pIts=%d pSolveRel=%.3e\n",
                    step, massStar, massAfter, velRel, pRel,
                    gL2U, gInfU, gL2P, gInfP,
                    rx.iterations, ry.iterations, rz.iterations,
                    pCorrResult.lastSolveInfo.iterations,
                    pCorrResult.lastSolveInfo.finalRelResNorm);
        std::fflush(stdout);
      }

      if (step >= minSteps && massAfter < tolMass && velRel < tolVel) {
        if (rank == 0) {
          std::printf("STOKES_SIMPLE_DECOMP_MMS Converged at iteration %d : massRes=%.12e velRel=%.12e\n",
                      step, massAfter, velRel);
        }
        break;
      }
    }

    if (rank == 0) {
      std::printf("STOKES_SIMPLE_DECOMP_MMS FINAL: massRes=%.12e\n", lastMass);
      std::printf("STOKES_SIMPLE_DECOMP_MMS PASS_RAN\n");
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
