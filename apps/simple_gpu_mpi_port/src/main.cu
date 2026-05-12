#include <mpi.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <exception>
#include <fstream>
#include <map>
#include <regex>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "poisson_decomp_library.h"
#include "scalar_decomp_library.h"
#include "bc_runtime_config.h"
#include "patch_geometry.h"
#include "velocity_bc_eval.h"

static void cuda_check(cudaError_t e, const char* what) {
  if (e != cudaSuccess) {
    throw std::runtime_error(std::string(what) + ": " + cudaGetErrorString(e));
  }
}



struct BasicPatchInfoLocal {
  std::string name;
  std::string type;
  int nFaces = 0;
  int startFace = 0;
};

static std::string read_text_file_local(const std::string& path) {
  std::ifstream in(path);
  if (!in) {
    throw std::runtime_error("Could not open " + path);
  }
  std::ostringstream ss;
  ss << in.rdbuf();
  return ss.str();
}

static int find_int_entry_local(const std::string& body, const std::string& key, int def = -1) {
  std::regex re("\\b" + key + R"(\s+([-+]?[0-9]+)\s*;)");
  std::smatch m;
  if (std::regex_search(body, m, re)) return std::stoi(m[1].str());
  return def;
}

static std::string find_word_entry_local(const std::string& body, const std::string& key) {
  std::regex re("\\b" + key + R"(\s+([A-Za-z0-9_]+)\s*;)");
  std::smatch m;
  if (std::regex_search(body, m, re)) return m[1].str();
  return "";
}

static std::vector<BasicPatchInfoLocal> read_basic_patch_table_local(const std::string& boundaryPath) {
  const std::string txt = read_text_file_local(boundaryPath);

  std::vector<BasicPatchInfoLocal> out;

  std::regex blockRe(R"(([A-Za-z0-9_]+)\s*\{([^{}]*)\})");
  auto begin = std::sregex_iterator(txt.begin(), txt.end(), blockRe);
  auto end = std::sregex_iterator();

  for (auto it = begin; it != end; ++it) {
    BasicPatchInfoLocal p;
    p.name = (*it)[1].str();
    const std::string body = (*it)[2].str();

    p.type = find_word_entry_local(body, "type");
    p.nFaces = find_int_entry_local(body, "nFaces");
    p.startFace = find_int_entry_local(body, "startFace");

    if (p.nFaces >= 0 && p.startFace >= 0) {
      out.push_back(p);
    }
  }

  if (out.empty()) {
    throw std::runtime_error("No patch table entries found in " + boundaryPath);
  }

  return out;
}


static double dot3_local(const std::array<double,3>& a, const std::array<double,3>& b) {
  return a[0]*b[0] + a[1]*b[1] + a[2]*b[2];
}

static int find_patch_index_local(const std::vector<std::string>& names, const std::string& name) {
  for (int i = 0; i < static_cast<int>(names.size()); ++i) {
    if (names[i] == name) return i;
  }
  return -1;
}

static void print_patch_audit_rank0(
    int rank,
    const Mesh& mesh,
    const std::string& patchName,
    const std::vector<std::string>& bcUType,
    const std::vector<std::string>& bcPType,
    const std::vector<double>& uFaceBC,
    const std::vector<double>& vFaceBC,
    const std::vector<double>& wFaceBC,
    MPI_Comm comm)
{
  const int pidx = find_patch_index_local(mesh.patchNames, patchName);

  double localArea = 0.0;
  double localFlux = 0.0;
  double localUMin = 1.0e300;
  double localUMax = -1.0e300;
  double localUSumA = 0.0;

  int localFaces = 0;

  if (pidx >= 0) {
    const int start = mesh.patchStartFace[pidx];
    const int nFaces = mesh.patchNFaces[pidx];

    for (int i = 0; i < nFaces; ++i) {
      const int f = start + i;

      const double ux = uFaceBC[f];
      const double uy = vFaceBC[f];
      const double uz = wFaceBC[f];

      const double Umag = std::sqrt(ux*ux + uy*uy + uz*uz);
      const double flux = ux*mesh.Sf[f][0] + uy*mesh.Sf[f][1] + uz*mesh.Sf[f][2];

      localArea += mesh.Af[f];
      localFlux += flux;
      localUMin = std::min(localUMin, Umag);
      localUMax = std::max(localUMax, Umag);
      localUSumA += Umag * mesh.Af[f];
      localFaces += 1;
    }
  }

  double globalArea = 0.0, globalFlux = 0.0, globalUSumA = 0.0;
  double globalUMin = 1.0e300, globalUMax = -1.0e300;
  int globalFaces = 0;

  MPI_Reduce(&localArea, &globalArea, 1, MPI_DOUBLE, MPI_SUM, 0, comm);
  MPI_Reduce(&localFlux, &globalFlux, 1, MPI_DOUBLE, MPI_SUM, 0, comm);
  MPI_Reduce(&localUSumA, &globalUSumA, 1, MPI_DOUBLE, MPI_SUM, 0, comm);
  MPI_Reduce(&localUMin, &globalUMin, 1, MPI_DOUBLE, MPI_MIN, 0, comm);
  MPI_Reduce(&localUMax, &globalUMax, 1, MPI_DOUBLE, MPI_MAX, 0, comm);
  MPI_Reduce(&localFaces, &globalFaces, 1, MPI_INT, MPI_SUM, 0, comm);

  if (rank == 0) {
    const double avgU = globalUSumA / std::max(globalArea, 1.0e-300);

    std::printf("Runtime BC patch audit [%s]\n", patchName.c_str());
    std::printf("  global faces       : %d\n", globalFaces);
    std::printf("  global area        : %.15e\n", globalArea);
    std::printf("  BC |U| min/max/avg : %.15e %.15e %.15e\n", globalUMin, globalUMax, avgU);
    std::printf("  BC outward flux    : %.15e\n", globalFlux);

    if (pidx >= 0) {
      std::printf("  rank0 type U/P     : %s / %s\n",
                  bcUType[pidx].c_str(), bcPType[pidx].c_str());
    } else {
      std::printf("  rank0 type U/P     : patch not present on rank0\n");
    }
  }
}



static double face_lambda_local_mpi_port(const Mesh& mesh, int f) {
  const int P = mesh.owner[f];
  const int N = mesh.neigh[f];

  const auto d = std::array<double,3>{
      mesh.cc[N][0] - mesh.cc[P][0],
      mesh.cc[N][1] - mesh.cc[P][1],
      mesh.cc[N][2] - mesh.cc[P][2]
  };

  const auto dx = std::array<double,3>{
      mesh.xf[f][0] - mesh.cc[P][0],
      mesh.xf[f][1] - mesh.cc[P][1],
      mesh.xf[f][2] - mesh.cc[P][2]
  };

  const double dd = dot3_local(d, d);
  const double lam = dot3_local(dx, d) / std::max(dd, 1.0e-300);
  return std::min(1.0, std::max(0.0, lam));
}

static double face_lambda_proc_mpi_port(const DecompMesh& dm, int f) {
  const Mesh& mesh = dm.mesh;
  const int P = mesh.owner[f];

  const auto d = std::array<double,3>{
      dm.remoteCCForFace[f][0] - mesh.cc[P][0],
      dm.remoteCCForFace[f][1] - mesh.cc[P][1],
      dm.remoteCCForFace[f][2] - mesh.cc[P][2]
  };

  const auto dx = std::array<double,3>{
      mesh.xf[f][0] - mesh.cc[P][0],
      mesh.xf[f][1] - mesh.cc[P][1],
      mesh.xf[f][2] - mesh.cc[P][2]
  };

  const double dd = dot3_local(d, d);
  const double lam = dot3_local(dx, d) / std::max(dd, 1.0e-300);
  return std::min(1.0, std::max(0.0, lam));
}

static int patch_index_for_face_mpi_port(const Mesh& meshBC, int f) {
  for (int ip = 0; ip < static_cast<int>(meshBC.patchNames.size()); ++ip) {
    const int start = meshBC.patchStartFace[ip];
    const int end = start + meshBC.patchNFaces[ip];
    if (f >= start && f < end) return ip;
  }
  return -1;
}

static std::vector<double> build_face_flux_mpi_port(
    const DecompMesh& dm,
    const Mesh& meshBC,
    const std::vector<std::string>& bcUType,
    const std::vector<double>& U,
    const std::vector<double>& V,
    const std::vector<double>& W,
    const std::vector<double>& uFaceBC,
    const std::vector<double>& vFaceBC,
    const std::vector<double>& wFaceBC)
{
  const Mesh& mesh = dm.mesh;

  const auto rU = exchange_proc_face_scalar_owner_values(dm, U);
  const auto rV = exchange_proc_face_scalar_owner_values(dm, V);
  const auto rW = exchange_proc_face_scalar_owner_values(dm, W);

  std::vector<double> phi(mesh.nFaces, 0.0);

  for (int f = 0; f < mesh.nInternalFaces; ++f) {
    const int P = mesh.owner[f];
    const int N = mesh.neigh[f];

    const double lam = face_lambda_local_mpi_port(mesh, f);

    const double uf = (1.0 - lam) * U[P] + lam * U[N];
    const double vf = (1.0 - lam) * V[P] + lam * V[N];
    const double wf = (1.0 - lam) * W[P] + lam * W[N];

    phi[f] = uf * mesh.Sf[f][0] + vf * mesh.Sf[f][1] + wf * mesh.Sf[f][2];
  }

  for (int f = mesh.nInternalFaces; f < mesh.nFaces; ++f) {
    const int P = mesh.owner[f];

    if (dm.isProcFace[f]) {
      const double lam = face_lambda_proc_mpi_port(dm, f);

      const double uf = (1.0 - lam) * U[P] + lam * rU[f];
      const double vf = (1.0 - lam) * V[P] + lam * rV[f];
      const double wf = (1.0 - lam) * W[P] + lam * rW[f];

      phi[f] = uf * mesh.Sf[f][0] + vf * mesh.Sf[f][1] + wf * mesh.Sf[f][2];
    } else {
      const int ip = patch_index_for_face_mpi_port(meshBC, f);

      double uf = U[P];
      double vf = V[P];
      double wf = W[P];

      if (ip >= 0 && ip < static_cast<int>(bcUType.size()) && bcUType[ip] == "Dirichlet") {
        uf = uFaceBC[f];
        vf = vFaceBC[f];
        wf = wFaceBC[f];
      }

      phi[f] = uf * mesh.Sf[f][0] + vf * mesh.Sf[f][1] + wf * mesh.Sf[f][2];
    }
  }

  return phi;
}

static std::vector<double> compute_cell_div_sum_mpi_port(
    const DecompMesh& dm,
    const std::vector<double>& phi)
{
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

static void print_flux_divergence_audit_rank0(
    int rank,
    const DecompMesh& dm,
    const std::vector<double>& phi,
    MPI_Comm comm)
{
  const Mesh& mesh = dm.mesh;

  double localPhysicalFlux = 0.0;
  double localProcFlux = 0.0;
  double localAbsPhysicalFlux = 0.0;
  double localAbsProcFlux = 0.0;

  for (int f = mesh.nInternalFaces; f < mesh.nFaces; ++f) {
    if (dm.isProcFace[f]) {
      localProcFlux += phi[f];
      localAbsProcFlux += std::abs(phi[f]);
    } else {
      localPhysicalFlux += phi[f];
      localAbsPhysicalFlux += std::abs(phi[f]);
    }
  }

  const auto div = compute_cell_div_sum_mpi_port(dm, phi);

  double localDivLinf = 0.0;
  double localDivL1 = 0.0;
  double localDivSum = 0.0;

  for (double d : div) {
    localDivLinf = std::max(localDivLinf, std::abs(d));
    localDivL1 += std::abs(d);
    localDivSum += d;
  }

  double globalPhysicalFlux = 0.0;
  double globalProcFlux = 0.0;
  double globalAbsPhysicalFlux = 0.0;
  double globalAbsProcFlux = 0.0;
  double globalDivLinf = 0.0;
  double globalDivL1 = 0.0;
  double globalDivSum = 0.0;

  MPI_Reduce(&localPhysicalFlux, &globalPhysicalFlux, 1, MPI_DOUBLE, MPI_SUM, 0, comm);
  MPI_Reduce(&localProcFlux, &globalProcFlux, 1, MPI_DOUBLE, MPI_SUM, 0, comm);
  MPI_Reduce(&localAbsPhysicalFlux, &globalAbsPhysicalFlux, 1, MPI_DOUBLE, MPI_SUM, 0, comm);
  MPI_Reduce(&localAbsProcFlux, &globalAbsProcFlux, 1, MPI_DOUBLE, MPI_SUM, 0, comm);
  MPI_Reduce(&localDivLinf, &globalDivLinf, 1, MPI_DOUBLE, MPI_MAX, 0, comm);
  MPI_Reduce(&localDivL1, &globalDivL1, 1, MPI_DOUBLE, MPI_SUM, 0, comm);
  MPI_Reduce(&localDivSum, &globalDivSum, 1, MPI_DOUBLE, MPI_SUM, 0, comm);

  if (rank == 0) {
    std::printf("Initial flux/divergence audit\n");
    std::printf("  physical boundary net flux      : %.15e\n", globalPhysicalFlux);
    std::printf("  physical boundary abs flux sum  : %.15e\n", globalAbsPhysicalFlux);
    std::printf("  processor boundary net flux     : %.15e\n", globalProcFlux);
    std::printf("  processor boundary abs flux sum : %.15e\n", globalAbsProcFlux);
    std::printf("  cell divergence sum             : %.15e\n", globalDivSum);
    std::printf("  cell divergence L1              : %.15e\n", globalDivL1);
    std::printf("  cell divergence Linf            : %.15e\n", globalDivLinf);
  }
}



static ScalarBCSet make_component_scalar_bc_from_legacy_mpi_port(
    const Mesh& meshBC,
    const std::set<std::string>& procPatchNames,
    const std::vector<std::string>& bcUType,
    const std::vector<double>& uFaceBC,
    const std::vector<double>& vFaceBC,
    const std::vector<double>& wFaceBC,
    int comp)
{
  ScalarBCSet bc;

  for (int ip = 0; ip < static_cast<int>(meshBC.patchNames.size()); ++ip) {
    const std::string& name = meshBC.patchNames[ip];

    if (procPatchNames.find(name) != procPatchNames.end()) {
      continue;
    }

    const int start = meshBC.patchStartFace[ip];
    const int nFaces = meshBC.patchNFaces[ip];

    if (ip < static_cast<int>(bcUType.size()) && bcUType[ip] == "Dirichlet") {
      double sumA = 0.0;
      double sumVA = 0.0;

      for (int i = 0; i < nFaces; ++i) {
        const int f = start + i;
        const double val = (comp == 0) ? uFaceBC[f] :
                           (comp == 1) ? vFaceBC[f] :
                                         wFaceBC[f];

        sumA += meshBC.Af[f];
        sumVA += val * meshBC.Af[f];
      }

      const double avg = sumVA / std::max(sumA, 1.0e-300);
      bc.patches.push_back(make_dirichlet_constant_bc(name, avg));
    } else {
      // Zero-gradient velocity outlet / open style.
      bc.patches.push_back(make_neumann_constant_bc(name, 0.0));
    }
  }

  return bc;
}

static double global_max_abs_vec_mpi_port(
    const std::vector<double>& a,
    MPI_Comm comm)
{
  double local = 0.0;
  for (double v : a) local = std::max(local, std::abs(v));

  double global = 0.0;
  MPI_Allreduce(&local, &global, 1, MPI_DOUBLE, MPI_MAX, comm);
  return global;
}

static void print_velocity_predictor_audit_rank0(
    int rank,
    const DecompMesh& dm,
    const std::vector<double>& U,
    const std::vector<double>& V,
    const std::vector<double>& W,
    const std::vector<double>& phiPred,
    int uIts,
    int vIts,
    int wIts,
    double uRel,
    double vRel,
    double wRel,
    MPI_Comm comm)
{
  const auto div = compute_cell_div_sum_mpi_port(dm, phiPred);

  double localDivInf = 0.0;
  double localDivL1 = 0.0;
  double localDivSum = 0.0;

  for (double d : div) {
    localDivInf = std::max(localDivInf, std::abs(d));
    localDivL1 += std::abs(d);
    localDivSum += d;
  }

  double globalDivInf = 0.0;
  double globalDivL1 = 0.0;
  double globalDivSum = 0.0;

  MPI_Reduce(&localDivInf, &globalDivInf, 1, MPI_DOUBLE, MPI_MAX, 0, comm);
  MPI_Reduce(&localDivL1, &globalDivL1, 1, MPI_DOUBLE, MPI_SUM, 0, comm);
  MPI_Reduce(&localDivSum, &globalDivSum, 1, MPI_DOUBLE, MPI_SUM, 0, comm);

  const double Umax = global_max_abs_vec_mpi_port(U, comm);
  const double Vmax = global_max_abs_vec_mpi_port(V, comm);
  const double Wmax = global_max_abs_vec_mpi_port(W, comm);

  if (rank == 0) {
    std::printf("First distributed momentum predictor audit\n");
    std::printf("  U/V/W max          : %.15e %.15e %.15e\n", Umax, Vmax, Wmax);
    std::printf("  velocity iterations: [%d %d %d]\n", uIts, vIts, wIts);
    std::printf("  velocity finalRel  : %.15e %.15e %.15e\n", uRel, vRel, wRel);
    std::printf("  predictor div sum  : %.15e\n", globalDivSum);
    std::printf("  predictor div L1   : %.15e\n", globalDivL1);
    std::printf("  predictor div Linf : %.15e\n", globalDivInf);
  }
}



static double momentum_diag_diffusion_cell_mpi_port(
    const DecompMesh& dm,
    double mu,
    int P)
{
  const Mesh& mesh = dm.mesh;
  double diag = 0.0;

  for (int f = 0; f < mesh.nInternalFaces; ++f) {
    if (mesh.owner[f] != P && mesh.neigh[f] != P) continue;

    const int C0 = mesh.owner[f];
    const int C1 = mesh.neigh[f];

    const auto d = std::array<double,3>{
      mesh.cc[C1][0] - mesh.cc[C0][0],
      mesh.cc[C1][1] - mesh.cc[C0][1],
      mesh.cc[C1][2] - mesh.cc[C0][2]
    };

    const double dDotS = dot3_local(d, mesh.Sf[f]);
    const double D = mu * dot3_local(mesh.Sf[f], mesh.Sf[f]) / std::max(dDotS, 1.0e-300);
    diag += D;
  }

  for (int f : mesh.cellBFace[P]) {
    std::array<double,3> d;

    if (dm.isProcFace[f]) {
      d = {
        dm.remoteCCForFace[f][0] - mesh.cc[P][0],
        dm.remoteCCForFace[f][1] - mesh.cc[P][1],
        dm.remoteCCForFace[f][2] - mesh.cc[P][2]
      };
    } else {
      d = {
        mesh.xf[f][0] - mesh.cc[P][0],
        mesh.xf[f][1] - mesh.cc[P][1],
        mesh.xf[f][2] - mesh.cc[P][2]
      };
    }

    const double dDotS = dot3_local(d, mesh.Sf[f]);
    const double D = mu * dot3_local(mesh.Sf[f], mesh.Sf[f]) / std::max(dDotS, 1.0e-300);
    diag += D;
  }

  return std::max(diag, 1.0e-300);
}

static std::vector<double> build_rAU_diffusion_mpi_port(
    const DecompMesh& dm,
    double mu)
{
  std::vector<double> rAU(dm.mesh.nCells, 0.0);

  for (int c = 0; c < dm.mesh.nCells; ++c) {
    rAU[c] = 1.0 / momentum_diag_diffusion_cell_mpi_port(dm, mu, c);
  }

  return rAU;
}

static std::vector<double> build_pressure_gamma_faces_mpi_port(
    const DecompMesh& dm,
    const std::vector<double>& rAU)
{
  const Mesh& mesh = dm.mesh;
  const auto rRAU = exchange_proc_face_scalar_owner_values(dm, rAU);

  std::vector<double> gamma(mesh.nFaces, 0.0);

  for (int f = 0; f < mesh.nInternalFaces; ++f) {
    const int P = mesh.owner[f];
    const int N = mesh.neigh[f];
    const double lam = face_lambda_local_mpi_port(mesh, f);
    gamma[f] = (1.0 - lam) * rAU[P] + lam * rAU[N];
  }

  for (int f = mesh.nInternalFaces; f < mesh.nFaces; ++f) {
    const int P = mesh.owner[f];

    if (dm.isProcFace[f]) {
      const double lam = face_lambda_proc_mpi_port(dm, f);
      gamma[f] = (1.0 - lam) * rAU[P] + lam * rRAU[f];
    } else {
      gamma[f] = rAU[P];
    }
  }

  return gamma;
}

static ScalarBCSet make_pcorr_bc_from_legacy_pressure_mpi_port(
    const Mesh& meshBC,
    const std::set<std::string>& procPatchNames,
    const std::vector<std::string>& bcPType)
{
  ScalarBCSet bc;

  for (int ip = 0; ip < static_cast<int>(meshBC.patchNames.size()); ++ip) {
    const std::string& name = meshBC.patchNames[ip];

    if (procPatchNames.find(name) != procPatchNames.end()) {
      continue;
    }

    if (ip < static_cast<int>(bcPType.size()) && bcPType[ip] == "Dirichlet") {
      // Fixed pressure outlet => pCorr = 0.
      bc.patches.push_back(make_dirichlet_constant_bc(name, 0.0));
    } else {
      // Pressure zeroGradient patches => pCorr zeroGradient.
      bc.patches.push_back(make_neumann_constant_bc(name, 0.0));
    }
  }

  return bc;
}

static std::vector<double> correct_flux_orthogonal_pcorr_mpi_port(
    const DecompMesh& dm,
    const Mesh& meshBC,
    const std::vector<std::string>& bcPType,
    const std::vector<double>& phiPred,
    const std::vector<double>& pCorr,
    const std::vector<double>& gammaFace)
{
  const Mesh& mesh = dm.mesh;

  const auto rPCorr = exchange_proc_face_scalar_owner_values(dm, pCorr);

  std::vector<double> phi = phiPred;

  for (int f = 0; f < mesh.nInternalFaces; ++f) {
    const int P = mesh.owner[f];
    const int N = mesh.neigh[f];

    const auto d = std::array<double,3>{
      mesh.cc[N][0] - mesh.cc[P][0],
      mesh.cc[N][1] - mesh.cc[P][1],
      mesh.cc[N][2] - mesh.cc[P][2]
    };

    const double dDotS = dot3_local(d, mesh.Sf[f]);
    const double D = gammaFace[f] * dot3_local(mesh.Sf[f], mesh.Sf[f]) / std::max(dDotS, 1.0e-300);

    const double q = D * (pCorr[N] - pCorr[P]);
    phi[f] -= q;
  }

  for (int f = mesh.nInternalFaces; f < mesh.nFaces; ++f) {
    if (!dm.isProcFace[f]) continue;

    const int P = mesh.owner[f];

    const auto d = std::array<double,3>{
      dm.remoteCCForFace[f][0] - mesh.cc[P][0],
      dm.remoteCCForFace[f][1] - mesh.cc[P][1],
      dm.remoteCCForFace[f][2] - mesh.cc[P][2]
    };

    const double dDotS = dot3_local(d, mesh.Sf[f]);
    const double D = gammaFace[f] * dot3_local(mesh.Sf[f], mesh.Sf[f]) / std::max(dDotS, 1.0e-300);

    const double q = D * (rPCorr[f] - pCorr[P]);
    phi[f] -= q;
  }

  // Physical pCorr boundaries: fixed pressure gives pCorr=0 but no flux correction
  // through the physical boundary in this first checkpoint. Inlet/wall are zeroGradient.
  // Physical fixed-pressure boundaries also need a pressure-correction
  // flux update. For fixed pressure, pCorr_B = 0, but grad(pCorr) at the
  // outlet face is generally nonzero. Without this, the outlet cannot adjust
  // to balance inlet mass flux.
  for (int f = mesh.nInternalFaces; f < mesh.nFaces; ++f) {
    if (dm.isProcFace[f]) continue;

    const int ip = patch_index_for_face_mpi_port(meshBC, f);

    const bool fixedPressure =
        (ip >= 0 &&
         ip < static_cast<int>(bcPType.size()) &&
         bcPType[ip] == "Dirichlet");

    if (!fixedPressure) continue;

    const int P = mesh.owner[f];

    const std::array<double,3> d = {
      mesh.xf[f][0] - mesh.cc[P][0],
      mesh.xf[f][1] - mesh.cc[P][1],
      mesh.xf[f][2] - mesh.cc[P][2]
    };

    const double dDotS = dot3_local(d, mesh.Sf[f]);
    const double D = gammaFace[f] * dot3_local(mesh.Sf[f], mesh.Sf[f]) / std::max(dDotS, 1.0e-300);

    const double pCorrB = 0.0;
    const double q = D * (pCorrB - pCorr[P]);

    phi[f] -= q;
  }

  return phi;
}

static void print_pressure_correction_audit_rank0(
    int rank,
    const DecompMesh& dm,
    const std::vector<double>& phiPred,
    const std::vector<double>& phiCorr,
    const std::vector<double>& pCorr,
    int pIts,
    double pRel,
    MPI_Comm comm)
{
  const auto divPred = compute_cell_div_sum_mpi_port(dm, phiPred);
  const auto divCorr = compute_cell_div_sum_mpi_port(dm, phiCorr);

  double localPredInf = 0.0, localCorrInf = 0.0;
  double localPredL1 = 0.0, localCorrL1 = 0.0;
  double localPredSum = 0.0, localCorrSum = 0.0;
  double localPCorrMax = 0.0;

  for (double d : divPred) {
    localPredInf = std::max(localPredInf, std::abs(d));
    localPredL1 += std::abs(d);
    localPredSum += d;
  }

  for (double d : divCorr) {
    localCorrInf = std::max(localCorrInf, std::abs(d));
    localCorrL1 += std::abs(d);
    localCorrSum += d;
  }

  for (double v : pCorr) {
    localPCorrMax = std::max(localPCorrMax, std::abs(v));
  }

  double predInf = 0.0, corrInf = 0.0;
  double predL1 = 0.0, corrL1 = 0.0;
  double predSum = 0.0, corrSum = 0.0;
  double pCorrMax = 0.0;

  MPI_Reduce(&localPredInf, &predInf, 1, MPI_DOUBLE, MPI_MAX, 0, comm);
  MPI_Reduce(&localCorrInf, &corrInf, 1, MPI_DOUBLE, MPI_MAX, 0, comm);
  MPI_Reduce(&localPredL1, &predL1, 1, MPI_DOUBLE, MPI_SUM, 0, comm);
  MPI_Reduce(&localCorrL1, &corrL1, 1, MPI_DOUBLE, MPI_SUM, 0, comm);
  MPI_Reduce(&localPredSum, &predSum, 1, MPI_DOUBLE, MPI_SUM, 0, comm);
  MPI_Reduce(&localCorrSum, &corrSum, 1, MPI_DOUBLE, MPI_SUM, 0, comm);
  MPI_Reduce(&localPCorrMax, &pCorrMax, 1, MPI_DOUBLE, MPI_MAX, 0, comm);

  if (rank == 0) {
    std::printf("First distributed pressure correction audit\n");
    std::printf("  pCorr max abs       : %.15e\n", pCorrMax);
    std::printf("  pressure iterations : %d\n", pIts);
    std::printf("  pressure finalRel   : %.15e\n", pRel);
    std::printf("  div before sum/L1/Linf : %.15e %.15e %.15e\n", predSum, predL1, predInf);
    std::printf("  div after  sum/L1/Linf : %.15e %.15e %.15e\n", corrSum, corrL1, corrInf);
    std::printf("  div Linf reduction     : %.15e\n", predInf / std::max(corrInf, 1.0e-300));
  }
}


static bool is_proc_patch_name(const std::set<std::string>& procNames, const std::string& name) {
  return procNames.find(name) != procNames.end();
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

    // Keep common serial-style options accepted even if unused in this checkpoint.
    std::string wallPatch = "patch_0_0";
    std::string inletPatch = "patch_2_0";
    std::string outletPatch = "patch_1_0";
    double rho = 1.0;
    double mu = 0.05;
    double uMean = 1.0;
    int nsteps = 1;

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
      } else if (a == "-wall-patch") {
        need("-wall-patch");
        wallPatch = argv[++i];
      } else if (a == "-inlet-patch") {
        need("-inlet-patch");
        inletPatch = argv[++i];
      } else if (a == "-outlet-patch") {
        need("-outlet-patch");
        outletPatch = argv[++i];
      } else if (a == "-rho") {
        need("-rho");
        rho = std::atof(argv[++i]);
      } else if (a == "-mu") {
        need("-mu");
        mu = std::atof(argv[++i]);
      } else if (a == "-uMean") {
        need("-uMean");
        uMean = std::atof(argv[++i]);
      } else if (a == "-nsteps") {
        need("-nsteps");
        nsteps = std::atoi(argv[++i]);
      } else {
        if (rank == 0) {
          std::printf("simple_gpu_mpi_port checkpoint: ignoring unimplemented option %s\n", a.c_str());
        }
      }
    }

    int devCount = 0;
    cuda_check(cudaGetDeviceCount(&devCount), "cudaGetDeviceCount");
    if (devCount > 0) {
      cuda_check(cudaSetDevice(device % devCount), "cudaSetDevice");
    }

    cudaDeviceProp prop{};
    if (devCount > 0) {
      cuda_check(cudaGetDeviceProperties(&prop, device % devCount), "cudaGetDeviceProperties");
    }

    DecompMesh dm = read_decomposed_openfoam_case(caseRoot, MPI_COMM_WORLD);
    const Mesh& mesh = dm.mesh;

    const std::string localBoundaryPath =
        caseRoot + "/processor" + std::to_string(rank) + "/constant/polyMesh/boundary";

    const auto basicPatchTable = read_basic_patch_table_local(localBoundaryPath);

    std::vector<std::string> patchNamesForBC;
    std::vector<int> patchStartFaceForBC;
    std::vector<int> patchNFacesForBC;

    patchNamesForBC.reserve(basicPatchTable.size());
    patchStartFaceForBC.reserve(basicPatchTable.size());
    patchNFacesForBC.reserve(basicPatchTable.size());

    for (const auto& p : basicPatchTable) {
      patchNamesForBC.push_back(p.name);
      patchStartFaceForBC.push_back(p.startFace);
      patchNFacesForBC.push_back(p.nFaces);
    }

    // Serial BC helpers require a Mesh-like view with populated patch arrays.
    Mesh meshBC = mesh;
    meshBC.patchNames = patchNamesForBC;
    meshBC.patchStartFace = patchStartFaceForBC;
    meshBC.patchNFaces = patchNFacesForBC;

    std::set<std::string> procPatchNames;
    for (const auto& pp : dm.procPatches) {
      procPatchNames.insert(pp.name);
    }

    std::vector<std::string> physicalPatchNames;
    for (const auto& name : patchNamesForBC) {
      if (!is_proc_patch_name(procPatchNames, name)) {
        physicalPatchNames.push_back(name);
      }
    }

    pipebc::RuntimeBCConfig bcConfig;
    bool haveBC = !bcConfigPath.empty();

    if (haveBC) {
      bcConfig = pipebc::load_runtime_bc_config(bcConfigPath);

      // Important: validate only physical patches.
      // OpenFOAM processor patches are coupled internal faces, not user BC patches.
      pipebc::validate_runtime_bc_config_against_patches(bcConfig, physicalPatchNames);
    }

    // Build patch geometry and evaluate the same modular BC specs used by serial simple_gpu.
    pipebc::PatchGeometryInput patchGeomIn;
    patchGeomIn.nInternalFaces = mesh.nInternalFaces;
    patchGeomIn.nFaces = mesh.nFaces;
    patchGeomIn.xf = &mesh.xf;
    patchGeomIn.nf = &mesh.nf;
    patchGeomIn.Sf = &mesh.Sf;
    patchGeomIn.Af = &mesh.Af;
    patchGeomIn.patchNames = &meshBC.patchNames;
    patchGeomIn.patchStartFace = &meshBC.patchStartFace;
    patchGeomIn.patchNFaces = &meshBC.patchNFaces;

    const auto patchGeometryTable = pipebc::build_patch_geometry_table(patchGeomIn);

    pipebc::LegacyBCMeshView legacyBCMesh;
    legacyBCMesh.nFaces = mesh.nFaces;
    legacyBCMesh.nInternalFaces = mesh.nInternalFaces;
    legacyBCMesh.patchNames = &meshBC.patchNames;
    legacyBCMesh.patchStartFace = &meshBC.patchStartFace;
    legacyBCMesh.patchNFaces = &meshBC.patchNFaces;
    legacyBCMesh.xf = &mesh.xf;
    legacyBCMesh.nf = &mesh.nf;

    std::vector<std::string> bcUType(meshBC.patchNames.size(), "Neumann");
    std::vector<std::string> bcVType(meshBC.patchNames.size(), "Neumann");
    std::vector<std::string> bcWType(meshBC.patchNames.size(), "Neumann");
    std::vector<std::string> bcPType(meshBC.patchNames.size(), "Neumann");

    std::vector<double> uFaceBC(mesh.nFaces, 0.0);
    std::vector<double> vFaceBC(mesh.nFaces, 0.0);
    std::vector<double> wFaceBC(mesh.nFaces, 0.0);
    std::vector<double> pFaceBC(mesh.nFaces, 0.0);

    if (haveBC) {
      pipebc::apply_bc_specs_to_legacy_face_arrays(
          legacyBCMesh,
          patchGeometryTable,
          bcConfig.velocityPatchSpecs,
          bcConfig.pressurePatchSpecs,
          0.0,
          bcUType, bcVType, bcWType, bcPType,
          uFaceBC, vFaceBC, wFaceBC, pFaceBC);
    }

    int localProcFaces = 0;
    for (int f = mesh.nInternalFaces; f < mesh.nFaces; ++f) {
      if (dm.isProcFace[f]) ++localProcFaces;
    }

    int localPhysicalFaces = mesh.nFaces - mesh.nInternalFaces - localProcFaces;

    int globalCells = 0;
    int localCellsInt = mesh.nCells;
    MPI_Allreduce(&localCellsInt, &globalCells, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);

    int globalProcFaces = 0;
    int globalPhysicalFaces = 0;
    MPI_Allreduce(&localProcFaces, &globalProcFaces, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
    MPI_Allreduce(&localPhysicalFaces, &globalPhysicalFaces, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);

    MPI_Barrier(MPI_COMM_WORLD);

    std::printf(
      "simple_gpu_mpi_port rank %d/%d: rows=[%lld,%lld] nLocal=%d nFaces=%d internalFaces=%d physicalBoundaryFaces=%d procFaces=%d cudaDevice=%d name=%s\n",
      rank, size,
      (long long)dm.ilower, (long long)dm.iupper,
      mesh.nCells,
      mesh.nFaces,
      mesh.nInternalFaces,
      localPhysicalFaces,
      localProcFaces,
      devCount > 0 ? device % devCount : -1,
      devCount > 0 ? prop.name : "NO_CUDA_DEVICE");

    std::fflush(stdout);
    MPI_Barrier(MPI_COMM_WORLD);

    if (rank == 0) {
      std::printf("====================================================================\n");
      std::printf("simple_gpu_mpi_port FRONTEND CHECKPOINT PASS\n");
      std::printf("caseRoot       : %s\n", caseRoot.c_str());
      std::printf("bcConfig       : %s\n", haveBC ? bcConfigPath.c_str() : "<none>\n");
      std::printf("worldSize      : %d\n", size);
      std::printf("globalCells    : %d\n", globalCells);
      std::printf("globalProcFaces: %d\n", globalProcFaces);
      std::printf("globalPhysFaces: %d\n", globalPhysicalFaces);
      std::printf("rho/mu/uMean   : %.6e / %.6e / %.6e\n", rho, mu, uMean);
      std::printf("patch aliases  : wall=%s inlet=%s outlet=%s\n",
                  wallPatch.c_str(), inletPatch.c_str(), outletPatch.c_str());
      std::printf("physical patches validated against BC config:\n");
      for (const auto& name : physicalPatchNames) {
        std::printf("  physical patch: %s\n", name.c_str());
      }
      std::printf("all local patch table entries on rank0:\n");
      for (size_t ip = 0; ip < meshBC.patchNames.size(); ++ip) {
        std::printf("  patch[%zu] name=%s start=%d nFaces=%d%s\n",
                    ip,
                    meshBC.patchNames[ip].c_str(),
                    meshBC.patchStartFace[ip],
                    meshBC.patchNFaces[ip],
                    is_proc_patch_name(procPatchNames, meshBC.patchNames[ip]) ? " processor" : " physical");
      }
      std::printf("BC entries:\n");
      std::printf("  velocityPatchSpecs = %zu\n", bcConfig.velocityPatchSpecs.size());
      std::printf("  pressurePatchSpecs = %zu\n", bcConfig.pressurePatchSpecs.size());
      std::printf("Patch geometry table local rank0 entries: %zu\n", patchGeometryTable.size());
      std::printf("====================================================================\n");
      std::fflush(stdout);
    }

    if (haveBC) {
      print_patch_audit_rank0(rank, meshBC, wallPatch, bcUType, bcPType,
                              uFaceBC, vFaceBC, wFaceBC, MPI_COMM_WORLD);
      print_patch_audit_rank0(rank, meshBC, inletPatch, bcUType, bcPType,
                              uFaceBC, vFaceBC, wFaceBC, MPI_COMM_WORLD);
      print_patch_audit_rank0(rank, meshBC, outletPatch, bcUType, bcPType,
                              uFaceBC, vFaceBC, wFaceBC, MPI_COMM_WORLD);


      // Initial field/flux checkpoint.
      std::vector<double> U(mesh.nCells, 0.0);
      std::vector<double> V(mesh.nCells, 0.0);
      std::vector<double> W(mesh.nCells, 0.0);
      std::vector<double> pField(mesh.nCells, 0.0);

      const auto phi0 = build_face_flux_mpi_port(
          dm, meshBC, bcUType,
          U, V, W,
          uFaceBC, vFaceBC, wFaceBC);

      print_flux_divergence_audit_rank0(rank, dm, phi0, MPI_COMM_WORLD);

      // First distributed momentum predictor.
      // This is not yet the final serial simple_gpu momentum physics;
      // it is the first MPI momentum-solve checkpoint using libscalar_decomp.
      libscalar_decomp::DistScalarTransportInputs momIn;
      momIn.faceFlux.assign(mesh.nFaces, 0.0);
      momIn.gammaFace.assign(mesh.nFaces, mu);
      momIn.Su.assign(mesh.nCells, 0.0);
      momIn.Sp.assign(mesh.nCells, 0.0);

      for (int f = 0; f < mesh.nFaces; ++f) {
        momIn.faceFlux[f] = rho * phi0[f];
      }

      ScalarBCSet uCompBC = make_component_scalar_bc_from_legacy_mpi_port(
          meshBC, procPatchNames, bcUType, uFaceBC, vFaceBC, wFaceBC, 0);
      ScalarBCSet vCompBC = make_component_scalar_bc_from_legacy_mpi_port(
          meshBC, procPatchNames, bcUType, uFaceBC, vFaceBC, wFaceBC, 1);
      ScalarBCSet wCompBC = make_component_scalar_bc_from_legacy_mpi_port(
          meshBC, procPatchNames, bcUType, uFaceBC, vFaceBC, wFaceBC, 2);

      libscalar_decomp::DistScalarTransportOptions momOpt;
      momOpt.convectionScheme = libscalar_decomp::DistConvectionScheme::Upwind;
      momOpt.diffusionScheme = libscalar_decomp::DistDiffusionScheme::Orth;
      momOpt.gradScheme = "lsq";
      momOpt.nNonOrthCorr = 0;
      momOpt.solver.maxIter = 500;
      momOpt.solver.absTol = 1.0e-7;
      momOpt.solver.relTol = 1.0e-5;
      momOpt.solver.monitor = 0;

      auto uRes = libscalar_decomp::solve_scalar_transport_decomp(
          dm, momIn, uCompBC, momOpt, U);
      auto vRes = libscalar_decomp::solve_scalar_transport_decomp(
          dm, momIn, vCompBC, momOpt, V);
      auto wRes = libscalar_decomp::solve_scalar_transport_decomp(
          dm, momIn, wCompBC, momOpt, W);

      U = std::move(uRes.phi);
      V = std::move(vRes.phi);
      W = std::move(wRes.phi);

      const auto phiPred = build_face_flux_mpi_port(
          dm, meshBC, bcUType,
          U, V, W,
          uFaceBC, vFaceBC, wFaceBC);

      print_velocity_predictor_audit_rank0(
          rank, dm, U, V, W, phiPred,
          uRes.iterations, vRes.iterations, wRes.iterations,
          uRes.finalRelRes, vRes.finalRelRes, wRes.finalRelRes,
          MPI_COMM_WORLD);


      // First distributed pressure correction checkpoint.
      const auto rAU = build_rAU_diffusion_mpi_port(dm, mu);
      const auto pGammaFace = build_pressure_gamma_faces_mpi_port(dm, rAU);
      const auto divPred = compute_cell_div_sum_mpi_port(dm, phiPred);

      std::vector<double> pCorrSource(mesh.nCells, 0.0);
      for (int c = 0; c < mesh.nCells; ++c) {
        pCorrSource[c] = -divPred[c] / std::max(mesh.vol[c], 1.0e-300);
      }

      ScalarBCSet pCorrBC = make_pcorr_bc_from_legacy_pressure_mpi_port(
          meshBC, procPatchNames, bcPType);

      DistEllipticOptions pOpt;
      pOpt.gradScheme = "lsq";
      pOpt.laplacianScheme = "orth";
      pOpt.nNonOrthCorr = 0;
      pOpt.useReferenceCell = false;
      pOpt.referenceGlobalCell = 0;
      pOpt.referenceValue = 0.0;
      pOpt.hypre.maxIter = 1000;
      pOpt.hypre.absTol = 1.0e-9;
      pOpt.hypre.relTol = 0.0;
      pOpt.hypre.tol = 1.0e-9;
      pOpt.hypre.monitor = 0;
      pOpt.hypre.amgMaxIter = 1;
      pOpt.hypre.amgRelaxType = 18;
      pOpt.hypre.amgCoarsenType = 8;
      pOpt.hypre.amgInterpType = 6;
      pOpt.hypre.amgAggLevels = 1;
      pOpt.hypre.amgPmax = 4;
      pOpt.hypre.amgKeepTranspose = 1;

      auto pCorrRes = solve_scalar_elliptic_decomp(
          dm, pGammaFace, pCorrSource, pCorrBC, pOpt);

      const auto phiCorr = correct_flux_orthogonal_pcorr_mpi_port(
          dm, meshBC, bcPType, phiPred, pCorrRes.phi, pGammaFace);

      print_pressure_correction_audit_rank0(
          rank, dm, phiPred, phiCorr, pCorrRes.phi,
          pCorrRes.lastSolveInfo.iterations,
          pCorrRes.lastSolveInfo.finalRelResNorm,
          MPI_COMM_WORLD);


      if (rank == 0) {
        std::printf("NOTE: this checkpoint exits after first distributed momentum predictor + pressure correction. Next step wraps these in the SIMPLE loop.\n");
        std::printf("====================================================================\n");
        std::fflush(stdout);
      }
    }

    MPI_Finalize();
    return 0;
  } catch (const std::exception& e) {
    std::fprintf(stderr, "simple_gpu_mpi_port rank %d ERROR: %s\n", rank, e.what());
    MPI_Abort(MPI_COMM_WORLD, 1);
  }

  MPI_Finalize();
  return 1;
}
