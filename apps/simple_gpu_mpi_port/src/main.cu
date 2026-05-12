#include <mpi.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <exception>
#include <fstream>
#include <iomanip>
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

static double dot3_local(const std::array<double,3>& a, const std::array<double,3>& b) {
  return a[0]*b[0] + a[1]*b[1] + a[2]*b[2];
}

static std::array<double,3> sub3_local(const std::array<double,3>& a,
                                       const std::array<double,3>& b) {
  return {a[0]-b[0], a[1]-b[1], a[2]-b[2]};
}

static std::array<double,3> mul3_local(double s, const std::array<double,3>& a) {
  return {s*a[0], s*a[1], s*a[2]};
}

struct BasicPatchInfoLocal {
  std::string name;
  std::string type;
  int nFaces = 0;
  int startFace = 0;
};

static std::string read_text_file_local(const std::string& path) {
  std::ifstream in(path);
  if (!in) throw std::runtime_error("Could not open " + path);
  std::ostringstream ss;
  ss << in.rdbuf();
  return ss.str();
}

static int find_int_entry_local(const std::string& body, const std::string& key, int def=-1) {
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
    if (p.nFaces >= 0 && p.startFace >= 0) out.push_back(p);
  }

  if (out.empty()) throw std::runtime_error("No patch table entries found in " + boundaryPath);
  return out;
}

static bool is_proc_patch_name(const std::set<std::string>& procNames, const std::string& name) {
  return procNames.find(name) != procNames.end();
}

static int find_patch_index_local(const std::vector<std::string>& names, const std::string& name) {
  for (int i = 0; i < static_cast<int>(names.size()); ++i) {
    if (names[i] == name) return i;
  }
  return -1;
}

static int patch_index_for_face_mpi_port(const Mesh& meshBC, int f) {
  for (int ip = 0; ip < static_cast<int>(meshBC.patchNames.size()); ++ip) {
    const int start = meshBC.patchStartFace[ip];
    const int end = start + meshBC.patchNFaces[ip];
    if (f >= start && f < end) return ip;
  }
  return -1;
}

static double face_lambda_local_mpi_port(const Mesh& mesh, int f) {
  const int P = mesh.owner[f];
  const int N = mesh.neigh[f];

  const auto d = sub3_local(mesh.cc[N], mesh.cc[P]);
  const auto dx = sub3_local(mesh.xf[f], mesh.cc[P]);

  const double lam = dot3_local(dx, d) / std::max(dot3_local(d, d), 1.0e-300);
  return std::min(1.0, std::max(0.0, lam));
}

static double face_lambda_proc_mpi_port(const DecompMesh& dm, int f) {
  const Mesh& mesh = dm.mesh;
  const int P = mesh.owner[f];

  const auto d = sub3_local(dm.remoteCCForFace[f], mesh.cc[P]);
  const auto dx = sub3_local(mesh.xf[f], mesh.cc[P]);

  const double lam = dot3_local(dx, d) / std::max(dot3_local(d, d), 1.0e-300);
  return std::min(1.0, std::max(0.0, lam));
}



struct PressureBCFaceData {
  std::vector<ScalarBCType> type;
  std::vector<double> value;
};

static void compute_lsq_gradient_scalar_mpi_port(
    const DecompMesh& dm,
    const std::vector<double>& field,
    const PressureBCFaceData& bc,
    double lsqWeightPower,
    std::vector<std::array<double,3>>& grad);

static double pressure_delta_coeff_mpi_port(
    const std::array<double,3>& d,
    const std::array<double,3>& n,
    int pDeltaMode,
    double pDeltaMinCos)
{
  const double nd = dot3_local(n, d);
  const double dmag = std::sqrt(std::max(dot3_local(d, d), 1.0e-300));

  if (pDeltaMode == 0) {
    // legacy/v1 signed projected
    return 1.0 / std::max(nd, 1.0e-300);
  }

  if (pDeltaMode == 1) {
    // OpenFOAM-like stabilised projected delta
    const double denom = std::max(std::fabs(nd), pDeltaMinCos * dmag);
    return 1.0 / std::max(denom, 1.0e-300);
  }

  if (pDeltaMode == 2) {
    return 1.0 / std::max(std::fabs(nd), 1.0e-300);
  }

  // distance mode
  return 1.0 / std::max(dmag, 1.0e-300);
}

static void compute_gauss_gradient_scalar_mpi_port(
    const DecompMesh& dm,
    const Mesh& meshBC,
    const std::vector<double>& field,
    const PressureBCFaceData& bc,
    std::vector<std::array<double,3>>& grad)
{
  const Mesh& mesh = dm.mesh;
  const auto remoteField = exchange_proc_face_scalar_owner_values(dm, field);

  grad.assign(mesh.nCells, {0.0, 0.0, 0.0});

  for (int f = 0; f < mesh.nInternalFaces; ++f) {
    const int P = mesh.owner[f];
    const int N = mesh.neigh[f];
    const double lam = face_lambda_local_mpi_port(mesh, f);
    const double phif = (1.0 - lam) * field[P] + lam * field[N];

    for (int k = 0; k < 3; ++k) {
      const double contrib = phif * mesh.Sf[f][k];
      grad[P][k] += contrib;
      grad[N][k] -= contrib;
    }
  }

  for (int f = mesh.nInternalFaces; f < mesh.nFaces; ++f) {
    const int P = mesh.owner[f];

    double phif = field[P];

    if (dm.isProcFace[f]) {
      const double lam = face_lambda_proc_mpi_port(dm, f);
      phif = (1.0 - lam) * field[P] + lam * remoteField[f];
    } else {
      if (bc.type[f] == ScalarBCType::Dirichlet) {
        phif = bc.value[f];
      } else {
        // zero-gradient: extrapolate cell-centre value
        phif = field[P];
      }
    }

    for (int k = 0; k < 3; ++k) {
      grad[P][k] += phif * mesh.Sf[f][k];
    }
  }

  for (int c = 0; c < mesh.nCells; ++c) {
    const double invV = 1.0 / std::max(mesh.vol[c], 1.0e-300);
    grad[c][0] *= invV;
    grad[c][1] *= invV;
    grad[c][2] *= invV;
  }
}

static void compute_pressure_gradient_selected_mpi_port(
    const DecompMesh& dm,
    const Mesh& meshBC,
    const std::vector<double>& field,
    const PressureBCFaceData& bc,
    const std::string& pGradScheme,
    double lsqWeightPower,
    std::vector<std::array<double,3>>& grad)
{
  if (pGradScheme == "gauss") {
    compute_gauss_gradient_scalar_mpi_port(dm, meshBC, field, bc, grad);
  } else {
    compute_lsq_gradient_scalar_mpi_port(dm, field, bc, lsqWeightPower, grad);
  }
}

static void write_rank_cellcenter_vtu_mpi_port(
    const std::string& filename,
    const DecompMesh& dm,
    const std::vector<double>& U,
    const std::vector<double>& V,
    const std::vector<double>& W,
    const std::vector<double>& pField,
    const std::vector<double>& div)
{
  const Mesh& mesh = dm.mesh;
  std::ofstream out(filename);
  if (!out) throw std::runtime_error("could not write VTU file " + filename);

  out << "<?xml version=\"1.0\"?>\n";
  out << "<VTKFile type=\"UnstructuredGrid\" version=\"0.1\" byte_order=\"LittleEndian\">\n";
  out << "<UnstructuredGrid>\n";
  out << "<Piece NumberOfPoints=\"" << mesh.nCells << "\" NumberOfCells=\"" << mesh.nCells << "\">\n";

  out << "<Points>\n";
  out << "<DataArray type=\"Float64\" NumberOfComponents=\"3\" format=\"ascii\">\n";
  out << std::setprecision(17);
  for (int c = 0; c < mesh.nCells; ++c) {
    out << mesh.cc[c][0] << " " << mesh.cc[c][1] << " " << mesh.cc[c][2] << "\n";
  }
  out << "</DataArray>\n";
  out << "</Points>\n";

  out << "<Cells>\n";
  out << "<DataArray type=\"Int32\" Name=\"connectivity\" format=\"ascii\">\n";
  for (int c = 0; c < mesh.nCells; ++c) out << c << "\n";
  out << "</DataArray>\n";

  out << "<DataArray type=\"Int32\" Name=\"offsets\" format=\"ascii\">\n";
  for (int c = 0; c < mesh.nCells; ++c) out << (c + 1) << "\n";
  out << "</DataArray>\n";

  out << "<DataArray type=\"UInt8\" Name=\"types\" format=\"ascii\">\n";
  for (int c = 0; c < mesh.nCells; ++c) out << "1\n"; // VTK_VERTEX
  out << "</DataArray>\n";
  out << "</Cells>\n";

  out << "<PointData Vectors=\"U\">\n";

  out << "<DataArray type=\"Float64\" Name=\"U\" NumberOfComponents=\"3\" format=\"ascii\">\n";
  for (int c = 0; c < mesh.nCells; ++c) out << U[c] << " " << V[c] << " " << W[c] << "\n";
  out << "</DataArray>\n";

  out << "<DataArray type=\"Float64\" Name=\"p\" format=\"ascii\">\n";
  for (int c = 0; c < mesh.nCells; ++c) out << pField[c] << "\n";
  out << "</DataArray>\n";

  out << "<DataArray type=\"Float64\" Name=\"umag\" format=\"ascii\">\n";
  for (int c = 0; c < mesh.nCells; ++c) {
    const double m = std::sqrt(U[c]*U[c] + V[c]*V[c] + W[c]*W[c]);
    out << m << "\n";
  }
  out << "</DataArray>\n";

  out << "<DataArray type=\"Float64\" Name=\"div\" format=\"ascii\">\n";
  for (int c = 0; c < mesh.nCells; ++c) out << div[c] << "\n";
  out << "</DataArray>\n";

  out << "<DataArray type=\"Int32\" Name=\"rank\" format=\"ascii\">\n";
  int rank = 0;
  MPI_Comm_rank(dm.comm, &rank);
  for (int c = 0; c < mesh.nCells; ++c) out << rank << "\n";
  out << "</DataArray>\n";

  out << "</PointData>\n";
  out << "</Piece>\n";
  out << "</UnstructuredGrid>\n";
  out << "</VTKFile>\n";
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

static double global_max_abs_vec_mpi_port(const std::vector<double>& a, MPI_Comm comm) {
  double local = 0.0;
  for (double v : a) local = std::max(local, std::abs(v));

  double global = 0.0;
  MPI_Allreduce(&local, &global, 1, MPI_DOUBLE, MPI_MAX, comm);
  return global;
}

static double global_sum_vec_mpi_port(const std::vector<double>& a, MPI_Comm comm) {
  double local = 0.0;
  for (double v : a) local += v;

  double global = 0.0;
  MPI_Allreduce(&local, &global, 1, MPI_DOUBLE, MPI_SUM, comm);
  return global;
}

static double global_l1_vec_mpi_port(const std::vector<double>& a, MPI_Comm comm) {
  double local = 0.0;
  for (double v : a) local += std::abs(v);

  double global = 0.0;
  MPI_Allreduce(&local, &global, 1, MPI_DOUBLE, MPI_SUM, comm);
  return global;
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



static std::vector<double> build_rhiechow_phi_star_mpi_port(
    const DecompMesh& dm,
    const Mesh& meshBC,
    const std::vector<std::string>& bcUType,
    const std::vector<std::string>& bcVType,
    const std::vector<std::string>& bcWType,
    const std::vector<double>& U,
    const std::vector<double>& V,
    const std::vector<double>& W,
    const std::vector<double>& pField,
    const std::vector<std::array<double,3>>& gradP,
    const std::vector<double>& rAU,
    const std::vector<double>& uFaceBC,
    const std::vector<double>& vFaceBC,
    const std::vector<double>& wFaceBC,
    double rho,
    int rcModeInt,
    int pDeltaModeInt,
    double pDeltaMinCos)
{
  const Mesh& mesh = dm.mesh;

  const auto rU = exchange_proc_face_scalar_owner_values(dm, U);
  const auto rV = exchange_proc_face_scalar_owner_values(dm, V);
  const auto rW = exchange_proc_face_scalar_owner_values(dm, W);
  const auto rP = exchange_proc_face_scalar_owner_values(dm, pField);
  const auto rRAU = exchange_proc_face_scalar_owner_values(dm, rAU);
  const auto rGradP = exchange_proc_face_vector_owner_values(dm, gradP);

  std::vector<double> phiStar(mesh.nFaces, 0.0);

  for (int f = 0; f < mesh.nInternalFaces; ++f) {
    const int P = mesh.owner[f];
    const int N = mesh.neigh[f];

    const auto d = sub3_local(mesh.cc[N], mesh.cc[P]);
    const double dpn = dot3_local(mesh.nf[f], d);
    const double lam = face_lambda_local_mpi_port(mesh, f);

    const double uf = (1.0 - lam) * U[P] + lam * U[N];
    const double vf = (1.0 - lam) * V[P] + lam * V[N];
    const double wf = (1.0 - lam) * W[P] + lam * W[N];

    const std::array<double,3> gradpf = {
      (1.0 - lam) * gradP[P][0] + lam * gradP[N][0],
      (1.0 - lam) * gradP[P][1] + lam * gradP[N][1],
      (1.0 - lam) * gradP[P][2] + lam * gradP[N][2]
    };

    const double rAUf = (1.0 - lam) * rAU[P] + lam * rAU[N];

    const double phiInterp =
        rho * mesh.Af[f] *
        (uf * mesh.nf[f][0] + vf * mesh.nf[f][1] + wf * mesh.nf[f][2]);

    const double deltaCoeff = pressure_delta_coeff_mpi_port(d, mesh.nf[f], pDeltaModeInt, pDeltaMinCos);
    const double rc = (rcModeInt == 0)
        ? rho * mesh.Af[f] * rAUf * deltaCoeff *
          ((pField[N] - pField[P]) - dot3_local(gradpf, d))
        : 0.0;

    phiStar[f] = phiInterp - rc;
  }

  for (int f = mesh.nInternalFaces; f < mesh.nFaces; ++f) {
    const int P = mesh.owner[f];

    if (dm.isProcFace[f]) {
      const double lam = face_lambda_proc_mpi_port(dm, f);

      const double uf = (1.0 - lam) * U[P] + lam * rU[f];
      const double vf = (1.0 - lam) * V[P] + lam * rV[f];
      const double wf = (1.0 - lam) * W[P] + lam * rW[f];

      const std::array<double,3> gradpf = {
        (1.0 - lam) * gradP[P][0] + lam * rGradP[f][0],
        (1.0 - lam) * gradP[P][1] + lam * rGradP[f][1],
        (1.0 - lam) * gradP[P][2] + lam * rGradP[f][2]
      };

      const auto d = sub3_local(dm.remoteCCForFace[f], mesh.cc[P]);
      const double dpn = dot3_local(mesh.nf[f], d);
      const double rAUf = (1.0 - lam) * rAU[P] + lam * rRAU[f];

      const double phiInterp =
          rho * mesh.Af[f] *
          (uf * mesh.nf[f][0] + vf * mesh.nf[f][1] + wf * mesh.nf[f][2]);

      const double deltaCoeff = pressure_delta_coeff_mpi_port(d, mesh.nf[f], pDeltaModeInt, pDeltaMinCos);
      const double rc = (rcModeInt == 0)
          ? rho * mesh.Af[f] * rAUf * deltaCoeff *
            ((rP[f] - pField[P]) - dot3_local(gradpf, d))
          : 0.0;

      phiStar[f] = phiInterp - rc;
    } else {
      const int ip = patch_index_for_face_mpi_port(meshBC, f);

      double uf = U[P];
      double vf = V[P];
      double wf = W[P];

      if (ip >= 0 && ip < static_cast<int>(bcUType.size()) && bcUType[ip] == "Dirichlet") {
        uf = uFaceBC[f];
      }
      if (ip >= 0 && ip < static_cast<int>(bcVType.size()) && bcVType[ip] == "Dirichlet") {
        vf = vFaceBC[f];
      }
      if (ip >= 0 && ip < static_cast<int>(bcWType.size()) && bcWType[ip] == "Dirichlet") {
        wf = wFaceBC[f];
      }

      // Serial boundary treatment: no Rhie-Chow term on physical boundary faces.
      phiStar[f] =
          rho * mesh.Af[f] *
          (uf * mesh.nf[f][0] + vf * mesh.nf[f][1] + wf * mesh.nf[f][2]);
    }
  }

  return phiStar;
}

static PressureBCFaceData build_pressure_bc_face_data_mpi_port(
    const DecompMesh& dm,
    const Mesh& meshBC,
    const std::vector<std::string>& bcPType,
    const std::vector<double>& pFaceBC)
{
  const Mesh& mesh = dm.mesh;
  PressureBCFaceData out;
  out.type.assign(mesh.nFaces, ScalarBCType::Neumann);
  out.value.assign(mesh.nFaces, 0.0);

  for (int f = mesh.nInternalFaces; f < mesh.nFaces; ++f) {
    if (dm.isProcFace[f]) continue;

    const int ip = patch_index_for_face_mpi_port(meshBC, f);
    if (ip >= 0 && ip < static_cast<int>(bcPType.size()) && bcPType[ip] == "Dirichlet") {
      out.type[f] = ScalarBCType::Dirichlet;
      out.value[f] = pFaceBC[f];
    } else {
      out.type[f] = ScalarBCType::Neumann;
      out.value[f] = 0.0;
    }
  }

  return out;
}

static void compute_lsq_gradient_scalar_mpi_port(
    const DecompMesh& dm,
    const std::vector<double>& field,
    const PressureBCFaceData& bc,
    double weightPower,
    std::vector<std::array<double,3>>& grad)
{
  const Mesh& mesh = dm.mesh;
  const auto remoteField = exchange_proc_face_scalar_owner_values(dm, field);

  grad.assign(mesh.nCells, {0.0, 0.0, 0.0});

  for (int P = 0; P < mesh.nCells; ++P) {
    const auto xP = mesh.cc[P];
    const double phiP = field[P];

    double M[3][3] = {{0,0,0},{0,0,0},{0,0,0}};
    double rhs[3] = {0,0,0};

    auto add_constraint = [&](const std::array<double,3>& r, double dphi) {
      const double rmag = std::sqrt(std::max(dot3_local(r, r), 1.0e-300));
      const double w = 1.0 / std::pow(rmag, weightPower);
      for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) M[i][j] += w * r[i] * r[j];
        rhs[i] += w * r[i] * dphi;
      }
    };

    for (int N : mesh.cellNbrs[P]) {
      add_constraint(sub3_local(mesh.cc[N], xP), field[N] - phiP);
    }

    for (int f : mesh.cellBFace[P]) {
      if (dm.isProcFace[f]) {
        add_constraint(sub3_local(dm.remoteCCForFace[f], xP), remoteField[f] - phiP);
      } else {
        const auto rcf = sub3_local(mesh.xf[f], xP);

        if (bc.type[f] == ScalarBCType::Dirichlet) {
          add_constraint(rcf, bc.value[f] - phiP);
        } else {
          // Match serial v1.1b LSQ boundary operator:
          // include the boundary face-center vector in the LSQ matrix.
          // For zero-gradient pressure, dphi = 0 but r must remain xf - xP,
          // not projected to dn*n. This matters strongly on wall-rich
          // industrial meshes.
          const double dn = dot3_local(rcf, mesh.nf[f]);
          add_constraint(rcf, bc.value[f] * dn);
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
    if (procPatchNames.find(name) != procPatchNames.end()) continue;

    const int start = meshBC.patchStartFace[ip];
    const int nFaces = meshBC.patchNFaces[ip];

    if (ip < static_cast<int>(bcUType.size()) && bcUType[ip] == "Dirichlet") {
      double sumA = 0.0, sumVA = 0.0;

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
      bc.patches.push_back(make_neumann_constant_bc(name, 0.0));
    }
  }

  return bc;
}

static double momentum_diag_cell_mpi_port(
    const DecompMesh& dm,
    const Mesh& meshBC,
    const std::vector<std::string>& bcUType,
    const std::vector<double>& faceFlux,
    double rho,
    double mu,
    int P)
{
  const Mesh& mesh = dm.mesh;
  double diag = 0.0;

  for (int f = 0; f < mesh.nInternalFaces; ++f) {
    const int owner = mesh.owner[f];
    const int neigh = mesh.neigh[f];

    if (owner != P && neigh != P) continue;

    const auto d = sub3_local(mesh.cc[neigh], mesh.cc[owner]);
    const double dDotS = dot3_local(d, mesh.Sf[f]);
    const double D = mu * dot3_local(mesh.Sf[f], mesh.Sf[f]) / std::max(dDotS, 1.0e-300);

    diag += D;

    const double F = rho * faceFlux[f];

    if (owner == P) {
      diag += std::max(F, 0.0);
    } else {
      diag += std::max(-F, 0.0);
    }
  }

  for (int f : mesh.cellBFace[P]) {
    std::array<double,3> d;

    if (dm.isProcFace[f]) {
      d = sub3_local(dm.remoteCCForFace[f], mesh.cc[P]);
    } else {
      d = sub3_local(mesh.xf[f], mesh.cc[P]);
    }

    const double dDotS = dot3_local(d, mesh.Sf[f]);
    const double D = mu * dot3_local(mesh.Sf[f], mesh.Sf[f]) / std::max(dDotS, 1.0e-300);

    diag += D;

    const double F = rho * faceFlux[f];

    if (dm.isProcFace[f]) {
      diag += std::max(F, 0.0);
    } else {
      const int ip = patch_index_for_face_mpi_port(meshBC, f);
      const bool dirichletU = (ip >= 0 &&
                               ip < static_cast<int>(bcUType.size()) &&
                               bcUType[ip] == "Dirichlet");
      if (!dirichletU) diag += F;
    }
  }

  return std::max(diag, 1.0e-300);
}

static std::vector<double> build_rAU_mpi_port(
    const DecompMesh& dm,
    const Mesh& meshBC,
    const std::vector<std::string>& bcUType,
    const std::vector<double>& faceFlux,
    double rho,
    double mu)
{
  std::vector<double> rAU(dm.mesh.nCells, 0.0);

  for (int c = 0; c < dm.mesh.nCells; ++c) {
    rAU[c] = 1.0 / momentum_diag_cell_mpi_port(dm, meshBC, bcUType, faceFlux, rho, mu, c);
  }

  return rAU;
}

static std::vector<double> build_pressure_coeff_faces_v11b_mpi_port(
    const DecompMesh& dm,
    const std::vector<double>& rAU,
    double rho,
    double pCoeffScale,
    int pDeltaModeInt,
    double pDeltaMinCos)
{
  const Mesh& mesh = dm.mesh;
  const auto rRAU = exchange_proc_face_scalar_owner_values(dm, rAU);

  std::vector<double> coeff(mesh.nFaces, 0.0);

  auto make_coeff = [&](int f, double rAUf, const std::array<double,3>& d) {
    const double delta = pressure_delta_coeff_mpi_port(d, mesh.nf[f], pDeltaModeInt, pDeltaMinCos);
    // Strict serial v1.1b pressure coefficient:
    //   coeff = pCoeffScale * rho * Af * rAUf * pressure_delta_coeff_runtime(...)
    // This is used both in the pressure matrix and in the flux correction.
    return pCoeffScale * rho * mesh.Af[f] * rAUf * delta;
  };

  for (int f = 0; f < mesh.nInternalFaces; ++f) {
    const int P = mesh.owner[f];
    const int N = mesh.neigh[f];
    const double lam = face_lambda_local_mpi_port(mesh, f);
    const double rAUf = (1.0 - lam) * rAU[P] + lam * rAU[N];
    coeff[f] = make_coeff(f, rAUf, sub3_local(mesh.cc[N], mesh.cc[P]));
  }

  for (int f = mesh.nInternalFaces; f < mesh.nFaces; ++f) {
    const int P = mesh.owner[f];

    if (dm.isProcFace[f]) {
      const double lam = face_lambda_proc_mpi_port(dm, f);
      const double rAUf = (1.0 - lam) * rAU[P] + lam * rRAU[f];
      coeff[f] = make_coeff(f, rAUf, sub3_local(dm.remoteCCForFace[f], mesh.cc[P]));
    } else {
      coeff[f] = make_coeff(f, rAU[P], sub3_local(mesh.xf[f], mesh.cc[P]));
    }
  }

  return coeff;
}

static ScalarBCSet make_pcorr_bc_from_legacy_pressure_mpi_port(
    const Mesh& meshBC,
    const std::set<std::string>& procPatchNames,
    const std::vector<std::string>& bcPType)
{
  ScalarBCSet bc;

  for (int ip = 0; ip < static_cast<int>(meshBC.patchNames.size()); ++ip) {
    const std::string& name = meshBC.patchNames[ip];

    if (procPatchNames.find(name) != procPatchNames.end()) continue;

    if (ip < static_cast<int>(bcPType.size()) && bcPType[ip] == "Dirichlet") {
      bc.patches.push_back(make_dirichlet_constant_bc(name, 0.0));
    } else {
      bc.patches.push_back(make_neumann_constant_bc(name, 0.0));
    }
  }

  return bc;
}

static std::vector<double> correct_flux_v11b_coeff_mpi_port(
    const DecompMesh& dm,
    const Mesh& meshBC,
    const std::vector<std::string>& bcPType,
    const std::vector<double>& phiPred,
    const std::vector<double>& pCorr,
    const std::vector<double>& faceCoeff)
{
  const Mesh& mesh = dm.mesh;
  const auto rPCorr = exchange_proc_face_scalar_owner_values(dm, pCorr);

  std::vector<double> phi = phiPred;

  for (int f = 0; f < mesh.nInternalFaces; ++f) {
    const int P = mesh.owner[f];
    const int N = mesh.neigh[f];
    phi[f] -= faceCoeff[f] * (pCorr[N] - pCorr[P]);
  }

  for (int f = mesh.nInternalFaces; f < mesh.nFaces; ++f) {
    if (dm.isProcFace[f]) {
      const int P = mesh.owner[f];
      phi[f] -= faceCoeff[f] * (rPCorr[f] - pCorr[P]);
    } else {
      const int ip = patch_index_for_face_mpi_port(meshBC, f);
      const bool fixedPressure = (ip >= 0 &&
                                  ip < static_cast<int>(bcPType.size()) &&
                                  bcPType[ip] == "Dirichlet");
      if (!fixedPressure) continue;

      const int P = mesh.owner[f];
      const double pCorrB = 0.0;
      phi[f] -= faceCoeff[f] * (pCorrB - pCorr[P]);
    }
  }

  return phi;
}

static void print_div_metrics_rank0(
    int rank,
    const std::string& label,
    const DecompMesh& dm,
    const std::vector<double>& phi,
    MPI_Comm comm)
{
  const auto div = compute_cell_div_sum_mpi_port(dm, phi);

  const double divSum = global_sum_vec_mpi_port(div, comm);
  const double divL1 = global_l1_vec_mpi_port(div, comm);
  const double divInf = global_max_abs_vec_mpi_port(div, comm);

  if (rank == 0) {
    std::printf("%s div sum/L1/Linf : %.15e %.15e %.15e\n",
                label.c_str(), divSum, divL1, divInf);
  }
}


static void print_minmax_abs_rank0_mpi_port(
    int rank,
    const std::string& name,
    const std::vector<double>& a,
    MPI_Comm comm)
{
  double lmin = 1.0e300;
  double lmax = -1.0e300;
  double labs = 0.0;

  for (double v : a) {
    lmin = std::min(lmin, v);
    lmax = std::max(lmax, v);
    labs = std::max(labs, std::abs(v));
  }

  double gmin = 0.0, gmax = 0.0, gabs = 0.0;
  MPI_Reduce(&lmin, &gmin, 1, MPI_DOUBLE, MPI_MIN, 0, comm);
  MPI_Reduce(&lmax, &gmax, 1, MPI_DOUBLE, MPI_MAX, 0, comm);
  MPI_Reduce(&labs, &gabs, 1, MPI_DOUBLE, MPI_MAX, 0, comm);

  if (rank == 0) {
    std::printf("DEBUG %-20s min/max/maxAbs = %.12e %.12e %.12e\\n",
                name.c_str(), gmin, gmax, gabs);
  }
}

static void print_minmax_abs_grad_rank0_mpi_port(
    int rank,
    const std::string& name,
    const std::vector<std::array<double,3>>& g,
    MPI_Comm comm)
{
  double labs = 0.0;

  for (const auto& v : g) {
    const double m = std::sqrt(v[0]*v[0] + v[1]*v[1] + v[2]*v[2]);
    labs = std::max(labs, m);
  }

  double gabs = 0.0;
  MPI_Reduce(&labs, &gabs, 1, MPI_DOUBLE, MPI_MAX, 0, comm);

  if (rank == 0) {
    std::printf("DEBUG %-20s maxMag = %.12e\\n", name.c_str(), gabs);
  }
}


static void print_outer_iter_rank0(
    int rank,
    int iter,
    const DecompMesh& dm,
    const std::vector<double>& phiPred,
    const std::vector<double>& phiCorr,
    const std::vector<double>& U,
    const std::vector<double>& V,
    const std::vector<double>& W,
    const std::vector<double>& pCorr,
    int uIts, int vIts, int wIts,
    double uRel, double vRel, double wRel,
    int pIts, double pRel,
    MPI_Comm comm)
{
  const auto divPred = compute_cell_div_sum_mpi_port(dm, phiPred);
  const auto divCorr = compute_cell_div_sum_mpi_port(dm, phiCorr);

  const double predInf = global_max_abs_vec_mpi_port(divPred, comm);
  const double corrInf = global_max_abs_vec_mpi_port(divCorr, comm);
  const double corrL1 = global_l1_vec_mpi_port(divCorr, comm);
  const double corrSum = global_sum_vec_mpi_port(divCorr, comm);

  const double Umax = global_max_abs_vec_mpi_port(U, comm);
  const double Vmax = global_max_abs_vec_mpi_port(V, comm);
  const double Wmax = global_max_abs_vec_mpi_port(W, comm);
  const double pCorrMax = global_max_abs_vec_mpi_port(pCorr, comm);

  if (rank == 0) {
    std::printf("iter %4d : divPredInf=%.12e divCorrInf=%.12e divCorrL1=%.12e divCorrSum=%.12e "
                "pCorrMax=%.12e Umax/Vmax/Wmax=%.6e %.6e %.6e "
                "velIts=[%d %d %d] velSolveRel=[%.3e %.3e %.3e] pIts=%d pRel=%.3e\n",
                iter, predInf, corrInf, corrL1, corrSum,
                pCorrMax, Umax, Vmax, Wmax,
                uIts, vIts, wIts, uRel, vRel, wRel,
                pIts, pRel);
  }
}

int main(int argc, char** argv) {
  MPI_Init(&argc, &argv);

  int rank = 0, size = 1;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  try {
    std::string caseRoot = "/tmp/case";
    std::string bcConfigPath = "/tmp/case/anabasis_pipe.bc";
    int device = rank;

    std::string wallPatch = "patch_0_0";
    std::string inletPatch = "patch_2_0";
    std::string outletPatch = "patch_1_0";

    double rho = 1.0;
    double mu = 0.05;
    double uMean = 1.0;
    double uRelax = 0.7;
    double pRelax = 0.3;
    double pCoeffScale = 1.0;

    // v1.1b industrial reference defaults.
    std::string pMode = "absolute";
    std::string pSolveMode = "ofAbsolute";
    std::string rcMode = "oflike";
    std::string pGradScheme = "lsq";
    double lsqWeightPower = 1.0;
    std::string rAUMode = "raw";
    std::string pDeltaMode = "of";
    double pDeltaMinCos = 0.05;
    std::string momentumConvectionScheme = "central";
    double momNonOrthScale = 0.0;
    double pNonOrthScale = 0.0;
    int nVelNonOrthCorr = 0;
    int nNonOrthCorr = 0;
    int nPressureCorr = 0;

    int writeVtu = 1;
    int writeEvery = 0;
    std::string outPrefix = "/tmp/case/simple_gpu_mpi_port";

    int nsteps = 50;
    int minSteps = 5;
    int printEvery = 1;
    double tolMass = 1.0e-8;
    double tolVel = 1.0e-6;

    int velMaxit = 500;
    double velAbsTol = 1.0e-7;
    double velRelTol = 1.0e-5;
    std::string velSolver = "bicgstab";
    
    int velSweeps = 3;
    double velSmootherOmega = 0.8;
    int velGsSymmetric = 0;
    int velSmootherMonitor = 0;
int velGmresRestart = 30;

    int pMaxit = 1000;
    double pAbsTol = 1.0e-9;
    double pRelTol = 0.0;
    int pAmgMaxit = 1;
    int pAmgRelaxType = 18;
    int pAmgCoarsenType = 8;
    int pAmgInterpType = 6;
    int pAmgAggLevels = 1;
    int pAmgAggInterpType = 4;
    int pAmgPmax = 4;
    int pAmgKeepTranspose = 1;
    double pAmgTruncFactor = 0.0;
    int pAmgRebuildEvery = 1; // parsed for compatibility; current host-IJ path rebuilds every pressure solve

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
      } else if (a == "-out-prefix" || a == "-outPrefix") {
        need(a.c_str());
        outPrefix = argv[++i];
      } else if (a == "-write-vtu" || a == "-writeVtu") {
        need(a.c_str());
        writeVtu = std::atoi(argv[++i]);
      } else if (a == "-write-every" || a == "-writeEvery") {
        need(a.c_str());
        writeEvery = std::atoi(argv[++i]);
      } else if (a == "-p-mode" || a == "-pMode") {
        need(a.c_str());
        pMode = argv[++i];
      } else if (a == "-p-solve-mode" || a == "-pSolveMode") {
        need(a.c_str());
        pSolveMode = argv[++i];
      } else if (a == "-rc-mode" || a == "-rcMode" || a == "-rhie-chow-mode") {
        need(a.c_str());
        rcMode = argv[++i];
      } else if (a == "-p-grad-scheme" || a == "-pGradScheme") {
        need(a.c_str());
        pGradScheme = argv[++i];
      } else if (a == "-lsq-weight-power" || a == "-lsqWeightPower" || a == "-lsqWeight") {
        need(a.c_str());
        lsqWeightPower = std::atof(argv[++i]);
      } else if (a == "-rAU-mode" || a == "-rau-mode" || a == "-rAUMode") {
        need(a.c_str());
        rAUMode = argv[++i];
      } else if (a == "-p-delta-mode" || a == "-pDeltaMode") {
        need(a.c_str());
        pDeltaMode = argv[++i];
      } else if (a == "-p-delta-min-cos" || a == "-pDeltaMinCos") {
        need(a.c_str());
        pDeltaMinCos = std::atof(argv[++i]);
      } else if (a == "-momentum-convection-scheme" || a == "-momentumConvectionScheme") {
        need(a.c_str());
        momentumConvectionScheme = argv[++i];
      } else if (a == "-mom-nonorth-scale" || a == "-momNonOrthScale") {
        need(a.c_str());
        momNonOrthScale = std::atof(argv[++i]);
      } else if (a == "-p-nonorth-scale" || a == "-pNonOrthScale") {
        need(a.c_str());
        pNonOrthScale = std::atof(argv[++i]);
      } else if (a == "-nVelNonOrthCorr" || a == "-n-vel-nonorth-corr") {
        need(a.c_str());
        nVelNonOrthCorr = std::atoi(argv[++i]);
      } else if (a == "-nNonOrthCorr" || a == "-n-nonorth-corr") {
        need(a.c_str());
        nNonOrthCorr = std::atoi(argv[++i]);
      } else if (a == "-nPressureCorr" || a == "-n-pressure-corr") {
        need(a.c_str());
        nPressureCorr = std::atoi(argv[++i]);
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
      } else if (a == "-min-steps") {
        need("-min-steps");
        minSteps = std::atoi(argv[++i]);
      } else if (a == "-print-every") {
        need("-print-every");
        printEvery = std::atoi(argv[++i]);
      } else if (a == "-u-relax") {
        need("-u-relax");
        uRelax = std::atof(argv[++i]);
      } else if (a == "-p-relax") {
        need("-p-relax");
        pRelax = std::atof(argv[++i]);
      } else if (a == "-p-coeff-scale" || a == "-pCoeffScale") {
        need(a.c_str());
        pCoeffScale = std::atof(argv[++i]);
      } else if (a == "-tolMass") {
        need("-tolMass");
        tolMass = std::atof(argv[++i]);
      } else if (a == "-tolVel") {
        need("-tolVel");
        tolVel = std::atof(argv[++i]);
      } else if (a == "-vel-maxit") {
        need("-vel-maxit");
        velMaxit = std::atoi(argv[++i]);
      } else if (a == "-vel-tol") {
        need("-vel-tol");
        velAbsTol = std::atof(argv[++i]);
      } else if (a == "-vel-reltol") {
        need("-vel-reltol");
        velRelTol = std::atof(argv[++i]);
      } else if (a == "-vel-solver" || a == "-velSolver") {
        need(a.c_str());
        velSolver = argv[++i];
      } else if (a == "-vel-gmres-restart" || a == "-velGmresRestart") {
        need(a.c_str());
        velGmresRestart = std::atoi(argv[++i]);
      } else if (a == "-vel-sweeps" || a == "-velSweeps") {
        need(a.c_str());
        velSweeps = std::atoi(argv[++i]);
      } else if (a == "-vel-smoother-omega" || a == "-velSmootherOmega") {
        need(a.c_str());
        velSmootherOmega = std::atof(argv[++i]);
      } else if (a == "-vel-gs-symmetric" || a == "-velGsSymmetric") {
        need(a.c_str());
        velGsSymmetric = std::atoi(argv[++i]);
      } else if (a == "-vel-smoother-monitor" || a == "-velSmootherMonitor") {
        need(a.c_str());
        velSmootherMonitor = std::atoi(argv[++i]);
      } else if (a == "-p-maxit") {
        need("-p-maxit");
        pMaxit = std::atoi(argv[++i]);
      } else if (a == "-p-tol") {
        need("-p-tol");
        pAbsTol = std::atof(argv[++i]);
      } else if (a == "-p-reltol") {
        need("-p-reltol");
        pRelTol = std::atof(argv[++i]);
      } else if (a == "-p-amg-maxit") {
        need("-p-amg-maxit");
        pAmgMaxit = std::atoi(argv[++i]);
      } else if (a == "-p-amg-relax-type") {
        need("-p-amg-relax-type");
        pAmgRelaxType = std::atoi(argv[++i]);
      } else if (a == "-p-amg-coarsen-type") {
        need("-p-amg-coarsen-type");
        pAmgCoarsenType = std::atoi(argv[++i]);
      } else if (a == "-p-amg-interp-type") {
        need("-p-amg-interp-type");
        pAmgInterpType = std::atoi(argv[++i]);
      } else if (a == "-p-amg-agg-levels") {
        need("-p-amg-agg-levels");
        pAmgAggLevels = std::atoi(argv[++i]);
      } else if (a == "-p-amg-agg-interp-type") {
        need("-p-amg-agg-interp-type");
        pAmgAggInterpType = std::atoi(argv[++i]);
      } else if (a == "-p-amg-pmax") {
        need("-p-amg-pmax");
        pAmgPmax = std::atoi(argv[++i]);
      } else if (a == "-p-amg-keep-transpose") {
        need("-p-amg-keep-transpose");
        pAmgKeepTranspose = std::atoi(argv[++i]);
      } else if (a == "-p-amg-trunc-factor") {
        need("-p-amg-trunc-factor");
        pAmgTruncFactor = std::atof(argv[++i]);
      } else if (a == "-p-amg-rebuild-every") {
        need("-p-amg-rebuild-every");
        pAmgRebuildEvery = std::atoi(argv[++i]);
      } else if (a == "-p-use-amg" || a == "-p-amg-setup-scope") {
        // Accepted for serial command-line compatibility. This MPI pressure path
        // currently always uses PCG+BoomerAMG and rebuilds the host-IJ matrix each solve.
        need(a.c_str());
        ++i;
      } else {
        if (rank == 0) {
          std::printf("simple_gpu_mpi_port: ignoring unimplemented option %s\n", a.c_str());
        }
      }
    }


    auto lower_mode = [](std::string v) {
      std::transform(v.begin(), v.end(), v.begin(), [](unsigned char c){ return std::tolower(c); });
      return v;
    };

    pMode = lower_mode(pMode);
    pSolveMode = lower_mode(pSolveMode);
    rcMode = lower_mode(rcMode);
    pGradScheme = lower_mode(pGradScheme);
    rAUMode = lower_mode(rAUMode);
    pDeltaMode = lower_mode(pDeltaMode);
    momentumConvectionScheme = lower_mode(momentumConvectionScheme);

    if (!(pMode == "absolute" || pMode == "abs")) {
      throw std::runtime_error("MPI port currently implements industrial pMode absolute only");
    }
    if (!(pSolveMode == "ofabsolute" || pSolveMode == "of-absolute" || pSolveMode == "openfoam" || pSolveMode == "of")) {
      throw std::runtime_error("MPI port currently implements pSolveMode ofAbsolute only");
    }
    if (!(rcMode == "oflike" || rcMode == "of" || rcMode == "openfoam" || rcMode == "old" || rcMode == "legacy")) {
      throw std::runtime_error("Unknown rcMode for MPI port");
    }
    if (!(pGradScheme == "gauss" || pGradScheme == "lsq")) {
      throw std::runtime_error("Unknown pGradScheme for MPI port");
    }
    if (!(rAUMode == "raw" || rAUMode == "relaxed")) {
      throw std::runtime_error("Unknown rAUMode for MPI port");
    }
    if (!(pDeltaMode == "of" || pDeltaMode == "openfoam" || pDeltaMode == "legacy" || pDeltaMode == "v1" || pDeltaMode == "normal" || pDeltaMode == "distance")) {
      throw std::runtime_error("Unknown pDeltaMode for MPI port");
    }
    if (!(momentumConvectionScheme == "central" || momentumConvectionScheme == "linear" || momentumConvectionScheme == "upwind")) {
      throw std::runtime_error("Unknown momentumConvectionScheme for MPI port");
    }
    if (std::abs(momNonOrthScale) > 1.0e-30) {
      throw std::runtime_error("momNonOrthScale != 0 not yet ported; reference.case uses 0");
    }
    if (std::abs(pNonOrthScale) > 1.0e-30) {
      throw std::runtime_error("pNonOrthScale != 0 not yet ported; reference.case uses 0");
    }
    if (nVelNonOrthCorr != 0 || nNonOrthCorr != 0 || nPressureCorr != 0) {
      throw std::runtime_error("nonzero pressure/velocity correction subloops not yet ported; reference.case uses zeros");
    }

    const int rcModeInt = (rcMode == "old" || rcMode == "legacy") ? 0 : 1;
    const int rAUModeInt = (rAUMode == "raw") ? 0 : 1;
    const int pDeltaModeInt =
        (pDeltaMode == "legacy" || pDeltaMode == "v1") ? 0 :
        (pDeltaMode == "normal") ? 2 :
        (pDeltaMode == "distance") ? 3 : 1;

    int devCount = 0;
    cuda_check(cudaGetDeviceCount(&devCount), "cudaGetDeviceCount");
    if (devCount > 0) cuda_check(cudaSetDevice(device % devCount), "cudaSetDevice");

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

    for (const auto& p : basicPatchTable) {
      patchNamesForBC.push_back(p.name);
      patchStartFaceForBC.push_back(p.startFace);
      patchNFacesForBC.push_back(p.nFaces);
    }

    Mesh meshBC = mesh;
    meshBC.patchNames = patchNamesForBC;
    meshBC.patchStartFace = patchStartFaceForBC;
    meshBC.patchNFaces = patchNFacesForBC;

    std::set<std::string> procPatchNames;
    for (const auto& pp : dm.procPatches) procPatchNames.insert(pp.name);

    std::vector<std::string> physicalPatchNames;
    for (const auto& name : patchNamesForBC) {
      if (!is_proc_patch_name(procPatchNames, name)) physicalPatchNames.push_back(name);
    }

    pipebc::RuntimeBCConfig bcConfig = pipebc::load_runtime_bc_config(bcConfigPath);
    pipebc::validate_runtime_bc_config_against_patches(bcConfig, physicalPatchNames);

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

    pipebc::apply_bc_specs_to_legacy_face_arrays(
        legacyBCMesh,
        patchGeometryTable,
        bcConfig.velocityPatchSpecs,
        bcConfig.pressurePatchSpecs,
        0.0,
        bcUType, bcVType, bcWType, bcPType,
        uFaceBC, vFaceBC, wFaceBC, pFaceBC);

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

    if (rank == 0) {
      std::printf("====================================================================\n");
      std::printf("simple_gpu_mpi_port SIMPLE LOOP CHECKPOINT\n");
      std::printf("caseRoot       : %s\n", caseRoot.c_str());
      std::printf("bcConfig       : %s\n", bcConfigPath.c_str());
      std::printf("worldSize      : %d\n", size);
      std::printf("globalCells    : %d\n", globalCells);
      std::printf("globalProcFaces: %d\n", globalProcFaces);
      std::printf("globalPhysFaces: %d\n", globalPhysicalFaces);
      std::printf("rho/mu/uMean   : %.6e / %.6e / %.6e\n", rho, mu, uMean);
      std::printf("relax U/P      : %.6e / %.6e\n", uRelax, pRelax);
      std::printf("pCoeffScale    : %.6e\n", pCoeffScale);
      std::printf("velSolver      : %s restart=%d\n", velSolver.c_str(), velGmresRestart);
      std::printf("lsqWeightPower : %.6e\n", lsqWeightPower);
      std::printf("modes          : pMode=%s pSolveMode=%s rcMode=%s pGradScheme=%s rAUMode=%s pDeltaMode=%s pDeltaMinCos=%.6g momentumConvection=%s momNonOrthScale=%.6g pNonOrthScale=%.6g\n",
                  pMode.c_str(), pSolveMode.c_str(), rcMode.c_str(), pGradScheme.c_str(),
                  rAUMode.c_str(), pDeltaMode.c_str(), pDeltaMinCos,
                  momentumConvectionScheme.c_str(), momNonOrthScale, pNonOrthScale);
      std::printf("writeVtu/writeEvery/outPrefix : %d / %d / %s\n",
                  writeVtu, writeEvery, outPrefix.c_str());
      std::printf("nsteps/minSteps/tolMass/tolVel : %d / %d / %.6e / %.6e\n", nsteps, minSteps, tolMass, tolVel);
      std::printf("velocityPatchSpecs = %zu pressurePatchSpecs = %zu\n",
                  bcConfig.velocityPatchSpecs.size(), bcConfig.pressurePatchSpecs.size());
      std::fflush(stdout);
    }

    print_patch_audit_rank0(rank, meshBC, wallPatch, bcUType, bcPType,
                            uFaceBC, vFaceBC, wFaceBC, MPI_COMM_WORLD);
    print_patch_audit_rank0(rank, meshBC, inletPatch, bcUType, bcPType,
                            uFaceBC, vFaceBC, wFaceBC, MPI_COMM_WORLD);
    print_patch_audit_rank0(rank, meshBC, outletPatch, bcUType, bcPType,
                            uFaceBC, vFaceBC, wFaceBC, MPI_COMM_WORLD);

    std::vector<double> U(mesh.nCells, 0.0);
    std::vector<double> V(mesh.nCells, 0.0);
    std::vector<double> W(mesh.nCells, 0.0);
    std::vector<double> pField(mesh.nCells, 0.0);

    ScalarBCSet uCompBC = make_component_scalar_bc_from_legacy_mpi_port(
        meshBC, procPatchNames, bcUType, uFaceBC, vFaceBC, wFaceBC, 0);
    ScalarBCSet vCompBC = make_component_scalar_bc_from_legacy_mpi_port(
        meshBC, procPatchNames, bcUType, uFaceBC, vFaceBC, wFaceBC, 1);
    ScalarBCSet wCompBC = make_component_scalar_bc_from_legacy_mpi_port(
        meshBC, procPatchNames, bcUType, uFaceBC, vFaceBC, wFaceBC, 2);
    ScalarBCSet pCorrBC = make_pcorr_bc_from_legacy_pressure_mpi_port(
        meshBC, procPatchNames, bcPType);

    PressureBCFaceData pBCData = build_pressure_bc_face_data_mpi_port(
        dm, meshBC, bcPType, pFaceBC);

    libscalar_decomp::DistScalarTransportOptions momOpt;
    momOpt.convectionScheme =
        (momentumConvectionScheme == "upwind")
        ? libscalar_decomp::DistConvectionScheme::Upwind
        : libscalar_decomp::DistConvectionScheme::Central;
    momOpt.diffusionScheme = libscalar_decomp::DistDiffusionScheme::Orth;
    momOpt.gradScheme = "lsq";
    momOpt.nNonOrthCorr = 0;
    momOpt.underRelax = uRelax;
    momOpt.rAUMode = rAUModeInt;
    momOpt.rAUScale = 1.0;
    momOpt.solver.maxIter = velMaxit;
    momOpt.solver.absTol = velAbsTol;
    momOpt.solver.relTol = velRelTol;
    momOpt.solver.monitor = 0;
    momOpt.solver.solverType = velSolver;
    momOpt.solver.gmresRestart = velGmresRestart;
    momOpt.solver.smootherSweeps = velSweeps;
    momOpt.solver.smootherOmega = velSmootherOmega;
    momOpt.solver.smootherSymmetric = velGsSymmetric;
    momOpt.solver.smootherMonitor = velSmootherMonitor;

    DistEllipticOptions pOpt;
    pOpt.gradScheme = "lsq";
    pOpt.laplacianScheme = "orth";
    pOpt.nNonOrthCorr = 0;
    pOpt.useReferenceCell = false;
    pOpt.referenceGlobalCell = 0;
    pOpt.referenceValue = 0.0;
    pOpt.hypre.maxIter = pMaxit;
    pOpt.hypre.absTol = pAbsTol;
    pOpt.hypre.relTol = pRelTol;
    pOpt.hypre.tol = pAbsTol;
    pOpt.hypre.monitor = 0;
    pOpt.hypre.amgMaxIter = pAmgMaxit;
    pOpt.hypre.amgRelaxType = pAmgRelaxType;
    pOpt.hypre.amgCoarsenType = pAmgCoarsenType;
    pOpt.hypre.amgInterpType = pAmgInterpType;
    pOpt.hypre.amgAggLevels = pAmgAggLevels;
    pOpt.hypre.amgAggInterpType = pAmgAggInterpType;
    pOpt.hypre.amgPmax = pAmgPmax;
    pOpt.hypre.amgKeepTranspose = pAmgKeepTranspose;
    pOpt.hypre.amgTruncFactor = pAmgTruncFactor;

    std::vector<double> phi = build_face_flux_mpi_port(
        dm, meshBC, bcUType,
        U, V, W,
        uFaceBC, vFaceBC, wFaceBC);
    for (double& pf : phi) pf *= rho;

    print_div_metrics_rank0(rank, "initial", dm, phi, MPI_COMM_WORLD);

    double finalMass = 1.0e300;

    for (int iter = 1; iter <= nsteps; ++iter) {
      std::vector<double> Uold = U;
      std::vector<double> Vold = V;
      std::vector<double> Wold = W;
      std::vector<double> pOld = pField;

      std::vector<std::array<double,3>> gradP;
      compute_pressure_gradient_selected_mpi_port(dm, meshBC, pField, pBCData, pGradScheme, lsqWeightPower, gradP);
      if (iter <= 3 || iter % 10 == 0) {
        print_minmax_abs_rank0_mpi_port(rank, "pOld", pField, MPI_COMM_WORLD);
        print_minmax_abs_grad_rank0_mpi_port(rank, "gradPold", gradP, MPI_COMM_WORLD);
      }

      libscalar_decomp::DistScalarTransportInputs momIn;
      momIn.faceFlux.assign(mesh.nFaces, 0.0);
      momIn.gammaFace.assign(mesh.nFaces, mu);
      momIn.Su.assign(mesh.nCells, 0.0);
      momIn.Sp.assign(mesh.nCells, 0.0);

      for (int f = 0; f < mesh.nFaces; ++f) {
        // phi is already mass flux, matching serial simple_gpu convention.
        momIn.faceFlux[f] = phi[f];
      }

      for (int c = 0; c < mesh.nCells; ++c) {
        momIn.Su[c] = -gradP[c][0];
      }
      auto uRes = libscalar_decomp::solve_scalar_transport_decomp(dm, momIn, uCompBC, momOpt, U);

      for (int c = 0; c < mesh.nCells; ++c) {
        momIn.Su[c] = -gradP[c][1];
      }
      auto vRes = libscalar_decomp::solve_scalar_transport_decomp(dm, momIn, vCompBC, momOpt, V);

      for (int c = 0; c < mesh.nCells; ++c) {
        momIn.Su[c] = -gradP[c][2];
      }
      auto wRes = libscalar_decomp::solve_scalar_transport_decomp(dm, momIn, wCompBC, momOpt, W);

      U = std::move(uRes.phi);
      V = std::move(vRes.phi);
      W = std::move(wRes.phi);

      // Use rAU extracted from the actual relaxed momentum matrix.
      // The U/V/W component matrices have the same operator for this pipe case;
      // W is the driven component, so use its returned rAU.
      std::vector<double> rAU = wRes.rAU;
      if (rAU.empty()) {
        rAU = build_rAU_mpi_port(dm, meshBC, bcUType, phi, rho, mu);
      }

      std::vector<double> Hx = U;
      std::vector<double> Hy = V;
      std::vector<double> Hz = W;
      for (int c = 0; c < mesh.nCells; ++c) {
        // ofAbsolute path: reconstruct HbyA = Ustar + rAU*grad(pOld)
        Hx[c] += rAU[c] * gradP[c][0];
        Hy[c] += rAU[c] * gradP[c][1];
        Hz[c] += rAU[c] * gradP[c][2];
      }

      const auto phiPred = build_rhiechow_phi_star_mpi_port(
          dm, meshBC,
          bcUType, bcVType, bcWType,
          Hx, Hy, Hz,
          pField, gradP,
          rAU,
          uFaceBC, vFaceBC, wFaceBC,
          rho, rcModeInt, pDeltaModeInt, pDeltaMinCos);

      const auto pCoeffFace = build_pressure_coeff_faces_v11b_mpi_port(
          dm, rAU, rho, pCoeffScale, pDeltaModeInt, pDeltaMinCos);

      const auto divPred = compute_cell_div_sum_mpi_port(dm, phiPred);

      if (iter <= 3 || iter % 10 == 0) {
        print_minmax_abs_rank0_mpi_port(rank, "rAU", rAU, MPI_COMM_WORLD);
        print_minmax_abs_rank0_mpi_port(rank, "pCoeffFace", pCoeffFace, MPI_COMM_WORLD);
        print_minmax_abs_rank0_mpi_port(rank, "phiPred", phiPred, MPI_COMM_WORLD);
        print_minmax_abs_rank0_mpi_port(rank, "divPred", divPred, MPI_COMM_WORLD);
      }

      std::vector<double> pRhsFlux(mesh.nCells, 0.0);
      for (int c = 0; c < mesh.nCells; ++c) {
        // Strict v1.1b pressure RHS: flux-divergence units directly.
        // Do not divide by volume; the direct-coefficient solver does not
        // multiply this RHS by volume internally.
        pRhsFlux[c] = -divPred[c];
      }

      if (iter <= 3 || iter % 10 == 0) {
        print_minmax_abs_rank0_mpi_port(rank, "pRHS_flux", pRhsFlux, MPI_COMM_WORLD);
      }

      auto pCorrRes = solve_scalar_elliptic_direct_coeff_decomp(
          dm, pCoeffFace, pRhsFlux, pCorrBC, pOpt);

      if (iter <= 3 || iter % 10 == 0) {
        print_minmax_abs_rank0_mpi_port(rank, "pSolved_or_pCorr", pCorrRes.phi, MPI_COMM_WORLD);
      }

      const auto phiCorr = correct_flux_v11b_coeff_mpi_port(
          dm, meshBC, bcPType, phiPred, pCorrRes.phi, pCoeffFace);

      PressureBCFaceData pCorrBCData;
      pCorrBCData.type.assign(mesh.nFaces, ScalarBCType::Neumann);
      pCorrBCData.value.assign(mesh.nFaces, 0.0);
      for (int f = mesh.nInternalFaces; f < mesh.nFaces; ++f) {
        if (dm.isProcFace[f]) continue;
        const int ip = patch_index_for_face_mpi_port(meshBC, f);
        if (ip >= 0 && ip < static_cast<int>(bcPType.size()) && bcPType[ip] == "Dirichlet") {
          pCorrBCData.type[f] = ScalarBCType::Dirichlet;
          pCorrBCData.value[f] = 0.0;
        }
      }

      std::vector<std::array<double,3>> gradPCorr;
      compute_pressure_gradient_selected_mpi_port(dm, meshBC, pCorrRes.phi, pCorrBCData, pGradScheme, lsqWeightPower, gradPCorr);

      // ofAbsolute path:
      // pCorrRes.phi is the newly solved absolute pressure candidate.
      // Relax pressure absolutely, then set U = HbyA - rAU*grad(pNew).
      for (int c = 0; c < mesh.nCells; ++c) {
        pField[c] = (1.0 - pRelax) * pField[c] + pRelax * pCorrRes.phi[c];
      }

      std::vector<std::array<double,3>> gradPNew;
      compute_pressure_gradient_selected_mpi_port(dm, meshBC, pField, pBCData, pGradScheme, lsqWeightPower, gradPNew);

      for (int c = 0; c < mesh.nCells; ++c) {
        U[c] = Hx[c] - rAU[c] * gradPNew[c][0];
        V[c] = Hy[c] - rAU[c] * gradPNew[c][1];
        W[c] = Hz[c] - rAU[c] * gradPNew[c][2];
      }

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

        localDP = std::max(localDP, std::abs(pField[c] - pOld[c]));
        localPScale = std::max(localPScale, std::abs(pField[c]));
      }

      double gDU = 0.0, gUScale = 0.0, gDP = 0.0, gPScale = 0.0;
      MPI_Allreduce(&localDU, &gDU, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
      MPI_Allreduce(&localUScale, &gUScale, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
      MPI_Allreduce(&localDP, &gDP, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
      MPI_Allreduce(&localPScale, &gPScale, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);

      const double velRel = gDU / std::max(gUScale, 1.0e-300);
      const double pChangeRel = gDP / std::max(gPScale, 1.0e-300);

      phi = phiCorr;

      const auto divCorr = compute_cell_div_sum_mpi_port(dm, phi);
      finalMass = global_max_abs_vec_mpi_port(divCorr, MPI_COMM_WORLD);

      if (iter % std::max(printEvery, 1) == 0 || iter == 1 || finalMass < tolMass) {
        print_outer_iter_rank0(
            rank, iter, dm, phiPred, phi,
            U, V, W,
            pCorrRes.phi,
            uRes.iterations, vRes.iterations, wRes.iterations,
            uRes.finalRelRes, vRes.finalRelRes, wRes.finalRelRes,
            pCorrRes.lastSolveInfo.iterations,
            pCorrRes.lastSolveInfo.finalRelResNorm,
            MPI_COMM_WORLD);

        if (rank == 0) {
          std::printf("          convergence check: velRel=%.12e pChangeRel=%.12e minSteps=%d\n",
                      velRel, pChangeRel, minSteps);
        }
      }

      if (iter >= minSteps && finalMass < tolMass && velRel < tolVel) {
        if (rank == 0) {
          std::printf("simple_gpu_mpi_port CONVERGED at iter %d massLinf=%.12e velRel=%.12e\n",
                      iter, finalMass, velRel);
        }
        break;
      }
    }


    if (writeVtu) {
      const auto divFinal = compute_cell_div_sum_mpi_port(dm, phi);

      std::ostringstream fn;
      fn << outPrefix << "_rank" << std::setw(4) << std::setfill('0') << rank << "_final.vtu";

      write_rank_cellcenter_vtu_mpi_port(fn.str(), dm, U, V, W, pField, divFinal);

      if (rank == 0) {
        std::printf("simple_gpu_mpi_port wrote per-rank VTU files: %s_rank####_final.vtu\n",
                    outPrefix.c_str());
      }
    }

    const double finalUmax = global_max_abs_vec_mpi_port(U, MPI_COMM_WORLD);
    const double finalVmax = global_max_abs_vec_mpi_port(V, MPI_COMM_WORLD);
    const double finalWmax = global_max_abs_vec_mpi_port(W, MPI_COMM_WORLD);
    const double finalPmax = global_max_abs_vec_mpi_port(pField, MPI_COMM_WORLD);

    if (rank == 0) {
      std::printf("simple_gpu_mpi_port FINAL massLinf=%.12e\n", finalMass);
      std::printf("simple_gpu_mpi_port FINAL max|u/v/w/p| = %.12e %.12e %.12e %.12e\n",
                  finalUmax, finalVmax, finalWmax, finalPmax);
      std::printf("simple_gpu_mpi_port PASS_RAN\n");
      std::printf("====================================================================\n");
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
