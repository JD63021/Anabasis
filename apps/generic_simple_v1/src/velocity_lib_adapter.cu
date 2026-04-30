#include "velocity_lib_adapter.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <map>
#include <stdexcept>
#include <string>
#include <vector>

#include <mpi.h>

namespace {

enum class BCType {
  Dirichlet,
  ZeroGradient,
  NeumannGradient
};

struct BoundaryFaceData {
  std::vector<BCType> type;
  std::vector<double> value;
};

struct CSRPattern {
  int nRows = 0;
  int nnz = 0;
  std::vector<int> rowOffsets;
  std::vector<int> diagPos;
  std::vector<int> facePP, facePN, faceNP, faceNN;
};

struct VelocityAsmImpl {
  PressureLibMeshData mesh;
  CSRPattern pat;
  std::map<int,int> rawToCompact;
};

static inline std::array<double,3> sub3(
    const std::array<double,3>& a,
    const std::array<double,3>& b) {
  return {a[0]-b[0], a[1]-b[1], a[2]-b[2]};
}

static inline std::array<double,3> add3(
    const std::array<double,3>& a,
    const std::array<double,3>& b) {
  return {a[0]+b[0], a[1]+b[1], a[2]+b[2]};
}

static inline std::array<double,3> mul3(
    double s,
    const std::array<double,3>& a) {
  return {s*a[0], s*a[1], s*a[2]};
}

static inline double dot3(
    const std::array<double,3>& a,
    const std::array<double,3>& b) {
  return a[0]*b[0] + a[1]*b[1] + a[2]*b[2];
}

static bool is_dirichlet_type(const std::string& s) {
  return s == "Dirichlet";
}

static bool is_neumann_type(const std::string& s) {
  return s == "Neumann" || s == "NeumannGradient";
}

static double face_interp_lambda(
    const PressureLibMeshData& mesh,
    int f) {
  const int P = mesh.owner[f];
  const int N = mesh.neigh[f];
  const auto d  = sub3(mesh.cc[N], mesh.cc[P]);
  const auto dx = sub3(mesh.xf[f], mesh.cc[P]);
  const double denom = std::max(dot3(d, d), 1.0e-30);
  const double lam = dot3(dx, d) / denom;
  return std::min(1.0, std::max(0.0, lam));
}

static std::map<int,int> build_raw_to_compact_patch_map(const PressureLibMeshData& mesh) {
  std::map<int,int> rawToCompact;
  int next = 0;
  for (int f = mesh.nInternalFaces; f < mesh.nFaces; ++f) {
    const int raw = mesh.bPatch[f];
    if (!rawToCompact.count(raw)) rawToCompact[raw] = next++;
  }
  return rawToCompact;
}

static int compact_patch_index(
    const std::map<int,int>& rawToCompact,
    int rawPatch) {
  auto it = rawToCompact.find(rawPatch);
  if (it == rawToCompact.end()) {
    throw std::runtime_error("Could not map boundary raw patch index");
  }
  return it->second;
}

static CSRPattern build_scalar_pattern(const PressureLibMeshData& mesh) {
  CSRPattern pat;
  pat.nRows = mesh.nCells;
  pat.rowOffsets.resize(mesh.nCells + 1);
  pat.diagPos.resize(mesh.nCells);
  pat.facePP.resize(mesh.nInternalFaces);
  pat.facePN.resize(mesh.nInternalFaces);
  pat.faceNP.resize(mesh.nInternalFaces);
  pat.faceNN.resize(mesh.nInternalFaces);

  std::vector<std::map<int,int>> pos(mesh.nCells);
  pat.rowOffsets[0] = 0;

  for (int c = 0; c < mesh.nCells; ++c) {
    std::vector<int> cols = mesh.cellNbrs[c];
    cols.insert(cols.begin(), c);
    std::sort(cols.begin(), cols.end());
    cols.erase(std::unique(cols.begin(), cols.end()), cols.end());
    pat.rowOffsets[c+1] = pat.rowOffsets[c] + static_cast<int>(cols.size());
    for (int j = 0; j < static_cast<int>(cols.size()); ++j) {
      pos[c][cols[j]] = pat.rowOffsets[c] + j;
    }
  }

  pat.nnz = pat.rowOffsets.back();

  for (int c = 0; c < mesh.nCells; ++c) {
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

static BoundaryFaceData build_boundary_face_data(
    const PressureLibMeshData& mesh,
    const std::map<int,int>& rawToCompact,
    const std::vector<std::string>& bcQType,
    const std::vector<double>& bcQFaceVal) {

  if (static_cast<int>(bcQFaceVal.size()) != mesh.nFaces) {
    throw std::runtime_error("bcQFaceVal must have size mesh.nFaces");
  }

  BoundaryFaceData out;
  out.type.assign(mesh.nFaces, BCType::ZeroGradient);
  out.value.assign(mesh.nFaces, 0.0);

  for (int f = mesh.nInternalFaces; f < mesh.nFaces; ++f) {
    const int patch = compact_patch_index(rawToCompact, mesh.bPatch[f]);
    const std::string& t = bcQType[patch];

    if (is_dirichlet_type(t)) {
      out.type[f] = BCType::Dirichlet;
      out.value[f] = bcQFaceVal[f];
    } else if (is_neumann_type(t)) {
      out.type[f] = BCType::NeumannGradient;
      out.value[f] = bcQFaceVal[f];
    } else {
      out.type[f] = BCType::ZeroGradient;
      out.value[f] = 0.0;
    }
  }

  return out;
}

static void compute_lsq_gradient(
    const PressureLibMeshData& mesh,
    const BoundaryFaceData& bcFaceData,
    const std::vector<double>& phi,
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

      if (bcFaceData.type[f] == BCType::Dirichlet) {
        r = rcf;
        dphi = bcFaceData.value[f] - phiP;
      } else {
        const double dn = std::max(dot3(rcf, mesh.nf[f]), 1.0e-30);
        r = mul3(dn, mesh.nf[f]);
        dphi = bcFaceData.value[f] * dn;
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

static std::vector<double> build_face_flux(
    const PressureLibMeshData& mesh,
    const std::map<int,int>& rawToCompact,
    const std::vector<double>& uConv,
    const std::vector<double>& vConv,
    const std::vector<double>& wConv,
    const std::vector<std::string>& bcUType,
    const std::vector<double>& bcUFaceVal,
    const std::vector<std::string>& bcVType,
    const std::vector<double>& bcVFaceVal,
    const std::vector<std::string>& bcWType,
    const std::vector<double>& bcWFaceVal,
    double rho) {

  std::vector<double> faceFlux(mesh.nFaces, 0.0);

  for (int f = 0; f < mesh.nInternalFaces; ++f) {
    const int P = mesh.owner[f];
    const int N = mesh.neigh[f];
    const double lam = face_interp_lambda(mesh, f);

    const double uf = (1.0 - lam) * uConv[P] + lam * uConv[N];
    const double vf = (1.0 - lam) * vConv[P] + lam * vConv[N];
    const double wf = (1.0 - lam) * wConv[P] + lam * wConv[N];

    faceFlux[f] = rho * (
        uf * mesh.Sf[f][0] +
        vf * mesh.Sf[f][1] +
        wf * mesh.Sf[f][2]);
  }

  for (int f = mesh.nInternalFaces; f < mesh.nFaces; ++f) {
    const int P = mesh.owner[f];
    const int patch = compact_patch_index(rawToCompact, mesh.bPatch[f]);

    const double ub = is_dirichlet_type(bcUType[patch]) ? bcUFaceVal[f] : uConv[P];
    const double vb = is_dirichlet_type(bcVType[patch]) ? bcVFaceVal[f] : vConv[P];
    const double wb = is_dirichlet_type(bcWType[patch]) ? bcWFaceVal[f] : wConv[P];

    faceFlux[f] = rho * (
        ub * mesh.Sf[f][0] +
        vb * mesh.Sf[f][1] +
        wb * mesh.Sf[f][2]);
  }

  return faceFlux;
}

static std::vector<double> build_unrelaxed_diag(
    const PressureLibMeshData& mesh,
    const std::map<int,int>& rawToCompact,
    const std::vector<double>& faceFlux,
    const std::vector<std::string>& bcQType,
    double mu,
    const std::string& convectionScheme) {

  const bool upwind = (convectionScheme == "upwind" || convectionScheme == "upwind1");
  std::vector<double> diag(mesh.nCells, 0.0);

  for (int f = 0; f < mesh.nInternalFaces; ++f) {
    const int P = mesh.owner[f];
    const int N = mesh.neigh[f];

    const auto d = sub3(mesh.cc[N], mesh.cc[P]);
    const double dDotS = dot3(d, mesh.Sf[f]);
    const double D = mu * dot3(mesh.Sf[f], mesh.Sf[f]) / std::max(dDotS, 1.0e-30);
    const double F = faceFlux[f];

    diag[P] += D;
    diag[N] += D;

    if (!upwind) {
      const double lam = face_interp_lambda(mesh, f);
      diag[P] += F * (1.0 - lam);
      diag[N] -= F * lam;
    } else {
      if (F >= 0.0) diag[P] += F;
      else          diag[N] -= F;
    }
  }

  for (int f = mesh.nInternalFaces; f < mesh.nFaces; ++f) {
    const int P = mesh.owner[f];
    const int patch = compact_patch_index(rawToCompact, mesh.bPatch[f]);

    const auto d = sub3(mesh.xf[f], mesh.cc[P]);
    const double dDotS = dot3(d, mesh.Sf[f]);
    const double D = mu * dot3(mesh.Sf[f], mesh.Sf[f]) / std::max(dDotS, 1.0e-30);
    const double F = faceFlux[f];

    if (is_dirichlet_type(bcQType[patch])) diag[P] += D;
    else                                   diag[P] += F;
  }

  return diag;
}

static void assemble_scalar_transport_system(
    const PressureLibMeshData& mesh,
    const CSRPattern& pat,
    const std::vector<double>& faceFlux,
    const std::vector<double>& gammaFace,
    const std::vector<double>& Su,
    const std::vector<double>& Sp,
    const BoundaryFaceData& bcFaceData,
    const std::vector<std::array<double,3>>& grad,
    const std::string& convectionScheme,
    const std::string& diffusionScheme,
    std::vector<double>& values,
    std::vector<double>& rhs) {

  const bool includeNonOrth = (diffusionScheme == "nonorth");
  const bool upwind = (convectionScheme == "upwind" || convectionScheme == "upwind1");

  values.assign(pat.nnz, 0.0);
  rhs.assign(mesh.nCells, 0.0);

  for (int c = 0; c < mesh.nCells; ++c) {
    rhs[c] = Su[c] * mesh.vol[c];
    values[pat.diagPos[c]] += -Sp[c] * mesh.vol[c];
  }

  for (int f = 0; f < mesh.nInternalFaces; ++f) {
    const int P = mesh.owner[f];
    const int N = mesh.neigh[f];

    const auto d = sub3(mesh.cc[N], mesh.cc[P]);
    const double dDotS = dot3(d, mesh.Sf[f]);
    const double gamma = gammaFace[f];
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
      rhs[P] += corr;
      rhs[N] -= corr;
    }

    const double F = faceFlux[f];

    if (!upwind) {
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
    const double gamma = gammaFace[f];
    const double D = gamma * dot3(mesh.Sf[f], mesh.Sf[f]) / std::max(dDotS, 1.0e-30);
    const double F = faceFlux[f];

    if (bcFaceData.type[f] == BCType::Dirichlet) {
      const double phiB = bcFaceData.value[f];
      values[pat.diagPos[P]] += D;
      rhs[P] += D * phiB;

      if (includeNonOrth) {
        const auto T = sub3(mesh.Sf[f], mul3(D / std::max(gamma, 1.0e-30), d));
        const double corr = gamma * dot3(T, grad[P]);
        rhs[P] += corr;
      }

      rhs[P] += -F * phiB;
    } else {
      const double gradn =
          (bcFaceData.type[f] == BCType::ZeroGradient) ? 0.0 : bcFaceData.value[f];
      rhs[P] += gamma * gradn * mesh.Af[f];
      values[pat.diagPos[P]] += F;
    }
  }
}

} // namespace

void init_velocity_lib_assembly_handle(
    VelocityLibAssemblyHandle& handle,
    const PressureLibMeshData& meshData) {
  destroy_velocity_lib_assembly_handle(handle);
  auto* impl = new VelocityAsmImpl;
  impl->mesh = meshData;
  impl->rawToCompact = build_raw_to_compact_patch_map(impl->mesh);
  impl->pat = build_scalar_pattern(impl->mesh);
  handle.impl = impl;
  handle.nRows = impl->pat.nRows;
  handle.nnz = impl->pat.nnz;
}

VelocityLibAssembly assemble_velocity_with_libscalar(
    VelocityLibAssemblyHandle& handle,
    const std::vector<double>& qLag,
    const std::vector<double>& uConv,
    const std::vector<double>& vConv,
    const std::vector<double>& wConv,
    const std::vector<double>& gradPcomp,
    const std::vector<std::string>& bcQType,
    const std::vector<double>& bcQFaceVal,
    const std::vector<std::string>& bcUType,
    const std::vector<double>& bcUFaceVal,
    const std::vector<std::string>& bcVType,
    const std::vector<double>& bcVFaceVal,
    const std::vector<std::string>& bcWType,
    const std::vector<double>& bcWFaceVal,
    const VelocityLibOptions& opt) {

  if (!handle.impl) {
    throw std::runtime_error("VelocityLibAssemblyHandle is not initialized");
  }

  auto* impl = static_cast<VelocityAsmImpl*>(handle.impl);
  const auto& mesh = impl->mesh;
  const auto& pat  = impl->pat;

  if (static_cast<int>(qLag.size()) != mesh.nCells ||
      static_cast<int>(gradPcomp.size()) != mesh.nCells) {
    throw std::runtime_error("qLag and gradPcomp must have size mesh.nCells");
  }

  const double t0 = MPI_Wtime();

  const auto bcFaceData = build_boundary_face_data(mesh, impl->rawToCompact, bcQType, bcQFaceVal);
  const auto faceFlux = build_face_flux(
      mesh, impl->rawToCompact,
      uConv, vConv, wConv,
      bcUType, bcUFaceVal,
      bcVType, bcVFaceVal,
      bcWType, bcWFaceVal,
      opt.rho);

  const auto diag0 = build_unrelaxed_diag(
      mesh, impl->rawToCompact, faceFlux, bcQType, opt.mu, opt.convectionScheme);

  std::vector<double> gammaFace(mesh.nFaces, opt.mu);
  std::vector<double> Su(mesh.nCells, 0.0);
  std::vector<double> Sp(mesh.nCells, 0.0);

  const double useRelax = std::max(opt.uRelax, 1.0e-14);
  const double invRelax = 1.0 / useRelax;

  for (int c = 0; c < mesh.nCells; ++c) {
    const double V  = std::max(mesh.vol[c], 1.0e-30);
    const double aP = diag0[c];

    Su[c] = -gradPcomp[c];
    Sp[c] = 0.0;

    if (opt.uRelax < 0.999999) {
      Su[c] += ((invRelax - 1.0) * aP * qLag[c]) / V;
      Sp[c]  = -((invRelax - 1.0) * aP) / V;
    }
  }

  std::vector<std::array<double,3>> grad(mesh.nCells, {0.0, 0.0, 0.0});
  compute_lsq_gradient(mesh, bcFaceData, qLag, grad);

  VelocityLibAssembly out;
  assemble_scalar_transport_system(
      mesh, pat, faceFlux, gammaFace, Su, Sp,
      bcFaceData, grad,
      opt.convectionScheme, opt.diffusionScheme,
      out.values, out.rhs);

  out.rAU.resize(mesh.nCells, 0.0);
  for (int c = 0; c < mesh.nCells; ++c) {
    double aRelaxed = diag0[c];
    if (opt.uRelax < 0.999999) aRelaxed *= invRelax;
    out.rAU[c] = (std::fabs(aRelaxed) > 1.0e-30) ? (mesh.vol[c] / aRelaxed) : 0.0;
  }

  out.wallTime = MPI_Wtime() - t0;
  return out;
}

void destroy_velocity_lib_assembly_handle(VelocityLibAssemblyHandle& handle) {
  if (handle.impl) {
    delete static_cast<VelocityAsmImpl*>(handle.impl);
    handle.impl = nullptr;
  }
  handle.nRows = 0;
  handle.nnz = 0;
}
