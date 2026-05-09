#include "gradient.h"

namespace {

std::string lower_copy(std::string s) {
  for (char& c : s) c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
  return s;
}

double boundary_equivalent_face_value(
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

double face_interp_lambda(const Mesh& mesh, int f) {
  const int P = mesh.owner[f];
  const int N = mesh.neigh[f];
  const auto d = sub3(mesh.cc[N], mesh.cc[P]);
  const auto dx = sub3(mesh.xf[f], mesh.cc[P]);
  const double lam = dot3(dx, d) / std::max(dot3(d, d), 1.0e-30);
  return std::min(1.0, std::max(0.0, lam));
}

} // namespace

GradientScheme gradient_scheme_from_string(const std::string& name) {
  const std::string v = lower_copy(name);
  if (v == "lsq" || v == "least-squares" || v == "leastsquares" || v == "least_squares") {
    return GradientScheme::LSQ;
  }
  if (v == "gauss" || v == "green-gauss" || v == "greengauss" || v == "gauss-linear" || v == "gausslinear") {
    return GradientScheme::Gauss;
  }
  throw std::runtime_error("Unknown gradient scheme '" + name + "'. Use lsq or gauss.");
}

const char* gradient_scheme_name(GradientScheme scheme) {
  return scheme == GradientScheme::Gauss ? "gauss" : "lsq";
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
    double rhs[3] = {0,0,0};

    for (int N : mesh.cellNbrs[P]) {
      auto r = sub3(mesh.cc[N], xP);
      double dphi = phi[N] - phiP;
      double w = 1.0 / std::max(dot3(r,r), 1e-30);
      for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) M[i][j] += w * r[i] * r[j];
        rhs[i] += w * r[i] * dphi;
      }
    }

    for (int f : mesh.cellBFace[P]) {
      auto rcf = sub3(mesh.xf[f], xP);
      std::array<double,3> r{};
      double dphi = 0.0;

      if (bcFaceData.type[f] == ScalarBCType::Dirichlet) {
        r = rcf;
        double phiB = bcFaceData.value[f];
        dphi = phiB - phiP;
      } else {
        const double dn = std::max(dot3(rcf, mesh.nf[f]), 1e-30);
        r = mul3(dn, mesh.nf[f]);            // pure normal constraint
        dphi = bcFaceData.value[f] * dn;     // (dphi/dn) * dn
      }

      double w = 1.0 / std::max(dot3(r,r), 1e-30);
      for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) M[i][j] += w * r[i] * r[j];
        rhs[i] += w * r[i] * dphi;
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


void compute_gauss_gradient(
    const Mesh& mesh,
    const std::vector<double>& phi,
    const BoundaryFaceData& bcFaceData,
    std::vector<std::array<double,3>>& grad) {
  if (static_cast<int>(phi.size()) != mesh.nCells) {
    throw std::runtime_error("compute_gauss_gradient: phi size must equal mesh.nCells");
  }

  grad.assign(mesh.nCells, {0.0, 0.0, 0.0});

  // Internal faces: Sf is owner-outward. Add to owner, subtract from neighbour.
  for (int f = 0; f < mesh.nInternalFaces; ++f) {
    const int P = mesh.owner[f];
    const int N = mesh.neigh[f];
    const double lam = face_interp_lambda(mesh, f);
    const double phiF = (1.0 - lam) * phi[P] + lam * phi[N];

    for (int d = 0; d < 3; ++d) {
      const double contrib = phiF * mesh.Sf[f][d];
      grad[P][d] += contrib;
      grad[N][d] -= contrib;
    }
  }

  // Boundary faces: Sf is owner-outward. Use Dirichlet face value or equivalent
  // Neumann face value obtained by extrapolating along the boundary normal.
  for (int f = mesh.nInternalFaces; f < mesh.nFaces; ++f) {
    const int P = mesh.owner[f];
    const double phiF = boundary_equivalent_face_value(mesh, bcFaceData, phi, P, f);
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

void compute_gradient(
    const Mesh& mesh,
    const std::vector<double>& phi,
    const BoundaryFaceData& bcFaceData,
    GradientScheme scheme,
    std::vector<std::array<double,3>>& grad) {
  if (scheme == GradientScheme::Gauss) {
    compute_gauss_gradient(mesh, phi, bcFaceData, grad);
  } else {
    compute_lsq_gradient(mesh, phi, bcFaceData, grad);
  }
}
