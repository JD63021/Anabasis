#include "gradient.h"

void compute_lsq_gradient(
    const Mesh& mesh,
    const std::vector<double>& phi,
    const std::vector<double>& boundaryFaceValue,
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
      auto r = sub3(mesh.xf[f], xP);
      double phiB = boundaryFaceValue[f];
      double dphi = phiB - phiP;
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
