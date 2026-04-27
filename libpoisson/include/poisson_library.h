#pragma once

#include "hypre_backend.h"

struct EllipticOptions {
  int nNonOrthCorr = 2;
  std::string gradScheme = "lsq";
  std::string laplacianScheme = "nonorth";
  bool useReferenceCell = false;
  int referenceCell = 0;
  double referenceValue = 0.0;
  HypreOptions hypre;
};

struct EllipticResult {
  std::vector<double> phi;
  HypreSolveInfo lastSolveInfo;
};

EllipticResult solve_scalar_elliptic(
    const Mesh& mesh,
    const std::vector<double>& gammaFace,
    const std::vector<double>& cellSource,
    const ScalarBCSet& bcSet,
    const EllipticOptions& opts);

inline EllipticResult solve_poisson(
    const Mesh& mesh,
    const std::vector<double>& cellSource,
    const ScalarBCSet& bcSet,
    const EllipticOptions& opts) {
  std::vector<double> gammaFace(mesh.nFaces, 1.0);
  return solve_scalar_elliptic(mesh, gammaFace, cellSource, bcSet, opts);
}
