#include "poisson_library.h"

EllipticResult solve_scalar_elliptic(
    const Mesh& mesh,
    const std::vector<double>& gammaFace,
    const std::vector<double>& cellSource,
    const ScalarBCSet& bcSet,
    const EllipticOptions& opts) {
  if (opts.gradScheme != "lsq") {
    throw std::runtime_error("Only gradScheme=lsq is implemented in this build.");
  }
  if (opts.laplacianScheme != "orth" && opts.laplacianScheme != "nonorth") {
    throw std::runtime_error("Use laplacianScheme=orth or laplacianScheme=nonorth.");
  }

  const auto bcFaceData = build_boundary_face_data(mesh, bcSet);
  const bool includeNonOrth = (opts.laplacianScheme == "nonorth");
  const int nOuter = includeNonOrth ? std::max(opts.nNonOrthCorr, 0) + 1 : 1;

  CSRPattern pat = build_scalar_pattern(mesh);
  std::vector<double> phi(mesh.nCells, 0.0);
  std::vector<std::array<double,3>> grad(mesh.nCells, {0.0, 0.0, 0.0});
  std::vector<HYPRE_Complex> values, rhs;
  HypreSolveInfo lastInfo{};

  bool anyDirichlet = false;
  for (int f = mesh.nInternalFaces; f < mesh.nFaces; ++f) {
    if (bcFaceData.type[f] == ScalarBCType::Dirichlet) { anyDirichlet = true; break; }
  }
  if (!anyDirichlet && !opts.useReferenceCell) {
    throw std::runtime_error("Pure-Neumann scalar elliptic problem needs useReferenceCell=true.");
  }

  for (int outer = 0; outer < nOuter; ++outer) {
    compute_lsq_gradient(mesh, phi, bcFaceData, grad);
    assemble_scalar_elliptic_system(
        mesh, pat, gammaFace, cellSource, bcFaceData, grad,
        values, rhs, includeNonOrth,
        opts.useReferenceCell, opts.referenceCell, opts.referenceValue);
    lastInfo = solve_system_hypre_gpu(pat, values, rhs, phi, opts.hypre);
  }

  EllipticResult out;
  out.phi = std::move(phi);
  out.lastSolveInfo = lastInfo;
  return out;
}
