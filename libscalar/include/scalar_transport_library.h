#pragma once

#include <array>
#include <functional>
#include <string>
#include <vector>

#include "mesh.h"

namespace libscalar {

enum class ScalarBCType {
  Dirichlet,
  ZeroGradient,
  NeumannGradient
};

using ScalarBCFunc = std::function<double(
    const std::array<double,3>& x,
    const std::array<double,3>& outwardNormal)>;

struct ScalarPatchBC {
  std::string patchName;
  ScalarBCType type = ScalarBCType::Dirichlet;
  ScalarBCFunc evaluator;
};

struct ScalarBCSet {
  std::vector<ScalarPatchBC> patches;
};

inline ScalarPatchBC make_dirichlet_patch_bc(const std::string& patchName, ScalarBCFunc fn) {
  return ScalarPatchBC{patchName, ScalarBCType::Dirichlet, std::move(fn)};
}

inline ScalarPatchBC make_zero_gradient_patch_bc(const std::string& patchName) {
  return ScalarPatchBC{
      patchName,
      ScalarBCType::ZeroGradient,
      [](const std::array<double,3>&, const std::array<double,3>&) { return 0.0; }};
}

inline ScalarPatchBC make_neumann_gradient_patch_bc(const std::string& patchName, ScalarBCFunc fn) {
  return ScalarPatchBC{patchName, ScalarBCType::NeumannGradient, std::move(fn)};
}

inline ScalarPatchBC make_dirichlet_constant_bc(const std::string& patchName, double value) {
  return make_dirichlet_patch_bc(
      patchName,
      [value](const std::array<double,3>&, const std::array<double,3>&) { return value; });
}

inline ScalarPatchBC make_neumann_gradient_constant_bc(const std::string& patchName, double value) {
  return make_neumann_gradient_patch_bc(
      patchName,
      [value](const std::array<double,3>&, const std::array<double,3>&) { return value; });
}

enum class ConvectionScheme {
  Central,
  Upwind
};

enum class DiffusionScheme {
  Orth,
  NonOrth
};

enum class LinearSolverType {
  BiCGSTAB
};

enum class PreconditionerType {
  Jacobi
};

struct ScalarTransportInputs {
  std::vector<double> faceFlux;   // size = mesh.nFaces, oriented owner->neigh/outward
  std::vector<double> gammaFace;  // size = mesh.nFaces
  std::vector<double> Su;         // size = mesh.nCells
  std::vector<double> Sp;         // size = mesh.nCells, implicit linearized source
};

struct ScalarTransportOptions {
  ConvectionScheme convectionScheme = ConvectionScheme::Central;
  DiffusionScheme diffusionScheme = DiffusionScheme::NonOrth;
  int nNonOrthCorr = 2;

  LinearSolverType linearSolver = LinearSolverType::BiCGSTAB;
  PreconditionerType preconditioner = PreconditionerType::Jacobi;

  int maxIter = 4000;
  double relTol = 0.0;
  double absTol = 1.0e-10;
  int monitor = 0;
};

struct ScalarTransportResult {
  std::vector<double> phi;
  int iterations = 0;
  double finalRelRes = 0.0;
};

ScalarTransportResult solve_steady_scalar_transport(
    const Mesh& mesh,
    const ScalarTransportInputs& in,
    const ScalarBCSet& bcSet,
    const ScalarTransportOptions& opt,
    const std::vector<double>& x0 = {});

} // namespace libscalar
