#pragma once

#include "poisson_decomp_library.h"

#include <string>
#include <vector>

namespace libscalar_decomp {

enum class DistConvectionScheme {
  Upwind,
  Central
};

DistConvectionScheme convection_scheme_from_string(const std::string& name);
const char* convection_scheme_name(DistConvectionScheme scheme);

enum class DistDiffusionScheme {
  Orth,
  NonOrth
};

DistDiffusionScheme diffusion_scheme_from_string(const std::string& name);
const char* diffusion_scheme_name(DistDiffusionScheme scheme);

struct DistBiCGSTABOptions {
  int maxIter = 1000;
  double relTol = 0.0;
  double absTol = 1e-8;
  int monitor = 0;
};

struct DistScalarTransportInputs {
  std::vector<double> faceFlux;   // size = local mesh.nFaces, oriented owner->neighbour/outward
  std::vector<double> gammaFace;  // size = local mesh.nFaces
  std::vector<double> Su;         // size = local mesh.nCells, source per volume
  std::vector<double> Sp;         // size = local mesh.nCells, implicit source per volume
};

struct DistScalarTransportOptions {
  DistConvectionScheme convectionScheme = DistConvectionScheme::Upwind;
  DistDiffusionScheme diffusionScheme = DistDiffusionScheme::NonOrth;

  std::string gradScheme = "lsq";
  int nNonOrthCorr = 2;

  DistBiCGSTABOptions solver;
};

struct DistScalarTransportResult {
  std::vector<double> phi;
  int iterations = 0;
  double finalRelRes = 0.0;
  int nOuter = 0;
  long long globalNnz = 0;
};

DistScalarTransportResult solve_scalar_transport_decomp(
    const DecompMesh& dm,
    const DistScalarTransportInputs& in,
    const ScalarBCSet& bcSet,
    const DistScalarTransportOptions& opt,
    const std::vector<double>& x0 = {});

} // namespace libscalar_decomp
