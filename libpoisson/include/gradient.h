#pragma once

#include "bc.h"

enum class GradientScheme {
  LSQ,
  Gauss
};

GradientScheme gradient_scheme_from_string(const std::string& name);
const char* gradient_scheme_name(GradientScheme scheme);

void compute_lsq_gradient(
    const Mesh& mesh,
    const std::vector<double>& phi,
    const BoundaryFaceData& bcFaceData,
    std::vector<std::array<double,3>>& grad);

// Green-Gauss / Gauss-linear gradient:
//   grad(phi)_P = 1/V_P * sum_f phi_f S_f
// where internal face values use linear interpolation to face centre and
// boundary face values are taken from Dirichlet or equivalent Neumann data.
void compute_gauss_gradient(
    const Mesh& mesh,
    const std::vector<double>& phi,
    const BoundaryFaceData& bcFaceData,
    std::vector<std::array<double,3>>& grad);

void compute_gradient(
    const Mesh& mesh,
    const std::vector<double>& phi,
    const BoundaryFaceData& bcFaceData,
    GradientScheme scheme,
    std::vector<std::array<double,3>>& grad);
