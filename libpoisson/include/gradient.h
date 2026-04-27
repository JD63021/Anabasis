#pragma once

#include "bc.h"

void compute_lsq_gradient(
    const Mesh& mesh,
    const std::vector<double>& phi,
    const BoundaryFaceData& bcFaceData,
    std::vector<std::array<double,3>>& grad);
