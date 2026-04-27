#pragma once

#include "mesh.h"

void compute_lsq_gradient(
    const Mesh& mesh,
    const std::vector<double>& phi,
    const std::vector<double>& boundaryFaceValue,
    std::vector<std::array<double,3>>& grad);
