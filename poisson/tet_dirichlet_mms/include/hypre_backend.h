#pragma once

#include "laplacian.h"

struct HypreOptions {
  int maxIter = 4000;
  double tol = 1e-12;
  int monitor = 1;
  int amgMaxIter = 1;
  int amgRelaxType = 18;
  int amgCoarsenType = 8;
  int amgInterpType = 6;
  int amgAggLevels = 1;
  int amgPmax = 4;
  int amgKeepTranspose = 1;
  double amgTruncFactor = 0.0;
};

struct HypreSolveInfo {
  int iterations = 0;
  double finalRelResNorm = 0.0;
};

HypreSolveInfo solve_system_hypre_gpu(
    const CSRPattern& pat,
    const std::vector<HYPRE_Complex>& values,
    const std::vector<HYPRE_Complex>& rhs,
    std::vector<double>& x,
    const HypreOptions& opt);
