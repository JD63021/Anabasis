#pragma once

#include "scalar_elliptic.h"

struct HypreOptions {
  int maxIter = 4000;  double relTol = 0.0;
  double absTol = -1.0;
  double tol = -1.0; // backward-compatible alias: if absTol < 0, use tol as absolute tol
  int monitor = 1;
  int amgMaxIter = 1;
  int amgRelaxType = 18;
  int amgCoarsenType = 8;
  int amgInterpType = 6;
  int amgAggLevels = 1;
  int amgAggInterpType = 4;
  int amgNumSweeps = 1;
  int amgRAP2 = 0;
  int amgPmax = 4;
  int amgKeepTranspose = 1;
  double amgTruncFactor = 0.0;
  double amgStrongThreshold = -1.0; // <0 means: do not explicitly set it
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

// ===== BEGIN reusable hypre system =====

struct HypreReusableSystem {
  bool initialized = false;
  bool isSetup = false;

  int nRows = 0;
  int nnz = 0;

  std::vector<HYPRE_BigInt> rows;
  std::vector<HYPRE_Int> ncols;
  std::vector<HYPRE_BigInt> cols;

  HypreOptions opt;

  HYPRE_IJMatrix Aij = nullptr;
  HYPRE_ParCSRMatrix Apar = nullptr;

  HYPRE_IJVector bij = nullptr;
  HYPRE_ParVector bpar = nullptr;

  HYPRE_IJVector xij = nullptr;
  HYPRE_ParVector xpar = nullptr;

  HYPRE_Solver solver = nullptr;
  HYPRE_Solver prec = nullptr;
};

void init_reusable_hypre_system_gpu(
    HypreReusableSystem& sys,
    const std::vector<HYPRE_BigInt>& rows,
    const std::vector<HYPRE_Int>& ncols,
    const std::vector<HYPRE_BigInt>& cols,
    const HypreOptions& opt);

void update_reusable_hypre_system_gpu(
    HypreReusableSystem& sys,
    const std::vector<HYPRE_Complex>& values,
    const std::vector<HYPRE_Complex>& rhs,
    const HypreOptions& opt,
    bool doSetup);

HypreSolveInfo solve_reusable_hypre_system_gpu(
    HypreReusableSystem& sys,
    std::vector<double>& x);

void destroy_reusable_hypre_system_gpu(HypreReusableSystem& sys);

// ===== END reusable hypre system =====
