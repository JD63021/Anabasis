#include "hypre_backend.h"

HypreSolveInfo solve_system_hypre_gpu(
    const CSRPattern& pat,
    const std::vector<HYPRE_Complex>& values,
    const std::vector<HYPRE_Complex>& rhs,
    std::vector<double>& x,
    const HypreOptions& opt) {

  const HYPRE_BigInt ilower = 0;
  const HYPRE_BigInt iupper = static_cast<HYPRE_BigInt>(pat.nRows) - 1;

  HYPRE_IJMatrix Aij = nullptr;
  HYPRE_ParCSRMatrix Apar = nullptr;
  HYPRE_IJVector bij = nullptr, xij = nullptr;
  HYPRE_ParVector bpar = nullptr, xpar = nullptr;
  HYPRE_Solver solver = nullptr, prec = nullptr;

  HYPRE_CALL(HYPRE_IJMatrixCreate(MPI_COMM_WORLD, ilower, iupper, ilower, iupper, &Aij));
  HYPRE_CALL(HYPRE_IJMatrixSetObjectType(Aij, HYPRE_PARCSR));
  HYPRE_CALL(HYPRE_IJMatrixSetRowSizes(Aij, const_cast<HYPRE_Int*>(pat.ncols.data())));
  HYPRE_CALL(HYPRE_IJMatrixInitialize_v2(Aij, HYPRE_MEMORY_HOST));
  HYPRE_CALL(HYPRE_IJMatrixSetValues(
      Aij,
      pat.nRows,
      const_cast<HYPRE_Int*>(pat.ncols.data()),
      const_cast<HYPRE_BigInt*>(pat.rows.data()),
      const_cast<HYPRE_BigInt*>(pat.cols.data()),
      const_cast<HYPRE_Complex*>(values.data())));
  HYPRE_CALL(HYPRE_IJMatrixAssemble(Aij));
  HYPRE_CALL(HYPRE_IJMatrixMigrate(Aij, HYPRE_MEMORY_DEVICE));
  HYPRE_CALL(HYPRE_IJMatrixGetObject(Aij, reinterpret_cast<void**>(&Apar)));

  std::vector<HYPRE_BigInt> idx(pat.nRows);
  std::vector<HYPRE_Complex> x0(pat.nRows, 0.0);
  for (int i = 0; i < pat.nRows; ++i) idx[i] = static_cast<HYPRE_BigInt>(i);

  HYPRE_CALL(HYPRE_IJVectorCreate(MPI_COMM_WORLD, ilower, iupper, &bij));
  HYPRE_CALL(HYPRE_IJVectorSetObjectType(bij, HYPRE_PARCSR));
  HYPRE_CALL(HYPRE_IJVectorInitialize_v2(bij, HYPRE_MEMORY_HOST));
  HYPRE_CALL(HYPRE_IJVectorSetValues(bij, pat.nRows, idx.data(), const_cast<HYPRE_Complex*>(rhs.data())));
  HYPRE_CALL(HYPRE_IJVectorAssemble(bij));
  HYPRE_CALL(HYPRE_IJVectorMigrate(bij, HYPRE_MEMORY_DEVICE));
  HYPRE_CALL(HYPRE_IJVectorGetObject(bij, reinterpret_cast<void**>(&bpar)));

  HYPRE_CALL(HYPRE_IJVectorCreate(MPI_COMM_WORLD, ilower, iupper, &xij));
  HYPRE_CALL(HYPRE_IJVectorSetObjectType(xij, HYPRE_PARCSR));
  HYPRE_CALL(HYPRE_IJVectorInitialize_v2(xij, HYPRE_MEMORY_HOST));
  HYPRE_CALL(HYPRE_IJVectorSetValues(xij, pat.nRows, idx.data(), x0.data()));
  HYPRE_CALL(HYPRE_IJVectorAssemble(xij));
  HYPRE_CALL(HYPRE_IJVectorMigrate(xij, HYPRE_MEMORY_DEVICE));
  HYPRE_CALL(HYPRE_IJVectorGetObject(xij, reinterpret_cast<void**>(&xpar)));

  HYPRE_CALL(HYPRE_ParCSRPCGCreate(MPI_COMM_WORLD, &solver));
  const double pcgRelTol = std::max(opt.relTol, 0.0);
  const double pcgAbsTol = (opt.absTol >= 0.0) ? opt.absTol
                           : ((opt.tol >= 0.0) ? opt.tol : 0.0);

  HYPRE_CALL(HYPRE_PCGSetMaxIter(solver, opt.maxIter));
  HYPRE_CALL(HYPRE_PCGSetTol(solver, pcgRelTol));
  HYPRE_CALL(HYPRE_PCGSetAbsoluteTol(solver, pcgAbsTol));
  HYPRE_CALL(HYPRE_PCGSetTwoNorm(solver, 1));
  HYPRE_CALL(HYPRE_PCGSetLogging(solver, 1));
  HYPRE_CALL(HYPRE_PCGSetPrintLevel(solver, opt.monitor ? 2 : 0));

  HYPRE_CALL(HYPRE_BoomerAMGCreate(&prec));
  HYPRE_CALL(HYPRE_BoomerAMGSetPrintLevel(prec, opt.monitor ? 1 : 0));
  HYPRE_CALL(HYPRE_BoomerAMGSetMaxIter(prec, opt.amgMaxIter));
  HYPRE_CALL(HYPRE_BoomerAMGSetTol(prec, 0.0));
  HYPRE_CALL(HYPRE_BoomerAMGSetRelaxType(prec, opt.amgRelaxType));
  HYPRE_CALL(HYPRE_BoomerAMGSetCoarsenType(prec, opt.amgCoarsenType));
  HYPRE_CALL(HYPRE_BoomerAMGSetInterpType(prec, opt.amgInterpType));
  HYPRE_CALL(HYPRE_BoomerAMGSetNumSweeps(prec, opt.amgNumSweeps));
  HYPRE_CALL(HYPRE_BoomerAMGSetPMaxElmts(prec, opt.amgPmax));
  HYPRE_CALL(HYPRE_BoomerAMGSetKeepTranspose(prec, opt.amgKeepTranspose));
  HYPRE_CALL(HYPRE_BoomerAMGSetTruncFactor(prec, opt.amgTruncFactor));
  HYPRE_CALL(HYPRE_BoomerAMGSetRAP2(prec, opt.amgRAP2));

  if (opt.amgAggLevels > 0) {
    HYPRE_CALL(HYPRE_BoomerAMGSetAggNumLevels(prec, opt.amgAggLevels));
    HYPRE_CALL(HYPRE_BoomerAMGSetAggInterpType(prec, opt.amgAggInterpType));
  }

  if (opt.amgStrongThreshold >= 0.0) {
    HYPRE_CALL(HYPRE_BoomerAMGSetStrongThreshold(prec, opt.amgStrongThreshold));
  }

  HYPRE_CALL(HYPRE_PCGSetPrecond(
      solver,
      reinterpret_cast<HYPRE_PtrToSolverFcn>(HYPRE_BoomerAMGSolve),
      reinterpret_cast<HYPRE_PtrToSolverFcn>(HYPRE_BoomerAMGSetup),
      prec));

  HYPRE_CALL(HYPRE_ParCSRPCGSetup(solver, Apar, bpar, xpar));
  HYPRE_CALL(HYPRE_ParCSRPCGSolve(solver, Apar, bpar, xpar));

  HypreSolveInfo info;
  HYPRE_CALL(HYPRE_PCGGetNumIterations(solver, &info.iterations));
  HYPRE_CALL(HYPRE_PCGGetFinalRelativeResidualNorm(solver, &info.finalRelResNorm));

  HYPRE_CALL(HYPRE_IJVectorMigrate(xij, HYPRE_MEMORY_HOST));
  std::vector<HYPRE_Complex> xhost(pat.nRows, 0.0);
  HYPRE_CALL(HYPRE_IJVectorGetValues(xij, pat.nRows, idx.data(), xhost.data()));
  x.assign(pat.nRows, 0.0);
  for (int i = 0; i < pat.nRows; ++i) x[i] = static_cast<double>(xhost[i]);

  if (prec)   HYPRE_CALL(HYPRE_BoomerAMGDestroy(prec));
  if (solver) HYPRE_CALL(HYPRE_ParCSRPCGDestroy(solver));
  if (bij)    HYPRE_CALL(HYPRE_IJVectorDestroy(bij));
  if (xij)    HYPRE_CALL(HYPRE_IJVectorDestroy(xij));
  if (Aij)    HYPRE_CALL(HYPRE_IJMatrixDestroy(Aij));

  return info;
}

// ===== BEGIN reusable hypre system impl =====

namespace {

static void apply_reusable_hypre_options(HypreReusableSystem& sys) {
  const double pcgRelTol = std::max(sys.opt.relTol, 0.0);
  const double pcgAbsTol = (sys.opt.absTol >= 0.0) ? sys.opt.absTol
                           : ((sys.opt.tol >= 0.0) ? sys.opt.tol : 0.0);

  HYPRE_CALL(HYPRE_PCGSetMaxIter(sys.solver, sys.opt.maxIter));
  HYPRE_CALL(HYPRE_PCGSetTol(sys.solver, pcgRelTol));
  HYPRE_CALL(HYPRE_PCGSetAbsoluteTol(sys.solver, pcgAbsTol));
  HYPRE_CALL(HYPRE_PCGSetTwoNorm(sys.solver, 1));
  HYPRE_CALL(HYPRE_PCGSetLogging(sys.solver, 1));
  HYPRE_CALL(HYPRE_PCGSetPrintLevel(sys.solver, sys.opt.monitor ? 2 : 0));

  HYPRE_CALL(HYPRE_BoomerAMGSetPrintLevel(sys.prec, sys.opt.monitor ? 1 : 0));
  HYPRE_CALL(HYPRE_BoomerAMGSetMaxIter(sys.prec, sys.opt.amgMaxIter));
  HYPRE_CALL(HYPRE_BoomerAMGSetTol(sys.prec, 0.0));
  HYPRE_CALL(HYPRE_BoomerAMGSetRelaxType(sys.prec, sys.opt.amgRelaxType));
  HYPRE_CALL(HYPRE_BoomerAMGSetCoarsenType(sys.prec, sys.opt.amgCoarsenType));
  HYPRE_CALL(HYPRE_BoomerAMGSetInterpType(sys.prec, sys.opt.amgInterpType));
  HYPRE_CALL(HYPRE_BoomerAMGSetNumSweeps(sys.prec, sys.opt.amgNumSweeps));
  HYPRE_CALL(HYPRE_BoomerAMGSetPMaxElmts(sys.prec, sys.opt.amgPmax));
  HYPRE_CALL(HYPRE_BoomerAMGSetKeepTranspose(sys.prec, sys.opt.amgKeepTranspose));
  HYPRE_CALL(HYPRE_BoomerAMGSetTruncFactor(sys.prec, sys.opt.amgTruncFactor));
  HYPRE_CALL(HYPRE_BoomerAMGSetRAP2(sys.prec, sys.opt.amgRAP2));

  if (sys.opt.amgAggLevels > 0) {
    HYPRE_CALL(HYPRE_BoomerAMGSetAggNumLevels(sys.prec, sys.opt.amgAggLevels));
    HYPRE_CALL(HYPRE_BoomerAMGSetAggInterpType(sys.prec, sys.opt.amgAggInterpType));
  }

  if (sys.opt.amgStrongThreshold >= 0.0) {
    HYPRE_CALL(HYPRE_BoomerAMGSetStrongThreshold(sys.prec, sys.opt.amgStrongThreshold));
  }
}

} // namespace

void init_reusable_hypre_system_gpu(
    HypreReusableSystem& sys,
    const std::vector<HYPRE_BigInt>& rows,
    const std::vector<HYPRE_Int>& ncols,
    const std::vector<HYPRE_BigInt>& cols,
    const HypreOptions& opt) {

  if (sys.initialized) {
    sys.opt = opt;
    return;
  }

  sys.nRows = static_cast<int>(rows.size());
  sys.nnz = static_cast<int>(cols.size());
  sys.rows = rows;
  sys.ncols = ncols;
  sys.cols = cols;
  sys.opt = opt;

  const HYPRE_BigInt ilower = 0;
  const HYPRE_BigInt iupper = static_cast<HYPRE_BigInt>(sys.nRows - 1);

  HYPRE_CALL(HYPRE_IJMatrixCreate(MPI_COMM_WORLD, ilower, iupper, ilower, iupper, &sys.Aij));
  HYPRE_CALL(HYPRE_IJMatrixSetObjectType(sys.Aij, HYPRE_PARCSR));
  HYPRE_CALL(HYPRE_IJMatrixSetRowSizes(sys.Aij, sys.ncols.data()));
  HYPRE_CALL(HYPRE_IJMatrixInitialize(sys.Aij));

  std::vector<HYPRE_Complex> zeroVals(sys.nnz, 0.0);
  HYPRE_CALL(HYPRE_IJMatrixSetValues(
      sys.Aij,
      sys.nRows,
      sys.ncols.data(),
      sys.rows.data(),
      sys.cols.data(),
      zeroVals.data()));
  HYPRE_CALL(HYPRE_IJMatrixAssemble(sys.Aij));
  HYPRE_CALL(HYPRE_IJMatrixGetObject(sys.Aij, reinterpret_cast<void**>(&sys.Apar)));

  HYPRE_CALL(HYPRE_IJVectorCreate(MPI_COMM_WORLD, ilower, iupper, &sys.bij));
  HYPRE_CALL(HYPRE_IJVectorSetObjectType(sys.bij, HYPRE_PARCSR));
  HYPRE_CALL(HYPRE_IJVectorInitialize(sys.bij));
  std::vector<HYPRE_Complex> zeroB(sys.nRows, 0.0);
  HYPRE_CALL(HYPRE_IJVectorSetValues(sys.bij, sys.nRows, sys.rows.data(), zeroB.data()));
  HYPRE_CALL(HYPRE_IJVectorAssemble(sys.bij));
  HYPRE_CALL(HYPRE_IJVectorGetObject(sys.bij, reinterpret_cast<void**>(&sys.bpar)));

  HYPRE_CALL(HYPRE_IJVectorCreate(MPI_COMM_WORLD, ilower, iupper, &sys.xij));
  HYPRE_CALL(HYPRE_IJVectorSetObjectType(sys.xij, HYPRE_PARCSR));
  HYPRE_CALL(HYPRE_IJVectorInitialize(sys.xij));
  std::vector<HYPRE_Complex> zeroX(sys.nRows, 0.0);
  HYPRE_CALL(HYPRE_IJVectorSetValues(sys.xij, sys.nRows, sys.rows.data(), zeroX.data()));
  HYPRE_CALL(HYPRE_IJVectorAssemble(sys.xij));
  HYPRE_CALL(HYPRE_IJVectorGetObject(sys.xij, reinterpret_cast<void**>(&sys.xpar)));

  HYPRE_CALL(HYPRE_ParCSRPCGCreate(MPI_COMM_WORLD, &sys.solver));
  HYPRE_CALL(HYPRE_BoomerAMGCreate(&sys.prec));
  apply_reusable_hypre_options(sys);

  HYPRE_CALL(HYPRE_PCGSetPrecond(
      sys.solver,
      reinterpret_cast<HYPRE_PtrToSolverFcn>(HYPRE_BoomerAMGSolve),
      reinterpret_cast<HYPRE_PtrToSolverFcn>(HYPRE_BoomerAMGSetup),
      sys.prec));

  sys.initialized = true;
  sys.isSetup = false;
}

void update_reusable_hypre_system_gpu(
    HypreReusableSystem& sys,
    const std::vector<HYPRE_Complex>& values,
    const std::vector<HYPRE_Complex>& rhs,
    const HypreOptions& opt,
    bool doSetup) {

  if (!sys.initialized) {
    throw std::runtime_error("Reusable Hypre system not initialized");
  }
  if (static_cast<int>(values.size()) != sys.nnz) {
    throw std::runtime_error("update_reusable_hypre_system_gpu: values size mismatch");
  }
  if (static_cast<int>(rhs.size()) != sys.nRows) {
    throw std::runtime_error("update_reusable_hypre_system_gpu: rhs size mismatch");
  }

  sys.opt = opt;

  HYPRE_CALL(HYPRE_IJMatrixSetValues(
      sys.Aij,
      sys.nRows,
      sys.ncols.data(),
      sys.rows.data(),
      sys.cols.data(),
      const_cast<HYPRE_Complex*>(values.data())));
  HYPRE_CALL(HYPRE_IJMatrixAssemble(sys.Aij));
  HYPRE_CALL(HYPRE_IJMatrixGetObject(sys.Aij, reinterpret_cast<void**>(&sys.Apar)));

  HYPRE_CALL(HYPRE_IJVectorSetValues(
      sys.bij,
      sys.nRows,
      sys.rows.data(),
      const_cast<HYPRE_Complex*>(rhs.data())));
  HYPRE_CALL(HYPRE_IJVectorAssemble(sys.bij));
  HYPRE_CALL(HYPRE_IJVectorGetObject(sys.bij, reinterpret_cast<void**>(&sys.bpar)));

  if (doSetup || !sys.isSetup) {
    apply_reusable_hypre_options(sys);
    HYPRE_CALL(HYPRE_ParCSRPCGSetup(sys.solver, sys.Apar, sys.bpar, sys.xpar));
    sys.isSetup = true;
  }
}

HypreSolveInfo solve_reusable_hypre_system_gpu(
    HypreReusableSystem& sys,
    std::vector<double>& x) {

  if (!sys.initialized) {
    throw std::runtime_error("Reusable Hypre system not initialized");
  }

  if (static_cast<int>(x.size()) != sys.nRows) {
    x.assign(sys.nRows, 0.0);
  }

  std::vector<HYPRE_Complex> xHypre(sys.nRows, 0.0);
  for (int i = 0; i < sys.nRows; ++i) xHypre[i] = static_cast<HYPRE_Complex>(x[i]);

  HYPRE_CALL(HYPRE_IJVectorSetValues(
      sys.xij,
      sys.nRows,
      sys.rows.data(),
      xHypre.data()));
  HYPRE_CALL(HYPRE_IJVectorAssemble(sys.xij));
  HYPRE_CALL(HYPRE_IJVectorGetObject(sys.xij, reinterpret_cast<void**>(&sys.xpar)));

  apply_reusable_hypre_options(sys);
  HYPRE_CALL(HYPRE_ParCSRPCGSolve(sys.solver, sys.Apar, sys.bpar, sys.xpar));

  HYPRE_Int its = 0;
  HYPRE_Real relres = 0.0;
  HYPRE_CALL(HYPRE_ParCSRPCGGetNumIterations(sys.solver, &its));
  HYPRE_CALL(HYPRE_ParCSRPCGGetFinalRelativeResidualNorm(sys.solver, &relres));

  HYPRE_CALL(HYPRE_IJVectorGetValues(
      sys.xij,
      sys.nRows,
      sys.rows.data(),
      xHypre.data()));

  for (int i = 0; i < sys.nRows; ++i) x[i] = static_cast<double>(xHypre[i]);

  HypreSolveInfo out;
  out.iterations = static_cast<int>(its);
  out.finalRelResNorm = static_cast<double>(relres);
  return out;
}

void destroy_reusable_hypre_system_gpu(HypreReusableSystem& sys) {
  if (sys.solver) { HYPRE_ParCSRPCGDestroy(sys.solver); sys.solver = nullptr; }
  if (sys.prec)   { HYPRE_BoomerAMGDestroy(sys.prec); sys.prec = nullptr; }

  if (sys.Aij) { HYPRE_IJMatrixDestroy(sys.Aij); sys.Aij = nullptr; sys.Apar = nullptr; }
  if (sys.bij) { HYPRE_IJVectorDestroy(sys.bij); sys.bij = nullptr; sys.bpar = nullptr; }
  if (sys.xij) { HYPRE_IJVectorDestroy(sys.xij); sys.xij = nullptr; sys.xpar = nullptr; }

  sys.initialized = false;
  sys.isSetup = false;
  sys.nRows = 0;
  sys.nnz = 0;
  sys.rows.clear();
  sys.ncols.clear();
  sys.cols.clear();
}

// ===== END reusable hypre system impl =====
