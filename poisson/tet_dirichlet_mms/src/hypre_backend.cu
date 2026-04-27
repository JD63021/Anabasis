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
  HYPRE_CALL(HYPRE_PCGSetMaxIter(solver, opt.maxIter));
  HYPRE_CALL(HYPRE_PCGSetTol(solver, opt.tol));
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
  HYPRE_CALL(HYPRE_BoomerAMGSetAggNumLevels(prec, opt.amgAggLevels));
  HYPRE_CALL(HYPRE_BoomerAMGSetPMaxElmts(prec, opt.amgPmax));
  HYPRE_CALL(HYPRE_BoomerAMGSetKeepTranspose(prec, opt.amgKeepTranspose));
  HYPRE_CALL(HYPRE_BoomerAMGSetTruncFactor(prec, opt.amgTruncFactor));

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
