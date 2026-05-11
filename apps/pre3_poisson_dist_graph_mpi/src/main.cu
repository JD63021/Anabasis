#include <mpi.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <map>
#include <regex>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "mesh.h"
#include "hypre_backend.h"

static std::string read_file(const std::string &path)
{
  std::ifstream in(path);
  if(!in) throw std::runtime_error("Could not open " + path);
  std::ostringstream ss;
  ss << in.rdbuf();
  return ss.str();
}

static std::vector<long long> read_label_list(const std::string &path)
{
  const std::string txt = read_file(path);
  std::regex re(R"(\n\s*([0-9]+)\s*\n\s*\()");
  std::smatch m;
  if(!std::regex_search(txt, m, re)) {
    throw std::runtime_error("Could not parse labelList " + path);
  }

  const int n = std::stoi(m[1].str());
  const size_t bodyStart = (size_t)m.position(0) + (size_t)m.length(0);

  std::vector<long long> vals;
  vals.reserve(n);

  std::regex intRe(R"([-+]?[0-9]+)");
  auto it = std::sregex_iterator(txt.begin() + (long long)bodyStart, txt.end(), intRe);
  auto end = std::sregex_iterator();

  for(; it != end && (int)vals.size() < n; ++it) {
    vals.push_back(std::stoll((*it).str()));
  }

  if((int)vals.size() != n) {
    throw std::runtime_error("Wrong count in labelList " + path);
  }

  return vals;
}

struct ProcPatch {
  std::string name;
  int nFaces = 0;
  int startFace = 0;
  int myProcNo = -1;
  int neighbProcNo = -1;
};

static int find_int_entry(const std::string &body, const std::string &key, int def = -1)
{
  std::regex re("\\b" + key + R"(\s+([-+]?[0-9]+)\s*;)");
  std::smatch m;
  if(std::regex_search(body, m, re)) return std::stoi(m[1].str());
  return def;
}

static std::string find_word_entry(const std::string &body, const std::string &key)
{
  std::regex re("\\b" + key + R"(\s+([A-Za-z0-9_]+)\s*;)");
  std::smatch m;
  if(std::regex_search(body, m, re)) return m[1].str();
  return "";
}

static std::vector<ProcPatch> read_processor_patches(const std::string &boundaryPath)
{
  const std::string txt = read_file(boundaryPath);

  std::vector<ProcPatch> out;

  std::regex blockRe(R"(([A-Za-z0-9_]+)\s*\{([^{}]*)\})");
  auto begin = std::sregex_iterator(txt.begin(), txt.end(), blockRe);
  auto end = std::sregex_iterator();

  for(auto it = begin; it != end; ++it) {
    const std::string name = (*it)[1].str();
    const std::string body = (*it)[2].str();

    if(find_word_entry(body, "type") != "processor") continue;

    ProcPatch p;
    p.name = name;
    p.nFaces = find_int_entry(body, "nFaces");
    p.startFace = find_int_entry(body, "startFace");
    p.myProcNo = find_int_entry(body, "myProcNo");
    p.neighbProcNo = find_int_entry(body, "neighbProcNo");

    if(p.nFaces < 0 || p.startFace < 0 || p.myProcNo < 0 || p.neighbProcNo < 0) {
      throw std::runtime_error("Incomplete processor patch in " + boundaryPath);
    }

    out.push_back(p);
  }

  return out;
}

static double u_exact_row(long long globalRow)
{
  const double x = double(globalRow + 1);
  return std::sin(0.013579 * x) + 0.25 * std::cos(0.00731 * x);
}

static void add_coeff(std::vector<std::map<HYPRE_BigInt, double>> &rows,
                      int lr,
                      HYPRE_BigInt col,
                      double val)
{
  rows[lr][col] += val;
}

int main(int argc, char **argv)
{
  MPI_Init(&argc, &argv);

  int rank = 0, size = 1;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  try {
    std::string caseRoot = "/tmp/case";
    int maxit = 500;
    double tol = 1e-7;

    for(int i = 1; i < argc; ++i) {
      std::string a = argv[i];
      auto need = [&](const char *key){
        if(i + 1 >= argc) {
          std::fprintf(stderr, "Missing value after %s\n", key);
          MPI_Abort(MPI_COMM_WORLD, 1);
        }
      };

      if(a == "-case-root") {
        need("-case-root");
        caseRoot = argv[++i];
      } else if(a == "-maxit") {
        need("-maxit");
        maxit = std::atoi(argv[++i]);
      } else if(a == "-tol") {
        need("-tol");
        tol = std::atof(argv[++i]);
      }
    }

    const std::string polyMeshDir =
      caseRoot + "/processor" + std::to_string(rank) + "/constant/polyMesh";

    Mesh mesh = read_openfoam_polymesh(polyMeshDir);
    const auto procPatches = read_processor_patches(polyMeshDir + "/boundary");

    const int nLocal = mesh.nCells;

    std::vector<int> counts(size, 0);
    MPI_Allgather(&nLocal, 1, MPI_INT, counts.data(), 1, MPI_INT, MPI_COMM_WORLD);

    std::vector<int> offsets(size + 1, 0);
    for(int r = 0; r < size; ++r) offsets[r+1] = offsets[r] + counts[r];

    const HYPRE_BigInt ilower = offsets[rank];
    const HYPRE_BigInt iupper = offsets[rank] + nLocal - 1;
    const HYPRE_BigInt globalN = offsets[size];

    auto local_row = [&](int c) -> HYPRE_BigInt {
      return HYPRE_BigInt(offsets[rank] + c);
    };

    std::vector<HYPRE_BigInt> remoteRowForFace(mesh.nFaces, -1);

    for(const auto &pp : procPatches) {
      const int nbr = pp.neighbProcNo;

      std::vector<long long> sendRows(pp.nFaces, -1);
      for(int i = 0; i < pp.nFaces; ++i) {
        const int f = pp.startFace + i;
        const int P = mesh.owner[f];
        sendRows[i] = (long long)local_row(P);
      }

      int recvN = 0;
      MPI_Sendrecv(&pp.nFaces, 1, MPI_INT, nbr, 100,
                   &recvN, 1, MPI_INT, nbr, 100,
                   MPI_COMM_WORLD, MPI_STATUS_IGNORE);

      if(recvN != pp.nFaces) {
        throw std::runtime_error("processor patch face-count mismatch");
      }

      std::vector<long long> recvRows(recvN, -1);
      MPI_Sendrecv(sendRows.data(), pp.nFaces, MPI_LONG_LONG, nbr, 101,
                   recvRows.data(), recvN, MPI_LONG_LONG, nbr, 101,
                   MPI_COMM_WORLD, MPI_STATUS_IGNORE);

      for(int i = 0; i < pp.nFaces; ++i) {
        const int f = pp.startFace + i;
        remoteRowForFace[f] = HYPRE_BigInt(recvRows[i]);
      }
    }

    std::vector<char> isProcFace(mesh.nFaces, 0);
    for(const auto &pp : procPatches) {
      for(int i = 0; i < pp.nFaces; ++i) {
        isProcFace[pp.startFace + i] = 1;
      }
    }

    std::vector<std::map<HYPRE_BigInt, double>> rows(nLocal);

    long long nLocalInternalContrib = 0;
    long long nProcContrib = 0;
    long long nBoundaryContrib = 0;

    // Internal local faces.
    for(int f = 0; f < mesh.nInternalFaces; ++f) {
      const int P = mesh.owner[f];
      const int N = mesh.neigh[f];

      const HYPRE_BigInt gP = local_row(P);
      const HYPRE_BigInt gN = local_row(N);

      add_coeff(rows, P, gP, +1.0);
      add_coeff(rows, P, gN, -1.0);

      add_coeff(rows, N, gN, +1.0);
      add_coeff(rows, N, gP, -1.0);

      nLocalInternalContrib += 2;
    }

    // Boundary and processor faces.
    for(int f = mesh.nInternalFaces; f < mesh.nFaces; ++f) {
      const int P = mesh.owner[f];
      const HYPRE_BigInt gP = local_row(P);

      if(isProcFace[f]) {
        const HYPRE_BigInt gRemote = remoteRowForFace[f];
        if(gRemote < 0) throw std::runtime_error("missing remote row for processor face");

        add_coeff(rows, P, gP, +1.0);
        add_coeff(rows, P, gRemote, -1.0);
        nProcContrib++;
      } else {
        // Physical Dirichlet-like graph anchor.
        add_coeff(rows, P, gP, +1.0);
        nBoundaryContrib++;
      }
    }

    CSRPattern pat;
    pat.nRows = nLocal;
    pat.rows.resize(nLocal);
    pat.ncols.resize(nLocal);
    pat.rowOffsets.resize(nLocal + 1);
    pat.diagPos.resize(nLocal);

    pat.rowOffsets[0] = 0;
    for(int c = 0; c < nLocal; ++c) {
      pat.rows[c] = local_row(c);
      pat.ncols[c] = HYPRE_Int(rows[c].size());
      pat.rowOffsets[c + 1] = pat.rowOffsets[c] + int(rows[c].size());
    }

    pat.nnz = pat.rowOffsets[nLocal];
    pat.cols.resize(pat.nnz);

    std::vector<HYPRE_Complex> values(pat.nnz, 0.0);
    std::vector<HYPRE_Complex> rhs(nLocal, 0.0);

    for(int c = 0; c < nLocal; ++c) {
      int k = pat.rowOffsets[c];
      const HYPRE_BigInt diag = local_row(c);

      double b = 0.0;
      for(const auto &kv : rows[c]) {
        pat.cols[k] = kv.first;
        values[k] = HYPRE_Complex(kv.second);
        if(kv.first == diag) pat.diagPos[c] = k;

        b += kv.second * u_exact_row((long long)kv.first);
        ++k;
      }

      rhs[c] = HYPRE_Complex(b);
    }

    long long localNnz = pat.nnz;
    long long globalNnz = 0;
    MPI_Allreduce(&localNnz, &globalNnz, 1, MPI_LONG_LONG, MPI_SUM, MPI_COMM_WORLD);

    if(rank == 0) {
      std::printf("PRE3C1 distributed graph Poisson setup: ranks=%d globalRows=%lld globalNnz=%lld\n",
                  size, (long long)globalN, globalNnz);
    }

    std::printf("rank %d/%d rows=[%lld,%lld] nLocal=%d localNnz=%d internalContrib=%lld procContrib=%lld boundaryContrib=%lld\n",
                rank, size, (long long)ilower, (long long)iupper, nLocal, pat.nnz,
                nLocalInternalContrib, nProcContrib, nBoundaryContrib);
    std::fflush(stdout);

    HYPRE_CALL(HYPRE_Initialize());
#if defined(HYPRE_USING_GPU)
    HYPRE_CALL(HYPRE_DeviceInitialize());
    HYPRE_CALL(HYPRE_SetMemoryLocation(HYPRE_MEMORY_HOST));
    HYPRE_CALL(HYPRE_SetExecutionPolicy(HYPRE_EXEC_HOST));
#endif

    HypreOptions opt;
    opt.maxIter = maxit;
    opt.relTol = 0.0;
    opt.absTol = tol;
    opt.tol = tol;
    opt.monitor = 0;
    opt.amgRelaxType = 18;
    opt.amgCoarsenType = 8;
    opt.amgInterpType = 6;
    opt.amgAggLevels = 1;
    opt.amgPmax = 4;
    opt.amgKeepTranspose = 1;

    // This backend now uses the selected communicator from the earlier patch.
    // But for true distributed solve, it still needs correct global ilower/iupper.
    // So we do the HYPRE solve here directly, not through solve_system_hypre_gpu().
    HYPRE_IJMatrix Aij = nullptr;
    HYPRE_ParCSRMatrix A = nullptr;
    HYPRE_IJVector bij = nullptr, xij = nullptr;
    HYPRE_ParVector bpar = nullptr, xpar = nullptr;
    HYPRE_Solver solver = nullptr, prec = nullptr;

    HYPRE_CALL(HYPRE_IJMatrixCreate(MPI_COMM_WORLD, ilower, iupper, ilower, iupper, &Aij));
    HYPRE_CALL(HYPRE_IJMatrixSetObjectType(Aij, HYPRE_PARCSR));
    HYPRE_CALL(HYPRE_IJMatrixSetRowSizes(Aij, pat.ncols.data()));
    HYPRE_CALL(HYPRE_IJMatrixInitialize_v2(Aij, HYPRE_MEMORY_HOST));
    HYPRE_CALL(HYPRE_IJMatrixSetValues(Aij, pat.nRows, pat.ncols.data(),
                                       pat.rows.data(), pat.cols.data(), values.data()));
    HYPRE_CALL(HYPRE_IJMatrixAssemble(Aij));
    HYPRE_CALL(HYPRE_IJMatrixMigrate(Aij, HYPRE_MEMORY_DEVICE));
    HYPRE_CALL(HYPRE_IJMatrixGetObject(Aij, (void**)&A));

    std::vector<HYPRE_Complex> x0(nLocal, 0.0);
    std::vector<HYPRE_Complex> xhost(nLocal, 0.0);

    HYPRE_CALL(HYPRE_IJVectorCreate(MPI_COMM_WORLD, ilower, iupper, &bij));
    HYPRE_CALL(HYPRE_IJVectorSetObjectType(bij, HYPRE_PARCSR));
    HYPRE_CALL(HYPRE_IJVectorInitialize_v2(bij, HYPRE_MEMORY_HOST));
    HYPRE_CALL(HYPRE_IJVectorSetValues(bij, nLocal, pat.rows.data(), rhs.data()));
    HYPRE_CALL(HYPRE_IJVectorAssemble(bij));
    HYPRE_CALL(HYPRE_IJVectorMigrate(bij, HYPRE_MEMORY_DEVICE));
    HYPRE_CALL(HYPRE_IJVectorGetObject(bij, (void**)&bpar));

    HYPRE_CALL(HYPRE_IJVectorCreate(MPI_COMM_WORLD, ilower, iupper, &xij));
    HYPRE_CALL(HYPRE_IJVectorSetObjectType(xij, HYPRE_PARCSR));
    HYPRE_CALL(HYPRE_IJVectorInitialize_v2(xij, HYPRE_MEMORY_HOST));
    HYPRE_CALL(HYPRE_IJVectorSetValues(xij, nLocal, pat.rows.data(), x0.data()));
    HYPRE_CALL(HYPRE_IJVectorAssemble(xij));
    HYPRE_CALL(HYPRE_IJVectorMigrate(xij, HYPRE_MEMORY_DEVICE));
    HYPRE_CALL(HYPRE_IJVectorGetObject(xij, (void**)&xpar));

    HYPRE_CALL(HYPRE_ParCSRPCGCreate(MPI_COMM_WORLD, &solver));
    HYPRE_CALL(HYPRE_PCGSetMaxIter(solver, opt.maxIter));
    HYPRE_CALL(HYPRE_PCGSetTol(solver, opt.relTol));
    HYPRE_CALL(HYPRE_PCGSetAbsoluteTol(solver, opt.absTol));
    HYPRE_CALL(HYPRE_PCGSetTwoNorm(solver, 1));
    HYPRE_CALL(HYPRE_PCGSetLogging(solver, 1));
    HYPRE_CALL(HYPRE_PCGSetPrintLevel(solver, 0));

    HYPRE_CALL(HYPRE_BoomerAMGCreate(&prec));
    HYPRE_CALL(HYPRE_BoomerAMGSetPrintLevel(prec, 0));
    HYPRE_CALL(HYPRE_BoomerAMGSetMaxIter(prec, 1));
    HYPRE_CALL(HYPRE_BoomerAMGSetTol(prec, 0.0));
    HYPRE_CALL(HYPRE_BoomerAMGSetRelaxType(prec, opt.amgRelaxType));
    HYPRE_CALL(HYPRE_BoomerAMGSetCoarsenType(prec, opt.amgCoarsenType));
    HYPRE_CALL(HYPRE_BoomerAMGSetInterpType(prec, opt.amgInterpType));
    HYPRE_CALL(HYPRE_BoomerAMGSetNumSweeps(prec, opt.amgNumSweeps));
    HYPRE_CALL(HYPRE_BoomerAMGSetPMaxElmts(prec, opt.amgPmax));
    HYPRE_CALL(HYPRE_BoomerAMGSetKeepTranspose(prec, opt.amgKeepTranspose));

    HYPRE_CALL(HYPRE_PCGSetPrecond(
      solver,
      (HYPRE_PtrToSolverFcn)HYPRE_BoomerAMGSolve,
      (HYPRE_PtrToSolverFcn)HYPRE_BoomerAMGSetup,
      prec));

    MPI_Barrier(MPI_COMM_WORLD);
    const double t0 = MPI_Wtime();

    HYPRE_CALL(HYPRE_ParCSRPCGSetup(solver, A, bpar, xpar));
    HYPRE_Int solveErr = HYPRE_ParCSRPCGSolve(solver, A, bpar, xpar);

    MPI_Barrier(MPI_COMM_WORLD);
    const double t1 = MPI_Wtime();

    HYPRE_Int its = 0;
    HYPRE_Real rel = 0.0;
    HYPRE_CALL(HYPRE_PCGGetNumIterations(solver, &its));
    HYPRE_CALL(HYPRE_PCGGetFinalRelativeResidualNorm(solver, &rel));

    HYPRE_CALL(HYPRE_IJVectorMigrate(xij, HYPRE_MEMORY_HOST));
    HYPRE_CALL(HYPRE_IJVectorGetValues(xij, nLocal, pat.rows.data(), xhost.data()));

    double localL2 = 0.0;
    double localInf = 0.0;

    for(int c = 0; c < nLocal; ++c) {
      const double ue = u_exact_row((long long)local_row(c));
      const double e = std::abs(double(xhost[c]) - ue);
      localL2 += e * e;
      localInf = std::max(localInf, e);
    }

    double globalL2 = 0.0;
    double globalInf = 0.0;
    MPI_Allreduce(&localL2, &globalL2, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    MPI_Allreduce(&localInf, &globalInf, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);

    globalL2 = std::sqrt(globalL2 / double(globalN));

    double wall = t1 - t0;
    double maxWall = 0.0;
    MPI_Reduce(&wall, &maxWall, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    if(rank == 0) {
      std::printf("PRE3C1 distributed graph Poisson RESULT: solveErr=%d its=%d finalRel=%.12e L2=%.12e Linf=%.12e wall=%.6e s\n",
                  (int)solveErr, (int)its, double(rel), globalL2, globalInf, maxWall);
      if(std::isfinite(globalL2) && std::isfinite(globalInf) && globalInf < 1e-4) {
        std::printf("PRE3C1 RESULT: PASS\n");
      } else {
        std::printf("PRE3C1 RESULT: CHECK\n");
      }
    }

    HYPRE_CALL(HYPRE_BoomerAMGDestroy(prec));
    HYPRE_CALL(HYPRE_ParCSRPCGDestroy(solver));
    HYPRE_CALL(HYPRE_IJVectorDestroy(xij));
    HYPRE_CALL(HYPRE_IJVectorDestroy(bij));
    HYPRE_CALL(HYPRE_IJMatrixDestroy(Aij));
    HYPRE_Finalize();

    MPI_Finalize();
    return 0;
  }
  catch(const std::exception &e) {
    std::fprintf(stderr, "rank %d ERROR: %s\n", rank, e.what());
    MPI_Abort(MPI_COMM_WORLD, 1);
  }

  MPI_Finalize();
  return 1;
}
