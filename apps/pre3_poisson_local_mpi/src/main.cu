#include <mpi.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

#include "mesh.h"
#include "mms.h"
#include "poisson_library.h"
#include "hypre_backend.h"

static void repair_patch_ranges(Mesh &mesh)
{
  if(mesh.patchStartFace.size() == mesh.patchNames.size() &&
     mesh.patchNFaces.size() == mesh.patchNames.size()) {
    return;
  }

  const int nPatches = (int)mesh.patchNames.size();
  mesh.patchStartFace.assign(nPatches, -1);
  mesh.patchNFaces.assign(nPatches, 0);

  for(int f = mesh.nInternalFaces; f < mesh.nFaces; ++f) {
    const int raw = mesh.bPatch[f];
    const int pidx = raw - 1;
    if(pidx < 0 || pidx >= nPatches) {
      throw std::runtime_error("Bad bPatch index while repairing patch ranges");
    }
    if(mesh.patchNFaces[pidx] == 0) mesh.patchStartFace[pidx] = f;
    mesh.patchNFaces[pidx]++;
  }
}

int main(int argc, char **argv)
{
  MPI_Init(&argc, &argv);

  int rank = 0, size = 1;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  try {
    std::string caseRoot = "/tmp/case";
    int maxit = 4000;
    double tol = 1e-8;
    int nNonOrth = 2;
    std::string gradScheme = "lsq";
    std::string lapScheme = "nonorth";

    for(int i = 1; i < argc; ++i) {
      std::string a = argv[i];
      auto need = [&](const char *key){
        if(i + 1 >= argc) {
          std::fprintf(stderr, "Missing value after %s\\n", key);
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
      } else if(a == "-nonorth") {
        need("-nonorth");
        nNonOrth = std::atoi(argv[++i]);
      } else if(a == "-grad") {
        need("-grad");
        gradScheme = argv[++i];
      } else if(a == "-laplacian") {
        need("-laplacian");
        lapScheme = argv[++i];
      }
    }

    const std::string polyMeshDir =
      caseRoot + "/processor" + std::to_string(rank) + "/constant/polyMesh";

    Mesh mesh = read_openfoam_polymesh(polyMeshDir);
    repair_patch_ranges(mesh);

    // This is a local-per-rank test, not a distributed row solve.
    libpoisson_set_hypre_comm(MPI_COMM_SELF);

    HYPRE_CALL(HYPRE_Initialize());
    HYPRE_CALL(HYPRE_DeviceInitialize());
    HYPRE_CALL(HYPRE_SetMemoryLocation(HYPRE_MEMORY_DEVICE));
    HYPRE_CALL(HYPRE_SetExecutionPolicy(HYPRE_EXEC_DEVICE));

    // Exact MMS source for -laplacian(phi)=f.
    std::vector<double> source(mesh.nCells, 0.0);
    for(int c = 0; c < mesh.nCells; ++c) {
      source[c] = rhs_exact(mesh.cc[c]);
    }

    // For this local test, every boundary patch, including processor patches,
    // receives exact Dirichlet values. This proves decomposed-mesh geometry
    // and libpoisson assembly before we add off-rank columns.
    ScalarBCSet bc;
    for(const auto &name : mesh.patchNames) {
      bc.patches.push_back(make_dirichlet_patch_bc(
        name,
        [](const std::array<double,3>& x, const std::array<double,3>&) {
          return phi_exact(x);
        }));
    }

    EllipticOptions opts;
    opts.nNonOrthCorr = nNonOrth;
    opts.gradScheme = gradScheme;
    opts.laplacianScheme = lapScheme;
    opts.useReferenceCell = false;
    opts.hypre.maxIter = maxit;
    opts.hypre.relTol = 0.0;
    opts.hypre.absTol = tol;
    opts.hypre.tol = tol;
    opts.hypre.monitor = 0;
    opts.hypre.amgRelaxType = 18;
    opts.hypre.amgCoarsenType = 8;
    opts.hypre.amgInterpType = 6;
    opts.hypre.amgAggLevels = 1;
    opts.hypre.amgPmax = 4;
    opts.hypre.amgKeepTranspose = 1;

    const double t0 = MPI_Wtime();
    EllipticResult result = solve_poisson(mesh, source, bc, opts);
    const double t1 = MPI_Wtime();

    double localL2 = 0.0;
    double localInf = 0.0;
    double localVol = 0.0;

    for(int c = 0; c < mesh.nCells; ++c) {
      const double ex = phi_exact(mesh.cc[c]);
      const double e = std::abs(result.phi[c] - ex);
      localL2 += e * e * mesh.vol[c];
      localInf = std::max(localInf, e);
      localVol += mesh.vol[c];
    }

    double globalL2 = 0.0;
    double globalInf = 0.0;
    double globalVol = 0.0;

    MPI_Allreduce(&localL2, &globalL2, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    MPI_Allreduce(&localInf, &globalInf, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
    MPI_Allreduce(&localVol, &globalVol, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

    globalL2 = std::sqrt(globalL2 / std::max(globalVol, 1e-300));

    std::printf(
      "rank %d/%d PRE3C0 local libpoisson MMS: mesh=%s cells=%d faces=%d internalFaces=%d maxNonOrth=%.3f its=%d finalRel=%.3e wall=%.6e localInf=%.6e\\n",
      rank, size, polyMeshDir.c_str(),
      mesh.nCells, mesh.nFaces, mesh.nInternalFaces, mesh.maxNonOrthDeg,
      result.lastSolveInfo.iterations,
      result.lastSolveInfo.finalRelResNorm,
      t1 - t0,
      localInf);
    std::fflush(stdout);

    MPI_Barrier(MPI_COMM_WORLD);

    if(rank == 0) {
      std::printf(
        "PRE3C0 local libpoisson MMS RESULT: globalL2=%.12e globalInf=%.12e globalVol=%.12e\\n",
        globalL2, globalInf, globalVol);

      if(std::isfinite(globalL2) && std::isfinite(globalInf)) {
        std::printf("PRE3C0 RESULT: PASS_RAN\\n");
      } else {
        std::printf("PRE3C0 RESULT: FAIL_NAN_INF\\n");
      }
    }

    HYPRE_Finalize();
    MPI_Finalize();
    return 0;
  }
  catch(const std::exception &e) {
    std::fprintf(stderr, "rank %d ERROR: %s\\n", rank, e.what());
    MPI_Abort(MPI_COMM_WORLD, 1);
  }

  MPI_Finalize();
  return 1;
}
