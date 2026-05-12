#include <mpi.h>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <map>
#include <set>
#include <string>
#include <vector>

#include "poisson_decomp_library.h"
#include "mms.h"

#include "bc_runtime_config.h"

static ScalarBCSet make_exact_mms_dirichlet_bcs_for_physical_patches(const DecompMesh& dm) {
  std::set<std::string> procPatchNames;
  for (const auto& pp : dm.procPatches) procPatchNames.insert(pp.name);

  ScalarBCSet bc;
  for (const auto& name : dm.mesh.patchNames) {
    if (procPatchNames.count(name)) continue;
    bc.patches.push_back(make_dirichlet_patch_bc(
        name,
        [](const std::array<double,3>& x, const std::array<double,3>&) {
          return phi_exact(x);
        }));
  }
  return bc;
}

static ScalarBCSet make_scalar_bcs_from_pressure_config(
    const DecompMesh& dm,
    const std::string& bcConfigPath) {
  auto cfg = pipebc::load_runtime_bc_config(bcConfigPath);
  pipebc::validate_runtime_bc_config_against_patches(cfg, dm.mesh.patchNames);

  const auto dups = pipebc::duplicate_pressure_bc_patches(cfg.pressurePatchSpecs);
  if (!dups.empty()) {
    throw std::runtime_error("Duplicate pressure BC for patch '" + dups.front() + "' in " + bcConfigPath);
  }

  ScalarBCSet bc;
  for (const auto& spec : cfg.pressurePatchSpecs) {
    if (spec.type == pipebc::PressureBCType::FixedValue) {
      bc.patches.push_back(make_dirichlet_constant_bc(spec.patchName, spec.value));
    } else if (spec.type == pipebc::PressureBCType::FixedValueFunction) {
      auto fn = spec.scalarFunction;
      bc.patches.push_back(make_dirichlet_patch_bc(
          spec.patchName,
          [fn](const std::array<double,3>& x, const std::array<double,3>&) {
            return fn ? fn(x, 0.0) : 0.0;
          }));
    } else if (spec.type == pipebc::PressureBCType::ZeroGradient ||
               spec.type == pipebc::PressureBCType::Open) {
      bc.patches.push_back(make_neumann_constant_bc(spec.patchName, 0.0));
    } else {
      throw std::runtime_error("Unsupported pressure BC type in " + bcConfigPath);
    }
  }

  return bc;
}

int main(int argc, char** argv) {
  MPI_Init(&argc, &argv);

  int rank = 0, size = 1;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  try {
    std::string caseRoot = "/tmp/case";
    std::string bcConfigPath;
    int device = rank;
    double tol = 1e-7;
    int maxit = 500;
    int nNonOrthCorr = 2;
    std::string gradScheme = "lsq";
    std::string laplacianScheme = "nonorth";

    for (int i = 1; i < argc; ++i) {
      std::string a = argv[i];

      auto need = [&](const char* key) {
        if (i + 1 >= argc) {
          std::fprintf(stderr, "Missing value after %s\n", key);
          MPI_Abort(MPI_COMM_WORLD, 1);
        }
      };

      if (a == "-case-root") {
        need("-case-root");
        caseRoot = argv[++i];
      } else if (a == "-bc-config" || a == "-case-config") {
        need(a.c_str());
        bcConfigPath = argv[++i];
      } else if (a == "-device") {
        need("-device");
        device = std::atoi(argv[++i]);
      } else if (a == "-tol" || a == "-p-tol") {
        need(a.c_str());
        tol = std::atof(argv[++i]);
      } else if (a == "-maxit" || a == "-p-maxit") {
        need(a.c_str());
        maxit = std::atoi(argv[++i]);
      } else if (a == "-nNonOrthCorr" || a == "-n-nonorth-corr") {
        need(a.c_str());
        nNonOrthCorr = std::atoi(argv[++i]);
      } else if (a == "-grad-scheme") {
        need("-grad-scheme");
        gradScheme = argv[++i];
      } else if (a == "-laplacian-scheme") {
        need("-laplacian-scheme");
        laplacianScheme = argv[++i];
      }
    }

    int devCount = 0;
    CUDA_CALL(cudaGetDeviceCount(&devCount));
    if (devCount > 0) CUDA_CALL(cudaSetDevice(device % devCount));

    DecompMesh dm = read_decomposed_openfoam_case(caseRoot, MPI_COMM_WORLD);

    ScalarBCSet bcSet;
    if (!bcConfigPath.empty()) {
      bcSet = make_scalar_bcs_from_pressure_config(dm, bcConfigPath);
    } else {
      bcSet = make_exact_mms_dirichlet_bcs_for_physical_patches(dm);
    }

    std::vector<double> cellSource(dm.mesh.nCells, 0.0);
    for (int c = 0; c < dm.mesh.nCells; ++c) {
      cellSource[c] = rhs_exact(dm.mesh.cc[c]);
    }

    DistEllipticOptions opts;
    opts.gradScheme = gradScheme;
    opts.laplacianScheme = laplacianScheme;
    opts.nNonOrthCorr = nNonOrthCorr;
    opts.hypre.maxIter = maxit;
    opts.hypre.absTol = tol;
    opts.hypre.relTol = 0.0;
    opts.hypre.tol = tol;
    opts.hypre.monitor = 0;
    opts.hypre.amgMaxIter = 1;
    opts.hypre.amgRelaxType = 18;
    opts.hypre.amgCoarsenType = 8;
    opts.hypre.amgInterpType = 6;
    opts.hypre.amgAggLevels = 1;
    opts.hypre.amgPmax = 4;
    opts.hypre.amgKeepTranspose = 1;

    MPI_Barrier(MPI_COMM_WORLD);
    const double t0 = MPI_Wtime();

    DistEllipticResult result = solve_poisson_decomp(dm, cellSource, bcSet, opts);

    MPI_Barrier(MPI_COMM_WORLD);
    const double t1 = MPI_Wtime();

    double localL2 = 0.0;
    double localInf = 0.0;
    double localVol = 0.0;

    for (int c = 0; c < dm.mesh.nCells; ++c) {
      const double ue = phi_exact(dm.mesh.cc[c]);
      const double e = std::abs(result.phi[c] - ue);
      localL2 += e * e * dm.mesh.vol[c];
      localInf = std::max(localInf, e);
      localVol += dm.mesh.vol[c];
    }

    double globalL2 = 0.0, globalInf = 0.0, globalVol = 0.0;
    MPI_Allreduce(&localL2, &globalL2, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    MPI_Allreduce(&localInf, &globalInf, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
    MPI_Allreduce(&localVol, &globalVol, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

    globalL2 = std::sqrt(globalL2 / std::max(globalVol, 1e-300));

    double wall = t1 - t0;
    double maxWall = 0.0;
    MPI_Reduce(&wall, &maxWall, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    long long localNnz = build_decomp_scalar_pattern(dm).nnz;
    long long globalNnz = 0;
    MPI_Allreduce(&localNnz, &globalNnz, 1, MPI_LONG_LONG, MPI_SUM, MPI_COMM_WORLD);

    std::printf("rank %d/%d rows=[%lld,%lld] nLocal=%d nnz=%lld internalFaces=%d procFaces=%zu boundaryFaces=%d\n",
                rank, size,
                (long long)dm.ilower, (long long)dm.iupper,
                dm.nLocal, localNnz,
                dm.mesh.nInternalFaces,
                build_decomp_scalar_pattern(dm).procFace.size(),
                dm.mesh.nFaces - dm.mesh.nInternalFaces - (int)build_decomp_scalar_pattern(dm).procFace.size());
    std::fflush(stdout);

    if (rank == 0) {
      std::printf("POISSON_DECOMP_MMS setup: ranks=%d globalRows=%lld globalNnz=%lld bcConfig=%s grad=%s laplacian=%s nNonOrthCorr=%d nOuter=%d\n",
                  size, (long long)dm.globalN, globalNnz,
                  bcConfigPath.empty() ? "<exact-mms-default>" : bcConfigPath.c_str(),
                  gradScheme.c_str(), laplacianScheme.c_str(),
                  nNonOrthCorr, result.nOuter);

      std::printf("POISSON_DECOMP_MMS RESULT: its=%d finalRel=%.12e L2=%.12e Linf=%.12e wall=%.6e s\n",
                  result.lastSolveInfo.iterations,
                  result.lastSolveInfo.finalRelResNorm,
                  globalL2, globalInf, maxWall);

      if (std::isfinite(globalL2) && std::isfinite(globalInf)) {
        std::printf("POISSON_DECOMP_MMS PASS_RAN\n");
      } else {
        std::printf("POISSON_DECOMP_MMS FAIL_NAN_INF\n");
      }
    }

    MPI_Finalize();
    return 0;
  } catch (const std::exception& e) {
    std::fprintf(stderr, "rank %d ERROR: %s\n", rank, e.what());
    MPI_Abort(MPI_COMM_WORLD, 1);
  }

  MPI_Finalize();
  return 1;
}
