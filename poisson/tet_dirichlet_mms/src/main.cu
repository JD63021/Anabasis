#include "hypre_backend.h"
#include "mms.h"

struct Params {
  std::string polyMeshDir = "./mesh/constant/polyMesh";
  int device = 0;
  int nNonOrthCorr = 2;
  int monitor = 1;
  int maxIter = 4000;
  double tol = 1e-12;
  int amgMaxIter = 1;
  int amgRelaxType = 18;
  int amgCoarsenType = 8;
  int amgInterpType = 6;
  int amgAggLevels = 1;
  int amgPmax = 4;
  int amgKeepTranspose = 1;
  double amgTruncFactor = 0.0;
  std::string gradScheme = "lsq";
  std::string laplacianScheme = "nonorth";
};

static void parse_args(int argc, char** argv, Params& par) {
  for (int i = 1; i < argc; ++i) {
    auto need = [&](const char* opt) {
      if (i + 1 >= argc) {
        std::fprintf(stderr, "Missing value after %s\n", opt);
        MPI_Abort(MPI_COMM_WORLD, 1);
      }
    };
    if (!std::strcmp(argv[i], "-polyMeshDir")) {
      need(argv[i]); par.polyMeshDir = argv[++i];
    } else if (!std::strcmp(argv[i], "-device")) {
      need(argv[i]); par.device = std::atoi(argv[++i]);
    } else if (!std::strcmp(argv[i], "-nNonOrthCorr")) {
      need(argv[i]); par.nNonOrthCorr = std::atoi(argv[++i]);
    } else if (!std::strcmp(argv[i], "-monitor")) {
      need(argv[i]); par.monitor = std::atoi(argv[++i]);
    } else if (!std::strcmp(argv[i], "-maxit")) {
      need(argv[i]); par.maxIter = std::atoi(argv[++i]);
    } else if (!std::strcmp(argv[i], "-tol")) {
      need(argv[i]); par.tol = std::atof(argv[++i]);
    } else if (!std::strcmp(argv[i], "-amg-maxit")) {
      need(argv[i]); par.amgMaxIter = std::atoi(argv[++i]);
    } else if (!std::strcmp(argv[i], "-amg-relax-type")) {
      need(argv[i]); par.amgRelaxType = std::atoi(argv[++i]);
    } else if (!std::strcmp(argv[i], "-amg-coarsen-type")) {
      need(argv[i]); par.amgCoarsenType = std::atoi(argv[++i]);
    } else if (!std::strcmp(argv[i], "-amg-interp-type")) {
      need(argv[i]); par.amgInterpType = std::atoi(argv[++i]);
    } else if (!std::strcmp(argv[i], "-amg-agg-levels")) {
      need(argv[i]); par.amgAggLevels = std::atoi(argv[++i]);
    } else if (!std::strcmp(argv[i], "-amg-pmax")) {
      need(argv[i]); par.amgPmax = std::atoi(argv[++i]);
    } else if (!std::strcmp(argv[i], "-amg-keep-transpose")) {
      need(argv[i]); par.amgKeepTranspose = std::atoi(argv[++i]);
    } else if (!std::strcmp(argv[i], "-amg-trunc-factor")) {
      need(argv[i]); par.amgTruncFactor = std::atof(argv[++i]);
    } else if (!std::strcmp(argv[i], "-grad-scheme")) {
      need(argv[i]); par.gradScheme = argv[++i];
    } else if (!std::strcmp(argv[i], "-laplacian-scheme")) {
      need(argv[i]); par.laplacianScheme = argv[++i];
    } else {
      std::fprintf(stderr, "Unknown option: %s\n", argv[i]);
      MPI_Abort(MPI_COMM_WORLD, 1);
    }
  }
}

int main(int argc, char** argv) {
  MPI_Init(&argc, &argv);
  int rank = 0, size = 1;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  if (size != 1) {
    if (rank == 0) {
      std::fprintf(stderr, "This first Poisson MMS solver is intentionally single-rank/single-GPU. Run it with mpirun -n 1.\n");
    }
    MPI_Abort(MPI_COMM_WORLD, 1);
  }

  Params par;
  parse_args(argc, argv, par);

  if (par.gradScheme != "lsq") {
    if (rank == 0) std::fprintf(stderr, "Only -grad-scheme lsq is implemented in this first build.\n");
    MPI_Abort(MPI_COMM_WORLD, 1);
  }
  if (par.laplacianScheme != "nonorth" && par.laplacianScheme != "orth") {
    if (rank == 0) std::fprintf(stderr, "Use -laplacian-scheme orth or -laplacian-scheme nonorth.\n");
    MPI_Abort(MPI_COMM_WORLD, 1);
  }

  CUDA_CALL(cudaSetDevice(par.device));
  HYPRE_CALL(HYPRE_Initialize());
  HYPRE_CALL(HYPRE_SetMemoryLocation(HYPRE_MEMORY_DEVICE));
  HYPRE_CALL(HYPRE_SetExecutionPolicy(HYPRE_EXEC_DEVICE));

  if (rank == 0) {
    print_device_info(par.device);
    std::printf("Reading mesh from %s\n", par.polyMeshDir.c_str());
  }

  Mesh mesh = read_openfoam_polymesh(par.polyMeshDir);

  if (rank == 0) {
    std::printf("Mesh stats: cells=%d, faces=%d, internalFaces=%d, boundaryPatches=%zu, maxNonOrthDeg=%.3f\n",
                mesh.nCells, mesh.nFaces, mesh.nInternalFaces, mesh.patchNames.size(), mesh.maxNonOrthDeg);
    for (std::size_t i = 0; i < mesh.patchNames.size(); ++i) {
      std::printf("  patch[%zu] = %s\n", i, mesh.patchNames[i].c_str());
    }
  }

  std::vector<double> boundaryFaceValue(mesh.nFaces, 0.0);
  for (int f = mesh.nInternalFaces; f < mesh.nFaces; ++f) {
    boundaryFaceValue[f] = phi_exact(mesh.xf[f]);
  }

  std::vector<double> cellSource(mesh.nCells, 0.0);
  for (int c = 0; c < mesh.nCells; ++c) {
    cellSource[c] = rhs_exact(mesh.cc[c]);
  }

  CSRPattern pat = build_scalar_pattern(mesh);
  std::vector<double> phi(mesh.nCells, 0.0);
  std::vector<std::array<double,3>> grad(mesh.nCells, {0.0, 0.0, 0.0});
  std::vector<HYPRE_Complex> values, rhs;
  HypreSolveInfo lastInfo{};

  const bool includeNonOrth = (par.laplacianScheme == "nonorth");
  const int nOuter = includeNonOrth ? std::max(par.nNonOrthCorr, 0) + 1 : 1;

  HypreOptions hopts;
  hopts.maxIter = par.maxIter;
  hopts.tol = par.tol;
  hopts.monitor = par.monitor;
  hopts.amgMaxIter = par.amgMaxIter;
  hopts.amgRelaxType = par.amgRelaxType;
  hopts.amgCoarsenType = par.amgCoarsenType;
  hopts.amgInterpType = par.amgInterpType;
  hopts.amgAggLevels = par.amgAggLevels;
  hopts.amgPmax = par.amgPmax;
  hopts.amgKeepTranspose = par.amgKeepTranspose;
  hopts.amgTruncFactor = par.amgTruncFactor;

  for (int outer = 0; outer < nOuter; ++outer) {
    compute_lsq_gradient(mesh, phi, boundaryFaceValue, grad);
    assemble_poisson_system(mesh, pat, cellSource, boundaryFaceValue, grad, values, rhs, includeNonOrth);
    lastInfo = solve_system_hypre_gpu(pat, values, rhs, phi, hopts);
    if (rank == 0) {
      std::printf("nonOrthIter %d/%d: hypreIts=%d finalRelRes=%.6e\n",
                  outer + 1, nOuter, lastInfo.iterations, lastInfo.finalRelResNorm);
    }
  }

  double volSum = 0.0;
  double l1 = 0.0;
  double l2 = 0.0;
  double linf = 0.0;
  double phiMin = std::numeric_limits<double>::max();
  double phiMax = -std::numeric_limits<double>::max();
  for (int c = 0; c < mesh.nCells; ++c) {
    double exact = phi_exact(mesh.cc[c]);
    double err = std::fabs(phi[c] - exact);
    double V = mesh.vol[c];
    volSum += V;
    l1 += V * err;
    l2 += V * err * err;
    linf = std::max(linf, err);
    phiMin = std::min(phiMin, phi[c]);
    phiMax = std::max(phiMax, phi[c]);
  }
  l1 /= std::max(volSum, 1e-30);
  l2 = std::sqrt(l2 / std::max(volSum, 1e-30));

  if (rank == 0) {
    std::printf("\n=== Poisson MMS summary ===\n");
    std::printf("gradScheme      : %s\n", par.gradScheme.c_str());
    std::printf("laplacianScheme : %s\n", par.laplacianScheme.c_str());
    std::printf("nNonOrthCorr    : %d\n", par.nNonOrthCorr);
    std::printf("solution min/max: %.12e  %.12e\n", phiMin, phiMax);
    std::printf("L1 error        : %.12e\n", l1);
    std::printf("L2 error        : %.12e\n", l2);
    std::printf("Linf error      : %.12e\n", linf);
  }
  HYPRE_CALL(HYPRE_Finalize());
  MPI_Finalize();
  return 0;
}
