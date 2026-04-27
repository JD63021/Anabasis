#include "poisson_library.h"

#include <array>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <limits>
#include <string>
#include <vector>

static constexpr double PI = 3.141592653589793238462643383279502884;

struct Params {
  std::string polyMeshDir = "./mesh/constant/polyMesh";
  int device = 0;
  int nNonOrthCorr = 10;
  int monitor = 1;
  int maxIter = 4000;
  double tol = 1e-10;
  int amgMaxIter = 1;
  int amgRelaxType = 7;
  int amgCoarsenType = 8;
  int amgInterpType = 6;
  int amgAggLevels = 1;
  int amgPmax = 4;
  int amgKeepTranspose = 1;
  double amgTruncFactor = 0.0;
  double amgStrongThreshold = 0.6;
  std::string gradScheme = "lsq";
  std::string laplacianScheme = "nonorth";
  int useReferenceCell = 1;
  int referenceCell = 0;
  int referenceValueUserSet = 0;
  double referenceValue = 0.0;
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
    } else if (!std::strcmp(argv[i], "-amg-strong-threshold")) {
      need(argv[i]); par.amgStrongThreshold = std::atof(argv[++i]);
    } else if (!std::strcmp(argv[i], "-grad-scheme")) {
      need(argv[i]); par.gradScheme = argv[++i];
    } else if (!std::strcmp(argv[i], "-laplacian-scheme")) {
      need(argv[i]); par.laplacianScheme = argv[++i];
    } else if (!std::strcmp(argv[i], "-use-reference-cell")) {
      need(argv[i]); par.useReferenceCell = std::atoi(argv[++i]);
    } else if (!std::strcmp(argv[i], "-reference-cell")) {
      need(argv[i]); par.referenceCell = std::atoi(argv[++i]);
    } else if (!std::strcmp(argv[i], "-reference-value")) {
      need(argv[i]); par.referenceValue = std::atof(argv[++i]);
      par.referenceValueUserSet = 1;
    } else {
      std::fprintf(stderr, "Unknown option: %s\n", argv[i]);
      MPI_Abort(MPI_COMM_WORLD, 1);
    }
  }
}

static double phi_exact(const std::array<double,3>& x) {
  return std::cos(PI*x[0]) * std::cos(PI*x[1]) * std::cos(PI*x[2]);
}

static std::array<double,3> grad_exact(const std::array<double,3>& x) {
  const double cx = std::cos(PI*x[0]), cy = std::cos(PI*x[1]), cz = std::cos(PI*x[2]);
  const double sx = std::sin(PI*x[0]), sy = std::sin(PI*x[1]), sz = std::sin(PI*x[2]);
  return {
    -PI * sx * cy * cz,
    -PI * cx * sy * cz,
    -PI * cx * cy * sz
  };
}

static double normal_grad_exact(const std::array<double,3>& x, const std::array<double,3>& n) {
  const auto g = grad_exact(x);
  return g[0]*n[0] + g[1]*n[1] + g[2]*n[2];
}

static double rhs_exact(const std::array<double,3>& x) {
  return 3.0 * PI * PI * phi_exact(x);
}

int main(int argc, char** argv) {
  MPI_Init(&argc, &argv);
  int rank = 0, size = 1;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  if (size != 1) {
    if (rank == 0) {
      std::fprintf(stderr, "This Neumann Poisson MMS app is intentionally single-rank/single-GPU. Run it with mpirun -n 1.\n");
    }
    MPI_Abort(MPI_COMM_WORLD, 1);
  }

  Params par;
  parse_args(argc, argv, par);
  init_hypre_gpu_runtime(par.device);

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

  ScalarBCSet bcSet;
  for (const auto& patchName : mesh.patchNames) {
    bcSet.patches.push_back(make_neumann_patch_bc(
        patchName,
        [](const std::array<double,3>& x, const std::array<double,3>& n) {
          return normal_grad_exact(x, n);
        }));
  }

  std::vector<double> cellSource(mesh.nCells, 0.0);
  double srcVolSum = 0.0;
  double volSumSrc = 0.0;
  for (int c = 0; c < mesh.nCells; ++c) {
    cellSource[c] = rhs_exact(mesh.cc[c]);
    srcVolSum += cellSource[c] * mesh.vol[c];
    volSumSrc += mesh.vol[c];
  }
  const double srcMean = srcVolSum / std::max(volSumSrc, 1e-30);
  for (int c = 0; c < mesh.nCells; ++c) {
    cellSource[c] -= srcMean;  // enforce discrete compatibility for pure Neumann MMS
  }
  if (rank == 0) {
    std::printf("Discrete source mean removed: %.12e\n", srcMean);
  }

  EllipticOptions opts;
  opts.nNonOrthCorr = par.nNonOrthCorr;
  opts.gradScheme = par.gradScheme;
  opts.laplacianScheme = par.laplacianScheme;
  opts.useReferenceCell = (par.useReferenceCell != 0);
  opts.referenceCell = par.referenceCell;

  if (opts.useReferenceCell) {
    if (par.referenceCell < 0 || par.referenceCell >= mesh.nCells) {
      if (rank == 0) std::fprintf(stderr, "referenceCell %d out of range [0,%d)\n", par.referenceCell, mesh.nCells);
      MPI_Abort(MPI_COMM_WORLD, 1);
    }
    opts.referenceValue = par.referenceValueUserSet
                        ? par.referenceValue
                        : phi_exact(mesh.cc[par.referenceCell]);
  }

  opts.hypre.maxIter = par.maxIter;
  opts.hypre.relTol = 0.0;
  opts.hypre.absTol = par.tol;
  opts.hypre.tol = -1.0;
  opts.hypre.monitor = par.monitor;
  opts.hypre.amgMaxIter = par.amgMaxIter;
  opts.hypre.amgRelaxType = par.amgRelaxType;
  opts.hypre.amgCoarsenType = par.amgCoarsenType;
  opts.hypre.amgInterpType = par.amgInterpType;
  opts.hypre.amgAggLevels = par.amgAggLevels;
  opts.hypre.amgPmax = par.amgPmax;
  opts.hypre.amgKeepTranspose = par.amgKeepTranspose;
  opts.hypre.amgTruncFactor = par.amgTruncFactor;
  opts.hypre.amgStrongThreshold = par.amgStrongThreshold;

  EllipticResult result = solve_poisson(mesh, cellSource, bcSet, opts);
  const auto& phi = result.phi;

  double volSum = 0.0, l1 = 0.0, l2 = 0.0, linf = 0.0;
  double phiMin = std::numeric_limits<double>::max();
  double phiMax = -std::numeric_limits<double>::max();
  double errMeanNum = 0.0;

  for (int c = 0; c < mesh.nCells; ++c) {
    const double exact = phi_exact(mesh.cc[c]);
    const double rawErr = phi[c] - exact;
    const double errAbs = std::fabs(rawErr);
    const double V = mesh.vol[c];
    volSum += V;
    errMeanNum += V * rawErr;
    l1 += V * errAbs;
    l2 += V * errAbs * errAbs;
    linf = std::max(linf, errAbs);
    phiMin = std::min(phiMin, phi[c]);
    phiMax = std::max(phiMax, phi[c]);
  }

  l1 /= std::max(volSum, 1e-30);
  l2 = std::sqrt(l2 / std::max(volSum, 1e-30));

  const double errMean = errMeanNum / std::max(volSum, 1e-30);

  double l1Shift = 0.0, l2Shift = 0.0, linfShift = 0.0;
  for (int c = 0; c < mesh.nCells; ++c) {
    const double exact = phi_exact(mesh.cc[c]);
    const double shiftedErr = (phi[c] - exact) - errMean;
    const double errAbs = std::fabs(shiftedErr);
    const double V = mesh.vol[c];
    l1Shift += V * errAbs;
    l2Shift += V * errAbs * errAbs;
    linfShift = std::max(linfShift, errAbs);
  }
  l1Shift /= std::max(volSum, 1e-30);
  l2Shift = std::sqrt(l2Shift / std::max(volSum, 1e-30));

  if (rank == 0) {
    std::printf("\n=== Poisson Neumann MMS summary ===\n");
    std::printf("gradScheme      : %s\n", par.gradScheme.c_str());
    std::printf("laplacianScheme : %s\n", par.laplacianScheme.c_str());
    std::printf("nNonOrthCorr    : %d\n", par.nNonOrthCorr);
    std::printf("last hypreIts   : %d\n", result.lastSolveInfo.iterations);
    std::printf("last relRes     : %.12e\n", result.lastSolveInfo.finalRelResNorm);
    if (opts.useReferenceCell) {
      std::printf("referenceCell   : %d\n", opts.referenceCell);
      std::printf("referenceValue  : %.12e\n", opts.referenceValue);
    }
    std::printf("solution min/max: %.12e  %.12e\n", phiMin, phiMax);
    std::printf("L1 error        : %.12e\n", l1);
    std::printf("L2 error        : %.12e\n", l2);
    std::printf("Linf error      : %.12e\n", linf);
    std::printf("mean err shift  : %.12e\n", errMean);
    std::printf("L1 shifted      : %.12e\n", l1Shift);
    std::printf("L2 shifted      : %.12e\n", l2Shift);
    std::printf("Linf shifted    : %.12e\n", linfShift);
  }

  finalize_hypre_gpu_runtime();
  MPI_Finalize();
  return 0;
}
