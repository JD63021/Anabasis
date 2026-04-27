#include "poisson_library.h"
#include "mms.h"

struct Params {
  std::string polyMeshDir = "./mesh/constant/polyMesh";
  int device = 0;
  int nNonOrthCorr = 10;
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
  double amgStrongThreshold = 0.6;
  std::string gradScheme = "lsq";
  std::string laplacianScheme = "nonorth";
  std::string dirichletPatchesCsv = "";
  std::string neumannPatchesCsv = "";
  int useReferenceCell = 0;
  int referenceCell = 0;
  double referenceValue = 0.0;
};

static std::vector<std::string> split_csv(const std::string& s) {
  std::vector<std::string> out;
  std::stringstream ss(s);
  std::string item;
  while (std::getline(ss, item, ',')) {
    std::string t;
    for (char ch : item) if (!std::isspace(static_cast<unsigned char>(ch))) t.push_back(ch);
    if (!t.empty()) out.push_back(t);
  }
  return out;
}

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
    } else if (!std::strcmp(argv[i], "-dirichlet-patches")) {
      need(argv[i]); par.dirichletPatchesCsv = argv[++i];
    } else if (!std::strcmp(argv[i], "-neumann-patches")) {
      need(argv[i]); par.neumannPatchesCsv = argv[++i];
    } else if (!std::strcmp(argv[i], "-use-reference-cell")) {
      need(argv[i]); par.useReferenceCell = std::atoi(argv[++i]);
    } else if (!std::strcmp(argv[i], "-reference-cell")) {
      need(argv[i]); par.referenceCell = std::atoi(argv[++i]);
    } else if (!std::strcmp(argv[i], "-reference-value")) {
      need(argv[i]); par.referenceValue = std::atof(argv[++i]);
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
      std::fprintf(stderr, "This library-driven Poisson MMS app is intentionally single-rank/single-GPU. Run it with mpirun -n 1.\n");
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

  const auto dirichletList = split_csv(par.dirichletPatchesCsv);
  const auto neumannList = split_csv(par.neumannPatchesCsv);
  std::set<std::string> dirichletSet(dirichletList.begin(), dirichletList.end());
  std::set<std::string> neumannSet(neumannList.begin(), neumannList.end());

  ScalarBCSet bcSet;
  for (const auto& patchName : mesh.patchNames) {
    if (neumannSet.count(patchName)) {
      bcSet.patches.push_back(make_neumann_patch_bc(
          patchName,
          [](const std::array<double,3>& x, const std::array<double,3>& n) {
            return normal_grad_exact(x, n);
          }));
    } else {
      bcSet.patches.push_back(make_dirichlet_patch_bc(
          patchName,
          [](const std::array<double,3>& x, const std::array<double,3>&) {
            return phi_exact(x);
          }));
    }
  }

  std::vector<double> cellSource(mesh.nCells, 0.0);
  for (int c = 0; c < mesh.nCells; ++c) {
    cellSource[c] = rhs_exact(mesh.cc[c]);
  }

  EllipticOptions opts;
  opts.nNonOrthCorr = par.nNonOrthCorr;
  opts.gradScheme = par.gradScheme;
  opts.laplacianScheme = par.laplacianScheme;
  opts.useReferenceCell = (par.useReferenceCell != 0);
  opts.referenceCell = par.referenceCell;
  opts.referenceValue = par.referenceValue;
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
  const std::vector<double>& phi = result.phi;

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
    std::printf("last hypreIts   : %d\n", result.lastSolveInfo.iterations);
    std::printf("last relRes     : %.12e\n", result.lastSolveInfo.finalRelResNorm);
    std::printf("solution min/max: %.12e  %.12e\n", phiMin, phiMax);
    std::printf("L1 error        : %.12e\n", l1);
    std::printf("L2 error        : %.12e\n", l2);
    std::printf("Linf error      : %.12e\n", linf);
  }

  finalize_hypre_gpu_runtime();
  MPI_Finalize();
  return 0;
}
