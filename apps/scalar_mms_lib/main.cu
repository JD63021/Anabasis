#include "scalar_transport_library.h"
#include "hypre_backend.h"

#include <array>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <limits>
#include <set>
#include <sstream>
#include <string>
#include <vector>

static constexpr double PI = 3.141592653589793238462643383279502884;

struct Params {
  std::string polyMeshDir = "/tmp/unitcube_hex_50mm/constant/polyMesh";
  int device = 0;
  double gamma = 0.1;
  int maxIter = 4000;
  double tol = 1.0e-10;
  int monitor = 0;
  std::string convectionScheme = "central";
  std::string diffusionScheme = "orth";
  int nNonOrthCorr = 0;
  std::string transportMode = "convection_diffusion";   // convection_diffusion | diffusion_only

  std::string bcMode = "dirichlet";          // dirichlet | mixed
  std::string dirichletPatches = "";         // comma-separated, only used for mixed
};

static std::vector<std::string> split_csv(const std::string& s) {
  std::vector<std::string> out;
  std::stringstream ss(s);
  std::string item;
  while (std::getline(ss, item, ',')) {
    std::string trimmed;
    for (char ch : item) {
      if (ch != ' ' && ch != '\t' && ch != '\n' && ch != '\r') trimmed.push_back(ch);
    }
    if (!trimmed.empty()) out.push_back(trimmed);
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
    } else if (!std::strcmp(argv[i], "-gamma")) {
      need(argv[i]); par.gamma = std::atof(argv[++i]);
    } else if (!std::strcmp(argv[i], "-maxit")) {
      need(argv[i]); par.maxIter = std::atoi(argv[++i]);
    } else if (!std::strcmp(argv[i], "-tol")) {
      need(argv[i]); par.tol = std::atof(argv[++i]);
    } else if (!std::strcmp(argv[i], "-monitor")) {
      need(argv[i]); par.monitor = std::atoi(argv[++i]);
    } else if (!std::strcmp(argv[i], "-convection-scheme")) {
      need(argv[i]); par.convectionScheme = argv[++i];
    } else if (!std::strcmp(argv[i], "-diffusion-scheme")) {
      need(argv[i]); par.diffusionScheme = argv[++i];
    } else if (!std::strcmp(argv[i], "-nNonOrthCorr")) {
      need(argv[i]); par.nNonOrthCorr = std::atoi(argv[++i]);
    } else if (!std::strcmp(argv[i], "-transport-mode")) {
      need(argv[i]); par.transportMode = argv[++i];
    } else if (!std::strcmp(argv[i], "-bc-mode")) {
      need(argv[i]); par.bcMode = argv[++i];
    } else if (!std::strcmp(argv[i], "-dirichlet-patches")) {
      need(argv[i]); par.dirichletPatches = argv[++i];
    } else {
      std::fprintf(stderr, "Unknown option: %s\n", argv[i]);
      MPI_Abort(MPI_COMM_WORLD, 1);
    }
  }
}

static double phi_exact(const std::array<double,3>& x) {
  return std::sin(PI*x[0]) * std::sin(PI*x[1]) * std::sin(PI*x[2]);
}

static std::array<double,3> grad_phi_exact(const std::array<double,3>& x) {
  const double sx = std::sin(PI*x[0]), sy = std::sin(PI*x[1]), sz = std::sin(PI*x[2]);
  const double cx = std::cos(PI*x[0]), cy = std::cos(PI*x[1]), cz = std::cos(PI*x[2]);
  return {
    PI * cx * sy * sz,
    PI * sx * cy * sz,
    PI * sx * sy * cz
  };
}

static std::array<double,3> vel_exact(const std::array<double,3>& x) {
  return {
    1.0 + x[1],
    0.5 - x[0],
    0.25
  };
}

static double normal_grad_phi_exact(
    const std::array<double,3>& x,
    const std::array<double,3>& n) {
  const auto g = grad_phi_exact(x);
  return g[0]*n[0] + g[1]*n[1] + g[2]*n[2];
}

static double source_convection_diffusion(const std::array<double,3>& x, double gamma) {
  const auto u = vel_exact(x);
  const auto g = grad_phi_exact(x);
  const double conv = u[0]*g[0] + u[1]*g[1] + u[2]*g[2];
  const double diff = 3.0 * gamma * PI * PI * phi_exact(x);
  return conv + diff;
}

static double source_diffusion_only(const std::array<double,3>& x, double gamma) {
  return 3.0 * gamma * PI * PI * phi_exact(x);
}

int main(int argc, char** argv) {
  MPI_Init(&argc, &argv);
  int rank = 0, size = 1;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  if (size != 1) {
    if (rank == 0) std::fprintf(stderr, "Run with mpirun -n 1.\n");
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

  libscalar::ScalarTransportInputs in;
  in.faceFlux.assign(mesh.nFaces, 0.0);
  in.gammaFace.assign(mesh.nFaces, par.gamma);
  in.Su.assign(mesh.nCells, 0.0);
  in.Sp.assign(mesh.nCells, 0.0);

  const bool diffusionOnly = (par.transportMode == "diffusion_only");
  if (par.transportMode != "diffusion_only" &&
      par.transportMode != "convection_diffusion") {
    if (rank == 0) {
      std::fprintf(stderr,
          "Unsupported transportMode '%s'. Use convection_diffusion or diffusion_only.\n",
          par.transportMode.c_str());
    }
    MPI_Abort(MPI_COMM_WORLD, 1);
  }

  for (int f = 0; f < mesh.nFaces; ++f) {
    if (diffusionOnly) {
      in.faceFlux[f] = 0.0;
    } else {
      const auto u = vel_exact(mesh.xf[f]);
      in.faceFlux[f] = u[0]*mesh.Sf[f][0] + u[1]*mesh.Sf[f][1] + u[2]*mesh.Sf[f][2];
    }
  }

  for (int c = 0; c < mesh.nCells; ++c) {
    in.Su[c] = diffusionOnly
        ? source_diffusion_only(mesh.cc[c], par.gamma)
        : source_convection_diffusion(mesh.cc[c], par.gamma);
    in.Sp[c] = 0.0;
  }

  libscalar::ScalarBCSet bcSet;
  std::set<std::string> dirichletPatchSet;
  for (const auto& name : split_csv(par.dirichletPatches)) dirichletPatchSet.insert(name);

  if (par.bcMode != "dirichlet" && par.bcMode != "mixed") {
    if (rank == 0) {
      std::fprintf(stderr, "Unsupported bcMode '%s'. Use dirichlet or mixed.\n", par.bcMode.c_str());
    }
    MPI_Abort(MPI_COMM_WORLD, 1);
  }

  for (const auto& patchName : mesh.patchNames) {
    const bool useDirichlet =
        (par.bcMode == "dirichlet") ? true : (dirichletPatchSet.count(patchName) > 0);

    if (useDirichlet) {
      bcSet.patches.push_back(libscalar::make_dirichlet_patch_bc(
          patchName,
          [](const std::array<double,3>& x, const std::array<double,3>&) {
            return phi_exact(x);
          }));
    } else {
      bcSet.patches.push_back(libscalar::make_neumann_gradient_patch_bc(
          patchName,
          [gamma = par.gamma](const std::array<double,3>& x, const std::array<double,3>& n) {
            return normal_grad_phi_exact(x, n);
          }));
    }
  }

  libscalar::ScalarTransportOptions opt;
  opt.convectionScheme =
      (par.convectionScheme == "upwind" || par.convectionScheme == "upwind1")
      ? libscalar::ConvectionScheme::Upwind
      : libscalar::ConvectionScheme::Central;
  opt.diffusionScheme =
      (par.diffusionScheme == "nonorth")
      ? libscalar::DiffusionScheme::NonOrth
      : libscalar::DiffusionScheme::Orth;
  opt.nNonOrthCorr = par.nNonOrthCorr;
  opt.linearSolver = libscalar::LinearSolverType::BiCGSTAB;
  opt.preconditioner = libscalar::PreconditionerType::Jacobi;
  opt.maxIter = par.maxIter;
  opt.relTol = 0.0;
  opt.absTol = par.tol;
  opt.monitor = par.monitor;

  libscalar::ScalarTransportResult result =
      libscalar::solve_steady_scalar_transport(mesh, in, bcSet, opt);

  double volSum = 0.0, l1 = 0.0, l2 = 0.0, linf = 0.0;
  double phiMin = std::numeric_limits<double>::max();
  double phiMax = -std::numeric_limits<double>::max();

  for (int c = 0; c < mesh.nCells; ++c) {
    const double exact = phi_exact(mesh.cc[c]);
    const double err = std::fabs(result.phi[c] - exact);
    const double V = mesh.vol[c];
    volSum += V;
    l1 += V * err;
    l2 += V * err * err;
    linf = std::max(linf, err);
    phiMin = std::min(phiMin, result.phi[c]);
    phiMax = std::max(phiMax, result.phi[c]);
  }

  l1 /= std::max(volSum, 1.0e-30);
  l2 = std::sqrt(l2 / std::max(volSum, 1.0e-30));

  if (rank == 0) {
    std::printf("\n=== Scalar convection-diffusion MMS summary ===\n");
    std::printf("convectionScheme: %s\n", par.convectionScheme.c_str());
    std::printf("diffusionScheme : %s\n", par.diffusionScheme.c_str());
    std::printf("nNonOrthCorr    : %d\n", par.nNonOrthCorr);
    std::printf("gamma           : %.6e\n", par.gamma);
    std::printf("transportMode   : %s\n", par.transportMode.c_str());
    std::printf("bcMode          : %s\n", par.bcMode.c_str());
    if (par.bcMode == "mixed") {
      std::printf("dirichletPatches: %s\n", par.dirichletPatches.c_str());
    }
    std::printf("last solver its : %d\n", result.iterations);
    std::printf("last relRes     : %.12e\n", result.finalRelRes);
    std::printf("solution min/max: %.12e  %.12e\n", phiMin, phiMax);
    std::printf("L1 error        : %.12e\n", l1);
    std::printf("L2 error        : %.12e\n", l2);
    std::printf("Linf error      : %.12e\n", linf);
  }

  MPI_Finalize();
  return 0;
}
