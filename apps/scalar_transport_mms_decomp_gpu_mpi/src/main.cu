#include <mpi.h>
#include <cuda_runtime.h>

#include <array>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <set>
#include <stdexcept>
#include <string>
#include <vector>

#include "scalar_decomp_library.h"
#include "bc_runtime_config.h"

namespace {

constexpr double PI = 3.141592653589793238462643383279502884;

double phi_exact_scalar(const std::array<double,3>& x) {
  return std::sin(PI*x[0]) * std::sin(PI*x[1]) * std::sin(PI*x[2]);
}

std::array<double,3> grad_phi_exact_scalar(const std::array<double,3>& x) {
  return {
    PI * std::cos(PI*x[0]) * std::sin(PI*x[1]) * std::sin(PI*x[2]),
    PI * std::sin(PI*x[0]) * std::cos(PI*x[1]) * std::sin(PI*x[2]),
    PI * std::sin(PI*x[0]) * std::sin(PI*x[1]) * std::cos(PI*x[2])
  };
}

double source_scalar_mms(
    const std::array<double,3>& x,
    const std::array<double,3>& u,
    double gamma) {
  const double phi = phi_exact_scalar(x);
  const auto g = grad_phi_exact_scalar(x);

  const double adv = u[0]*g[0] + u[1]*g[1] + u[2]*g[2];
  const double diff = 3.0 * gamma * PI * PI * phi;

  return adv + diff;
}

ScalarBCSet make_exact_mms_dirichlet_bcs_for_physical_patches(const DecompMesh& dm) {
  std::set<std::string> procPatchNames;
  for (const auto& pp : dm.procPatches) procPatchNames.insert(pp.name);

  ScalarBCSet bc;
  for (const auto& name : dm.mesh.patchNames) {
    if (procPatchNames.count(name)) continue;

    bc.patches.push_back(make_dirichlet_patch_bc(
        name,
        [](const std::array<double,3>& x, const std::array<double,3>&) {
          return phi_exact_scalar(x);
        }));
  }

  return bc;
}

ScalarBCSet make_scalar_bcs_from_pressure_config(
    const DecompMesh& dm,
    const std::string& bcConfigPath) {
  auto cfg = pipebc::load_runtime_bc_config(bcConfigPath);
  pipebc::validate_runtime_bc_config_against_patches(cfg, dm.mesh.patchNames);

  const auto dups = pipebc::duplicate_pressure_bc_patches(cfg.pressurePatchSpecs);
  if (!dups.empty()) {
    throw std::runtime_error("Duplicate pressure/scalar BC for patch '" + dups.front() + "'");
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
      throw std::runtime_error("Unsupported pressure BC type in scalar MMS config");
    }
  }

  return bc;
}

void cuda_check(cudaError_t e, const char* what) {
  if (e != cudaSuccess) {
    throw std::runtime_error(std::string(what) + ": " + cudaGetErrorString(e));
  }
}

} // namespace

int main(int argc, char** argv) {
  MPI_Init(&argc, &argv);

  int rank = 0, size = 1;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  try {
    std::string caseRoot = "/tmp/case_cube_0p03mm";
    std::string bcConfigPath;

    int device = rank;
    double tol = 1e-8;
    int maxit = 1000;

    std::string gradScheme = "lsq";
    std::string diffusionScheme = "nonorth";
    std::string convectionScheme = "upwind";
    int nNonOrthCorr = 4;

    double gamma = 0.01;
    std::array<double,3> u = {1.0, 0.0, 0.0};

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
      } else if (a == "-tol" || a == "-absTol" || a == "-scalar-tol") {
        need(a.c_str());
        tol = std::atof(argv[++i]);
      } else if (a == "-maxit" || a == "-scalar-maxit") {
        need(a.c_str());
        maxit = std::atoi(argv[++i]);
      } else if (a == "-grad-scheme") {
        need("-grad-scheme");
        gradScheme = argv[++i];
      } else if (a == "-diffusion-scheme" || a == "-laplacian-scheme") {
        need(a.c_str());
        diffusionScheme = argv[++i];
      } else if (a == "-convection-scheme") {
        need("-convection-scheme");
        convectionScheme = argv[++i];
      } else if (a == "-nNonOrthCorr") {
        need("-nNonOrthCorr");
        nNonOrthCorr = std::atoi(argv[++i]);
      } else if (a == "-gamma") {
        need("-gamma");
        gamma = std::atof(argv[++i]);
      } else if (a == "-ux") {
        need("-ux");
        u[0] = std::atof(argv[++i]);
      } else if (a == "-uy") {
        need("-uy");
        u[1] = std::atof(argv[++i]);
      } else if (a == "-uz") {
        need("-uz");
        u[2] = std::atof(argv[++i]);
      }
    }

    int devCount = 0;
    cuda_check(cudaGetDeviceCount(&devCount), "cudaGetDeviceCount");
    if (devCount > 0) cuda_check(cudaSetDevice(device % devCount), "cudaSetDevice");

    DecompMesh dm = read_decomposed_openfoam_case(caseRoot, MPI_COMM_WORLD);

    ScalarBCSet bcSet;
    if (!bcConfigPath.empty()) {
      bcSet = make_scalar_bcs_from_pressure_config(dm, bcConfigPath);
    } else {
      bcSet = make_exact_mms_dirichlet_bcs_for_physical_patches(dm);
    }

    libscalar_decomp::DistScalarTransportInputs in;
    in.faceFlux.assign(dm.mesh.nFaces, 0.0);
    in.gammaFace.assign(dm.mesh.nFaces, gamma);
    in.Su.assign(dm.mesh.nCells, 0.0);
    in.Sp.assign(dm.mesh.nCells, 0.0);

    for (int f = 0; f < dm.mesh.nFaces; ++f) {
      in.faceFlux[f] =
          u[0] * dm.mesh.Sf[f][0] +
          u[1] * dm.mesh.Sf[f][1] +
          u[2] * dm.mesh.Sf[f][2];
    }

    for (int c = 0; c < dm.mesh.nCells; ++c) {
      in.Su[c] = source_scalar_mms(dm.mesh.cc[c], u, gamma);
    }

    libscalar_decomp::DistScalarTransportOptions opt;
    opt.convectionScheme = libscalar_decomp::convection_scheme_from_string(convectionScheme);
    opt.diffusionScheme = libscalar_decomp::diffusion_scheme_from_string(diffusionScheme);
    opt.gradScheme = gradScheme;
    opt.nNonOrthCorr = nNonOrthCorr;
    opt.solver.maxIter = maxit;
    opt.solver.absTol = tol;
    opt.solver.relTol = 0.0;
    opt.solver.monitor = 0;

    MPI_Barrier(MPI_COMM_WORLD);
    const double t0 = MPI_Wtime();

    auto result = libscalar_decomp::solve_scalar_transport_decomp(dm, in, bcSet, opt);

    MPI_Barrier(MPI_COMM_WORLD);
    const double t1 = MPI_Wtime();

    double localL2 = 0.0;
    double localInf = 0.0;
    double localVol = 0.0;

    for (int c = 0; c < dm.mesh.nCells; ++c) {
      const double exact = phi_exact_scalar(dm.mesh.cc[c]);
      const double err = std::abs(result.phi[c] - exact);
      localL2 += err * err * dm.mesh.vol[c];
      localInf = std::max(localInf, err);
      localVol += dm.mesh.vol[c];
    }

    double globalL2 = 0.0;
    double globalInf = 0.0;
    double globalVol = 0.0;

    MPI_Allreduce(&localL2, &globalL2, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    MPI_Allreduce(&localInf, &globalInf, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
    MPI_Allreduce(&localVol, &globalVol, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

    globalL2 = std::sqrt(globalL2 / std::max(globalVol, 1.0e-300));

    double wall = t1 - t0;
    double maxWall = 0.0;
    MPI_Reduce(&wall, &maxWall, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    DistCSRPattern pat = build_decomp_scalar_pattern(dm);

    std::printf("rank %d/%d rows=[%lld,%lld] nLocal=%d nnz=%d internalFaces=%d procFaces=%zu boundaryFaces=%d\n",
                rank, size,
                (long long)dm.ilower, (long long)dm.iupper,
                dm.nLocal,
                pat.nnz,
                dm.mesh.nInternalFaces,
                pat.procFace.size(),
                dm.mesh.nFaces - dm.mesh.nInternalFaces - (int)pat.procFace.size());
    std::fflush(stdout);

    if (rank == 0) {
      std::printf("SCALAR_DECOMP_MMS setup: ranks=%d globalRows=%lld globalNnz=%lld bcConfig=%s grad=%s diffusion=%s convection=%s nNonOrthCorr=%d nOuter=%d gamma=%.6e U=(%.6e %.6e %.6e)\n",
                  size,
                  (long long)dm.globalN,
                  result.globalNnz,
                  bcConfigPath.empty() ? "<exact-mms-default>" : bcConfigPath.c_str(),
                  gradScheme.c_str(),
                  diffusionScheme.c_str(),
                  convectionScheme.c_str(),
                  nNonOrthCorr,
                  result.nOuter,
                  gamma,
                  u[0], u[1], u[2]);

      std::printf("SCALAR_DECOMP_MMS RESULT: its=%d finalRel=%.12e L2=%.12e Linf=%.12e wall=%.6e s\n",
                  result.iterations,
                  result.finalRelRes,
                  globalL2,
                  globalInf,
                  maxWall);

      if (std::isfinite(globalL2) && std::isfinite(globalInf)) {
        std::printf("SCALAR_DECOMP_MMS PASS_RAN\n");
      } else {
        std::printf("SCALAR_DECOMP_MMS FAIL_NAN_INF\n");
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
