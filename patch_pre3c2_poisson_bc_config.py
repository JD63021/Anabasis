#!/usr/bin/env python3
from pathlib import Path

root = Path.cwd()
main = root / "apps/pre3_poisson_dist_geom_gpu_mpi/src/main.cu"
cmake = root / "apps/pre3_poisson_dist_geom_gpu_mpi/CMakeLists.txt"

if not main.exists():
    raise SystemExit(f"ERROR: run this from repo root; missing {main}")
if not cmake.exists():
    raise SystemExit(f"ERROR: missing {cmake}")

s = main.read_text()
orig = s

# 1) include serial modular BC parser
s = s.replace(
    '#include "hypre_backend.h"\n',
    '#include "hypre_backend.h"\n\n#include "bc_runtime_config.h"\n'
)

# 2) replace hardcoded exact-MMS boundary kernel with configurable boundary kernel
old = '''__global__ static void k_boundary_dirichlet_geom_poisson(
    int nB,
    const int *bFace,
    const int *bOwner,
    const int *bDiag,
    const double *ccx,
    const double *ccy,
    const double *ccz,
    const double *xfx,
    const double *xfy,
    const double *xfz,
    const double *Sfx,
    const double *Sfy,
    const double *Sfz,
    HYPRE_Complex *vals,
    HYPRE_Complex *rhs)
{
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if(i >= nB) return;

  const int f = bFace[i];
  const int P = bOwner[i];

  const double dx = xfx[f] - ccx[P];
  const double dy = xfy[f] - ccy[P];
  const double dz = xfz[f] - ccz[P];

  const double sx = Sfx[f];
  const double sy = Sfy[f];
  const double sz = Sfz[f];

  const double ss = sx*sx + sy*sy + sz*sz;
  double dDotS = dx*sx + dy*sy + dz*sz;
  if(dDotS < 1e-300) dDotS = 1e-300;

  const double D = ss / dDotS;
  const double phiB = d_phi_exact_xyz(xfx[f], xfy[f], xfz[f]);

  atomicAdd(&vals[bDiag[i]], (HYPRE_Complex)D);
  atomicAdd(&rhs[P], (HYPRE_Complex)(D * phiB));
}
'''
new = '''// Boundary BC mode for scalar Poisson MMS:
//   0 = exact MMS Dirichlet, original PRE3C2 behavior.
//   1 = constant Dirichlet from modular BC config:
//       pressure <patch> fixed_value <value>
//   2 = homogeneous zeroGradient/open from modular BC config:
//       no matrix/RHS contribution for boundary Laplacian face.
__global__ static void k_boundary_configurable_geom_poisson(
    int nB,
    const int *bFace,
    const int *bOwner,
    const int *bDiag,
    const int *bMode,
    const double *bValue,
    const double *ccx,
    const double *ccy,
    const double *ccz,
    const double *xfx,
    const double *xfy,
    const double *xfz,
    const double *Sfx,
    const double *Sfy,
    const double *Sfz,
    HYPRE_Complex *vals,
    HYPRE_Complex *rhs)
{
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if(i >= nB) return;

  const int mode = bMode[i];

  if(mode == 2) return;

  const int f = bFace[i];
  const int P = bOwner[i];

  const double dx = xfx[f] - ccx[P];
  const double dy = xfy[f] - ccy[P];
  const double dz = xfz[f] - ccz[P];

  const double sx = Sfx[f];
  const double sy = Sfy[f];
  const double sz = Sfz[f];

  const double ss = sx*sx + sy*sy + sz*sz;
  double dDotS = dx*sx + dy*sy + dz*sz;
  if(dDotS < 1e-300) dDotS = 1e-300;

  const double D = ss / dDotS;
  const double phiB = (mode == 1) ? bValue[i]
                                  : d_phi_exact_xyz(xfx[f], xfy[f], xfz[f]);

  atomicAdd(&vals[bDiag[i]], (HYPRE_Complex)D);
  atomicAdd(&rhs[P], (HYPRE_Complex)(D * phiB));
}
'''
if old not in s:
    raise SystemExit("ERROR: could not find old boundary kernel block")
s = s.replace(old, new)

# 3) add bc config CLI variable
s = s.replace(
    '    double tol = 1e-7;\n    int device = rank;\n',
    '    double tol = 1e-7;\n    int device = rank;\n    std::string bcConfigPath;\n'
)

# 4) parse -bc-config / -case-config
s = s.replace(
    '''      } else if(a == "-device") {
        need("-device");
        device = std::atoi(argv[++i]);
      }
''',
    '''      } else if(a == "-device") {
        need("-device");
        device = std::atoi(argv[++i]);
      } else if(a == "-bc-config" || a == "-case-config") {
        need(a.c_str());
        bcConfigPath = argv[++i];
      }
'''
)

# 5) load BC config after mesh read
marker = '''    Mesh mesh = read_openfoam_polymesh(polyMeshDir);
    const auto procPatches = read_processor_patches(polyMeshDir + "/boundary");

    const int nLocal = mesh.nCells;
'''
insert = '''    Mesh mesh = read_openfoam_polymesh(polyMeshDir);
    const auto procPatches = read_processor_patches(polyMeshDir + "/boundary");

    const bool useBCConfig = !bcConfigPath.empty();
    pipebc::RuntimeBCConfig bcConfig;
    std::map<std::string, pipebc::PressurePatchBCSpec> pressureBCByPatch;

    if(useBCConfig) {
      bcConfig = pipebc::load_runtime_bc_config(bcConfigPath);
      pipebc::validate_runtime_bc_config_against_patches(bcConfig, mesh.patchNames);

      const auto dups = pipebc::duplicate_pressure_bc_patches(bcConfig.pressurePatchSpecs);
      if(!dups.empty()) {
        throw std::runtime_error("Duplicate pressure BC for patch '" + dups.front() + "' in " + bcConfigPath);
      }

      for(const auto &spec : bcConfig.pressurePatchSpecs) {
        pressureBCByPatch[spec.patchName] = spec;
      }
    }

    if(rank == 0) {
      if(useBCConfig) {
        std::printf("PRE3C2 BC config mode: file=%s ; using pressure patch specs as scalar Poisson BCs\\n",
                    bcConfigPath.c_str());
        std::printf("PRE3C2 BC config note: for cube MMS, use: pressure patch_0_0 fixed_value 0.0\\n");
      } else {
        std::printf("PRE3C2 BC config mode: default exact MMS Dirichlet on all physical boundary faces\\n");
      }
      std::fflush(stdout);
    }

    const int nLocal = mesh.nCells;
'''
if marker not in s:
    raise SystemExit("ERROR: could not find mesh read marker")
s = s.replace(marker, insert)

# 6) add boundary mode/value arrays
s = s.replace(
    '    std::vector<int> bFace, bOwner, bDiag;\n    std::vector<int> pFace, pOwner, pDiag, pOff;\n',
    '    std::vector<int> bFace, bOwner, bDiag, bMode;\n    std::vector<double> bValue;\n    std::vector<int> pFace, pOwner, pDiag, pOff;\n'
)

# 7) fill boundary modes from pressure BC config
old = '''      } else {
        bFace.push_back(f);
        bOwner.push_back(P);
        bDiag.push_back(pos[P][local_row(P)]);
      }
'''
new = '''      } else {
        int mode = 0;
        double value = 0.0;

        if(useBCConfig) {
          const int pidx = mesh.bPatch[f] - 1;
          if(pidx < 0 || pidx >= (int)mesh.patchNames.size()) {
            throw std::runtime_error("Boundary face has invalid patch index");
          }

          const std::string &patchName = mesh.patchNames[pidx];
          auto it = pressureBCByPatch.find(patchName);
          if(it == pressureBCByPatch.end()) {
            throw std::runtime_error(
              "No pressure/scalar Poisson BC supplied for physical patch '" + patchName +
              "' in " + bcConfigPath);
          }

          const auto &spec = it->second;
          if(spec.type == pipebc::PressureBCType::FixedValue) {
            mode = 1;
            value = spec.value;
          } else if(spec.type == pipebc::PressureBCType::ZeroGradient ||
                    spec.type == pipebc::PressureBCType::Open) {
            mode = 2;
            value = 0.0;
          } else {
            throw std::runtime_error(
              "Poisson BC config supports pressure fixed_value, zero_gradient, or open only for patch '" +
              patchName + "'");
          }
        }

        bFace.push_back(f);
        bOwner.push_back(P);
        bDiag.push_back(pos[P][local_row(P)]);
        bMode.push_back(mode);
        bValue.push_back(value);
      }
'''
if old not in s:
    raise SystemExit("ERROR: could not find bFace fill block")
s = s.replace(old, new)

# 8) print physical BC face counts
marker = '''    if(rank == 0) {
      long long globalNnz = 0;
      long long localNnz = pat.nnz;
'''
insert = '''    int localExactDir = 0, localConstDir = 0, localZeroGrad = 0;
    for(int m : bMode) {
      if(m == 0) ++localExactDir;
      else if(m == 1) ++localConstDir;
      else if(m == 2) ++localZeroGrad;
    }

    int globalExactDir = 0, globalConstDir = 0, globalZeroGrad = 0;
    MPI_Allreduce(&localExactDir, &globalExactDir, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
    MPI_Allreduce(&localConstDir, &globalConstDir, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
    MPI_Allreduce(&localZeroGrad, &globalZeroGrad, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);

    if(rank == 0) {
      std::printf("PRE3C2 physical BC face counts: exactDirichlet=%d configFixedValue=%d configZeroGradientOpen=%d\\n",
                  globalExactDir, globalConstDir, globalZeroGrad);
      std::fflush(stdout);
    }

    if(rank == 0) {
      long long globalNnz = 0;
      long long localNnz = pat.nnz;
'''
if marker not in s:
    raise SystemExit("ERROR: could not find setup print marker")
s = s.replace(marker, insert)

# 9) copy bMode/bValue to device
s = s.replace(
    '''    int *d_bFace = copy_vec_to_device(bFace);
    int *d_bOwner = copy_vec_to_device(bOwner);
    int *d_bDiag = copy_vec_to_device(bDiag);

    int *d_pFace = copy_vec_to_device(pFace);
''',
    '''    int *d_bFace = copy_vec_to_device(bFace);
    int *d_bOwner = copy_vec_to_device(bOwner);
    int *d_bDiag = copy_vec_to_device(bDiag);
    int *d_bMode = copy_vec_to_device(bMode);
    double *d_bValue = copy_vec_to_device(bValue);

    int *d_pFace = copy_vec_to_device(pFace);
'''
)

# 10) replace boundary kernel call
old = '''    k_boundary_dirichlet_geom_poisson<<<((int)bFace.size() + block - 1)/block, block>>>(
      (int)bFace.size(),
      d_bFace, d_bOwner, d_bDiag,
      d_ccx, d_ccy, d_ccz,
      d_xfx, d_xfy, d_xfz,
      d_Sfx, d_Sfy, d_Sfz,
      d_values, d_rhs);
'''
new = '''    k_boundary_configurable_geom_poisson<<<((int)bFace.size() + block - 1)/block, block>>>(
      (int)bFace.size(),
      d_bFace, d_bOwner, d_bDiag,
      d_bMode, d_bValue,
      d_ccx, d_ccy, d_ccz,
      d_xfx, d_xfy, d_xfz,
      d_Sfx, d_Sfy, d_Sfz,
      d_values, d_rhs);
'''
if old not in s:
    raise SystemExit("ERROR: could not find old boundary kernel call")
s = s.replace(old, new)

if s == orig:
    raise SystemExit("ERROR: no changes made to main.cu")

main.write_text(s)

# 11) patch app CMakeLists
cs = cmake.read_text()
origc = cs

cs = cs.replace(
    '''add_executable(pre3_poisson_dist_geom_gpu_mpi
  src/main.cu
)
''',
    '''add_executable(pre3_poisson_dist_geom_gpu_mpi
  src/main.cu
  ${CMAKE_SOURCE_DIR}/apps/simple_gpu/src/bc_specs.cu
  ${CMAKE_SOURCE_DIR}/apps/simple_gpu/src/bc_runtime_config.cu
)
'''
)

cs = cs.replace(
    '''target_include_directories(pre3_poisson_dist_geom_gpu_mpi PRIVATE
  ${CMAKE_SOURCE_DIR}/libpoisson/include
)
''',
    '''target_include_directories(pre3_poisson_dist_geom_gpu_mpi PRIVATE
  ${CMAKE_SOURCE_DIR}/libpoisson/include
  ${CMAKE_SOURCE_DIR}/apps/simple_gpu/src
)
'''
)

if cs == origc:
    raise SystemExit("ERROR: no changes made to app CMakeLists.txt")

cmake.write_text(cs)

print("OK: patched pre3_poisson_dist_geom_gpu_mpi with -bc-config / -case-config support.")
print("Default no-bc-config behavior remains exact MMS Dirichlet.")
