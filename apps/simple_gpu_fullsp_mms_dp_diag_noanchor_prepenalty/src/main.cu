#include <mpi.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <string>
#include <vector>
#include <array>
#include <map>
#include <set>
#include <fstream>
#include <sstream>
#include <iostream>
#include <iomanip>
#include <algorithm>
#include <stdexcept>
#include <cfloat>
#include <cctype>
#include <cuda_runtime.h>
#include "patch_geometry.h"
#include "bc_specs.h"
#include "velocity_bc_eval.h"
#include "bc_runtime_config.h"
#include "mesh.h"
#include "scalar_transport_library.h"
extern "C" {
#include "HYPRE.h"
#include "HYPRE_IJ_mv.h"
#include "HYPRE_parcsr_ls.h"
#include "HYPRE_krylov.h"
#include "HYPRE_utilities.h"
#include "_hypre_parcsr_mv.h"

// Memory diagnostic rank used by setup/helper functions.
// This is diagnostic-only and does not affect solver numerics.
static int g_memdiag_rank = 0;
static void gpu_mem_mark(const char *tag, int rank, bool sync);
static void gpu_mem_mark0(const char *tag)
{
  gpu_mem_mark(tag, g_memdiag_rank, true);
}
}

#ifdef ANABASIS_EXPECT_HYPRE_COMPLEX_BYTES
static_assert(sizeof(HYPRE_Complex) == ANABASIS_EXPECT_HYPRE_COMPLEX_BYTES,
              "HYPRE_Complex precision does not match this Anabasis build script");
#endif

#ifdef ANABASIS_FULL_SP_APP
using cfdScalar = HYPRE_Real;
static_assert(sizeof(cfdScalar) == 4,
              "simple_gpu_fullsp must be built against single-precision HYPRE");
#else
using cfdScalar = double;
#endif
using cfdVector3 = std::array<cfdScalar,3>;

static inline cfdScalar to_cfd_scalar(double x) { return static_cast<cfdScalar>(x); }
static inline std::vector<cfdScalar> to_cfd_scalar_vec(const std::vector<double>& x) {
  std::vector<cfdScalar> y(x.size());
  for(std::size_t i=0;i<x.size();++i) y[i] = static_cast<cfdScalar>(x[i]);
  return y;
}
static inline std::vector<double> to_double_vec(const std::vector<cfdScalar>& x) {
  std::vector<double> y(x.size());
  for(std::size_t i=0;i<x.size();++i) y[i] = static_cast<double>(x[i]);
  return y;
}

#define CUDA_CALL(stmt) do { \
  cudaError_t _err = (stmt); \
  if (_err != cudaSuccess) { \
    std::fprintf(stderr, "CUDA ERROR at %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(_err)); \
    MPI_Abort(MPI_COMM_WORLD, -1); \
  } \
} while (0)

#define CUDA_CHECK_LAST() do { \
  cudaError_t _err = cudaGetLastError(); \
  if (_err != cudaSuccess) { \
    std::fprintf(stderr, "CUDA KERNEL ERROR at %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(_err)); \
    MPI_Abort(MPI_COMM_WORLD, -1); \
  } \
} while (0)

#define HYPRE_CALL(stmt) do { \
  HYPRE_Int _ierr = (stmt); \
  if (_ierr) { \
    int _rank = 0; MPI_Comm_rank(MPI_COMM_WORLD, &_rank); \
    std::fprintf(stderr, "[%d] HYPRE ERROR %s:%d code=%d\n", _rank, __FILE__, __LINE__, (int)_ierr); \
    MPI_Abort(MPI_COMM_WORLD, (int)_ierr); \
  } \
} while (0)

__device__ __forceinline__ void hypreAtomicAdd(HYPRE_Complex *addr, HYPRE_Complex val){
  // HYPRE_Complex is double in the normal build and float in the single-HYPRE build.
  // Use the native CUDA atomicAdd overload so both precision variants are valid.
  atomicAdd(addr, val);
}

struct Params {
  std::string polyMeshDir="/tmp/meshCase/constant/polyMesh";
  std::string outPrefix="pipe_poiseuille_gpu";
  double rho=1.0, mu=0.05, Re=1.0, Umean=1.0, CFL=1.0;
  bool muExplicit=false;
  double pipeDiameter=0.05, pipeLength=0.50;
  std::string wallPatchName="patch_0_0", inletPatchName="patch_1_0", outletPatchName="patch_2_0";
  std::string bcConfigPath="";
  int device=0, velRestart=40, velMaxit=100, monitor=1, write_vtu=1;
  int nVelNonOrthCorr=0, nNonOrthCorr=2, nPressureCorr=0, nsteps=1000, printEvery=25, writeEvery=0;
  double velTol=1e-10, velRelTol=0.0, corrTol=1e-12, tolMass=1e-10, tolVel=1e-10;
  double uRelax=0.7, pRelax=1.0;
  int p_use_amg=1, pMaxit=4000, pAmgMaxit=1, pAmgRelaxType=18, pAmgCoarsenType=8, pAmgInterpType=3, pAmgAggLevels=1, pAmgAggInterpType=4, pAmgPmax=4, pAmgKeepTranspose=0;
  double pTol=1e-10, pRelTol=0.0, pAmgTruncFactor=0.2;
  int profileSteps=0;
  int pAmgRebuildEvery=1; // rebuild AMG hierarchy on outer iter 1 and then every N outer iterations
  int pAmgSetupScope=0;    // 0 = setup once per outer iteration, 1 = setup before every pressure solve

  // Momentum linear solver selector:
  //   0 = HYPRE BiCGSTAB
  //   2 = GPU multi-color Gauss-Seidel defect smoother
  //   3 = GPU fused multi-color Gauss-Seidel defect smoother for U/V/W
  int velSolver=0;
  int velSweeps=2;
  double velSmootherOmega=0.8;
  int velGsSymmetric=0;      // 0=forward color pass, 1=forward+backward color pass
  int velCorrectionSolve=0;  // optional BiCGSTAB defect solve: A*dq = b - A*qOld

  // simple_gpu pressure-velocity coupling controls.
  // Defaults preserve the v1-style correction solve, while command-line options
  // can select the more OpenFOAM-like robust branch found useful on bad poly/tet meshes.
  double momNonOrthScale=1.0;
  int rcMode=0;              // 0=old explicit RC term, 1=OpenFOAM-like no explicit RC in phi predictor
  int rAUMode=1;             // 0=raw V/aP_raw, 1=relaxed V/aP_relaxed
  double rAUScale=1.0;       // diagnostic multiplier applied consistently to rAU
  double pNonOrthScale=1.0;  // 0 corresponds to uncorrected pressure laplacian/flux
  int pMode=0;               // 0=pressure correction, 1=absolute pressure/HbyA
  int pSolveMode=0;          // 0=current correction-compatible path, 1=OF absolute p/HbyA assignment path
  int pGradScheme=0;         // 0=LSQ gradient, 1=Gauss-linear gradient for final velocity correction
  double pCoeffScale=1.0;
  int hbyaBcMode=1;          // constrain fixed-velocity boundary fluxes in HbyA/phiHbyA
  int pFluxMode=0;           // retained for compatibility; flux is matrix-consistent
  int pDeltaMode=1;           // 0=v1-signed 1/(n.d), 1=OF-stabilised 1/max(|n.d|, minCos*|d|), 2=abs-projected 1/|n.d|, 3=distance 1/|d|
  double pDeltaMinCos=0.05;   // floor for pDeltaMode=1
  int geomMethod=1;          // 0=legacy geometry, 1=robust triangulated geometry
  int lsqStencilMode=0;      // 0=compact; extended is accepted but not yet implemented in this app
  double lsqWeightPower=2.0; // compact LSQ weight 1/|d|^p

  int momentumConvectionScheme=0; // 0=central/linear, 1=first-order upwind for div(phi,U) momentum convection

  // Optional geometry-consistent SIMPLE flux reconstruction.
  // Defaults stay OFF to preserve the memory-light production path.
  // They can also be overridden by environment variables:
  //   ANABASIS_SIMPLE_FACE_SKEW_CORR=1
  //   ANABASIS_SIMPLE_SUBTRI_CONTINUITY=1
  //   ANABASIS_SIMPLE_SUBTRI_FLUX_RECON=1
  int simpleFaceSkewCorr = 0;
  int simpleSubtriContinuity = 0;
  int simpleSubtriFluxRecon = 0;

  // Working polycube SIMPLE MMS route from simple_gpu_subtri.
  // This is the pressure-Dirichlet MMS, not the invalid arbitrary-polycube
  // sine/cosine all-Neumann MMS.
  int simpleMmsEnable = 0;
  int simpleMmsInitExact = 1;
  int simpleMmsPressureDirichlet = 1;
  double simpleMmsVelocityScale = 1.0;
  double simpleMmsPressureScale = 1.0;
  int simpleMmsFixedVelocityBoundaryPressureEqn = 1;
  int simpleMmsSourceQuad8 = 0;
  int simpleMmsStokesOnly = 0; // MMS-only: remove convective term from manufactured source

  // Diagnostic-only MMS clamps/probes.
  int simpleMmsDiagStages = 0;
  int simpleMmsForceExactPBeforeMomentum = 0;
  int simpleMmsForceExactUBeforePressure = 0;
  int simpleMmsForceExactPAfterUpdate = 0;
  int simpleMmsDiagPressureExactResidual = 0;
  int simpleMmsDiagWritePErrorVtk = 0;
  double simpleMmsPressurePenaltyCoeff = 0.0; // diagnostic: distributed pressure penalty assembled before AMG setup

  // Generic patch force postprocess.
  // C = 2F / (rho * Uref^2 * Aref)
  int forceEnable = 0;
  std::string forcePatchName = "";
  int forceNormalSign = -1; // use -1 for internal obstacle patches when mesh.nf points fluid -> solid

  std::array<double,3> forceDragDir{{1.0, 0.0, 0.0}};
  std::array<double,3> forceLiftDir{{0.0, 1.0, 0.0}};
  std::array<double,3> forceSpanDir{{0.0, 0.0, 1.0}};

  double forceUref = -1.0;
  double forceAreaRef = -1.0;

  // Optional OpenFOAM-like potential/projection initialisation.
  // 0 = current uniform-inlet-average initial field; 1 = solve one Laplace/projection
  // from runtime boundary fluxes and initialise U = -rAU*grad(potential).
  int initMode = 0;
  double potentialInitRAU = 1.0;
  int potentialInitMaxit = 4000;
  double potentialInitTol = 1e-10;
  double potentialInitRelTol = 0.0;
  int potentialInitWrite = 0;

  // v2.1 diagnostics for bad-cell / bad-face correction feedback.
  // Off by default. Set badCellAuditEvery > 0 for periodic dumps,
  // or badCellAuditOnGrowth=1 to dump when mass residual climbs.
  int badCellAuditEvery = 0;
  int badCellAuditTop = 12;
  int badCellAuditStart = 1;
  int badCellAuditOnGrowth = 0;
  double badCellAuditGrowthFactor = 1.20;
  double badCellAuditMassFloor = 0.0;
  int badCellAuditWriteCsv = 1;

  // v1 PDE-oriented case-file options. These are parsed and printed here so
  // future Poisson/scalar physics can share one case file without disturbing
  // the working SIMPLE path. Poisson gradient schemes are implemented in
  // libpoisson; scalar convection scheme strings map to libscalar enums.
  std::string poissonGradientScheme = "lsq";   // lsq or gauss
  std::string poissonLaplacianScheme = "nonorth";
  int poissonNonOrthCorr = 2;

  int scalarEnable = 0;
  std::string scalarSolveMode = "afterFlow";
  std::string scalarName = "scalar";
  std::string scalarBCConfigPath = "";
  std::string scalarConvectionScheme = "central"; // central or upwind
  std::string scalarDiffusionScheme = "nonorth";
  double scalarGamma = 1.0;
  double scalarRelax = 1.0;
  int scalarNonOrthCorr = 2;
  int scalarMaxit = 4000;
  double scalarTol = 1.0e-10;
  double scalarRelTol = 0.0;

  // Sine-wave manufactured solution for steady SIMPLE verification on unit cube.
  // Exact field:
  //   psi = sin^2(pi x) sin^2(pi y) sin^2(pi z)
  //   u = dpsi/dy, v = -dpsi/dx, w = 0
  //   p = cos(2pi x) cos(2pi y) cos(2pi z)
  // This gives U=0 and dp/dn=0 on all unit-cube boundaries.
  int sineMmsEnable = 0;
  int sineMmsInitExact = 1;
  double sineMmsVelocityScale = 0.05;
  double sineMmsPressureScale = 1.0;
};

// add3/sub3/mul3/dot3/cross3/norm3 are provided by libpoisson/common.h via mesh.h.

// OpenFOAM-like stabilized normal delta coefficient used by nonOrthDeltaCoeffs:
//   1/max(nHat.d, 0.05*|d|)
// This avoids huge pressure/diffusion coefficients on nearly 90-degree faces.
#define OF_STAB_DELTA_LIMIT 0.05
static __host__ __device__ __forceinline__ double of_delta_coeff_stabilised(
    double dx, double dy, double dz,
    double nx, double ny, double nz)
{
  const double magD = sqrt(dx*dx + dy*dy + dz*dz);
  const double nd   = nx*dx + ny*dy + nz*dz;
  const double dEff = fmax(nd, OF_STAB_DELTA_LIMIT*magD);
  return 1.0 / fmax(dEff, 1.0e-300);
}
static inline double of_delta_coeff_stabilised(const std::array<double,3>& n,
                                               const std::array<double,3>& d)
{
  return of_delta_coeff_stabilised(d[0], d[1], d[2], n[0], n[1], n[2]);
}


// Runtime pressure delta coefficient selector.
// pDeltaMode:
//   0 = legacy/v1 signed-projected: delta = 1/(n.d), guarded as in legacy SIMPLE
//   1 = OpenFOAM-like stabilised:  delta = 1/max(|n.d|, pDeltaMinCos*|d|)
//   2 = abs projected:            delta = 1/|n.d|
//   3 = distance:                 delta = 1/|d|  (this was the first v2 "legacy" implementation)
// Default mode 1 preserves current simple_gpu robust behavior.
__device__ __managed__ int g_pDeltaMode = 1;
__device__ __managed__ double g_pDeltaMinCos = 0.05;

static __host__ __device__ __forceinline__ double pressure_delta_coeff_runtime(
    double dx, double dy, double dz,
    double nfx, double nfy, double nfz)
{
  const double dmag = sqrt(dx*dx + dy*dy + dz*dz);
  if (dmag <= 1.0e-300) return 0.0;

  const double dpn = nfx*dx + nfy*dy + nfz*dz;

  // Exact v1 pressure coefficient: Af*rAU/(n.d), with the same small/negative
  // projected-distance guard used by legacy SIMPLE.
  if (g_pDeltaMode == 0) {
    return (dpn > 1.0e-14) ? (1.0 / dpn) : 0.0;
  }

  if (g_pDeltaMode == 2) {
    return 1.0 / fmax(fabs(dpn), 1.0e-300);
  }

  if (g_pDeltaMode == 3) {
    return 1.0 / fmax(dmag, 1.0e-300);
  }

  const double floorVal = fmax(g_pDeltaMinCos, 0.0) * dmag;
  return 1.0 / fmax(fabs(dpn), fmax(floorVal, 1.0e-300));
}

static inline double pressure_delta_coeff_runtime(
    const std::array<double,3>& n,
    const std::array<double,3>& d)
{
  return pressure_delta_coeff_runtime(d[0], d[1], d[2], n[0], n[1], n[2]);
}

static std::array<double,3> parse_vec3_arg(const std::string& raw, const char* optName){
  std::string t = raw;
  for(char& c : t){
    if(c == ',' || c == ';' || c == ':') c = ' ';
  }

  std::istringstream iss(t);
  std::array<double,3> v{{0.0, 0.0, 0.0}};
  if(!(iss >> v[0] >> v[1] >> v[2])){
    std::fprintf(stderr, "Could not parse %s value '%s'. Expected x,y,z\n",
                 optName, raw.c_str());
    MPI_Abort(MPI_COMM_WORLD, 1);
  }
  return v;
}

static std::array<double,3> normalized_vec3(std::array<double,3> v){
  const double m = norm3(v);
  if(m <= 1.0e-300) return std::array<double,3>{{0.0, 0.0, 0.0}};
  return mul3(1.0/m, v);
}

static int find_patch_index_local(const Mesh& mesh, const std::string& patchName){
  for(std::size_t k=0; k<mesh.patchNames.size(); ++k){
    if(mesh.patchNames[k] == patchName) return (int)k;
  }
  return -1;
}


static std::string trim_case_line(std::string line){
  const std::size_t hash = line.find('#');
  if(hash != std::string::npos) line = line.substr(0, hash);

  std::size_t a = 0;
  while(a < line.size() && std::isspace(static_cast<unsigned char>(line[a]))) ++a;

  std::size_t b = line.size();
  while(b > a && std::isspace(static_cast<unsigned char>(line[b-1]))) --b;

  return line.substr(a, b-a);
}

static std::vector<std::string> tokenize_case_line(const std::string& line){
  std::istringstream iss(line);
  std::vector<std::string> tok;
  for(std::string t; iss >> t; ) tok.push_back(t);
  return tok;
}

static std::vector<std::string> expand_case_config_args(int argc, char** argv){
  std::string casePath;

  for(int i=1; i<argc; ++i){
    if(!std::strcmp(argv[i], "-case-config")){
      if(i + 1 >= argc){
        throw std::runtime_error("Missing value after -case-config");
      }
      casePath = argv[i+1];
      break;
    }
  }

  if(casePath.empty()){
    std::vector<std::string> out;
    out.reserve(argc);
    for(int i=0; i<argc; ++i) out.emplace_back(argv[i]);
    return out;
  }

  std::map<std::string,std::string> keyToFlag = {
    {"polyMeshDir", "-polyMeshDir"},
    {"outPrefix", "-out-prefix"},
    {"rho", "-rho"},
    {"mu", "-mu"},
    {"re", "-re"},
    {"Re", "-re"},
    {"uMean", "-uMean"},
    {"Umean", "-uMean"},
    {"pipeD", "-pipeD"},
    {"pipeL", "-pipeL"},

    {"wallPatch", "-wall-patch"},
    {"inletPatch", "-inlet-patch"},
    {"outletPatch", "-outlet-patch"},
    {"bcConfig", "-bc-config"},

    {"device", "-device"},
    {"cfl", "-cfl"},
    {"CFL", "-cfl"},

    {"nsteps", "-nsteps"},
    {"printEvery", "-print-every"},
    {"writeVtu", "-write-vtu"},
    {"writeEvery", "-write-every"},
    {"monitor", "-monitor"},

    {"velRestart", "-vel-restart"},
    {"velMaxit", "-vel-maxit"},
    {"velTol", "-vel-tol"},
    {"velRelTol", "-vel-reltol"},
    {"velSolver", "-vel-solver"},
    {"velSweeps", "-vel-sweeps"},
    {"velSmootherOmega", "-vel-smoother-omega"},
    {"velGsSymmetric", "-vel-gs-symmetric"},
    {"velCorrectionSolve", "-vel-correction-solve"},

    {"nVelNonOrthCorr", "-nVelNonOrthCorr"},
    {"nNonOrthCorr", "-nNonOrthCorr"},
    {"nPressureCorr", "-nPressureCorr"},

    {"uRelax", "-u-relax"},
    {"pRelax", "-p-relax"},
    {"corrTol", "-corrTol"},
    {"tolMass", "-tolMass"},
    {"tolVel", "-tolVel"},

    {"pUseAmg", "-p-use-amg"},
    {"pMaxit", "-p-maxit"},
    {"pTol", "-p-tol"},
    {"pRelTol", "-p-reltol"},
    {"pAmgSetupScope", "-p-amg-setup-scope"},
    {"pAmgMaxit", "-p-amg-maxit"},
    {"pAmgCoarsenType", "-p-amg-coarsen-type"},
    {"pAmgInterpType", "-p-amg-interp-type"},
    {"pAmgAggLevels", "-p-amg-agg-levels"},
    {"pAmgAggInterpType", "-p-amg-agg-interp-type"},
    {"pAmgRelaxType", "-p-amg-relax-type"},
    {"pAmgPmax", "-p-amg-pmax"},
    {"pAmgTruncFactor", "-p-amg-trunc-factor"},
    {"pAmgKeepTranspose", "-p-amg-keep-transpose"},
    {"pAmgRebuildEvery", "-p-amg-rebuild-every"},

    {"forceEnable", "-force-enable"},
    {"forcePatch", "-force-patch"},
    {"forceNormalSign", "-force-normal-sign"},
    {"forceDragDir", "-force-drag-dir"},
    {"forceLiftDir", "-force-lift-dir"},
    {"forceSpanDir", "-force-span-dir"},
    {"forceUref", "-force-uref"},
    {"forceAreaRef", "-force-area-ref"},

    {"profileSteps", "-profile-steps"},
    {"assemblyBackend", "-assembly-backend"},

    // simple_gpu formulation / OpenFOAM-like controls
    {"geomMethod", "-geom-method"},
    {"geom-method", "-geom-method"},
    {"lsqStencil", "-lsq-stencil"},
    {"lsq-stencil", "-lsq-stencil"},
    {"lsqWeightPower", "-lsq-weight-power"},
    {"momentumConvectionScheme", "-momentum-convection-scheme"},
    {"momentumPhiScheme", "-momentum-convection-scheme"},
    {"momentumDivScheme", "-momentum-convection-scheme"},
    {"divPhiUScheme", "-momentum-convection-scheme"},
    {"lsq-weight-power", "-lsq-weight-power"},
    {"momNonOrthScale", "-mom-nonorth-scale"},
    {"mom-nonorth-scale", "-mom-nonorth-scale"},
    {"pNonOrthScale", "-p-nonorth-scale"},
    {"p-nonorth-scale", "-p-nonorth-scale"},
    {"pMode", "-p-mode"},
    {"p-mode", "-p-mode"},
    {"pSolveMode", "-p-solve-mode"},
    {"p-solve-mode", "-p-solve-mode"},
    {"pGradScheme", "-p-grad-scheme"},
    {"p-grad-scheme", "-p-grad-scheme"},
    {"pCoeffScale", "-p-coeff-scale"},
    {"p-coeff-scale", "-p-coeff-scale"},
    {"rcMode", "-rc-mode"},
    {"rc-mode", "-rc-mode"},
    {"rAUMode", "-rAU-mode"},
    {"rAU-mode", "-rAU-mode"},
    {"divdevScale", "-divdev-scale"},
    {"divdev-scale", "-divdev-scale"},
    {"pFluxMode", "-p-flux-mode"},
    {"p-flux-mode", "-p-flux-mode"},
    {"pDeltaMode", "-p-delta-mode"},
    {"p-delta-mode", "-p-delta-mode"},
    {"pressureDeltaMode", "-p-delta-mode"},
    {"pressure-delta-mode", "-p-delta-mode"},
    {"pDeltaMinCos", "-p-delta-min-cos"},
    {"p-delta-min-cos", "-p-delta-min-cos"},
    {"pressureDeltaMinCos", "-p-delta-min-cos"},
    {"pressure-delta-min-cos", "-p-delta-min-cos"},

    {"initMode", "-init-mode"},
    {"initialisationMode", "-init-mode"},
    {"initializationMode", "-init-mode"},
    {"potentialInitRAU", "-potential-init-rAU"},
    {"potentialInitRau", "-potential-init-rAU"},
    {"potentialInitMaxit", "-potential-init-maxit"},
    {"potentialInitTol", "-potential-init-tol"},
    {"potentialInitRelTol", "-potential-init-reltol"},
    {"potentialInitWrite", "-potential-init-write"},

    // PDE-oriented options accepted by v1 case files.
    {"poissonGradientScheme", "-poisson-gradient-scheme"},
    {"poissonGradScheme", "-poisson-gradient-scheme"},
    {"poissonLaplacianScheme", "-poisson-laplacian-scheme"},
    {"poissonNonOrthCorr", "-poisson-nonorth-corr"},

    {"scalarEnable", "-scalar-enable"},
    {"scalarSolveMode", "-scalar-solve-mode"},
    {"scalarName", "-scalar-name"},
    {"scalarBCConfig", "-scalar-bc-config"},
    {"scalarBcConfig", "-scalar-bc-config"},
    {"scalarConvectionScheme", "-scalar-convection-scheme"},
    {"scalarPhiScheme", "-scalar-convection-scheme"},
    {"phiScheme", "-scalar-convection-scheme"},
    {"scalarDiffusionScheme", "-scalar-diffusion-scheme"},
    {"scalarGamma", "-scalar-gamma"},
    {"scalarRelax", "-scalar-relax"},
    {"scalarNonOrthCorr", "-scalar-nonorth-corr"},
    {"scalarMaxit", "-scalar-maxit"},
    {"scalarTol", "-scalar-tol"},
    {"scalarRelTol", "-scalar-reltol"},

    {"simpleFaceSkewCorr", "-simple-face-skew-corr"},
    {"simple-face-skew-corr", "-simple-face-skew-corr"},
    {"simpleSubtriContinuity", "-simple-subtri-continuity"},
    {"simple-subtri-continuity", "-simple-subtri-continuity"},
    {"simpleSubtriFluxRecon", "-simple-subtri-flux-recon"},
    {"simple-subtri-flux-recon", "-simple-subtri-flux-recon"},
    {"simpleMmsEnable", "-simple-mms-enable"},
    {"simple-mms-enable", "-simple-mms-enable"},
    {"simpleMmsInitExact", "-simple-mms-init-exact"},
    {"simple-mms-init-exact", "-simple-mms-init-exact"},
    {"simpleMmsPressureDirichlet", "-simple-mms-pressure-dirichlet"},
    {"simple-mms-pressure-dirichlet", "-simple-mms-pressure-dirichlet"},
    {"simpleMmsVelocityScale", "-simple-mms-velocity-scale"},
    {"simple-mms-velocity-scale", "-simple-mms-velocity-scale"},
    {"simpleMmsPressureScale", "-simple-mms-pressure-scale"},
    {"simple-mms-pressure-scale", "-simple-mms-pressure-scale"},
    {"simpleMmsFixedVelocityBoundaryPressureEqn", "-simple-mms-fixed-velocity-boundary-pressure-eqn"},
    {"simple-mms-fixed-velocity-boundary-pressure-eqn", "-simple-mms-fixed-velocity-boundary-pressure-eqn"},
    {"simpleMmsSourceQuad8", "-simple-mms-source-quad8"},
    {"simple-mms-source-quad8", "-simple-mms-source-quad8"},
    {"simpleMmsStokesOnly", "-simple-mms-stokes-only"},
    {"simple-mms-stokes-only", "-simple-mms-stokes-only"},
    {"simpleMmsDiagStages", "-simple-mms-diag-stages"},
    {"simple-mms-diag-stages", "-simple-mms-diag-stages"},
    {"simpleMmsForceExactPBeforeMomentum", "-simple-mms-force-exact-p-before-momentum"},
    {"simple-mms-force-exact-p-before-momentum", "-simple-mms-force-exact-p-before-momentum"},
    {"simpleMmsForceExactUBeforePressure", "-simple-mms-force-exact-u-before-pressure"},
    {"simple-mms-force-exact-u-before-pressure", "-simple-mms-force-exact-u-before-pressure"},
    {"simpleMmsForceExactPAfterUpdate", "-simple-mms-force-exact-p-after-update"},
    {"simple-mms-force-exact-p-after-update", "-simple-mms-force-exact-p-after-update"},
    {"simpleMmsDiagPressureExactResidual", "-simple-mms-diag-pressure-exact-residual"},
    {"simple-mms-diag-pressure-exact-residual", "-simple-mms-diag-pressure-exact-residual"},
    {"simpleMmsDiagWritePErrorVtk", "-simple-mms-diag-write-p-error-vtk"},
    {"simple-mms-diag-write-p-error-vtk", "-simple-mms-diag-write-p-error-vtk"},
    {"simpleMmsPressurePenaltyCoeff", "-simple-mms-pressure-penalty-coeff"},
    {"simple-mms-pressure-penalty-coeff", "-simple-mms-pressure-penalty-coeff"},

    {"sineMmsEnable", "-sine-mms-enable"},
    {"sineMmsInitExact", "-sine-mms-init-exact"},
    {"sineMmsVelocityScale", "-sine-mms-velocity-scale"},
    {"sineMmsPressureScale", "-sine-mms-pressure-scale"},

    {"badCellAuditEvery", "-bad-cell-audit-every"},
    {"badCellAuditTop", "-bad-cell-audit-top"},
    {"badCellAuditStart", "-bad-cell-audit-start"},
    {"badCellAuditOnGrowth", "-bad-cell-audit-on-growth"},
    {"badCellAuditGrowthFactor", "-bad-cell-audit-growth-factor"},
    {"badCellAuditMassFloor", "-bad-cell-audit-mass-floor"},
    {"badCellAuditWriteCsv", "-bad-cell-audit-write-csv"},
  };

  std::ifstream in(casePath);
  if(!in){
    throw std::runtime_error("Could not open case config file: " + casePath);
  }

  std::vector<std::string> out;
  out.emplace_back(argv[0]);

  std::vector<std::string> bcLines;
  std::vector<std::string> scalarBCLines;
  bool explicitBCConfig = false;
  bool explicitScalarBCConfig = false;

  std::string raw;
  int lineNo = 0;

  while(std::getline(in, raw)){
    ++lineNo;
    const std::string line = trim_case_line(raw);
    if(line.empty()) continue;

    const auto tok = tokenize_case_line(line);
    if(tok.empty()) continue;

    if(tok[0] == "velocity" || tok[0] == "pressure"){
      bcLines.push_back(line);
      continue;
    }
    if(tok[0] == "scalar"){
      scalarBCLines.push_back(line);
      continue;
    }

    if(tok.size() != 2){
      std::ostringstream oss;
      oss << "Case config parse error in '" << casePath << "' at line " << lineNo
          << ": expected '<key> <value>' or a velocity/pressure BC line";
      throw std::runtime_error(oss.str());
    }

    const auto it = keyToFlag.find(tok[0]);
    if(it == keyToFlag.end()){
      std::ostringstream oss;
      oss << "Case config parse error in '" << casePath << "' at line " << lineNo
          << ": unknown key '" << tok[0] << "'";
      throw std::runtime_error(oss.str());
    }

    if(tok[0] == "bcConfig") explicitBCConfig = true;
    if(tok[0] == "scalarBCConfig" || tok[0] == "scalarBcConfig") explicitScalarBCConfig = true;

    out.push_back(it->second);
    out.push_back(tok[1]);
  }

  if(!bcLines.empty() && explicitBCConfig){
    throw std::runtime_error(
        "Case config cannot contain both bcConfig and inline velocity/pressure BC lines");
  }

  if(!bcLines.empty()){
    const std::string generatedBCPath = casePath + ".generated.bc";
    std::ofstream bcOut(generatedBCPath);
    if(!bcOut){
      throw std::runtime_error("Could not write generated BC file: " + generatedBCPath);
    }

    bcOut << "# Auto-generated from " << casePath << "\n";
    for(const auto& line : bcLines) bcOut << line << "\n";

    out.push_back("-bc-config");
    out.push_back(generatedBCPath);
  }

  if(!scalarBCLines.empty() && explicitScalarBCConfig){
    throw std::runtime_error(
        "Case config cannot contain both scalarBCConfig and inline scalar BC lines");
  }

  if(!scalarBCLines.empty()){
    const std::string generatedScalarBCPath = casePath + ".generated.scalar.bc";
    std::ofstream scalarBCOut(generatedScalarBCPath);
    if(!scalarBCOut){
      throw std::runtime_error("Could not write generated scalar BC file: " + generatedScalarBCPath);
    }

    scalarBCOut << "# Auto-generated scalar BCs from " << casePath << "\n";
    for(const auto& line : scalarBCLines) scalarBCOut << line << "\n";

    out.push_back("-scalar-bc-config");
    out.push_back(generatedScalarBCPath);
  }

  // Append explicit command-line options after case-file options.
  // This lets command-line arguments override the case file.
  for(int i=1; i<argc; ++i){
    if(!std::strcmp(argv[i], "-case-config")){
      ++i;
      continue;
    }
    out.emplace_back(argv[i]);
  }

  return out;
}

static void parse_args(int argc, char** argv, Params &par){
  for(int i=1;i<argc;++i){
    auto need=[&](const char* opt){ if(i+1>=argc){std::fprintf(stderr,"Missing value after %s\n",opt); MPI_Abort(MPI_COMM_WORLD,1);} };
    if(!std::strcmp(argv[i],"-polyMeshDir")){need(argv[i]); par.polyMeshDir=argv[++i];}
    else if(!std::strcmp(argv[i],"-out-prefix")){need(argv[i]); par.outPrefix=argv[++i];}
    else if(!std::strcmp(argv[i],"-rho")){need(argv[i]); par.rho=std::atof(argv[++i]);}
    else if(!std::strcmp(argv[i],"-mu")){need(argv[i]); par.mu=std::atof(argv[++i]); par.muExplicit=true;}
    else if(!std::strcmp(argv[i],"-re")){need(argv[i]); par.Re=std::atof(argv[++i]);}
    else if(!std::strcmp(argv[i],"-uMean") || !std::strcmp(argv[i],"-lidU")){need(argv[i]); par.Umean=std::atof(argv[++i]);}
    else if(!std::strcmp(argv[i],"-pipeD")){need(argv[i]); par.pipeDiameter=std::atof(argv[++i]);}
    else if(!std::strcmp(argv[i],"-pipeL")){need(argv[i]); par.pipeLength=std::atof(argv[++i]);}
    else if(!std::strcmp(argv[i],"-wall-patch")){need(argv[i]); par.wallPatchName=argv[++i];}
    else if(!std::strcmp(argv[i],"-inlet-patch") || !std::strcmp(argv[i],"-lid-patch")){need(argv[i]); par.inletPatchName=argv[++i];}
    else if(!std::strcmp(argv[i],"-outlet-patch")){need(argv[i]); par.outletPatchName=argv[++i];}
    else if(!std::strcmp(argv[i],"-bc-config")){need(argv[i]); par.bcConfigPath=argv[++i];}
    else if(!std::strcmp(argv[i],"-cfl")){need(argv[i]); par.CFL=std::atof(argv[++i]);}
    else if(!std::strcmp(argv[i],"-device")){need(argv[i]); par.device=std::atoi(argv[++i]);}
    else if(!std::strcmp(argv[i],"-vel-restart")){need(argv[i]); par.velRestart=std::atoi(argv[++i]);}
    else if(!std::strcmp(argv[i],"-force-enable")){need(argv[i]); par.forceEnable=std::atoi(argv[++i]);}
    else if(!std::strcmp(argv[i],"-force-patch")){need(argv[i]); par.forcePatchName=argv[++i];}
    else if(!std::strcmp(argv[i],"-force-normal-sign")){need(argv[i]); par.forceNormalSign=std::atoi(argv[++i]);}
    else if(!std::strcmp(argv[i],"-force-uref")){need(argv[i]); par.forceUref=std::atof(argv[++i]);}
    else if(!std::strcmp(argv[i],"-force-area-ref")){need(argv[i]); par.forceAreaRef=std::atof(argv[++i]);}
    else if(!std::strcmp(argv[i],"-force-drag-dir")){need(argv[i]); par.forceDragDir=parse_vec3_arg(argv[++i], "-force-drag-dir");}
    else if(!std::strcmp(argv[i],"-force-lift-dir")){need(argv[i]); par.forceLiftDir=parse_vec3_arg(argv[++i], "-force-lift-dir");}
    else if(!std::strcmp(argv[i],"-force-span-dir")){need(argv[i]); par.forceSpanDir=parse_vec3_arg(argv[++i], "-force-span-dir");}
    else if(!std::strcmp(argv[i],"-vel-maxit")){need(argv[i]); par.velMaxit=std::atoi(argv[++i]);}
    else if(!std::strcmp(argv[i],"-vel-tol")){need(argv[i]); par.velTol=std::atof(argv[++i]);}
    else if(!std::strcmp(argv[i],"-vel-reltol")){need(argv[i]); par.velRelTol=std::atof(argv[++i]);}
    else if(!std::strcmp(argv[i],"-vel-solver")){
      need(argv[i]);
      std::string v=argv[++i];
      for(char &c:v) c=(char)std::tolower((unsigned char)c);
      if(v=="bicgstab" || v=="krylov") par.velSolver=0;
      else if(v=="mcgs" || v=="colored-gs" || v=="multicolor-gs" || v=="multi-color-gs") { par.velSolver=2; par.velCorrectionSolve=1; }
      else if(v=="mcgs-fused" || v=="mcgs_fused" || v=="fused-mcgs" || v=="fused-colored-gs") { par.velSolver=3; par.velCorrectionSolve=1; }
      else { std::fprintf(stderr,"Unknown -vel-solver '%s'. Use bicgstab, mcgs, or mcgs-fused.\n", v.c_str()); MPI_Abort(MPI_COMM_WORLD,1); }
    }
    else if(!std::strcmp(argv[i],"-vel-sweeps")){need(argv[i]); par.velSweeps=std::atoi(argv[++i]);}
    else if(!std::strcmp(argv[i],"-vel-smoother-omega")){need(argv[i]); par.velSmootherOmega=std::atof(argv[++i]);}
    else if(!std::strcmp(argv[i],"-vel-gs-symmetric")){need(argv[i]); par.velGsSymmetric=std::atoi(argv[++i]);}
    else if(!std::strcmp(argv[i],"-vel-correction-solve")){need(argv[i]); par.velCorrectionSolve=std::atoi(argv[++i]);}
    else if(!std::strcmp(argv[i],"-nVelNonOrthCorr")){need(argv[i]); par.nVelNonOrthCorr=std::atoi(argv[++i]);}
    else if(!std::strcmp(argv[i],"-nNonOrthCorr")){need(argv[i]); par.nNonOrthCorr=std::atoi(argv[++i]);}
    else if(!std::strcmp(argv[i],"-nPressureCorr")){need(argv[i]); par.nPressureCorr=std::atoi(argv[++i]);}
    else if(!std::strcmp(argv[i],"-u-relax")){need(argv[i]); par.uRelax=std::atof(argv[++i]);}
    else if(!std::strcmp(argv[i],"-p-relax")){need(argv[i]); par.pRelax=std::atof(argv[++i]);}
    else if(!std::strcmp(argv[i],"-mom-nonorth-scale") || !std::strcmp(argv[i],"-momNonOrthScale")){need(argv[i]); par.momNonOrthScale=std::atof(argv[++i]);}
    else if(!std::strcmp(argv[i],"-momentum-convection-scheme") || !std::strcmp(argv[i],"-momentum-phi-scheme") || !std::strcmp(argv[i],"-div-phi-u-scheme")){
      need(argv[i]);
      std::string v=argv[++i];
      for(char &c:v) c=(char)std::tolower((unsigned char)c);
      if(v=="central" || v=="linear" || v=="gauss-linear" || v=="gausslinear" || v=="centered" || v=="centred") par.momentumConvectionScheme=0;
      else if(v=="upwind" || v=="first-order-upwind" || v=="firstorderupwind") par.momentumConvectionScheme=1;
      else if(v=="none" || v=="off" || v=="stokes") par.momentumConvectionScheme=2;
      else { std::fprintf(stderr,"Unknown -momentum-convection-scheme '%s'. Use central, upwind, or none.\n", v.c_str()); MPI_Abort(MPI_COMM_WORLD,1); }
    }
    else if(!std::strcmp(argv[i],"-rc-mode") || !std::strcmp(argv[i],"-rhie-chow-mode")){
      need(argv[i]);
      std::string v=argv[++i];
      for(char &c:v) c=(char)std::tolower((unsigned char)c);
      if(v=="old" || v=="explicit" || v=="legacy" || v=="rc") par.rcMode=0;
      else if(v=="of" || v=="oflike" || v=="openfoam" || v=="chalmers" || v=="none" || v=="noexplicit") par.rcMode=1;
      else { std::fprintf(stderr,"Unknown -rc-mode '%s'. Use old or oflike.\n", v.c_str()); MPI_Abort(MPI_COMM_WORLD,1);}
    }
    else if(!std::strcmp(argv[i],"-rAU-mode") || !std::strcmp(argv[i],"-rau-mode") || !std::strcmp(argv[i],"-rUA-mode") || !std::strcmp(argv[i],"-rua-mode")){
      need(argv[i]);
      std::string v=argv[++i];
      for(char &c:v) c=(char)std::tolower((unsigned char)c);
      if(v=="raw" || v=="unrelaxed" || v=="a_raw" || v=="apraw") par.rAUMode=0;
      else if(v=="relaxed" || v=="of" || v=="oflike" || v=="openfoam" || v=="relaxeddiag") par.rAUMode=1;
      else { std::fprintf(stderr,"Unknown -rAU-mode '%s'. Use raw or relaxed.\n", v.c_str()); MPI_Abort(MPI_COMM_WORLD,1);}
    }
    else if(!std::strcmp(argv[i],"-rAU-scale") || !std::strcmp(argv[i],"-rau-scale") || !std::strcmp(argv[i],"-rUA-scale") || !std::strcmp(argv[i],"-rua-scale")){need(argv[i]); par.rAUScale=std::atof(argv[++i]);}
    else if(!std::strcmp(argv[i],"-p-nonorth-scale") || !std::strcmp(argv[i],"-pressure-nonorth-scale") || !std::strcmp(argv[i],"-pNonOrthScale")){need(argv[i]); par.pNonOrthScale=std::atof(argv[++i]);}
    else if(!std::strcmp(argv[i],"-p-mode") || !std::strcmp(argv[i],"-pressure-mode")){
      need(argv[i]);
      std::string v=argv[++i];
      for(char &c:v) c=(char)std::tolower((unsigned char)c);
      if(v=="pcorr" || v=="correction" || v=="p-correction" || v=="piso" || v=="old") par.pMode=0;
      else if(v=="absolute" || v=="abs" || v=="openfoam" || v=="of" || v=="simple") par.pMode=1;
      else { std::fprintf(stderr,"Unknown -p-mode '%s'. Use pcorr or absolute.\n", v.c_str()); MPI_Abort(MPI_COMM_WORLD,1);}
    }
    else if(!std::strcmp(argv[i],"-p-solve-mode") || !std::strcmp(argv[i],"-psolve-mode")){
      need(argv[i]);
      std::string v=argv[++i];
      for(char &c:v) c=(char)std::tolower((unsigned char)c);
      if(v=="correction" || v=="pcorr" || v=="legacy" || v=="current") par.pSolveMode=0;
      else if(v=="ofabsolute" || v=="of-absolute" || v=="openfoam" || v=="of" || v=="absolute") { par.pSolveMode=1; par.pMode=1; }
      else { std::fprintf(stderr,"Unknown -p-solve-mode '%s'. Use correction or ofAbsolute.\n", v.c_str()); MPI_Abort(MPI_COMM_WORLD,1);}
    }
    else if(!std::strcmp(argv[i],"-p-grad-scheme") || !std::strcmp(argv[i],"-pgrad-scheme")){
      need(argv[i]);
      std::string v=argv[++i];
      for(char &c:v) c=(char)std::tolower((unsigned char)c);
      if(v=="lsq" || v=="least-squares" || v=="leastsquares") par.pGradScheme=0;
      else if(v=="gauss" || v=="gauss-linear" || v=="gausslinear" || v=="of") par.pGradScheme=1;
      else { std::fprintf(stderr,"Unknown -p-grad-scheme '%s'. Use lsq or gauss.\n", v.c_str()); MPI_Abort(MPI_COMM_WORLD,1);}
    }
    else if(!std::strcmp(argv[i],"-p-coeff-scale") || !std::strcmp(argv[i],"-pressure-coeff-scale") || !std::strcmp(argv[i],"-pCoeffScale")){need(argv[i]); par.pCoeffScale=std::atof(argv[++i]);}
    else if(!std::strcmp(argv[i],"-hbya-bc-mode") || !std::strcmp(argv[i],"-HbyA-bc-mode") || !std::strcmp(argv[i],"-constrain-hbya")){
      need(argv[i]);
      std::string v=argv[++i];
      for(char &c:v) c=(char)std::tolower((unsigned char)c);
      if(v=="constrained" || v=="constraint" || v=="fixed" || v=="of" || v=="openfoam" || v=="1" || v=="yes" || v=="true") par.hbyaBcMode=1;
      else if(v=="owner" || v=="cell" || v=="unconstrained" || v=="0" || v=="no" || v=="false") par.hbyaBcMode=0;
      else { std::fprintf(stderr,"Unknown -hbya-bc-mode '%s'. Use constrained or owner.\n", v.c_str()); MPI_Abort(MPI_COMM_WORLD,1); }
    }
    else if(!std::strcmp(argv[i],"-p-delta-mode") || !std::strcmp(argv[i],"-pressure-delta-mode")){
      need(argv[i]);
      std::string v = argv[++i];
      if(v=="legacy" || v=="v1" || v=="signed" || v=="signed-projected" || v=="old" || v=="0") par.pDeltaMode=0;
      else if(v=="of" || v=="openfoam" || v=="oflike" || v=="stabilised" || v=="stabilized" || v=="on" || v=="1") par.pDeltaMode=1;
      else if(v=="normal" || v=="projected" || v=="orthogonal" || v=="orth" || v=="abs" || v=="2") par.pDeltaMode=2;
      else if(v=="distance" || v=="magd" || v=="dmag" || v=="magnitude" || v=="3") par.pDeltaMode=3;
      else {
        std::fprintf(stderr,"Unknown -p-delta-mode '%s'. Use legacy/v1, of, normal, or distance.\n", v.c_str());
        MPI_Abort(MPI_COMM_WORLD,1);
      }
    }
    else if(!std::strcmp(argv[i],"-p-delta-min-cos") || !std::strcmp(argv[i],"-pressure-delta-min-cos")){
      need(argv[i]);
      par.pDeltaMinCos=std::atof(argv[++i]);
      if(par.pDeltaMinCos < 0.0) {
        std::fprintf(stderr,"-p-delta-min-cos must be non-negative.\n");
        MPI_Abort(MPI_COMM_WORLD,1);
      }
    }

    else if(!std::strcmp(argv[i],"-init-mode") || !std::strcmp(argv[i],"-initialisation-mode") || !std::strcmp(argv[i],"-initialization-mode")){
      need(argv[i]);
      std::string v = argv[++i];
      for(char &c:v) c=(char)std::tolower((unsigned char)c);
      if(v=="uniform" || v=="default" || v=="inlet-average" || v=="inlet_average" || v=="0") par.initMode=0;
      else if(v=="potential" || v=="potentialflow" || v=="projection" || v=="projected" || v=="1") par.initMode=1;
      else { std::fprintf(stderr,"Unknown -init-mode '%s'. Use uniform or potential.\n", v.c_str()); MPI_Abort(MPI_COMM_WORLD,1); }
    }
    else if(!std::strcmp(argv[i],"-potential-init-rAU") || !std::strcmp(argv[i],"-potential-init-rau")){need(argv[i]); par.potentialInitRAU=std::atof(argv[++i]);}
    else if(!std::strcmp(argv[i],"-potential-init-maxit")){need(argv[i]); par.potentialInitMaxit=std::atoi(argv[++i]);}
    else if(!std::strcmp(argv[i],"-potential-init-tol")){need(argv[i]); par.potentialInitTol=std::atof(argv[++i]);}
    else if(!std::strcmp(argv[i],"-potential-init-reltol")){need(argv[i]); par.potentialInitRelTol=std::atof(argv[++i]);}
    else if(!std::strcmp(argv[i],"-potential-init-write")){need(argv[i]); par.potentialInitWrite=std::atoi(argv[++i]);}

    else if(!std::strcmp(argv[i],"-poisson-gradient-scheme")){
      need(argv[i]);
      std::string v=argv[++i];
      for(char &c:v) c=(char)std::tolower((unsigned char)c);
      if(v=="lsq" || v=="least-squares" || v=="leastsquares" || v=="gauss" || v=="green-gauss" || v=="greengauss" || v=="gauss-linear" || v=="gausslinear") par.poissonGradientScheme=v;
      else { std::fprintf(stderr,"Unknown -poisson-gradient-scheme '%s'. Use lsq or gauss.\n", v.c_str()); MPI_Abort(MPI_COMM_WORLD,1); }
    }
    else if(!std::strcmp(argv[i],"-poisson-laplacian-scheme")){
      need(argv[i]);
      std::string v=argv[++i];
      for(char &c:v) c=(char)std::tolower((unsigned char)c);
      if(v=="orth" || v=="orthogonal" || v=="nonorth" || v=="nonorthogonal" || v=="corrected") par.poissonLaplacianScheme=v;
      else { std::fprintf(stderr,"Unknown -poisson-laplacian-scheme '%s'. Use orth or nonorth.\n", v.c_str()); MPI_Abort(MPI_COMM_WORLD,1); }
    }
    else if(!std::strcmp(argv[i],"-poisson-nonorth-corr")){need(argv[i]); par.poissonNonOrthCorr=std::atoi(argv[++i]);}

    else if(!std::strcmp(argv[i],"-scalar-enable")){need(argv[i]); par.scalarEnable=std::atoi(argv[++i]);}
    else if(!std::strcmp(argv[i],"-scalar-solve-mode")){need(argv[i]); par.scalarSolveMode=argv[++i];}
    else if(!std::strcmp(argv[i],"-scalar-name")){need(argv[i]); par.scalarName=argv[++i];}
    else if(!std::strcmp(argv[i],"-scalar-bc-config")){need(argv[i]); par.scalarBCConfigPath=argv[++i];}
    else if(!std::strcmp(argv[i],"-scalar-convection-scheme")){
      need(argv[i]);
      std::string v=argv[++i];
      for(char &c:v) c=(char)std::tolower((unsigned char)c);
      if(v=="central" || v=="linear" || v=="gauss-linear" || v=="gausslinear" || v=="upwind" || v=="first-order-upwind" || v=="firstorderupwind") par.scalarConvectionScheme=v;
      else { std::fprintf(stderr,"Unknown -scalar-convection-scheme '%s'. Use central or upwind.\n", v.c_str()); MPI_Abort(MPI_COMM_WORLD,1); }
    }
    else if(!std::strcmp(argv[i],"-scalar-diffusion-scheme")){
      need(argv[i]);
      std::string v=argv[++i];
      for(char &c:v) c=(char)std::tolower((unsigned char)c);
      if(v=="orth" || v=="orthogonal" || v=="nonorth" || v=="nonorthogonal" || v=="corrected") par.scalarDiffusionScheme=v;
      else { std::fprintf(stderr,"Unknown -scalar-diffusion-scheme '%s'. Use orth or nonorth.\n", v.c_str()); MPI_Abort(MPI_COMM_WORLD,1); }
    }
    else if(!std::strcmp(argv[i],"-scalar-gamma")){need(argv[i]); par.scalarGamma=std::atof(argv[++i]);}
    else if(!std::strcmp(argv[i],"-scalar-relax")){need(argv[i]); par.scalarRelax=std::atof(argv[++i]);}
    else if(!std::strcmp(argv[i],"-scalar-nonorth-corr")){need(argv[i]); par.scalarNonOrthCorr=std::atoi(argv[++i]);}
    else if(!std::strcmp(argv[i],"-scalar-maxit")){need(argv[i]); par.scalarMaxit=std::atoi(argv[++i]);}
    else if(!std::strcmp(argv[i],"-scalar-tol")){need(argv[i]); par.scalarTol=std::atof(argv[++i]);}
    else if(!std::strcmp(argv[i],"-scalar-reltol")){need(argv[i]); par.scalarRelTol=std::atof(argv[++i]);}

    else if(!std::strcmp(argv[i],"-simple-face-skew-corr")){need(argv[i]); par.simpleFaceSkewCorr=std::atoi(argv[++i]);}
    else if(!std::strcmp(argv[i],"-simple-subtri-continuity")){need(argv[i]); par.simpleSubtriContinuity=std::atoi(argv[++i]);}
    else if(!std::strcmp(argv[i],"-simple-subtri-flux-recon")){need(argv[i]); par.simpleSubtriFluxRecon=std::atoi(argv[++i]);}
    else if(!std::strcmp(argv[i],"-simple-mms-enable")){need(argv[i]); par.simpleMmsEnable=std::atoi(argv[++i]);}
    else if(!std::strcmp(argv[i],"-simple-mms-init-exact")){need(argv[i]); par.simpleMmsInitExact=std::atoi(argv[++i]);}
    else if(!std::strcmp(argv[i],"-simple-mms-pressure-dirichlet")){need(argv[i]); par.simpleMmsPressureDirichlet=std::atoi(argv[++i]);}
    else if(!std::strcmp(argv[i],"-simple-mms-velocity-scale")){need(argv[i]); par.simpleMmsVelocityScale=std::atof(argv[++i]);}
    else if(!std::strcmp(argv[i],"-simple-mms-pressure-scale")){need(argv[i]); par.simpleMmsPressureScale=std::atof(argv[++i]);}
    else if(!std::strcmp(argv[i],"-simple-mms-fixed-velocity-boundary-pressure-eqn")){need(argv[i]); par.simpleMmsFixedVelocityBoundaryPressureEqn=std::atoi(argv[++i]);}
    else if(!std::strcmp(argv[i],"-simple-mms-source-quad8")){need(argv[i]); par.simpleMmsSourceQuad8=std::atoi(argv[++i]);}
    else if(!std::strcmp(argv[i],"-simple-mms-stokes-only")){need(argv[i]); par.simpleMmsStokesOnly=std::atoi(argv[++i]);}
    else if(!std::strcmp(argv[i],"-simple-mms-diag-stages")){need(argv[i]); par.simpleMmsDiagStages=std::atoi(argv[++i]);}
    else if(!std::strcmp(argv[i],"-simple-mms-force-exact-p-before-momentum")){need(argv[i]); par.simpleMmsForceExactPBeforeMomentum=std::atoi(argv[++i]);}
    else if(!std::strcmp(argv[i],"-simple-mms-force-exact-u-before-pressure")){need(argv[i]); par.simpleMmsForceExactUBeforePressure=std::atoi(argv[++i]);}
    else if(!std::strcmp(argv[i],"-simple-mms-force-exact-p-after-update")){need(argv[i]); par.simpleMmsForceExactPAfterUpdate=std::atoi(argv[++i]);}
    else if(!std::strcmp(argv[i],"-simple-mms-diag-pressure-exact-residual")){need(argv[i]); par.simpleMmsDiagPressureExactResidual=std::atoi(argv[++i]);}
    else if(!std::strcmp(argv[i],"-simple-mms-diag-write-p-error-vtk")){need(argv[i]); par.simpleMmsDiagWritePErrorVtk=std::atoi(argv[++i]);}
    else if(!std::strcmp(argv[i],"-simple-mms-pressure-penalty-coeff")){need(argv[i]); par.simpleMmsPressurePenaltyCoeff=std::atof(argv[++i]);}

    else if(!std::strcmp(argv[i],"-sine-mms-enable")){need(argv[i]); par.sineMmsEnable=std::atoi(argv[++i]);}
    else if(!std::strcmp(argv[i],"-sine-mms-init-exact")){need(argv[i]); par.sineMmsInitExact=std::atoi(argv[++i]);}
    else if(!std::strcmp(argv[i],"-sine-mms-velocity-scale")){need(argv[i]); par.sineMmsVelocityScale=std::atof(argv[++i]);}
    else if(!std::strcmp(argv[i],"-sine-mms-pressure-scale")){need(argv[i]); par.sineMmsPressureScale=std::atof(argv[++i]);}

    else if(!std::strcmp(argv[i],"-bad-cell-audit-every")){need(argv[i]); par.badCellAuditEvery=std::atoi(argv[++i]);}
    else if(!std::strcmp(argv[i],"-bad-cell-audit-top")){need(argv[i]); par.badCellAuditTop=std::atoi(argv[++i]);}
    else if(!std::strcmp(argv[i],"-bad-cell-audit-start")){need(argv[i]); par.badCellAuditStart=std::atoi(argv[++i]);}
    else if(!std::strcmp(argv[i],"-bad-cell-audit-on-growth")){need(argv[i]); par.badCellAuditOnGrowth=std::atoi(argv[++i]);}
    else if(!std::strcmp(argv[i],"-bad-cell-audit-growth-factor")){need(argv[i]); par.badCellAuditGrowthFactor=std::atof(argv[++i]);}
    else if(!std::strcmp(argv[i],"-bad-cell-audit-mass-floor")){need(argv[i]); par.badCellAuditMassFloor=std::atof(argv[++i]);}
    else if(!std::strcmp(argv[i],"-bad-cell-audit-write-csv")){need(argv[i]); par.badCellAuditWriteCsv=std::atoi(argv[++i]);}

    else if(!std::strcmp(argv[i],"-p-flux-mode") || !std::strcmp(argv[i],"-pressure-flux-mode") || !std::strcmp(argv[i],"-pEqn-flux-mode")){
      need(argv[i]);
      std::string v=argv[++i];
      for(char &c:v) c=(char)std::tolower((unsigned char)c);
      if(v=="legacy" || v=="old" || v=="0") par.pFluxMode=0;
      else if(v=="matrix" || v=="mat" || v=="of" || v=="openfoam" || v=="1") par.pFluxMode=1;
      else { std::fprintf(stderr,"Unknown -p-flux-mode '%s'. Use legacy or matrix.\n", v.c_str()); MPI_Abort(MPI_COMM_WORLD,1); }
    }
    else if(!std::strcmp(argv[i],"-geom-method")){
      need(argv[i]);
      std::string v=argv[++i];
      for(char &c:v) c=(char)std::tolower((unsigned char)c);
      if(v=="legacy" || v=="0") par.geomMethod = 0;
      else if(v=="robust" || v=="nasa" || v=="1") par.geomMethod = 1;
      else { std::fprintf(stderr,"Unknown -geom-method '%s'. Use legacy or robust.\n", v.c_str()); MPI_Abort(MPI_COMM_WORLD,1); }
    }
    else if(!std::strcmp(argv[i],"-lsq-weight-power") || !std::strcmp(argv[i],"-lsqWeightPower")){need(argv[i]); par.lsqWeightPower=std::atof(argv[++i]);}
    else if(!std::strcmp(argv[i],"-lsq-stencil")){
      need(argv[i]);
      std::string v=argv[++i];
      for(char &c:v) c=(char)std::tolower((unsigned char)c);
      if(v=="compact" || v=="local" || v=="0") par.lsqStencilMode=0;
      else if(v=="extended" || v=="second" || v=="2" || v=="1") {
        par.lsqStencilMode=1;
        if(MPI_COMM_WORLD != MPI_COMM_NULL) {
          int rank=0; MPI_Comm_rank(MPI_COMM_WORLD,&rank);
  g_memdiag_rank = rank;
          if(rank==0) std::fprintf(stderr,"WARNING: simple_gpu v1.1 currently accepts -lsq-stencil extended but uses compact LSQ coefficients.\n");
        }
      } else { std::fprintf(stderr,"Unknown -lsq-stencil '%s'. Use compact or extended.\n", v.c_str()); MPI_Abort(MPI_COMM_WORLD,1); }
    }
    else if(!std::strcmp(argv[i],"-corrTol")){need(argv[i]); par.corrTol=std::atof(argv[++i]);}
    else if(!std::strcmp(argv[i],"-monitor")){need(argv[i]); par.monitor=std::atoi(argv[++i]);}
    else if(!std::strcmp(argv[i],"-write-vtu")){need(argv[i]); par.write_vtu=std::atoi(argv[++i]);}
    else if(!std::strcmp(argv[i],"-nsteps")){need(argv[i]); par.nsteps=std::atoi(argv[++i]);}
    else if(!std::strcmp(argv[i],"-print-every")){need(argv[i]); par.printEvery=std::atoi(argv[++i]);}
    else if(!std::strcmp(argv[i],"-write-every")){need(argv[i]); par.writeEvery=std::atoi(argv[++i]);}
    else if(!std::strcmp(argv[i],"-tolMass")){need(argv[i]); par.tolMass=std::atof(argv[++i]);}
    else if(!std::strcmp(argv[i],"-tolVel")){need(argv[i]); par.tolVel=std::atof(argv[++i]);}
    else if(!std::strcmp(argv[i],"-p-use-amg")){need(argv[i]); par.p_use_amg=std::atoi(argv[++i]);}
    else if(!std::strcmp(argv[i],"-p-maxit")){need(argv[i]); par.pMaxit=std::atoi(argv[++i]);}
    else if(!std::strcmp(argv[i],"-p-tol")){need(argv[i]); par.pTol=std::atof(argv[++i]);}
    else if(!std::strcmp(argv[i],"-p-reltol")){need(argv[i]); par.pRelTol=std::atof(argv[++i]);}
    else if(!std::strcmp(argv[i],"-p-amg-maxit")){need(argv[i]); par.pAmgMaxit=std::atoi(argv[++i]);}
    else if(!std::strcmp(argv[i],"-p-amg-relax-type")){need(argv[i]); par.pAmgRelaxType=std::atoi(argv[++i]);}
    else if(!std::strcmp(argv[i],"-p-amg-coarsen-type")){need(argv[i]); par.pAmgCoarsenType=std::atoi(argv[++i]);}
    else if(!std::strcmp(argv[i],"-p-amg-interp-type")){need(argv[i]); par.pAmgInterpType=std::atoi(argv[++i]);}
    else if(!std::strcmp(argv[i],"-p-amg-agg-levels")){need(argv[i]); par.pAmgAggLevels=std::atoi(argv[++i]);}
    else if(!std::strcmp(argv[i],"-p-amg-agg-interp-type")){need(argv[i]); par.pAmgAggInterpType=std::atoi(argv[++i]);}
    else if(!std::strcmp(argv[i],"-p-amg-pmax")){need(argv[i]); par.pAmgPmax=std::atoi(argv[++i]);}
    else if(!std::strcmp(argv[i],"-p-amg-trunc-factor")){need(argv[i]); par.pAmgTruncFactor=std::atof(argv[++i]);}
    else if(!std::strcmp(argv[i],"-p-amg-keep-transpose")){need(argv[i]); par.pAmgKeepTranspose=std::atoi(argv[++i]);}
    else if(!std::strcmp(argv[i],"-profile-steps")){need(argv[i]); par.profileSteps=std::atoi(argv[++i]);}
    else if(!std::strcmp(argv[i],"-p-amg-rebuild-every")){need(argv[i]); par.pAmgRebuildEvery=std::atoi(argv[++i]);}
    else if(!std::strcmp(argv[i],"-p-amg-setup-scope")){
      need(argv[i]);
      std::string v = argv[++i];
      if(v=="outer") par.pAmgSetupScope = 0;
      else if(v=="pressure" || v=="solve") par.pAmgSetupScope = 1;
      else {
        std::fprintf(stderr,"Unknown -p-amg-setup-scope '%s'. Use outer or pressure.\n", v.c_str());
        MPI_Abort(MPI_COMM_WORLD,1);
      }
    }
  }
}

// print_device_info is provided by libpoisson/common.h.


static double get_cpu_rss_mb(){
  std::ifstream in("/proc/self/status");
  std::string key;
  while(in>>key){
    if(key=="VmRSS:"){
      double kb=0.0; in>>kb; return kb/1024.0;
    }
    std::string rest;
    std::getline(in, rest);
  }
  return 0.0;
}

static double get_gpu_used_mb(){
  size_t free_b=0,total_b=0;
  CUDA_CALL(cudaMemGetInfo(&free_b,&total_b));
  return double(total_b-free_b)/(1024.0*1024.0);
}

struct PhaseStats {
  double time_sum=0.0;
  double cpu_before_sum=0.0, cpu_after_sum=0.0, cpu_delta_sum=0.0;
  double gpu_before_sum=0.0, gpu_after_sum=0.0, gpu_delta_sum=0.0;
  double max_cpu_before_mb=0.0, max_cpu_after_mb=0.0, max_cpu_delta_mb=0.0;
  double max_gpu_before_mb=0.0, max_gpu_after_mb=0.0, max_gpu_delta_mb=0.0;
};

struct PhaseMark {
  double t0=0.0;
  double cpu0_mb=0.0;
  double gpu0_mb=0.0;
};

enum ProfilePhase {
  PH_PGRAD=0,
  PH_UGRAD,
  PH_UASM,
  PH_USOLVE,
  PH_VGRAD,
  PH_VASM,
  PH_VSOLVE,
  PH_WGRAD,
  PH_WASM,
  PH_WSOLVE,
  PH_PSETUP,
  PH_PREDICTOR_PHI,
  PH_CONT_PRE_P,
  PH_PSOLVE_LOOP,
  PH_FLUX_CORR_LOOP,
  PH_CONT_IN_P_LOOP,
  PH_PCORR_GRAD,
  PH_VEL_CORRECT,
  PH_COUNT
};

static const char* kProfilePhaseNames[PH_COUNT] = {
  "grad(pOld)",
  "grad(uOld)",
  "assemble(Ux)",
  "solve(Ux)",
  "grad(vOld)",
  "assemble(Uy)",
  "solve(Uy)",
  "grad(wOld)",
  "assemble(Uz)",
  "solve(Uz)",
  "pressure AMG setup",
  "predictor phi",
  "continuity before p",
  "pressure solve loop",
  "flux correction loop",
  "continuity in p loop",
  "grad(pCorrRelax)",
  "velocity correction"
};

static bool g_profile_enabled = false;
static int g_p_amg_setup_scope = 0; // 0 outer, 1 pressure-solve


static inline PhaseMark profile_begin(){
  PhaseMark m;
  if(g_profile_enabled){
    CUDA_CALL(cudaDeviceSynchronize());
    m.cpu0_mb = get_cpu_rss_mb();
    m.gpu0_mb = get_gpu_used_mb();
  }
  m.t0 = MPI_Wtime();
  return m;
}

static inline void profile_record(PhaseStats &ps, const PhaseMark &m){
  // lightweight timing is always accumulated.
  // With -profile-steps 0 this does not call cudaDeviceSynchronize(),
  // so it preserves realistic runtime while still exposing wall-time buckets.
  if(!g_profile_enabled){
    const double t1 = MPI_Wtime();
    ps.time_sum += (t1 - m.t0);
    return;
  }

  CUDA_CALL(cudaDeviceSynchronize());
  const double t1 = MPI_Wtime();
  const double cpu1 = get_cpu_rss_mb();
  const double gpu1 = get_gpu_used_mb();
  const double cpu_delta = cpu1 - m.cpu0_mb;
  const double gpu_delta = gpu1 - m.gpu0_mb;
  ps.time_sum += (t1 - m.t0);
  ps.cpu_before_sum += m.cpu0_mb;
  ps.cpu_after_sum += cpu1;
  ps.cpu_delta_sum += cpu_delta;
  ps.gpu_before_sum += m.gpu0_mb;
  ps.gpu_after_sum += gpu1;
  ps.gpu_delta_sum += gpu_delta;
  ps.max_cpu_before_mb = std::max(ps.max_cpu_before_mb, m.cpu0_mb);
  ps.max_cpu_after_mb  = std::max(ps.max_cpu_after_mb,  cpu1);
  ps.max_cpu_delta_mb  = std::max(ps.max_cpu_delta_mb,  cpu_delta);
  ps.max_gpu_before_mb = std::max(ps.max_gpu_before_mb, m.gpu0_mb);
  ps.max_gpu_after_mb  = std::max(ps.max_gpu_after_mb,  gpu1);
  ps.max_gpu_delta_mb  = std::max(ps.max_gpu_delta_mb,  gpu_delta);
}

static std::string read_file_to_string(const std::string &filename){ std::ifstream in(filename.c_str()); if(!in) throw std::runtime_error("Could not open "+filename); std::ostringstream ss; ss<<in.rdbuf(); return ss.str(); }
static std::string strip_comments(const std::string &txt){
  std::string s=txt;
  for(;;){ std::size_t a=s.find("/*"); if(a==std::string::npos) break; std::size_t b=s.find("*/",a+2); if(b==std::string::npos) break; s.erase(a,b-a+2); }
  std::stringstream in(s); std::string line,out; while(std::getline(in,line)){ std::size_t p=line.find("//"); if(p!=std::string::npos) line=line.substr(0,p); out += line; out.push_back('\n'); } return out;
}
static std::string extract_main_list(const std::string &txt){
  std::size_t startIdx=std::string::npos, endIdx=std::string::npos;
  for(std::size_t i=0;i<txt.size();++i) if(std::isdigit((unsigned char)txt[i])){ std::size_t j=i; while(j<txt.size()&&std::isdigit((unsigned char)txt[j])) ++j; while(j<txt.size()&&std::isspace((unsigned char)txt[j])) ++j; if(j<txt.size()&&txt[j]=='('){ startIdx=i; endIdx=j; break; } }
  if(startIdx==std::string::npos) throw std::runtime_error("Could not locate top-level OpenFOAM list");
  std::size_t startPos=endIdx; int depth=0; std::size_t endPos=std::string::npos;
  for(std::size_t i=startPos;i<txt.size();++i){ if(txt[i]=='(') ++depth; else if(txt[i]==')'){ --depth; if(depth==0){ endPos=i; break; } } }
  if(endPos==std::string::npos) throw std::runtime_error("Failed to match parentheses");
  return txt.substr(startPos+1,endPos-startPos-1);
}
static std::vector<std::array<double,3>> read_foam_points(const std::string &filename){
  std::string inside=extract_main_list(strip_comments(read_file_to_string(filename))); std::vector<std::array<double,3>> P; std::size_t pos=0;
  while(true){ std::size_t a=inside.find('(',pos); if(a==std::string::npos) break; std::size_t b=inside.find(')',a+1); if(b==std::string::npos) break; std::stringstream ss(inside.substr(a+1,b-a-1)); std::array<double,3> p{}; ss>>p[0]>>p[1]>>p[2]; P.push_back(p); pos=b+1; } return P;
}
static std::vector<std::vector<int>> read_foam_faces(const std::string &filename){
  std::string inside=extract_main_list(strip_comments(read_file_to_string(filename))); std::vector<std::vector<int>> faces; std::size_t pos=0;
  while(pos<inside.size()){
    while(pos<inside.size()&&std::isspace((unsigned char)inside[pos])) ++pos; if(pos>=inside.size()) break;
    if(!std::isdigit((unsigned char)inside[pos])){ ++pos; continue; }
    std::size_t q=pos; while(q<inside.size()&&std::isdigit((unsigned char)inside[q])) ++q;
    int k=std::atoi(inside.substr(pos,q-pos).c_str());
    while(q<inside.size()&&std::isspace((unsigned char)inside[q])) ++q;
    if(q>=inside.size()||inside[q]!='(') throw std::runtime_error("Malformed faces list");
    std::size_t r=inside.find(')',q+1); if(r==std::string::npos) throw std::runtime_error("Malformed faces list");
    std::stringstream ss(inside.substr(q+1,r-q-1)); std::vector<int> fv(k); for(int i=0;i<k;++i) ss>>fv[i];
    faces.push_back(fv); pos=r+1;
  }
  return faces;
}
static std::vector<int> read_foam_labels(const std::string &filename){ std::string inside=extract_main_list(strip_comments(read_file_to_string(filename))); std::stringstream ss(inside); std::vector<int> vals; int v; while(ss>>v) vals.push_back(v); return vals; }
static std::vector<PatchInfo> read_foam_boundary(const std::string &filename){
  std::string inside=extract_main_list(strip_comments(read_file_to_string(filename))); std::vector<PatchInfo> patches; std::size_t pos=0;
  while(pos<inside.size()){
    while(pos<inside.size()&&std::isspace((unsigned char)inside[pos])) ++pos; if(pos>=inside.size()) break;
    if(!(std::isalpha((unsigned char)inside[pos])||inside[pos]=='_')){ ++pos; continue; }
    std::size_t a=pos; while(pos<inside.size()&&(std::isalnum((unsigned char)inside[pos])||inside[pos]=='_')) ++pos; std::string name=inside.substr(a,pos-a);
    while(pos<inside.size()&&std::isspace((unsigned char)inside[pos])) ++pos; if(pos>=inside.size()||inside[pos]!='{') continue;
    int depth=1; std::size_t bodyStart=++pos; while(pos<inside.size()&&depth>0){ if(inside[pos]=='{') ++depth; else if(inside[pos]=='}') --depth; ++pos; }
    std::string body=inside.substr(bodyStart,pos-bodyStart-1); PatchInfo p; p.name=name;
    auto find_int=[&](const std::string &key)->int{ std::size_t k=body.find(key); if(k==std::string::npos) return 0; k+=key.size(); while(k<body.size()&&!std::isdigit((unsigned char)body[k])&&body[k]!='-') ++k; std::size_t e=k; while(e<body.size()&&(std::isdigit((unsigned char)body[e])||body[e]=='-')) ++e; return std::atoi(body.substr(k,e-k).c_str()); };
    auto find_word=[&](const std::string &key)->std::string{ std::size_t k=body.find(key); if(k==std::string::npos) return ""; k+=key.size(); while(k<body.size()&&std::isspace((unsigned char)body[k])) ++k; std::size_t e=k; while(e<body.size()&&(std::isalnum((unsigned char)body[e])||body[e]=='_')) ++e; return body.substr(k,e-k); };
    p.nFaces=find_int("nFaces"); p.startFace=find_int("startFace"); p.type=find_word("type"); patches.push_back(p);
  }
  return patches;
}
struct FaceGeomCalc {
  std::array<double,3> centre{0.0,0.0,0.0};
  std::array<double,3> areaVec{0.0,0.0,0.0};
  double area=0.0;
};

static FaceGeomCalc calc_face_geom_legacy(const std::vector<std::array<double,3>> &P,
                                          const std::vector<int> &fv){
  FaceGeomCalc fg;
  for(int v: fv) fg.centre = add3(fg.centre, P[v]);
  fg.centre = mul3(1.0/std::max((int)fv.size(),1), fg.centre);
  const auto a = P[fv[0]];
  for(std::size_t i=1; i+1<fv.size(); ++i){
    const auto b = P[fv[i]];
    const auto d = P[fv[i+1]];
    fg.areaVec = add3(fg.areaVec, mul3(0.5, cross3(sub3(b,a), sub3(d,a))));
  }
  fg.area = norm3(fg.areaVec);
  return fg;
}

static FaceGeomCalc calc_face_geom_robust(const std::vector<std::array<double,3>> &P,
                                          const std::vector<int> &fv){
  // NASA/AIAA-style polygon handling: triangulate the possibly-warped face
  // about an internal point, update that point to the area-weighted triangle
  // centroid, and repeat a few times.  This avoids using the plain vertex
  // average as the face integration point on warped/non-planar faces.
  FaceGeomCalc fg;
  for(int v: fv) fg.centre = add3(fg.centre, P[v]);
  fg.centre = mul3(1.0/std::max((int)fv.size(),1), fg.centre);

  for(int iter=0; iter<4; ++iter){
    std::array<double,3> areaVec{0.0,0.0,0.0};
    for(std::size_t i=0; i<fv.size(); ++i){
      const auto &ri = P[fv[i]];
      const auto &rj = P[fv[(i+1)%fv.size()]];
      areaVec = add3(areaVec, mul3(0.5, cross3(sub3(ri,fg.centre), sub3(rj,fg.centre))));
    }
    const double amag = norm3(areaVec);
    if(amag <= 1e-300) break;
    const auto nhat = mul3(1.0/amag, areaVec);

    std::array<double,3> csum{0.0,0.0,0.0};
    double asum = 0.0;
    for(std::size_t i=0; i<fv.size(); ++i){
      const auto &ri = P[fv[i]];
      const auto &rj = P[fv[(i+1)%fv.size()]];
      const auto avec = mul3(0.5, cross3(sub3(ri,fg.centre), sub3(rj,fg.centre)));
      const double aSigned = dot3(avec, nhat);
      const auto ctri = mul3(1.0/3.0, add3(add3(ri,rj), fg.centre));
      csum = add3(csum, mul3(aSigned, ctri));
      asum += aSigned;
    }
    if(std::fabs(asum) <= 1e-300) break;
    fg.centre = mul3(1.0/asum, csum);
  }

  for(std::size_t i=0; i<fv.size(); ++i){
    const auto &ri = P[fv[i]];
    const auto &rj = P[fv[(i+1)%fv.size()]];
    fg.areaVec = add3(fg.areaVec, mul3(0.5, cross3(sub3(ri,fg.centre), sub3(rj,fg.centre))));
  }
  fg.area = norm3(fg.areaVec);
  return fg;
}

static Mesh read_openfoam_polymesh(const std::string &polyMeshDir, int geomMethod){
  Mesh mesh;
  auto patches = read_foam_boundary(polyMeshDir+"/boundary");
  mesh.P = read_foam_points(polyMeshDir+"/points");
  mesh.faces = read_foam_faces(polyMeshDir+"/faces");
  std::vector<int> owner0 = read_foam_labels(polyMeshDir+"/owner");
  std::vector<int> neigh0 = read_foam_labels(polyMeshDir+"/neighbour");

  mesh.nFaces=(int)mesh.faces.size();
  mesh.nInternalFaces=(int)neigh0.size();
  mesh.nCells=0;
  for(int v:owner0) mesh.nCells=std::max(mesh.nCells,v+1);
  for(int v:neigh0) mesh.nCells=std::max(mesh.nCells,v+1);
  mesh.owner.resize(mesh.nFaces);
  mesh.neigh.assign(mesh.nInternalFaces,0);
  for(int i=0;i<mesh.nFaces;++i) mesh.owner[i]=owner0[i];
  for(int i=0;i<mesh.nInternalFaces;++i) mesh.neigh[i]=neigh0[i];

  mesh.bPatch.assign(mesh.nFaces,0);
  mesh.patchNames.resize(patches.size());
  for(std::size_t k=0;k<patches.size();++k){
    mesh.patchNames[k]=patches[k].name;
    for(int f=patches[k].startFace; f<patches[k].startFace+patches[k].nFaces; ++f) mesh.bPatch[f]=(int)k+1;
  }

  mesh.cellFaces.assign(mesh.nCells,{});
  mesh.cellOrient.assign(mesh.nCells,{});
  for(int f=0;f<mesh.nFaces;++f){ int P=mesh.owner[f]; mesh.cellFaces[P].push_back(f); mesh.cellOrient[P].push_back(+1); }
  for(int f=0;f<mesh.nInternalFaces;++f){ int N=mesh.neigh[f]; mesh.cellFaces[N].push_back(f); mesh.cellOrient[N].push_back(-1); }

  // Face geometry first; robust mode uses area-weighted triangulated face centres.
  mesh.xf.assign(mesh.nFaces,{0,0,0});
  mesh.Af.assign(mesh.nFaces,0.0);
  mesh.nf.assign(mesh.nFaces,{0,0,0});
  mesh.Sf.assign(mesh.nFaces,{0,0,0});
  for(int f=0; f<mesh.nFaces; ++f){
    FaceGeomCalc fg = (geomMethod == 0)
        ? calc_face_geom_legacy(mesh.P, mesh.faces[f])
        : calc_face_geom_robust(mesh.P, mesh.faces[f]);
    if(fg.area <= 1e-30) throw std::runtime_error("Degenerate face area at face "+std::to_string(f));
    mesh.xf[f] = fg.centre;
    mesh.Af[f] = fg.area;
    mesh.nf[f] = mul3(1.0/fg.area, fg.areaVec);
    mesh.Sf[f] = fg.areaVec;
  }

  mesh.cc.assign(mesh.nCells,{0,0,0});
  mesh.vol.assign(mesh.nCells,0.0);

  if(geomMethod == 0){
    // true legacy/v1 cell volume and centroid construction: first-vertex fan
    // on each oriented face about a vertex-average cell point.
    for(int c=0;c<mesh.nCells;++c){
      std::set<int> vertsSet;
      for(int f:mesh.cellFaces[c]) for(int v:mesh.faces[f]) vertsSet.insert(v);
      std::array<double,3> c0{0,0,0};
      for(int v:vertsSet) c0=add3(c0,mesh.P[v]);
      c0=mul3(1.0/std::max((int)vertsSet.size(),1),c0);
      double V=0.0;
      std::array<double,3> M{0,0,0};
      for(std::size_t j=0;j<mesh.cellFaces[c].size();++j){
        int f=mesh.cellFaces[c][j];
        int ori=mesh.cellOrient[c][j];
        std::vector<int> fv=mesh.faces[f];
        if(ori<0) std::reverse(fv.begin(),fv.end());
        auto a=mesh.P[fv[0]];
        for(std::size_t i=1;i+1<fv.size();++i){
          auto b=mesh.P[fv[i]], d=mesh.P[fv[i+1]];
          double vTet=dot3(sub3(a,c0),cross3(sub3(b,c0),sub3(d,c0)))/6.0;
          auto cTet=mul3(0.25,add3(add3(c0,a),add3(b,d)));
          V += vTet;
          M = add3(M,mul3(vTet,cTet));
        }
      }
      if(V<=0) throw std::runtime_error("Non-positive cell volume at cell "+std::to_string(c));
      mesh.vol[c]=V;
      mesh.cc[c]=mul3(1.0/V,M);
    }
  } else {
    // Robust cell volume and centroid by tetrahedralizing each oriented face
    // about its own face centre.
    for(int c=0;c<mesh.nCells;++c){
      std::set<int> vertsSet;
      for(int f:mesh.cellFaces[c]) for(int v:mesh.faces[f]) vertsSet.insert(v);
      std::array<double,3> c0{0,0,0};
      for(int v:vertsSet) c0=add3(c0,mesh.P[v]);
      c0=mul3(1.0/std::max((int)vertsSet.size(),1),c0);

      double V=0.0;
      std::array<double,3> M{0,0,0};
      for(std::size_t jf=0; jf<mesh.cellFaces[c].size(); ++jf){
        const int f = mesh.cellFaces[c][jf];
        const int ori = mesh.cellOrient[c][jf];
        std::vector<int> fv = mesh.faces[f];
        if(ori<0) std::reverse(fv.begin(), fv.end());
        const auto fc = mesh.xf[f];
        for(std::size_t i=0; i<fv.size(); ++i){
          const auto b = mesh.P[fv[i]];
          const auto d = mesh.P[fv[(i+1)%fv.size()]];
          const double vTet = dot3(sub3(fc,c0), cross3(sub3(b,c0), sub3(d,c0)))/6.0;
          const auto cTet = mul3(0.25, add3(add3(c0,fc), add3(b,d)));
          V += vTet;
          M = add3(M, mul3(vTet,cTet));
        }
      }
      if(V<=0) throw std::runtime_error("Non-positive cell volume at cell "+std::to_string(c));
      mesh.vol[c]=V;
      mesh.cc[c]=mul3(1.0/V,M);
    }
  }

  // Orient face normals from owner to neighbour / boundary face centre after
  // cell centres are available.  Keep Sf consistent with nf and Af.
  for(int f=0; f<mesh.nFaces; ++f){
    int P=mesh.owner[f];
    std::array<double,3> dtest;
    if(f<mesh.nInternalFaces) dtest=sub3(mesh.cc[mesh.neigh[f]],mesh.cc[P]);
    else dtest=sub3(mesh.xf[f],mesh.cc[P]);
    if(dot3(mesh.nf[f],dtest)<0){
      mesh.nf[f]=mul3(-1.0,mesh.nf[f]);
      mesh.Sf[f]=mul3(-1.0,mesh.Sf[f]);
    }
  }

  mesh.cellNbrs.assign(mesh.nCells,{});
  mesh.cellBFace.assign(mesh.nCells,{});
  for(int f=0;f<mesh.nFaces;++f){
    int P=mesh.owner[f];
    if(f<mesh.nInternalFaces){ int N=mesh.neigh[f]; mesh.cellNbrs[P].push_back(N); mesh.cellNbrs[N].push_back(P); }
    else mesh.cellBFace[P].push_back(f);
  }
  for(int c=0;c<mesh.nCells;++c){
    std::sort(mesh.cellNbrs[c].begin(),mesh.cellNbrs[c].end());
    mesh.cellNbrs[c].erase(std::unique(mesh.cellNbrs[c].begin(),mesh.cellNbrs[c].end()),mesh.cellNbrs[c].end());
  }

  mesh.maxNonOrthDeg=0.0;
  for(int f=0;f<mesh.nInternalFaces;++f){
    auto d=sub3(mesh.cc[mesh.neigh[f]],mesh.cc[mesh.owner[f]]);
    double cosang=std::fabs(dot3(d,mesh.nf[f]))/std::max(norm3(d),1e-30);
    cosang=std::min(1.0,std::max(0.0,cosang));
    mesh.maxNonOrthDeg=std::max(mesh.maxNonOrthDeg, std::acos(cosang)*180.0/M_PI);
  }
  return mesh;
}


static void compute_lsq_gradient(const Mesh &mesh,const std::vector<cfdScalar> &phi,const std::vector<std::string> &bcType,const std::vector<cfdScalar> &bcFaceValue,std::vector<cfdVector3> &grad){
  grad.assign(mesh.nCells,{0.0,0.0,0.0});
  for(int P=0;P<mesh.nCells;++P){
    auto xP=mesh.cc[P]; double phiP=phi[P];
    double M[3][3]={{0,0,0},{0,0,0},{0,0,0}}; double rhs[3]={0,0,0};
    for(int N:mesh.cellNbrs[P]){ auto r=sub3(mesh.cc[N],xP); double dphi=phi[N]-phiP; double w=1.0/std::max(dot3(r,r),1e-30); for(int i=0;i<3;++i){ for(int j=0;j<3;++j) M[i][j]+=w*r[i]*r[j]; rhs[i]+=w*r[i]*dphi; } }
    for(int f:mesh.cellBFace[P]){ int patch=mesh.bPatch[f]-1; auto r=sub3(mesh.xf[f],xP); double phiB=(bcType[patch]=="Dirichlet")?bcFaceValue[f]:phiP; double dphi=phiB-phiP; double w=1.0/std::max(dot3(r,r),1e-30); for(int i=0;i<3;++i){ for(int j=0;j<3;++j) M[i][j]+=w*r[i]*r[j]; rhs[i]+=w*r[i]*dphi; } }
    double a=M[0][0], b=M[0][1], c=M[0][2], d=M[1][0], e=M[1][1], f=M[1][2], g=M[2][0], h=M[2][1], k=M[2][2];
    double det=a*(e*k-f*h)-b*(d*k-f*g)+c*(d*h-e*g);
    if(std::fabs(det)>1e-20){
      double inv[3][3];
      inv[0][0]=(e*k-f*h)/det; inv[0][1]=(c*h-b*k)/det; inv[0][2]=(b*f-c*e)/det;
      inv[1][0]=(f*g-d*k)/det; inv[1][1]=(a*k-c*g)/det; inv[1][2]=(c*d-a*f)/det;
      inv[2][0]=(d*h-e*g)/det; inv[2][1]=(b*g-a*h)/det; inv[2][2]=(a*e-b*d)/det;
      grad[P] = cfdVector3{
        static_cast<cfdScalar>(inv[0][0]*rhs[0]+inv[0][1]*rhs[1]+inv[0][2]*rhs[2]),
        static_cast<cfdScalar>(inv[1][0]*rhs[0]+inv[1][1]*rhs[1]+inv[1][2]*rhs[2]),
        static_cast<cfdScalar>(inv[2][0]*rhs[0]+inv[2][1]*rhs[1]+inv[2][2]*rhs[2])
      };
    }
  }
}

static void predictor_face_fluxes(const Mesh &mesh,const std::vector<cfdScalar> &u,const std::vector<cfdScalar> &v,const std::vector<cfdScalar> &w,const std::vector<std::string> &patchUType,const std::vector<cfdScalar> &uFaceBC,const std::vector<std::string> &patchVType,const std::vector<cfdScalar> &vFaceBC,const std::vector<std::string> &patchWType,const std::vector<cfdScalar> &wFaceBC,double rho,std::vector<cfdScalar> &phi){
  phi.assign(mesh.nFaces,0.0);
  for(int f=0;f<mesh.nFaces;++f){
    int P=mesh.owner[f];
    double uf=0.0,vf=0.0,wf=0.0;
    if(f<mesh.nInternalFaces){
      int N=mesh.neigh[f];
      auto d=sub3(mesh.cc[N],mesh.cc[P]);
      auto dx=sub3(mesh.xf[f],mesh.cc[P]);
      double lam=dot3(dx,d)/std::max(dot3(d,d),1e-30);
      lam=std::min(1.0,std::max(0.0,lam));
      uf=(1.0-lam)*u[P]+lam*u[N];
      vf=(1.0-lam)*v[P]+lam*v[N];
      wf=(1.0-lam)*w[P]+lam*w[N];
    } else {
      int patch=mesh.bPatch[f]-1;
      uf=(patchUType[patch]=="Dirichlet")?uFaceBC[f]:u[P];
      vf=(patchVType[patch]=="Dirichlet")?vFaceBC[f]:v[P];
      wf=(patchWType[patch]=="Dirichlet")?wFaceBC[f]:w[P];
    }
    phi[f]=rho*mesh.Af[f]*(uf*mesh.nf[f][0]+vf*mesh.nf[f][1]+wf*mesh.nf[f][2]);
  }
}

static void build_rhiechow_predicted_flux_stokes_3d(
    const Mesh &mesh,
    const std::vector<cfdScalar> &u,
    const std::vector<cfdScalar> &v,
    const std::vector<cfdScalar> &w,
    const std::vector<cfdScalar> &p,
    const std::vector<cfdVector3> &gradP,
    const std::vector<cfdScalar> &rAU,
    const std::vector<std::string> &bcUType,
    const std::vector<cfdScalar> &uFaceBC,
    const std::vector<std::string> &bcVType,
    const std::vector<cfdScalar> &vFaceBC,
    const std::vector<std::string> &bcWType,
    const std::vector<cfdScalar> &wFaceBC,
    double rho,
    std::vector<cfdScalar> &phiStar)
{
  phiStar.assign(mesh.nFaces, 0.0);

  for (int f = 0; f < mesh.nFaces; ++f)
  {
    const int P = mesh.owner[f];

    if (f < mesh.nInternalFaces)
    {
      const int N = mesh.neigh[f];

      auto d = sub3(mesh.cc[N], mesh.cc[P]);
      const double dpn = dot3(mesh.nf[f], d);
      const double denom = std::max(dot3(d, d), 1.0e-30);

      double lam = dot3(sub3(mesh.xf[f], mesh.cc[P]), d) / denom;
      lam = std::min(1.0, std::max(0.0, lam));

      const double ubar = (1.0 - lam) * u[P] + lam * u[N];
      const double vbar = (1.0 - lam) * v[P] + lam * v[N];
      const double wbar = (1.0 - lam) * w[P] + lam * w[N];

      std::array<double,3> gradpbar{
        (1.0 - lam) * gradP[P][0] + lam * gradP[N][0],
        (1.0 - lam) * gradP[P][1] + lam * gradP[N][1],
        (1.0 - lam) * gradP[P][2] + lam * gradP[N][2]
      };

      const double rAUf = (1.0 - lam) * rAU[P] + lam * rAU[N];

      const double phiInterp =
        rho * mesh.Af[f] *
        (ubar * mesh.nf[f][0] + vbar * mesh.nf[f][1] + wbar * mesh.nf[f][2]);

      // Rhie-Chow, linear-pressure-preserving
      const double rc =
        rho * mesh.Af[f] * rAUf / std::max(std::fabs(dpn), 1.0e-30) *
        ((p[N] - p[P]) - dot3(gradpbar, d));

      phiStar[f] = phiInterp - rc;
    }
    else
    {
      const int patch = mesh.bPatch[f] - 1;

      const double uf = (bcUType[patch] == "Dirichlet") ? uFaceBC[f] : u[P];
      const double vf = (bcVType[patch] == "Dirichlet") ? vFaceBC[f] : v[P];
      const double wf = (bcWType[patch] == "Dirichlet") ? wFaceBC[f] : w[P];

      // Same boundary treatment as the 2D reference:
      // no Rhie-Chow term on boundary faces
      phiStar[f] =
        rho * mesh.Af[f] *
        (uf * mesh.nf[f][0] + vf * mesh.nf[f][1] + wf * mesh.nf[f][2]);
    }
  }
}

static void continuity_residual(const Mesh &mesh,const std::vector<cfdScalar> &phi,std::vector<cfdScalar> &divCell){
  divCell.assign(mesh.nCells,0.0);
  for(int f=0;f<mesh.nFaces;++f){
    int P=mesh.owner[f];
    divCell[P]+=phi[f];
    if(f<mesh.nInternalFaces){
      int N=mesh.neigh[f];
      divCell[N]-=phi[f];
    }
  }
}
static void correct_face_fluxes_simple(const Mesh &mesh,const std::vector<cfdScalar> &rAU,const std::vector<cfdScalar> &phiStar,const std::vector<cfdScalar> &pCorr,std::vector<cfdScalar> &phi){
  phi=phiStar;
  for(int f=0;f<mesh.nInternalFaces;++f){
    int P=mesh.owner[f], N=mesh.neigh[f];
    auto d=sub3(mesh.cc[N],mesh.cc[P]);
    double dpn=dot3(mesh.nf[f],d);
    double denom = dot3(d,d);
    double lam = dot3(sub3(mesh.xf[f], mesh.cc[P]), d) / std::max(denom,1e-30);
    lam = std::min(1.0,std::max(0.0,lam));
    double rAUf = (1.0-lam)*rAU[P] + lam*rAU[N];
    double coeff=mesh.Af[f]*rAUf/dpn;
    phi[f]=phiStar[f]-coeff*(pCorr[N]-pCorr[P]);
  }
}

static void pressure_nonorth_flux_and_divergence(const Mesh &mesh,
                                                 const std::vector<cfdScalar> &rAU,
                                                 const std::vector<std::string> &bcPType,
                                                 const std::vector<cfdVector3> &gradP,
                                                 std::vector<cfdScalar> &phiNonOrth,
                                                 std::vector<cfdScalar> &divNonOrth)
{
  phiNonOrth.assign(mesh.nFaces, 0.0);
  divNonOrth.assign(mesh.nCells, 0.0);

  for(int f=0; f<mesh.nInternalFaces; ++f){
    int P = mesh.owner[f], N = mesh.neigh[f];
    auto d = sub3(mesh.cc[N], mesh.cc[P]);
    double dpn = dot3(mesh.nf[f], d);
    if(std::fabs(dpn) <= 1e-30) continue;
    double denom = dot3(d,d);
    double lam = dot3(sub3(mesh.xf[f], mesh.cc[P]), d) / std::max(denom,1e-30);
    lam = std::min(1.0,std::max(0.0,lam));
    double rAUf = (1.0-lam)*rAU[P] + lam*rAU[N];
    std::array<double,3> gradf{
      (1.0-lam)*gradP[P][0] + lam*gradP[N][0],
      (1.0-lam)*gradP[P][1] + lam*gradP[N][1],
      (1.0-lam)*gradP[P][2] + lam*gradP[N][2]
    };
    std::array<double,3> t = sub3(mesh.Sf[f], mul3(mesh.Af[f]/dpn, d));
    double flux = rAUf * dot3(gradf, t);
    phiNonOrth[f] = flux;
    divNonOrth[P] += flux;
    divNonOrth[N] -= flux;
  }

  for(int f=mesh.nInternalFaces; f<mesh.nFaces; ++f){
    int patch = mesh.bPatch[f] - 1;
    if (patch < 0 || bcPType[patch] != "Dirichlet") continue;

    int P = mesh.owner[f];
    auto d = sub3(mesh.xf[f], mesh.cc[P]);
    double dpn = dot3(mesh.nf[f], d);
    if(std::fabs(dpn) <= 1e-30) continue;

    std::array<double,3> t = sub3(mesh.Sf[f], mul3(mesh.Af[f]/dpn, d));
    double flux = rAU[P] * (static_cast<double>(gradP[P][0])*t[0] + static_cast<double>(gradP[P][1])*t[1] + static_cast<double>(gradP[P][2])*t[2]);
    phiNonOrth[f] = flux;
    divNonOrth[P] += flux;
  }
}

static void correct_face_fluxes_simple_nonorth(const Mesh &mesh,
                                               const std::vector<cfdScalar> &rAU,
                                               const std::vector<std::string> &bcPType,
                                               const std::vector<cfdScalar> &pFaceBC,
                                               const std::vector<cfdScalar> &phiStar,
                                               const std::vector<cfdScalar> &pCorr,
                                               const std::vector<cfdVector3> &gradP,
                                               std::vector<cfdScalar> &phi)
{
  std::vector<cfdScalar> phiNonOrth, divDummy;
  pressure_nonorth_flux_and_divergence(mesh, rAU, bcPType, gradP, phiNonOrth, divDummy);
  phi = phiStar;

  for(int f=0; f<mesh.nInternalFaces; ++f){
    int P=mesh.owner[f], N=mesh.neigh[f];
    auto d=sub3(mesh.cc[N],mesh.cc[P]);
    double dpn=dot3(mesh.nf[f],d);
    double denom = dot3(d,d);
    double lam = dot3(sub3(mesh.xf[f], mesh.cc[P]), d) / std::max(denom,1e-30);
    lam = std::min(1.0,std::max(0.0,lam));
    double rAUf = (1.0-lam)*rAU[P] + lam*rAU[N];
    double coeff=mesh.Af[f]*rAUf/dpn;
    phi[f]=phiStar[f]-coeff*(pCorr[N]-pCorr[P]) - phiNonOrth[f];
  }

  for(int f=mesh.nInternalFaces; f<mesh.nFaces; ++f){
    int patch = mesh.bPatch[f] - 1;
    if (patch < 0 || bcPType[patch] != "Dirichlet") continue;

    int P = mesh.owner[f];
    auto d = sub3(mesh.xf[f], mesh.cc[P]);
    double dpn = dot3(mesh.nf[f], d);
    if(std::fabs(dpn) <= 1e-30) continue;

    double coeff = mesh.Af[f]*rAU[P]/dpn;
    const double pB = pFaceBC[f];
    phi[f] = phiStar[f] - coeff*(pB - pCorr[P]) - phiNonOrth[f];
  }
}


static void print_patch_geometry_summary(const pipebc::PatchGeometrySummary& patch){
  std::printf("Patch summary [%s]\n", patch.patchName.c_str());
  std::printf("  faces      : %zu\n", patch.faces.size());
  std::printf("  area       : %.12e\n", patch.totalArea);
  std::printf("  centroid   : [%.6e, %.6e, %.6e]\n",
              patch.centroid[0], patch.centroid[1], patch.centroid[2]);
  std::printf("  avg normal : [%.6e, %.6e, %.6e]\n",
              patch.averageNormal[0], patch.averageNormal[1], patch.averageNormal[2]);
  std::printf("  planarity  : %.6f\n", patch.planarity);
}


struct CylinderForceReport {
  bool valid = false;
  int patchIndex = -1;
  std::string patchName;
  int nFaces = 0;
  double area = 0.0;

  double FD_pressure = 0.0;
  double FD_viscous  = 0.0;
  double FD_total    = 0.0;

  double FL_pressure = 0.0;
  double FL_viscous  = 0.0;
  double FL_total    = 0.0;

  double CD = 0.0;
  double CL = 0.0;

  double rho = 0.0;
  double mu = 0.0;
  double Ubar = 0.0;
  double D = 0.0;
  double H = 0.0;
  double coeffDenom = 0.0;

  double maxWallDistance = 0.0;
  double minWallDistance = 1.0e300;
  double maxAbsDvtDn = 0.0;
};

// DFG/Schaefer-Turek-style cylinder force postprocess.
// Uses only patch faces belonging to the separated cylinder patch.
// Formula:
//   F_D = int_S (mu * d(v_t)/dn * n_y - p*n_x) dS
//   F_L = -int_S (mu * d(v_t)/dn * n_x + p*n_y) dS
//
// Here mesh.nf[f] points from the fluid owner cell toward the boundary face.
// For an immersed solid/cylinder boundary this is opposite to the paper's
// cylinder normal, so we use n = -mesh.nf[f].
static CylinderForceReport compute_cylinder_forces_paper(
    const Mesh& mesh,
    int cylinderPatch,
    const std::vector<cfdScalar>& u,
    const std::vector<cfdScalar>& v,
    const std::vector<cfdScalar>& w,
    const std::vector<cfdScalar>& p,
    double rho,
    double mu,
    double D,
    double H,
    double Ubar)
{
  CylinderForceReport r;
  if(cylinderPatch < 0 || cylinderPatch >= (int)mesh.patchNames.size()) return r;

  r.valid = true;
  r.patchIndex = cylinderPatch;
  r.patchName = mesh.patchNames[cylinderPatch];
  r.rho = rho;
  r.mu = mu;
  r.D = D;
  r.H = H;
  r.Ubar = Ubar;
  r.coeffDenom = rho * Ubar * Ubar * D * H;

  const int f0 = mesh.patchStartFace[cylinderPatch];
  const int f1 = f0 + mesh.patchNFaces[cylinderPatch];

  for(int f = f0; f < f1; ++f){
    const int P = mesh.owner[f];
    const double area = mesh.Af[f];
    if(area <= 1.0e-300) continue;

    // Paper normal: cylinder/solid -> fluid.
    std::array<double,3> n{{-mesh.nf[f][0], -mesh.nf[f][1], -mesh.nf[f][2]}};
    const double nmag = norm3(n);
    if(nmag <= 1.0e-300) continue;
    n = mul3(1.0/nmag, n);

    const double nx = n[0];
    const double ny = n[1];

    // Tangent used in the paper: t = (n_y, -n_x, 0).
    // Normalize for robustness on triangulated 3D cylinder surfaces.
    std::array<double,3> t{{ny, -nx, 0.0}};
    const double tmag = norm3(t);
    if(tmag <= 1.0e-14) continue;
    t = mul3(1.0/tmag, t);

    const double vt = u[P]*t[0] + v[P]*t[1] + w[P]*t[2];

    // Distance from wall face center to owner cell center in the cylinder-normal direction.
    const std::array<double,3> dx = sub3(mesh.cc[P], mesh.xf[f]);
    double dn = dot3(dx, n);
    if(dn <= 1.0e-14) dn = std::fabs(dn);
    if(dn <= 1.0e-14) dn = norm3(dx);
    if(dn <= 1.0e-14) continue;

    const double dvt_dn = vt / dn;
    const double pf = p[P];

    const double fd_p = (-pf * nx) * area;
    const double fd_v = ( mu * dvt_dn * ny) * area;

    const double fl_p = (-pf * ny) * area;
    const double fl_v = (-mu * dvt_dn * nx) * area;

    r.FD_pressure += fd_p;
    r.FD_viscous  += fd_v;
    r.FL_pressure += fl_p;
    r.FL_viscous  += fl_v;
    r.area += area;
    r.nFaces += 1;

    r.minWallDistance = std::min(r.minWallDistance, dn);
    r.maxWallDistance = std::max(r.maxWallDistance, dn);
    r.maxAbsDvtDn = std::max(r.maxAbsDvtDn, std::fabs(dvt_dn));
  }

  r.FD_total = r.FD_pressure + r.FD_viscous;
  r.FL_total = r.FL_pressure + r.FL_viscous;

  if(r.coeffDenom > 1.0e-300){
    r.CD = 2.0 * r.FD_total / r.coeffDenom;
    r.CL = 2.0 * r.FL_total / r.coeffDenom;
  }

  if(r.minWallDistance == 1.0e300) r.minWallDistance = 0.0;
  return r;
}



static void write_runtime_velocity_bc_patch_vtp(
    const Mesh& mesh,
    const std::string& patchName,
    const std::vector<std::string>& bcUType,
    const std::vector<cfdScalar>& uFaceBC,
    const std::vector<std::string>& bcVType,
    const std::vector<cfdScalar>& vFaceBC,
    const std::vector<std::string>& bcWType,
    const std::vector<cfdScalar>& wFaceBC,
    double rho,
    const std::string& filename)
{
  const int patch = find_patch_index_local(mesh, patchName);
  if(patch < 0){
    std::printf("Runtime velocity BC VTP skipped: patch '%s' not found.\n",
                patchName.c_str());
    return;
  }

  const int f0 = mesh.patchStartFace[patch];
  const int nf = mesh.patchNFaces[patch];
  const int f1 = f0 + nf;

  if(nf <= 0){
    std::printf("Runtime velocity BC VTP skipped: patch '%s' has zero faces.\n",
                patchName.c_str());
    return;
  }

  std::vector<int> pointMap(mesh.P.size(), -1);
  std::vector<cfdVector3> outPts;
  std::vector<long long> conn;
  std::vector<long long> offsets;

  std::vector<cfdVector3> Ubc;
  std::vector<cfdScalar> Umag, UnOut, inwardSpeed, flux, massFlux;
  std::vector<int> isDirichlet;

  outPts.reserve((std::size_t)nf * 4);
  Ubc.reserve(nf);
  Umag.reserve(nf);
  UnOut.reserve(nf);
  inwardSpeed.reserve(nf);
  flux.reserve(nf);
  massFlux.reserve(nf);
  isDirichlet.reserve(nf);

  long long off = 0;

  double area = 0.0;
  double sumFlux = 0.0;
  double sumMassFlux = 0.0;
  int outwardFaces = 0;

  double minUx=1.0e300, maxUx=-1.0e300, sumUx=0.0;
  double minUy=1.0e300, maxUy=-1.0e300, sumUy=0.0;
  double minUz=1.0e300, maxUz=-1.0e300, sumUz=0.0;
  double minUm=1.0e300, maxUm=-1.0e300, sumUm=0.0;
  double minUn=1.0e300, maxUn=-1.0e300, sumUn=0.0;
  double minIn=1.0e300, maxIn=-1.0e300, sumIn=0.0;

  for(int f = f0; f < f1; ++f){
    const int dir =
      (bcUType[patch] == "Dirichlet" &&
       bcVType[patch] == "Dirichlet" &&
       bcWType[patch] == "Dirichlet") ? 1 : 0;

    const double ux = uFaceBC[f];
    const double uy = vFaceBC[f];
    const double uz = wFaceBC[f];

    const double un =
      ux*mesh.nf[f][0] +
      uy*mesh.nf[f][1] +
      uz*mesh.nf[f][2];

    const double q = mesh.Af[f] * un;
    const double mq = rho * q;
    const double um = std::sqrt(ux*ux + uy*uy + uz*uz);
    const double inSpeed = -un;

    area += mesh.Af[f];
    sumFlux += q;
    sumMassFlux += mq;
    if(q > 1.0e-14) ++outwardFaces;

    minUx = std::min(minUx, ux); maxUx = std::max(maxUx, ux); sumUx += ux;
    minUy = std::min(minUy, uy); maxUy = std::max(maxUy, uy); sumUy += uy;
    minUz = std::min(minUz, uz); maxUz = std::max(maxUz, uz); sumUz += uz;
    minUm = std::min(minUm, um); maxUm = std::max(maxUm, um); sumUm += um;
    minUn = std::min(minUn, un); maxUn = std::max(maxUn, un); sumUn += un;
    minIn = std::min(minIn, inSpeed); maxIn = std::max(maxIn, inSpeed); sumIn += inSpeed;

    Ubc.push_back(cfdVector3{
      static_cast<cfdScalar>(ux),
      static_cast<cfdScalar>(uy),
      static_cast<cfdScalar>(uz)
    });
    Umag.push_back(um);
    UnOut.push_back(un);
    inwardSpeed.push_back(inSpeed);
    flux.push_back(q);
    massFlux.push_back(mq);
    isDirichlet.push_back(dir);

    for(int p : mesh.faces[f]){
      if(pointMap[p] < 0){
        pointMap[p] = (int)outPts.size();
        outPts.push_back(cfdVector3{
          static_cast<cfdScalar>(mesh.P[p][0]),
          static_cast<cfdScalar>(mesh.P[p][1]),
          static_cast<cfdScalar>(mesh.P[p][2])
        });
      }
      conn.push_back((long long)pointMap[p]);
      ++off;
    }
    offsets.push_back(off);
  }

  auto avg = [&](double x){ return x / std::max(nf, 1); };

  std::printf("Runtime velocity BC patch audit [%s]\n", patchName.c_str());
  std::printf("  file                  : %s\n", filename.c_str());
  std::printf("  faces                 : %d\n", nf);
  std::printf("  area                  : %.15e\n", area);
  std::printf("  Ux min/max/avg         : %.15e %.15e %.15e\n", minUx, maxUx, avg(sumUx));
  std::printf("  Uy min/max/avg         : %.15e %.15e %.15e\n", minUy, maxUy, avg(sumUy));
  std::printf("  Uz min/max/avg         : %.15e %.15e %.15e\n", minUz, maxUz, avg(sumUz));
  std::printf("  |U| min/max/avg        : %.15e %.15e %.15e\n", minUm, maxUm, avg(sumUm));
  std::printf("  Un_out min/max/avg     : %.15e %.15e %.15e\n", minUn, maxUn, avg(sumUn));
  std::printf("  inward speed min/max/avg: %.15e %.15e %.15e\n", minIn, maxIn, avg(sumIn));
  std::printf("  sum volumetric flux    : %.15e\n", sumFlux);
  std::printf("  inward flow rate       : %.15e\n", -sumFlux);
  std::printf("  sum mass flux          : %.15e\n", sumMassFlux);
  std::printf("  faces with outward flux: %d\n", outwardFaces);

  {
    std::string csv = filename;
    const std::size_t dot = csv.find_last_of('.');
    if(dot != std::string::npos) csv = csv.substr(0, dot);
    csv += ".csv";

    std::ofstream c(csv.c_str());
    c << "localFace,globalFace,isDirichlet,Ux,Uy,Uz,Umag,Un_out,inwardSpeed,flux,massFlux,Af,nx,ny,nz\n";
    for(int i=0; i<nf; ++i){
      const int f = f0 + i;
      c << i << "," << f << "," << isDirichlet[i] << ","
        << std::setprecision(17)
        << Ubc[i][0] << "," << Ubc[i][1] << "," << Ubc[i][2] << ","
        << Umag[i] << "," << UnOut[i] << "," << inwardSpeed[i] << ","
        << flux[i] << "," << massFlux[i] << ","
        << mesh.Af[f] << ","
        << mesh.nf[f][0] << "," << mesh.nf[f][1] << "," << mesh.nf[f][2] << "\n";
    }
    std::printf("  csv                   : %s\n", csv.c_str());
  }

  std::ofstream out(filename.c_str());
  if(!out){
    std::printf("WARNING: Could not write runtime velocity BC VTP: %s\n", filename.c_str());
    return;
  }

  out << "<?xml version=\"1.0\"?>\n";
  out << "<VTKFile type=\"PolyData\" version=\"1.0\" byte_order=\"LittleEndian\">\n";
  out << "  <PolyData>\n";
  out << "    <Piece NumberOfPoints=\"" << outPts.size()
      << "\" NumberOfPolys=\"" << nf << "\">\n";

  out << "      <Points>\n";
  out << "        <DataArray type=\"Float64\" NumberOfComponents=\"3\" format=\"ascii\">\n";
  for(const auto& p : outPts){
    out << "          " << std::setprecision(17)
        << p[0] << " " << p[1] << " " << p[2] << "\n";
  }
  out << "        </DataArray>\n";
  out << "      </Points>\n";

  out << "      <Polys>\n";
  out << "        <DataArray type=\"Int64\" Name=\"connectivity\" format=\"ascii\">\n          ";
  for(long long x : conn) out << x << " ";
  out << "\n        </DataArray>\n";
  out << "        <DataArray type=\"Int64\" Name=\"offsets\" format=\"ascii\">\n          ";
  for(long long x : offsets) out << x << " ";
  out << "\n        </DataArray>\n";
  out << "      </Polys>\n";

  out << "      <CellData>\n";

  out << "        <DataArray type=\"Int32\" Name=\"isDirichlet\" NumberOfComponents=\"1\" format=\"ascii\">\n";
  for(int x : isDirichlet) out << "          " << x << "\n";
  out << "        </DataArray>\n";

  out << "        <DataArray type=\"Float64\" Name=\"U_bc\" NumberOfComponents=\"3\" format=\"ascii\">\n";
  for(const auto& U : Ubc){
    out << "          " << std::setprecision(17)
        << U[0] << " " << U[1] << " " << U[2] << "\n";
  }
  out << "        </DataArray>\n";

  auto write_scalar = [&](const char* name, const std::vector<cfdScalar>& a){
    out << "        <DataArray type=\"Float64\" Name=\"" << name
        << "\" NumberOfComponents=\"1\" format=\"ascii\">\n";
    for(double x : a) out << "          " << std::setprecision(17) << x << "\n";
    out << "        </DataArray>\n";
  };

  write_scalar("Umag", Umag);
  write_scalar("Un_out", UnOut);
  write_scalar("inwardSpeed", inwardSpeed);
  write_scalar("flux", flux);
  write_scalar("massFlux", massFlux);

  out << "      </CellData>\n";
  out << "    </Piece>\n";
  out << "  </PolyData>\n";
  out << "</VTKFile>\n";
}

static void write_vtu_polyhedron_cell_data(const std::string &filename,const Mesh &mesh,const std::vector<std::string> &scalarNames,const std::vector<std::vector<cfdScalar>> &scalarData,const std::string &vecName,const std::vector<cfdVector3> *vecData){
  int nPts=(int)mesh.P.size(), nCells=mesh.nCells; std::vector<long long> connectivity, offsets(nCells), facesStream, faceoffsets(nCells); std::vector<int> types(nCells,42); long long connCount=0, faceCount=0;
  for(int c=0;c<nCells;++c){ const auto &fids=mesh.cellFaces[c]; const auto &oris=mesh.cellOrient[c]; std::vector<std::vector<int>> cellFacePts(fids.size()); std::vector<int> allPts; for(std::size_t j=0;j<fids.size();++j){ int f=fids[j]; cellFacePts[j]=mesh.faces[f]; if(oris[j]<0) std::reverse(cellFacePts[j].begin(),cellFacePts[j].end()); allPts.insert(allPts.end(),cellFacePts[j].begin(),cellFacePts[j].end()); } std::vector<int> uniqPts; std::set<int> seen; for(int p:allPts) if(seen.insert(p).second) uniqPts.push_back(p); for(int p:uniqPts) connectivity.push_back((long long)p); connCount += (long long)uniqPts.size(); offsets[c]=connCount; facesStream.push_back((long long)fids.size()); for(const auto &fv:cellFacePts){ facesStream.push_back((long long)fv.size()); for(int p:fv) facesStream.push_back((long long)p); } faceCount += 1; for(const auto &fv:cellFacePts) faceCount += 1 + (long long)fv.size(); faceoffsets[c]=faceCount; }
  std::ofstream fid(filename.c_str()); if(!fid) throw std::runtime_error("Could not open "+filename+" for writing");
  fid << "<?xml version=\"1.0\"?>\n<VTKFile type=\"UnstructuredGrid\" version=\"1.0\" byte_order=\"LittleEndian\">\n  <UnstructuredGrid>\n    <Piece NumberOfPoints=\""<<nPts<<"\" NumberOfCells=\""<<nCells<<"\">\n      <Points>\n        <DataArray type=\"Float64\" NumberOfComponents=\"3\" format=\"ascii\">\n"; for(const auto &p:mesh.P) fid<<"          "<<std::setprecision(15)<<p[0]<<" "<<p[1]<<" "<<p[2]<<"\n"; fid<<"        </DataArray>\n      </Points>\n      <Cells>\n        <DataArray type=\"Int64\" Name=\"connectivity\" format=\"ascii\">\n          "; for(auto v:connectivity) fid<<v<<" "; fid<<"\n        </DataArray>\n        <DataArray type=\"Int64\" Name=\"offsets\" format=\"ascii\">\n          "; for(auto v:offsets) fid<<v<<" "; fid<<"\n        </DataArray>\n        <DataArray type=\"UInt8\" Name=\"types\" format=\"ascii\">\n          "; for(auto v:types) fid<<v<<" "; fid<<"\n        </DataArray>\n        <DataArray type=\"Int64\" Name=\"faces\" format=\"ascii\">\n          "; for(auto v:facesStream) fid<<v<<" "; fid<<"\n        </DataArray>\n        <DataArray type=\"Int64\" Name=\"faceoffsets\" format=\"ascii\">\n          "; for(auto v:faceoffsets) fid<<v<<" "; fid<<"\n        </DataArray>\n      </Cells>\n      <CellData>\n";
  if(vecData){ fid<<"        <DataArray type=\"Float64\" Name=\""<<vecName<<"\" NumberOfComponents=\"3\" format=\"ascii\">\n"; for(const auto &v:*vecData) fid<<"          "<<std::setprecision(15)<<v[0]<<" "<<v[1]<<" "<<v[2]<<"\n"; fid<<"        </DataArray>\n"; }
  for(std::size_t k=0;k<scalarNames.size();++k){ fid<<"        <DataArray type=\"Float64\" Name=\""<<scalarNames[k]<<"\" NumberOfComponents=\"1\" format=\"ascii\">\n"; for(double v:scalarData[k]) fid<<"          "<<std::setprecision(15)<<v<<"\n"; fid<<"        </DataArray>\n"; }
  fid<<"      </CellData>\n    </Piece>\n  </UnstructuredGrid>\n</VTKFile>\n";
}


// -----------------------------------------------------------------------------
// simple_gpu v1.1 bad-cell / bad-face diagnostics
// -----------------------------------------------------------------------------
struct AuditMetricRow {
  int id=-1;
  double value=0.0;
};

static std::vector<AuditMetricRow> top_abs_rows(const std::vector<cfdScalar>& a, int topN){
  std::vector<AuditMetricRow> rows;
  rows.reserve(a.size());
  for(int i=0; i<(int)a.size(); ++i){
    const double v = std::fabs(a[i]);
    if(std::isfinite(v)) rows.push_back({i, v});
  }
  std::sort(rows.begin(), rows.end(), [](const AuditMetricRow& x, const AuditMetricRow& y){return x.value > y.value;});
  if(topN > 0 && (int)rows.size() > topN) rows.resize(topN);
  return rows;
}

static void print_top_cell_metric(
    const char* name,
    const std::vector<cfdScalar>& metric,
    int topN,
    const Mesh& mesh,
    const std::vector<cfdScalar>& pCorr,
    const std::vector<cfdScalar>& gradx,
    const std::vector<cfdScalar>& grady,
    const std::vector<cfdScalar>& gradz,
    const std::vector<cfdScalar>& rAU,
    const std::vector<cfdScalar>& divCorr,
    const std::vector<cfdScalar>& dUmag)
{
  std::printf("  Top cells by %s\n", name);
  std::printf("    rank cell value vol ccx ccy ccz |pCorr| |gradCorr| rAU |div| |dU|\n");
  auto rows = top_abs_rows(metric, topN);
  for(int k=0; k<(int)rows.size(); ++k){
    const int c = rows[k].id;
    const double gmag = std::sqrt(gradx[c]*gradx[c] + grady[c]*grady[c] + gradz[c]*gradz[c]);
    std::printf("    %4d %8d %.6e %.6e %.6e %.6e %.6e %.6e %.6e %.6e %.6e %.6e\n",
        k+1, c, rows[k].value, mesh.vol[c], mesh.cc[c][0], mesh.cc[c][1], mesh.cc[c][2],
        std::fabs(pCorr[c]), gmag, rAU[c], std::fabs(divCorr[c]), dUmag[c]);
  }
}

struct AuditFaceRow {
  int f=-1, P=-1, N=-1, patch=-1;
  double score=0.0;
  double coeff=0.0;
  double cosTheta=0.0;
  double dpn=0.0;
  double dmag=0.0;
  double Af=0.0;
  double rAUf=0.0;
  double pJump=0.0;
  double phiStar=0.0;
  double phi=0.0;
  double phiNonOrth=0.0;
};

static std::vector<AuditFaceRow> top_face_rows_by(
    std::vector<AuditFaceRow> rows, int topN)
{
  rows.erase(std::remove_if(rows.begin(), rows.end(), [](const AuditFaceRow& r){return !std::isfinite(r.score);}), rows.end());
  std::sort(rows.begin(), rows.end(), [](const AuditFaceRow& a, const AuditFaceRow& b){return a.score > b.score;});
  if(topN > 0 && (int)rows.size() > topN) rows.resize(topN);
  return rows;
}

static void print_face_table(const char* name, const std::vector<AuditFaceRow>& rows){
  std::printf("  Top faces by %s\n", name);
  std::printf("    rank face P N patch score coeff cos dpn dmag Af rAUf pJump phiStar phi phiNonOrth\n");
  for(int k=0; k<(int)rows.size(); ++k){
    const auto& r = rows[k];
    std::printf("    %4d %8d %8d %8d %5d %.6e %.6e %.6e %.6e %.6e %.6e %.6e %.6e %.6e %.6e %.6e\n",
        k+1, r.f, r.P, r.N, r.patch, r.score, r.coeff, r.cosTheta, r.dpn, r.dmag,
        r.Af, r.rAUf, r.pJump, r.phiStar, r.phi, r.phiNonOrth);
  }
}

static void write_bad_cell_audit_csv(
    const Params& par, int step,
    const Mesh& mesh,
    const std::vector<cfdScalar>& pCorr,
    const std::vector<cfdScalar>& gradx,
    const std::vector<cfdScalar>& grady,
    const std::vector<cfdScalar>& gradz,
    const std::vector<cfdScalar>& rAU,
    const std::vector<cfdScalar>& divCorr,
    const std::vector<cfdScalar>& dUmag)
{
  std::ostringstream fn;
  fn << par.outPrefix << "_badcell_iter" << std::setw(6) << std::setfill('0') << step << ".csv";
  std::ofstream out(fn.str().c_str());
  if(!out){
    std::printf("WARNING: Could not write bad-cell audit CSV: %s\n", fn.str().c_str());
    return;
  }
  out << "cell,ccx,ccy,ccz,vol,pCorr,gradx,grady,gradz,gradMag,rAU,divCorr,dUmag\n";
  for(int c=0; c<mesh.nCells; ++c){
    const double gmag = std::sqrt(gradx[c]*gradx[c] + grady[c]*grady[c] + gradz[c]*gradz[c]);
    out << c << ',' << std::setprecision(17)
        << mesh.cc[c][0] << ',' << mesh.cc[c][1] << ',' << mesh.cc[c][2] << ','
        << mesh.vol[c] << ',' << pCorr[c] << ','
        << gradx[c] << ',' << grady[c] << ',' << gradz[c] << ',' << gmag << ','
        << rAU[c] << ',' << divCorr[c] << ',' << dUmag[c] << '\n';
  }
  std::printf("  Wrote bad-cell audit CSV: %s\n", fn.str().c_str());
}

static void run_bad_cell_audit(
    const Params& par,
    const Mesh& mesh,
    const std::vector<std::string>& bcPType,
    int step,
    const char* reason,
    double massRes, double duRel, double dvRel, double dwRel, double dpRel,
    const std::vector<cfdScalar>& pCorr,
    const std::vector<cfdScalar>& gradx,
    const std::vector<cfdScalar>& grady,
    const std::vector<cfdScalar>& gradz,
    const std::vector<cfdScalar>& rAU,
    const std::vector<cfdScalar>& divCorr,
    const std::vector<cfdScalar>& uStar,
    const std::vector<cfdScalar>& vStar,
    const std::vector<cfdScalar>& wStar,
    const std::vector<cfdScalar>& u,
    const std::vector<cfdScalar>& v,
    const std::vector<cfdScalar>& w,
    const std::vector<cfdScalar>& phiStar,
    const std::vector<cfdScalar>& phi,
    const std::vector<cfdScalar>& phiNonOrth)
{
  const int topN = std::max(par.badCellAuditTop, 1);
  std::vector<cfdScalar> gradMag(mesh.nCells, 0.0), dUmag(mesh.nCells, 0.0), pcorrAbs(mesh.nCells, 0.0), divAbs(mesh.nCells, 0.0), rauAbs(mesh.nCells, 0.0);
  for(int c=0; c<mesh.nCells; ++c){
    gradMag[c] = std::sqrt(gradx[c]*gradx[c] + grady[c]*grady[c] + gradz[c]*gradz[c]);
    const double du = u[c] - uStar[c];
    const double dv = v[c] - vStar[c];
    const double dw = w[c] - wStar[c];
    dUmag[c] = std::sqrt(du*du + dv*dv + dw*dw);
    pcorrAbs[c] = std::fabs(pCorr[c]);
    divAbs[c] = std::fabs(divCorr[c]);
    rauAbs[c] = std::fabs(rAU[c]);
  }

  std::printf("\n------------------------------------------------------------\n");
  std::printf("BAD-CELL AUDIT simple_gpu at iter %d, reason=%s\n", step, reason ? reason : "periodic");
  std::printf("  residuals: massRes=%.6e duRel=%.6e dvRel=%.6e dwRel=%.6e dpRel=%.6e\n", massRes, duRel, dvRel, dwRel, dpRel);
  std::printf("  modes: pMode=%s pSolveMode=%s pGradScheme=%s rcMode=%s rAUMode=%s pDeltaMode=%d pDeltaMinCos=%.6g momNonOrthScale=%.6g pNonOrthScale=%.6g\n",
      par.pMode == 0 ? "pcorr" : "absolute",
      par.pSolveMode == 1 ? "ofAbsolute" : "correction",
      par.pGradScheme == 1 ? "gauss" : "lsq",
      par.rcMode == 0 ? "old" : "oflike",
      par.rAUMode == 0 ? "raw" : "relaxed",
      par.pDeltaMode, par.pDeltaMinCos, par.momNonOrthScale, par.pNonOrthScale);

  print_top_cell_metric("|divCorr|", divAbs, topN, mesh, pCorr, gradx, grady, gradz, rAU, divCorr, dUmag);
  print_top_cell_metric("|pCorr|", pcorrAbs, topN, mesh, pCorr, gradx, grady, gradz, rAU, divCorr, dUmag);
  print_top_cell_metric("|grad(correction)|", gradMag, topN, mesh, pCorr, gradx, grady, gradz, rAU, divCorr, dUmag);
  print_top_cell_metric("|dU correction|", dUmag, topN, mesh, pCorr, gradx, grady, gradz, rAU, divCorr, dUmag);
  print_top_cell_metric("|rAU|", rauAbs, topN, mesh, pCorr, gradx, grady, gradz, rAU, divCorr, dUmag);

  std::vector<AuditFaceRow> byCoeff, byBadCos, byFluxChange;
  byCoeff.reserve(mesh.nFaces);
  byBadCos.reserve(mesh.nFaces);
  byFluxChange.reserve(mesh.nFaces);
  for(int f=0; f<mesh.nFaces; ++f){
    const int P = mesh.owner[f];
    int N = -1;
    int patch = -1;
    std::array<double,3> d{{0.0,0.0,0.0}};
    double rAUf = rAU[P];
    double pJump = 0.0;
    bool include = false;
    if(f < mesh.nInternalFaces){
      N = mesh.neigh[f];
      d = sub3(mesh.cc[N], mesh.cc[P]);
      const double denom = std::max(dot3(d,d), 1.0e-30);
      double lam = dot3(sub3(mesh.xf[f], mesh.cc[P]), d) / denom;
      lam = std::min(1.0, std::max(0.0, lam));
      rAUf = (1.0-lam)*rAU[P] + lam*rAU[N];
      pJump = pCorr[N] - pCorr[P];
      include = true;
    } else {
      patch = mesh.bPatch[f] - 1;
      if(patch >= 0 && patch < (int)bcPType.size() && bcPType[patch] == "Dirichlet"){
        d = sub3(mesh.xf[f], mesh.cc[P]);
        pJump = -pCorr[P]; // fixed-value pressure correction BC value is usually zero in this app audit
        include = true;
      }
    }
    if(!include) continue;
    const double dmag = std::max(norm3(d), 1.0e-300);
    const double dpn = dot3(mesh.nf[f], d);
    const double cosTheta = dpn / dmag;
    const double deltaCoeff = pressure_delta_coeff_runtime(mesh.nf[f], d);
    const double coeff = par.pCoeffScale * par.rho * mesh.Af[f] * rAUf * deltaCoeff;
    AuditFaceRow row;
    row.f = f; row.P = P; row.N = N; row.patch = patch;
    row.coeff = coeff; row.cosTheta = cosTheta; row.dpn = dpn; row.dmag = dmag;
    row.Af = mesh.Af[f]; row.rAUf = rAUf; row.pJump = pJump;
    row.phiStar = (f < (int)phiStar.size() ? phiStar[f] : 0.0);
    row.phi = (f < (int)phi.size() ? phi[f] : 0.0);
    row.phiNonOrth = (f < (int)phiNonOrth.size() ? phiNonOrth[f] : 0.0);
    row.score = std::fabs(coeff);
    byCoeff.push_back(row);
    row.score = 1.0 / std::max(std::fabs(cosTheta), 1.0e-12);
    byBadCos.push_back(row);
    row.score = std::fabs(row.phi - row.phiStar);
    byFluxChange.push_back(row);
  }
  print_face_table("|pressure coeff|", top_face_rows_by(byCoeff, topN));
  print_face_table("1/|cos(theta)|", top_face_rows_by(byBadCos, topN));
  print_face_table("|phi-phiStar|", top_face_rows_by(byFluxChange, topN));

  if(par.badCellAuditWriteCsv){
    write_bad_cell_audit_csv(par, step, mesh, pCorr, gradx, grady, gradz, rAU, divCorr, dUmag);
  }
  std::printf("------------------------------------------------------------\n\n");
}


// -----------------------------------------------------------------------------
// Pressure-Dirichlet SIMPLE MMS used by the working polycube/subtri verification.
// This is intentionally separate from sineMmsEnable, whose all-Neumann pressure
// route is only useful for structured blockMesh checks.
struct SimpleMmsExactValue {
  double u, v, w, p;
  double fx, fy, fz;
};

static __host__ __device__ __forceinline__ SimpleMmsExactValue simple_mms_exact_value(
    double x, double y, double z,
    double rho, double mu,
    double U0, double P0, int stokesOnly = 0)
{
  const double pi = 3.141592653589793238462643383279502884;
  const double sx = sin(pi*x), cx = cos(pi*x);
  const double sy = sin(pi*y), cy = cos(pi*y);
  const double sz = sin(pi*z), cz = cos(pi*z);
  const double s2x = sin(2.0*pi*x), c2x = cos(2.0*pi*x);
  const double s2y = sin(2.0*pi*y), c2y = cos(2.0*pi*y);

  SimpleMmsExactValue q;
  q.u = U0 * pi * sx*sx * s2y;
  q.v = -U0 * pi * s2x * sy*sy;
  q.w = 0.0;
  q.p = P0 * sx * sy * sz;

  const double ux = U0 * pi*pi * s2x * s2y;
  const double uy = 2.0 * U0 * pi*pi * sx*sx * c2y;
  const double vx = -2.0 * U0 * pi*pi * c2x * sy*sy;
  const double vy = -U0 * pi*pi * s2x * s2y;

  const double lapu = U0 * pi*pi*pi * (2.0*c2x - 4.0*sx*sx) * s2y;
  const double lapv = U0 * pi*pi*pi * s2x * (4.0*sy*sy - 2.0*c2y);

  const double px = P0 * pi * cx * sy * sz;
  const double py = P0 * pi * sx * cy * sz;
  const double pz = P0 * pi * sx * sy * cz;

  const double convx = stokesOnly ? 0.0 : rho * (q.u*ux + q.v*uy);
  const double convy = stokesOnly ? 0.0 : rho * (q.u*vx + q.v*vy);

  // Body force for: div(rho U q) - div(mu grad q) = -grad(p)_q + f_q.
  q.fx = convx - mu*lapu + px;
  q.fy = convy - mu*lapv + py;
  q.fz = pz;
  return q;
}

static void simple_mms_fill_cell_fields(
    const Mesh& mesh,
    double rho, double mu, double U0, double P0,
    std::vector<cfdScalar>& u,
    std::vector<cfdScalar>& v,
    std::vector<cfdScalar>& w,
    std::vector<cfdScalar>& p)
{
  u.resize(mesh.nCells); v.resize(mesh.nCells); w.resize(mesh.nCells); p.resize(mesh.nCells);
  for(int c=0; c<mesh.nCells; ++c){
    const auto q = simple_mms_exact_value(mesh.cc[c][0], mesh.cc[c][1], mesh.cc[c][2], rho, mu, U0, P0);
    u[c] = (cfdScalar)q.u;
    v[c] = (cfdScalar)q.v;
    w[c] = (cfdScalar)q.w;
    p[c] = (cfdScalar)q.p;
  }
}

static void simple_mms_accumulate_error_for_margin(
    const Mesh& mesh,
    const std::vector<cfdScalar>& u,
    const std::vector<cfdScalar>& v,
    const std::vector<cfdScalar>& w,
    const std::vector<cfdScalar>& p,
    double rho, double mu, double U0, double P0,
    double margin,
    double& uAbsL2, double& uRelL2,
    double& pAbsL2, double& pRelL2,
    double& pMeanOffset,
    int& nUsed)
{
  double uErr2 = 0.0, uEx2 = 0.0;
  double pErr2 = 0.0, pEx2 = 0.0;
  double volSum = 0.0;
  double pDiffMean = 0.0;
  nUsed = 0;

  for(int c=0; c<mesh.nCells; ++c){
    const double x = mesh.cc[c][0];
    const double y = mesh.cc[c][1];
    const double z = mesh.cc[c][2];
    if(margin >= 0.0 && (x <= margin || x >= 1.0 - margin ||
       y <= margin || y >= 1.0 - margin ||
       z <= margin || z >= 1.0 - margin)) continue;

    const double V = mesh.vol[c];
    const auto q = simple_mms_exact_value(x, y, z, rho, mu, U0, P0);
    const double du = static_cast<double>(u[c]) - q.u;
    const double dv = static_cast<double>(v[c]) - q.v;
    const double dw = static_cast<double>(w[c]) - q.w;
    const double dp = static_cast<double>(p[c]) - q.p;

    uErr2 += V*(du*du + dv*dv + dw*dw);
    uEx2  += V*(q.u*q.u + q.v*q.v + q.w*q.w);
    pDiffMean += V*dp;
    volSum += V;
    ++nUsed;
  }

  pMeanOffset = (volSum > 0.0) ? pDiffMean / volSum : 0.0;

  for(int c=0; c<mesh.nCells; ++c){
    const double x = mesh.cc[c][0];
    const double y = mesh.cc[c][1];
    const double z = mesh.cc[c][2];
    if(margin >= 0.0 && (x <= margin || x >= 1.0 - margin ||
       y <= margin || y >= 1.0 - margin ||
       z <= margin || z >= 1.0 - margin)) continue;

    const double V = mesh.vol[c];
    const auto q = simple_mms_exact_value(x, y, z, rho, mu, U0, P0);
    const double dp = (static_cast<double>(p[c]) - q.p) - pMeanOffset;
    const double pe = q.p;
    pErr2 += V*dp*dp;
    pEx2  += V*pe*pe;
  }

  uAbsL2 = sqrt(std::max(uErr2, 0.0));
  uRelL2 = uAbsL2 / sqrt(std::max(uEx2, 1.0e-300));
  pAbsL2 = sqrt(std::max(pErr2, 0.0));
  pRelL2 = pAbsL2 / sqrt(std::max(pEx2, 1.0e-300));
}

static void simple_mms_report_error(
    const Mesh& mesh,
    const std::vector<cfdScalar>& u,
    const std::vector<cfdScalar>& v,
    const std::vector<cfdScalar>& w,
    const std::vector<cfdScalar>& p,
    double rho, double mu, double U0, double P0,
    double& uAbsL2, double& uRelL2,
    double& pAbsL2, double& pRelL2,
    double& pMeanOffset)
{
  int nUsed = 0;
  simple_mms_accumulate_error_for_margin(
      mesh, u, v, w, p, rho, mu, U0, P0,
      -1.0, uAbsL2, uRelL2, pAbsL2, pRelL2, pMeanOffset, nUsed);
}

__global__ static void kernel_simple_mms_set_exact_u(
    int nCells,
    const cfdScalar *ccx, const cfdScalar *ccy, const cfdScalar *ccz,
    double rho, double mu, double U0, double P0, int stokesOnly,
    cfdScalar *u, cfdScalar *v, cfdScalar *w)
{
  const int c = blockIdx.x * blockDim.x + threadIdx.x;
  if(c >= nCells) return;
  const auto q = simple_mms_exact_value((double)ccx[c], (double)ccy[c], (double)ccz[c],
                                        rho, mu, U0, P0, stokesOnly);
  u[c] = (cfdScalar)q.u;
  v[c] = (cfdScalar)q.v;
  w[c] = (cfdScalar)q.w;
}

__global__ static void kernel_simple_mms_set_exact_p(
    int nCells,
    const cfdScalar *ccx, const cfdScalar *ccy, const cfdScalar *ccz,
    double rho, double mu, double U0, double P0, int stokesOnly,
    cfdScalar *p)
{
  const int c = blockIdx.x * blockDim.x + threadIdx.x;
  if(c >= nCells) return;
  const auto q = simple_mms_exact_value((double)ccx[c], (double)ccy[c], (double)ccz[c],
                                        rho, mu, U0, P0, stokesOnly);
  p[c] = (cfdScalar)q.p;
}

__global__ static void kernel_simple_mms_subtract_ref_cell(
    int nCells, cfdScalar *p, cfdScalar pref)
{
  const int c = blockIdx.x * blockDim.x + threadIdx.x;
  if(c >= nCells) return;
  p[c] = (cfdScalar)((double)p[c] - (double)pref);
}

__global__ static void kernel_simple_mms_difference_p(
    int nCells, const cfdScalar *a, const cfdScalar *b, cfdScalar *out)
{
  const int c = blockIdx.x * blockDim.x + threadIdx.x;
  if(c >= nCells) return;
  out[c] = (cfdScalar)((double)a[c] - (double)b[c]);
}

__global__ static void kernel_mms_pressure_exact_residual_csr(
    int nCells,
    const int *rowOffsets,
    const HYPRE_BigInt *cols,
    const HYPRE_Complex *vals,
    const cfdScalar *x,
    const HYPRE_Complex *rhs,
    cfdScalar *res)
{
  const int r = blockIdx.x * blockDim.x + threadIdx.x;
  if(r >= nCells) return;

  double ax = 0.0;
  const int p0 = rowOffsets[r];
  const int p1 = rowOffsets[r + 1];

  for(int k = p0; k < p1; ++k){
    const int c = (int)cols[k];
    ax += (double)vals[k] * (double)x[c];
  }

  res[r] = (cfdScalar)(ax - (double)rhs[r]);
}

__global__ static void kernel_mms_pressure_exact_residual_anchor_zero(
    int refCell, cfdScalar *res)
{
  res[refCell] = 0;
}

__global__ static void kernel_mms_pressure_delta(
    int nCells,
    const cfdScalar *pSolved,
    const cfdScalar *pExact,
    cfdScalar *delta)
{
  const int c = blockIdx.x * blockDim.x + threadIdx.x;
  if(c >= nCells) return;
  delta[c] = (cfdScalar)((double)pSolved[c] - (double)pExact[c]);
}

static void simple_mms_write_p_error_point_vtk(
    const Mesh& mesh,
    const std::vector<cfdScalar>& ph,
    double rho,
    double mu,
    double U0,
    double P0,
    int stokesOnly,
    const std::string& path)
{
  std::vector<double> pExact(mesh.nCells, 0.0);
  std::vector<double> pErrRaw(mesh.nCells, 0.0);
  std::vector<double> pErrMf(mesh.nCells, 0.0);

  double mean = 0.0;
  double volSum = 0.0;

  for(int c = 0; c < mesh.nCells; ++c){
    const double x = mesh.cc[c][0];
    const double y = mesh.cc[c][1];
    const double z = mesh.cc[c][2];

    const auto q = simple_mms_exact_value(x, y, z, rho, mu, U0, P0, stokesOnly);
    pExact[c] = q.p;
    pErrRaw[c] = (double)ph[c] - pExact[c];

    mean += mesh.vol[c] * pErrRaw[c];
    volSum += mesh.vol[c];
  }

  mean /= std::max(volSum, 1.0e-300);

  for(int c = 0; c < mesh.nCells; ++c){
    pErrMf[c] = pErrRaw[c] - mean;
  }

  std::ofstream os(path);
  if(!os){
    std::fprintf(stderr, "WARNING: could not write MMS pressure-error VTK: %s\n", path.c_str());
    return;
  }

  os << "# vtk DataFile Version 3.0\n";
  os << "MMS pressure error point cloud\n";
  os << "ASCII\n";
  os << "DATASET POLYDATA\n";
  os << "POINTS " << mesh.nCells << " double\n";
  for(int c = 0; c < mesh.nCells; ++c){
    os << mesh.cc[c][0] << " " << mesh.cc[c][1] << " " << mesh.cc[c][2] << "\n";
  }

  os << "VERTICES " << mesh.nCells << " " << 2 * mesh.nCells << "\n";
  for(int c = 0; c < mesh.nCells; ++c){
    os << "1 " << c << "\n";
  }

  auto write_scalar = [&](const char* name, const std::vector<double>& a){
    os << "SCALARS " << name << " double 1\n";
    os << "LOOKUP_TABLE default\n";
    for(int c = 0; c < mesh.nCells; ++c){
      os << a[c] << "\n";
    }
  };

  std::vector<double> pNum(mesh.nCells, 0.0);
  for(int c = 0; c < mesh.nCells; ++c){
    pNum[c] = (double)ph[c];
  }

  os << "POINT_DATA " << mesh.nCells << "\n";
  write_scalar("p_num", pNum);
  write_scalar("p_exact", pExact);
  write_scalar("p_error_raw", pErrRaw);
  write_scalar("p_error_mf", pErrMf);

  std::printf("SIMPLE_MMS wrote pressure-error VTK point cloud: %s\n", path.c_str());
}

__global__ static void kernel_add_simple_mms_momentum_source_quad8(
    int nCells,
    const cfdScalar *ccx, const cfdScalar *ccy, const cfdScalar *ccz,
    const cfdScalar *vol,
    double rho, double mu, double U0, double P0, int stokesOnly,
    int comp,
    HYPRE_Complex *rhs)
{
  const int c = blockIdx.x * blockDim.x + threadIdx.x;
  if(c >= nCells) return;

  const double h = cbrt(fmax((double)vol[c], 1.0e-300));
  const double a = 0.5 * h / sqrt(3.0);
  const double x0 = (double)ccx[c];
  const double y0 = (double)ccy[c];
  const double z0 = (double)ccz[c];
  double sum = 0.0;

  #pragma unroll
  for(int ix=0; ix<2; ++ix){
    const double x = x0 + (ix == 0 ? -a : a);
    #pragma unroll
    for(int iy=0; iy<2; ++iy){
      const double y = y0 + (iy == 0 ? -a : a);
      #pragma unroll
      for(int iz=0; iz<2; ++iz){
        const double z = z0 + (iz == 0 ? -a : a);
        const auto q = simple_mms_exact_value(x, y, z, rho, mu, U0, P0, stokesOnly);
        sum += (comp == 0 ? q.fx : (comp == 1 ? q.fy : q.fz));
      }
    }
  }

  const double favg = sum * 0.125;
  rhs[c] += (HYPRE_Complex)(favg * (double)vol[c]);
}

__global__ static void kernel_add_simple_mms_momentum_source(
    int nCells,
    const cfdScalar *ccx, const cfdScalar *ccy, const cfdScalar *ccz,
    const cfdScalar *vol,
    double rho, double mu, double U0, double P0, int stokesOnly,
    int comp,
    HYPRE_Complex *rhs)
{
  const int c = blockIdx.x * blockDim.x + threadIdx.x;
  if(c >= nCells) return;
  const auto q = simple_mms_exact_value((double)ccx[c], (double)ccy[c], (double)ccz[c], rho, mu, U0, P0, stokesOnly);
  const double f = (comp == 0 ? q.fx : (comp == 1 ? q.fy : q.fz));
  rhs[c] += (HYPRE_Complex)(f * (double)vol[c]);
}

__global__ static void kernel_mms_enforce_fixed_velocity_boundary_flux(
    int nFaces,
    int nInternalFaces,
    const cfdScalar *phiStar,
    cfdScalar *phi)
{
  const int k = blockIdx.x * blockDim.x + threadIdx.x;
  const int f = nInternalFaces + k;
  if(f >= nFaces) return;
  phi[f] = phiStar[f];
}

struct DeviceMesh {
  // Optional subtriangle quadrature arrays. These must be allocated lazily
  // only when subtri continuity / final subtri flux reconstruction is enabled.
  int nSubTris = 0;
  int *d_subFace = nullptr;
  cfdScalar *d_subx = nullptr, *d_suby = nullptr, *d_subz = nullptr;
  cfdScalar *d_subsx = nullptr, *d_subsy = nullptr, *d_subsz = nullptr;

  int nCells=0, nFaces=0, nInternalFaces=0;
  int *d_owner=nullptr, *d_neigh=nullptr, *d_bPatch=nullptr;
  cfdScalar *d_ccx=nullptr, *d_ccy=nullptr, *d_ccz=nullptr;
  cfdScalar *d_xfx=nullptr, *d_xfy=nullptr, *d_xfz=nullptr;
  cfdScalar *d_nfx=nullptr, *d_nfy=nullptr, *d_nfz=nullptr;
  cfdScalar *d_sfx=nullptr, *d_sfy=nullptr, *d_sfz=nullptr;
  cfdScalar *d_Af=nullptr, *d_vol=nullptr;
};

struct DeviceGradientOperator {
  int nCells=0, nTerms=0;
  int *d_offsets=nullptr;
  int *d_src=nullptr;
  int *d_face=nullptr;
  cfdScalar *d_cx=nullptr, *d_cy=nullptr, *d_cz=nullptr;
};

struct DeviceBC {
  int nPatch=0;
  std::vector<int> type;
  std::vector<cfdScalar> faceValue;
  int *d_type=nullptr;
  cfdScalar *d_faceValue=nullptr;
};

struct MatrixPattern {
  int nRows=0, nnz=0;
  std::vector<int> ncols, rowOffsets, diagPos;
  std::vector<HYPRE_BigInt> rows, cols;
  std::vector<int> facePP, facePN, faceNP, faceNN;
  int *d_ncols=nullptr, *d_rowOffsets=nullptr, *d_diagPos=nullptr;
  HYPRE_BigInt *d_rows=nullptr, *d_cols=nullptr;
  int *d_facePP=nullptr, *d_facePN=nullptr, *d_faceNP=nullptr, *d_faceNN=nullptr;
  HYPRE_Complex *d_vals=nullptr;
};

struct GPULinearSystem {
  int n=0;
  bool isPCG=false;
  MatrixPattern pat;
  HYPRE_IJMatrix Aij=nullptr; HYPRE_ParCSRMatrix Apar=nullptr;
  HYPRE_IJVector bij=nullptr, xij=nullptr; HYPRE_ParVector bpar=nullptr, xpar=nullptr;
  HYPRE_Solver solver=nullptr, prec=nullptr;
  std::vector<HYPRE_BigInt> h_idx;
  HYPRE_Complex *d_rhs=nullptr, *d_x=nullptr;
  HYPRE_Complex *A_diag_data_dev=nullptr, *b_data_dev=nullptr, *x_data_dev=nullptr;
  std::vector<int> A_diag_perm_h;
  int *A_diag_perm_d=nullptr;
  bool A_diag_identity_perm=true;
  bool direct_matrix_values=false;
  bool mcgs_only=false; // momentum-only capacity mode: app CSR, no HYPRE velocity matrix/vectors
  bool is_setup=false;
};

struct GPUMomentumAssembler {
  GPULinearSystem lin;
  cfdScalar *d_qOld=nullptr, *d_uConv=nullptr, *d_vConv=nullptr, *d_wConv=nullptr;
  cfdScalar *d_gradQx=nullptr, *d_gradQy=nullptr, *d_gradQz=nullptr;
  cfdScalar *d_gradPcomp=nullptr;
  cfdScalar *d_rAU=nullptr;
};


static void add_simple_mms_source_to_momentum_rhs(
    const Params& par,
    const DeviceMesh& dm,
    const Mesh& mesh,
    GPUMomentumAssembler& mom,
    double rho,
    double mu,
    int comp)
{
  if(!par.simpleMmsEnable) return;
  const int block = 256;
  const int grid = (mesh.nCells + block - 1) / block;
  if(par.simpleMmsSourceQuad8){
    kernel_add_simple_mms_momentum_source_quad8<<<grid, block>>>(
        mesh.nCells, dm.d_ccx, dm.d_ccy, dm.d_ccz, dm.d_vol,
        rho, mu, par.simpleMmsVelocityScale, par.simpleMmsPressureScale, par.simpleMmsStokesOnly,
        comp, mom.lin.d_rhs);
  } else {
    kernel_add_simple_mms_momentum_source<<<grid, block>>>(
        mesh.nCells, dm.d_ccx, dm.d_ccy, dm.d_ccz, dm.d_vol,
        rho, mu, par.simpleMmsVelocityScale, par.simpleMmsPressureScale, par.simpleMmsStokesOnly,
        comp, mom.lin.d_rhs);
  }
  CUDA_CHECK_LAST();
}

struct GPUSimpleScratch {
  int nCells=0, nFaces=0;
  cfdScalar *d_u=nullptr, *d_v=nullptr, *d_w=nullptr, *d_p=nullptr, *d_pCorr=nullptr;
  cfdScalar *d_uOld=nullptr, *d_vOld=nullptr, *d_wOld=nullptr, *d_pOld=nullptr;
  cfdScalar *d_gradx=nullptr, *d_grady=nullptr, *d_gradz=nullptr;
  cfdScalar *d_phiStar=nullptr, *d_phi=nullptr, *d_phiNonOrth=nullptr;
  cfdScalar *d_divStar=nullptr, *d_divCorr=nullptr, *d_divNonOrth=nullptr;
  cfdScalar *d_pCorrDelta=nullptr;
  cfdScalar *d_reduce=nullptr, *d_reduce2=nullptr;
  int reduceSize=0;
};

static inline int bc_to_int(const std::string &s){ return (s=="Dirichlet") ? 1 : 0; }

template<typename T>
static void device_alloc(T *&ptr, std::size_t n){ CUDA_CALL(cudaMalloc((void**)&ptr, n*sizeof(T))); }

template<typename T>
static void device_free(T *&ptr){ if(ptr) CUDA_CALL(cudaFree(ptr)); ptr=nullptr; }

template<typename T>
static void copy_vec_to_device(const std::vector<T> &h, T *d){ if(!h.empty()) CUDA_CALL(cudaMemcpy(d, h.data(), h.size()*sizeof(T), cudaMemcpyHostToDevice)); }

template<typename T>
static void copy_device_to_vec(const T *d, std::vector<T> &h){ if(!h.empty()) CUDA_CALL(cudaMemcpy(h.data(), d, h.size()*sizeof(T), cudaMemcpyDeviceToHost)); }

static void build_device_mesh(const Mesh &mesh, DeviceMesh &dm, bool buildSubTris){
  dm.nCells=mesh.nCells; dm.nFaces=mesh.nFaces; dm.nInternalFaces=mesh.nInternalFaces;
  std::vector<cfdScalar> ccx(mesh.nCells), ccy(mesh.nCells), ccz(mesh.nCells), vol(mesh.nCells);
  std::vector<cfdScalar> xfx(mesh.nFaces), xfy(mesh.nFaces), xfz(mesh.nFaces), nfx(mesh.nFaces), nfy(mesh.nFaces), nfz(mesh.nFaces), sfx(mesh.nFaces), sfy(mesh.nFaces), sfz(mesh.nFaces), Af(mesh.nFaces);
  for(int c=0;c<mesh.nCells;++c){ ccx[c]=mesh.cc[c][0]; ccy[c]=mesh.cc[c][1]; ccz[c]=mesh.cc[c][2]; vol[c]=mesh.vol[c]; }
  for(int f=0;f<mesh.nFaces;++f){ xfx[f]=mesh.xf[f][0]; xfy[f]=mesh.xf[f][1]; xfz[f]=mesh.xf[f][2]; nfx[f]=mesh.nf[f][0]; nfy[f]=mesh.nf[f][1]; nfz[f]=mesh.nf[f][2]; sfx[f]=mesh.Sf[f][0]; sfy[f]=mesh.Sf[f][1]; sfz[f]=mesh.Sf[f][2]; Af[f]=mesh.Af[f]; }

  if(buildSubTris){
    std::vector<int> h_subFace;
    std::vector<cfdScalar> h_subx, h_suby, h_subz;
    std::vector<cfdScalar> h_subsx, h_subsy, h_subsz;
    h_subFace.reserve(mesh.nFaces * 2);
    h_subx.reserve(mesh.nFaces * 2); h_suby.reserve(mesh.nFaces * 2); h_subz.reserve(mesh.nFaces * 2);
    h_subsx.reserve(mesh.nFaces * 2); h_subsy.reserve(mesh.nFaces * 2); h_subsz.reserve(mesh.nFaces * 2);
    for(int f=0; f<mesh.nFaces; ++f){
      const std::vector<int> &fv = mesh.faces[f];
      const int nv = (int)fv.size();
      if(nv < 3) continue;
      double cx = 0.0, cy = 0.0, cz = 0.0;
      for(int ii=0; ii<nv; ++ii){
        const auto &pp = mesh.P[fv[ii]];
        cx += pp[0]; cy += pp[1]; cz += pp[2];
      }
      cx /= (double)nv; cy /= (double)nv; cz /= (double)nv;
      for(int ii=0; ii<nv; ++ii){
        const auto &pa = mesh.P[fv[ii]];
        const auto &pb = mesh.P[fv[(ii+1)%nv]];
        const double ax = pa[0] - cx;
        const double ay = pa[1] - cy;
        const double az = pa[2] - cz;
        const double bx = pb[0] - cx;
        const double by = pb[1] - cy;
        const double bz = pb[2] - cz;
        double sx = 0.5*(ay*bz - az*by);
        double sy = 0.5*(az*bx - ax*bz);
        double sz = 0.5*(ax*by - ay*bx);
        const double alignParentSf = sx*mesh.Sf[f][0] + sy*mesh.Sf[f][1] + sz*mesh.Sf[f][2];
        if(alignParentSf < 0.0){ sx = -sx; sy = -sy; sz = -sz; }
        h_subFace.push_back(f);
        h_subx.push_back((cfdScalar)((cx + pa[0] + pb[0]) / 3.0));
        h_suby.push_back((cfdScalar)((cy + pa[1] + pb[1]) / 3.0));
        h_subz.push_back((cfdScalar)((cz + pa[2] + pb[2]) / 3.0));
        h_subsx.push_back((cfdScalar)sx);
        h_subsy.push_back((cfdScalar)sy);
        h_subsz.push_back((cfdScalar)sz);
      }
    }
    dm.nSubTris = (int)h_subFace.size();
    if(dm.nSubTris > 0){
      device_alloc(dm.d_subFace, h_subFace.size()); copy_vec_to_device(h_subFace, dm.d_subFace);
      device_alloc(dm.d_subx, h_subx.size()); copy_vec_to_device(h_subx, dm.d_subx);
      device_alloc(dm.d_suby, h_suby.size()); copy_vec_to_device(h_suby, dm.d_suby);
      device_alloc(dm.d_subz, h_subz.size()); copy_vec_to_device(h_subz, dm.d_subz);
      device_alloc(dm.d_subsx, h_subsx.size()); copy_vec_to_device(h_subsx, dm.d_subsx);
      device_alloc(dm.d_subsy, h_subsy.size()); copy_vec_to_device(h_subsy, dm.d_subsy);
      device_alloc(dm.d_subsz, h_subsz.size()); copy_vec_to_device(h_subsz, dm.d_subsz);
    }
  } else {
    dm.nSubTris = 0;
  }

  device_alloc(dm.d_owner, mesh.owner.size()); copy_vec_to_device(mesh.owner, dm.d_owner);
  device_alloc(dm.d_neigh, mesh.neigh.size()); copy_vec_to_device(mesh.neigh, dm.d_neigh);
  device_alloc(dm.d_bPatch, mesh.bPatch.size()); copy_vec_to_device(mesh.bPatch, dm.d_bPatch);
  device_alloc(dm.d_ccx, ccx.size()); copy_vec_to_device(ccx, dm.d_ccx);
  device_alloc(dm.d_ccy, ccy.size()); copy_vec_to_device(ccy, dm.d_ccy);
  device_alloc(dm.d_ccz, ccz.size()); copy_vec_to_device(ccz, dm.d_ccz);
  device_alloc(dm.d_xfx, xfx.size()); copy_vec_to_device(xfx, dm.d_xfx);
  device_alloc(dm.d_xfy, xfy.size()); copy_vec_to_device(xfy, dm.d_xfy);
  device_alloc(dm.d_xfz, xfz.size()); copy_vec_to_device(xfz, dm.d_xfz);
  device_alloc(dm.d_nfx, nfx.size()); copy_vec_to_device(nfx, dm.d_nfx);
  device_alloc(dm.d_nfy, nfy.size()); copy_vec_to_device(nfy, dm.d_nfy);
  device_alloc(dm.d_nfz, nfz.size()); copy_vec_to_device(nfz, dm.d_nfz);
  device_alloc(dm.d_sfx, sfx.size()); copy_vec_to_device(sfx, dm.d_sfx);
  device_alloc(dm.d_sfy, sfy.size()); copy_vec_to_device(sfy, dm.d_sfy);
  device_alloc(dm.d_sfz, sfz.size()); copy_vec_to_device(sfz, dm.d_sfz);
  device_alloc(dm.d_Af, Af.size()); copy_vec_to_device(Af, dm.d_Af);
  device_alloc(dm.d_vol, vol.size()); copy_vec_to_device(vol, dm.d_vol);
}

static void destroy_device_mesh(DeviceMesh &dm){
  device_free(dm.d_subFace);
  device_free(dm.d_subx); device_free(dm.d_suby); device_free(dm.d_subz);
  device_free(dm.d_subsx); device_free(dm.d_subsy); device_free(dm.d_subsz);
  dm.nSubTris = 0;
  device_free(dm.d_owner); device_free(dm.d_neigh); device_free(dm.d_bPatch);
  device_free(dm.d_ccx); device_free(dm.d_ccy); device_free(dm.d_ccz);
  device_free(dm.d_xfx); device_free(dm.d_xfy); device_free(dm.d_xfz);
  device_free(dm.d_nfx); device_free(dm.d_nfy); device_free(dm.d_nfz);
  device_free(dm.d_sfx); device_free(dm.d_sfy); device_free(dm.d_sfz);
  device_free(dm.d_Af); device_free(dm.d_vol);
  dm = DeviceMesh{};
}

static DeviceBC make_device_bc(int nFaces, const std::vector<std::string> &bcType, const std::vector<cfdScalar> &bcFaceValue){
  DeviceBC bc;
  bc.nPatch = static_cast<int>(bcType.size());
  bc.type.resize(bc.nPatch);
  bc.faceValue = bcFaceValue;
  if(static_cast<int>(bc.faceValue.size()) != nFaces) throw std::runtime_error("BC face-value size mismatch");
  for(int i=0;i<bc.nPatch;++i) bc.type[i]=bc_to_int(bcType[i]);
  device_alloc(bc.d_type, bc.type.size()); copy_vec_to_device(bc.type, bc.d_type);
  device_alloc(bc.d_faceValue, bc.faceValue.size()); copy_vec_to_device(bc.faceValue, bc.d_faceValue);
  return bc;
}
static void destroy_device_bc(DeviceBC &bc){ device_free(bc.d_type); device_free(bc.d_faceValue); bc=DeviceBC{}; }

static void init_simple_scratch(GPUSimpleScratch &ss, const Mesh &mesh){
  ss.nCells = mesh.nCells;
  ss.nFaces = mesh.nFaces;
  device_alloc(ss.d_u, mesh.nCells);
  gpu_mem_mark0("simple_scratch_alloc_01_d_u"); device_alloc(ss.d_v, mesh.nCells);
  gpu_mem_mark0("simple_scratch_alloc_02_d_v"); device_alloc(ss.d_w, mesh.nCells);
  gpu_mem_mark0("simple_scratch_alloc_03_d_w");
  device_alloc(ss.d_uOld, mesh.nCells);
  gpu_mem_mark0("simple_scratch_alloc_04_d_uOld"); device_alloc(ss.d_vOld, mesh.nCells);
  gpu_mem_mark0("simple_scratch_alloc_05_d_vOld"); device_alloc(ss.d_wOld, mesh.nCells);
  gpu_mem_mark0("simple_scratch_alloc_06_d_wOld");
  device_alloc(ss.d_p, mesh.nCells);
  gpu_mem_mark0("simple_scratch_alloc_07_d_p"); device_alloc(ss.d_pOld, mesh.nCells);
  gpu_mem_mark0("simple_scratch_alloc_08_d_pOld"); device_alloc(ss.d_pCorr, mesh.nCells);
  gpu_mem_mark0("simple_scratch_alloc_09_d_pCorr");
  device_alloc(ss.d_gradx, mesh.nCells);
  gpu_mem_mark0("simple_scratch_alloc_10_d_gradx"); device_alloc(ss.d_grady, mesh.nCells);
  gpu_mem_mark0("simple_scratch_alloc_11_d_grady"); device_alloc(ss.d_gradz, mesh.nCells);
  gpu_mem_mark0("simple_scratch_alloc_12_d_gradz");
  device_alloc(ss.d_phiStar, mesh.nFaces);
  gpu_mem_mark0("simple_scratch_alloc_13_d_phiStar"); device_alloc(ss.d_phi, mesh.nFaces);
  gpu_mem_mark0("simple_scratch_alloc_14_d_phi"); device_alloc(ss.d_phiNonOrth, mesh.nFaces);
  gpu_mem_mark0("simple_scratch_alloc_15_d_phiNonOrth");
  device_alloc(ss.d_divStar, mesh.nCells);
  gpu_mem_mark0("simple_scratch_alloc_16_d_divStar"); device_alloc(ss.d_divCorr, mesh.nCells);
  gpu_mem_mark0("simple_scratch_alloc_17_d_divCorr"); device_alloc(ss.d_divNonOrth, mesh.nCells);
  gpu_mem_mark0("simple_scratch_alloc_18_d_divNonOrth");
  device_alloc(ss.d_pCorrDelta, mesh.nCells);
  gpu_mem_mark0("simple_scratch_alloc_19_d_pCorrDelta");
  ss.reduceSize = std::max((mesh.nCells + 255)/256, 1);
  device_alloc(ss.d_reduce, ss.reduceSize);
  gpu_mem_mark0("simple_scratch_alloc_20_d_reduce"); device_alloc(ss.d_reduce2, ss.reduceSize);
  gpu_mem_mark0("simple_scratch_alloc_21_d_reduce2");
}

static void destroy_simple_scratch(GPUSimpleScratch &ss){
  device_free(ss.d_u); device_free(ss.d_v); device_free(ss.d_w); device_free(ss.d_p); device_free(ss.d_pCorr);
  device_free(ss.d_uOld); device_free(ss.d_vOld); device_free(ss.d_wOld); device_free(ss.d_pOld);
  device_free(ss.d_gradx); device_free(ss.d_grady); device_free(ss.d_gradz);
  device_free(ss.d_phiStar); device_free(ss.d_phi); device_free(ss.d_phiNonOrth);
  device_free(ss.d_divStar); device_free(ss.d_divCorr); device_free(ss.d_divNonOrth);
  device_free(ss.d_pCorrDelta); device_free(ss.d_reduce); device_free(ss.d_reduce2);
  ss = GPUSimpleScratch{};
}

static void upload_gradient_to_device(const std::vector<cfdVector3> &grad,
                                      std::vector<cfdScalar> &bufX, std::vector<cfdScalar> &bufY, std::vector<cfdScalar> &bufZ,
                                      cfdScalar *d_gx, cfdScalar *d_gy, cfdScalar *d_gz){
  const int n = (int)grad.size();
  if((int)bufX.size() != n){ bufX.resize(n); bufY.resize(n); bufZ.resize(n); }
  for(int c=0;c<n;++c){ bufX[c]=grad[c][0]; bufY[c]=grad[c][1]; bufZ[c]=grad[c][2]; }
  copy_vec_to_device(bufX, d_gx);
  copy_vec_to_device(bufY, d_gy);
  copy_vec_to_device(bufZ, d_gz);
}

// Forward declarations for CUDA kernels used before their definitions.
__global__ static void kernel_zero_scalar(cfdScalar *x, int n);

__global__ static void kernel_continuity_residual_from_flux(
    int nFaces, int nInternalFaces,
    const int *owner, const int *neigh,
    const cfdScalar *phi, cfdScalar *divCell);


static void build_lsq_gradient_operator(const Mesh &mesh, DeviceGradientOperator &op, double weightPower){
  op.nCells = mesh.nCells;
  std::vector<int> offsets(mesh.nCells + 1, 0), src, face;
  std::vector<cfdScalar> cx, cy, cz;
  src.reserve(mesh.nInternalFaces*2 + (mesh.nFaces-mesh.nInternalFaces));
  face.reserve(src.capacity());
  cx.reserve(src.capacity());
  cy.reserve(src.capacity());
  cz.reserve(src.capacity());

  for(int P=0; P<mesh.nCells; ++P){
    auto xP = mesh.cc[P];
    double M[3][3]={{0,0,0},{0,0,0},{0,0,0}};

    struct Term {
      int src;
      int face;
      double r[3];
      double w;
    };

    std::vector<Term> terms;
    terms.reserve(mesh.cellNbrs[P].size() + mesh.cellBFace[P].size());

    for(int N: mesh.cellNbrs[P]){
      auto r = sub3(mesh.cc[N], xP);
      double w = 1.0/std::pow(std::max(std::sqrt(dot3(r,r)),1e-300), weightPower);
      for(int i=0;i<3;++i)
        for(int j=0;j<3;++j)
          M[i][j] += w*r[i]*r[j];

      Term t;
      t.src = N;
      t.face = -1;
      t.r[0] = r[0]; t.r[1] = r[1]; t.r[2] = r[2];
      t.w = w;
      terms.push_back(t);
    }

    for(int f: mesh.cellBFace[P]){
      auto r = sub3(mesh.xf[f], xP);
      double w = 1.0/std::pow(std::max(std::sqrt(dot3(r,r)),1e-300), weightPower);
      for(int i=0;i<3;++i)
        for(int j=0;j<3;++j)
          M[i][j] += w*r[i]*r[j];

      Term t;
      t.src = -1;
      t.face = f;
      t.r[0] = r[0]; t.r[1] = r[1]; t.r[2] = r[2];
      t.w = w;
      terms.push_back(t);
    }

    double a=M[0][0], b=M[0][1], c=M[0][2];
    double d=M[1][0], e=M[1][1], f2=M[1][2];
    double g=M[2][0], h=M[2][1], k=M[2][2];

    double det = a*(e*k-f2*h) - b*(d*k-f2*g) + c*(d*h-e*g);

    double inv[3][3]={{0,0,0},{0,0,0},{0,0,0}};
    if(std::fabs(det)>1e-20){
      inv[0][0]=(e*k-f2*h)/det; inv[0][1]=(c*h-b*k)/det; inv[0][2]=(b*f2-c*e)/det;
      inv[1][0]=(f2*g-d*k)/det; inv[1][1]=(a*k-c*g)/det; inv[1][2]=(c*d-a*f2)/det;
      inv[2][0]=(d*h-e*g)/det; inv[2][1]=(b*g-a*h)/det; inv[2][2]=(a*e-b*d)/det;
    }

    offsets[P] = (int)src.size();

    for(const auto &t: terms){
      double bx = t.w*t.r[0];
      double by = t.w*t.r[1];
      double bz = t.w*t.r[2];

      src.push_back(t.src);
      face.push_back(t.face);

      cx.push_back(inv[0][0]*bx + inv[0][1]*by + inv[0][2]*bz);
      cy.push_back(inv[1][0]*bx + inv[1][1]*by + inv[1][2]*bz);
      cz.push_back(inv[2][0]*bx + inv[2][1]*by + inv[2][2]*bz);
    }
  }

  offsets[mesh.nCells] = (int)src.size();
  op.nTerms = (int)src.size();

  device_alloc(op.d_offsets, offsets.size()); copy_vec_to_device(offsets, op.d_offsets);
  device_alloc(op.d_src, src.size()); copy_vec_to_device(src, op.d_src);
  device_alloc(op.d_face, face.size()); copy_vec_to_device(face, op.d_face);
  device_alloc(op.d_cx, cx.size()); copy_vec_to_device(cx, op.d_cx);
  device_alloc(op.d_cy, cy.size()); copy_vec_to_device(cy, op.d_cy);
  device_alloc(op.d_cz, cz.size()); copy_vec_to_device(cz, op.d_cz);
}

static void destroy_lsq_gradient_operator(DeviceGradientOperator &op){
  device_free(op.d_offsets);
  device_free(op.d_src);
  device_free(op.d_face);
  device_free(op.d_cx);
  device_free(op.d_cy);
  device_free(op.d_cz);
  op = DeviceGradientOperator{};
}

// Forward declarations for CUDA kernels used before their definitions.
__global__ static void kernel_zero_scalar(cfdScalar *x, int n);

__global__ static void kernel_continuity_residual_from_flux(
    int nFaces, int nInternalFaces,
    const int *owner, const int *neigh,
    const cfdScalar *phi, cfdScalar *divCell);

__global__ static void kernel_apply_lsq_gradient(
    int nCells, const int *offsets, const int *src, const int *face,
    const cfdScalar *cx, const cfdScalar *cy, const cfdScalar *cz,
    const cfdScalar *phi, const int *bPatch, const int *bcType, const cfdScalar *bcFaceValue,
    cfdScalar *gx, cfdScalar *gy, cfdScalar *gz);

__global__ static void kernel_apply_gauss_linear_gradient(
    int nFaces, int nInternalFaces,
    const int *owner, const int *neigh, const int *bPatch,
    const cfdScalar *ccx, const cfdScalar *ccy, const cfdScalar *ccz,
    const cfdScalar *xfx, const cfdScalar *xfy, const cfdScalar *xfz,
    const cfdScalar *sfx, const cfdScalar *sfy, const cfdScalar *sfz,
    const cfdScalar *vol,
    const cfdScalar *phi, const int *bcType, const cfdScalar *bcFaceValue,
    cfdScalar *gx, cfdScalar *gy, cfdScalar *gz);

static void compute_gauss_linear_gradient_gpu(const DeviceMesh &dm, const DeviceBC &bc,
                                              const cfdScalar *d_phi, cfdScalar *d_gx, cfdScalar *d_gy, cfdScalar *d_gz);

__global__ static void kernel_add_scaled_inplace(int n, cfdScalar *y, const cfdScalar *x, double a);
__global__ static void kernel_update_pressure_relax(int n, cfdScalar *p, const cfdScalar *pcorr, double pRelax);
__global__ static void kernel_subtract_scalar_inplace(int n, cfdScalar *x, double a);
__global__ static void kernel_maxabs_reduce(int n, const cfdScalar *x, cfdScalar *blockMax);
__global__ static void kernel_relchg_reduce(int n, const cfdScalar *a, const cfdScalar *b, cfdScalar *blockNum, cfdScalar *blockDen);

static void continuity_residual_gpu(const DeviceMesh &dm, const cfdScalar *d_phi, cfdScalar *d_div){
  const int block = 256;
  kernel_zero_scalar<<<(dm.nCells + block - 1)/block, block>>>(d_div, dm.nCells);
  CUDA_CHECK_LAST();
  kernel_continuity_residual_from_flux<<<(dm.nFaces + block - 1)/block, block>>>(
      dm.nFaces, dm.nInternalFaces, dm.d_owner, dm.d_neigh, d_phi, d_div);
  CUDA_CHECK_LAST();
}


static void compute_lsq_gradient_gpu(const DeviceGradientOperator &gop, const DeviceMesh &dm, const DeviceBC &bc,
                                     const cfdScalar *d_phi, cfdScalar *d_gx, cfdScalar *d_gy, cfdScalar *d_gz){
  // Memory-light Gauss mode deliberately does not allocate the LSQ gradient operator.
  // The subtri/skew MMS correction still needs velocity gradients; fall back to
  // the already-available Gauss linear gradient rather than launching an LSQ
  // kernel with zero blocks / null LSQ arrays.
  if(gop.nCells <= 0 || gop.d_offsets == nullptr || gop.d_src == nullptr || gop.d_face == nullptr){
    compute_gauss_linear_gradient_gpu(dm, bc, d_phi, d_gx, d_gy, d_gz);
    return;
  }

  const int block = 256;
  const int grid = (gop.nCells + block - 1)/block;
  if(grid <= 0){
    compute_gauss_linear_gradient_gpu(dm, bc, d_phi, d_gx, d_gy, d_gz);
    return;
  }

  kernel_apply_lsq_gradient<<<grid, block>>>(
      gop.nCells, gop.d_offsets, gop.d_src, gop.d_face,
      gop.d_cx, gop.d_cy, gop.d_cz,
      d_phi, dm.d_bPatch, bc.d_type, bc.d_faceValue,
      d_gx, d_gy, d_gz);
  CUDA_CHECK_LAST();
}

static void compute_gauss_linear_gradient_gpu(const DeviceMesh &dm, const DeviceBC &bc,
                                              const cfdScalar *d_phi, cfdScalar *d_gx, cfdScalar *d_gy, cfdScalar *d_gz){
  const int block = 256;
  kernel_zero_scalar<<<(dm.nCells + block - 1)/block, block>>>(d_gx, dm.nCells);
  kernel_zero_scalar<<<(dm.nCells + block - 1)/block, block>>>(d_gy, dm.nCells);
  kernel_zero_scalar<<<(dm.nCells + block - 1)/block, block>>>(d_gz, dm.nCells);
  CUDA_CHECK_LAST();
  kernel_apply_gauss_linear_gradient<<<(dm.nFaces + block - 1)/block, block>>>(
      dm.nFaces, dm.nInternalFaces,
      dm.d_owner, dm.d_neigh, dm.d_bPatch,
      dm.d_ccx, dm.d_ccy, dm.d_ccz,
      dm.d_xfx, dm.d_xfy, dm.d_xfz,
      dm.d_sfx, dm.d_sfy, dm.d_sfz,
      dm.d_vol,
      d_phi, bc.d_type, bc.d_faceValue,
      d_gx, d_gy, d_gz);
  CUDA_CHECK_LAST();
}

static void compute_pressure_gradient_gpu(const Params &par, const DeviceGradientOperator &gop,
                                          const DeviceMesh &dm, const DeviceBC &bc,
                                          const cfdScalar *d_phi, cfdScalar *d_gx, cfdScalar *d_gy, cfdScalar *d_gz){
  if(par.pGradScheme == 1) compute_gauss_linear_gradient_gpu(dm, bc, d_phi, d_gx, d_gy, d_gz);
  else compute_lsq_gradient_gpu(gop, dm, bc, d_phi, d_gx, d_gy, d_gz);
}


static int simple_env_or_param_int(const char *envName, int paramValue){
  const char *e = std::getenv(envName);
  return e ? std::atoi(e) : paramValue;
}

__global__ static void kernel_simple_add_phi_velocity_skew_component(
    int nInternalFaces, int comp,
    const int *owner, const int *neigh,
    const cfdScalar *ccx, const cfdScalar *ccy, const cfdScalar *ccz,
    const cfdScalar *xfx, const cfdScalar *xfy, const cfdScalar *xfz,
    const cfdScalar *nfx, const cfdScalar *nfy, const cfdScalar *nfz,
    const cfdScalar *Af,
    const cfdScalar *gradx, const cfdScalar *grady, const cfdScalar *gradz,
    double rho, double faceSkewScale,
    cfdScalar *phi)
{
  const int f = blockIdx.x*blockDim.x + threadIdx.x;
  if(f >= nInternalFaces) return;
  const int P = owner[f];
  const int N = neigh[f];
  const double dx = (double)ccx[N] - (double)ccx[P];
  const double dy = (double)ccy[N] - (double)ccy[P];
  const double dz = (double)ccz[N] - (double)ccz[P];
  const double denom = fmax(dx*dx + dy*dy + dz*dz, 1.0e-30);
  double lam = (((double)xfx[f]-(double)ccx[P])*dx + ((double)xfy[f]-(double)ccy[P])*dy + ((double)xfz[f]-(double)ccz[P])*dz) / denom;
  lam = fmin(1.0, fmax(0.0, lam));
  const double xlinx = (double)ccx[P] + lam*dx;
  const double xliny = (double)ccy[P] + lam*dy;
  const double xlinz = (double)ccz[P] + lam*dz;
  const double skx = (double)xfx[f] - xlinx;
  const double sky = (double)xfy[f] - xliny;
  const double skz = (double)xfz[f] - xlinz;
  const double gx = (1.0-lam)*(double)gradx[P] + lam*(double)gradx[N];
  const double gy = (1.0-lam)*(double)grady[P] + lam*(double)grady[N];
  const double gz = (1.0-lam)*(double)gradz[P] + lam*(double)gradz[N];
  const double qSkew = faceSkewScale * (gx*skx + gy*sky + gz*skz);
  const double ncomp = (comp == 0 ? (double)nfx[f] : (comp == 1 ? (double)nfy[f] : (double)nfz[f]));
  hypreAtomicAdd((HYPRE_Complex*)&phi[f], (HYPRE_Complex)(rho * (double)Af[f] * qSkew * ncomp));
}

__global__ static void kernel_simple_subtri_velocity_flux_correction_component(
    int nSubTris, int nInternalFaces, int comp,
    const int *subFace,
    const cfdScalar *subx, const cfdScalar *suby, const cfdScalar *subz,
    const cfdScalar *subsx, const cfdScalar *subsy, const cfdScalar *subsz,
    const int *owner, const int *neigh, const int *bPatch,
    const cfdScalar *ccx, const cfdScalar *ccy, const cfdScalar *ccz,
    const cfdScalar *xfx, const cfdScalar *xfy, const cfdScalar *xfz,
    const cfdScalar *q,
    const cfdScalar *gradx, const cfdScalar *grady, const cfdScalar *gradz,
    const int *bcType, const cfdScalar *qFaceBC,
    double rho, double faceSkewScale,
    cfdScalar *phi)
{
  const int it = blockIdx.x*blockDim.x + threadIdx.x;
  if(it >= nSubTris) return;
  const int f = subFace[it];
  const double Sq = (comp == 0 ? (double)subsx[it] : (comp == 1 ? (double)subsy[it] : (double)subsz[it]));

  double qParent = 0.0;
  double qSub = 0.0;
  if(f < nInternalFaces){
    const int P = owner[f];
    const int N = neigh[f];
    const double dx = (double)ccx[N] - (double)ccx[P];
    const double dy = (double)ccy[N] - (double)ccy[P];
    const double dz = (double)ccz[N] - (double)ccz[P];
    const double denom = fmax(dx*dx + dy*dy + dz*dz, 1.0e-30);
    double lamF = (((double)xfx[f]-(double)ccx[P])*dx + ((double)xfy[f]-(double)ccy[P])*dy + ((double)xfz[f]-(double)ccz[P])*dz) / denom;
    lamF = fmin(1.0, fmax(0.0, lamF));
    const double xlinFx = (double)ccx[P] + lamF*dx;
    const double xlinFy = (double)ccy[P] + lamF*dy;
    const double xlinFz = (double)ccz[P] + lamF*dz;
    const double gFx = (1.0-lamF)*(double)gradx[P] + lamF*(double)gradx[N];
    const double gFy = (1.0-lamF)*(double)grady[P] + lamF*(double)grady[N];
    const double gFz = (1.0-lamF)*(double)gradz[P] + lamF*(double)gradz[N];
    qParent = (1.0-lamF)*(double)q[P] + lamF*(double)q[N]
            + faceSkewScale*(gFx*((double)xfx[f]-xlinFx) + gFy*((double)xfy[f]-xlinFy) + gFz*((double)xfz[f]-xlinFz));

    double lamQ = (((double)subx[it]-(double)ccx[P])*dx + ((double)suby[it]-(double)ccy[P])*dy + ((double)subz[it]-(double)ccz[P])*dz) / denom;
    lamQ = fmin(1.0, fmax(0.0, lamQ));
    const double xlinQx = (double)ccx[P] + lamQ*dx;
    const double xlinQy = (double)ccy[P] + lamQ*dy;
    const double xlinQz = (double)ccz[P] + lamQ*dz;
    const double gQx = (1.0-lamQ)*(double)gradx[P] + lamQ*(double)gradx[N];
    const double gQy = (1.0-lamQ)*(double)grady[P] + lamQ*(double)grady[N];
    const double gQz = (1.0-lamQ)*(double)gradz[P] + lamQ*(double)gradz[N];
    qSub = (1.0-lamQ)*(double)q[P] + lamQ*(double)q[N]
         + faceSkewScale*(gQx*((double)subx[it]-xlinQx) + gQy*((double)suby[it]-xlinQy) + gQz*((double)subz[it]-xlinQz));
  } else {
    const int P = owner[f];
    const int patch = bPatch[f] - 1;
    if(patch < 0) return;
    if(bcType[patch] != 1) return;
    qParent = (double)qFaceBC[f];
    qSub = (double)qFaceBC[f]
         + faceSkewScale*((double)gradx[P]*((double)subx[it]-(double)xfx[f])
                         + (double)grady[P]*((double)suby[it]-(double)xfy[f])
                         + (double)gradz[P]*((double)subz[it]-(double)xfz[f]));
  }

  hypreAtomicAdd((HYPRE_Complex*)&phi[f], (HYPRE_Complex)(rho * (qSub - qParent) * Sq));
}

static void add_simple_parent_velocity_skew_to_phi(
    const DeviceMesh &dmesh,
    const Mesh &mesh,
    GPUMomentumAssembler &mom,
    const DeviceGradientOperator &gop,
    const DeviceBC &dbcU,
    const DeviceBC &dbcV,
    const DeviceBC &dbcW,
    const cfdScalar *d_u,
    const cfdScalar *d_v,
    const cfdScalar *d_w,
    double rho,
    cfdScalar *d_phi)
{
  const int block = 256;
  const int gridFaces = (mesh.nInternalFaces + block - 1)/block;
  if(gridFaces <= 0) return;
  compute_lsq_gradient_gpu(gop, dmesh, dbcU, d_u, mom.d_gradQx, mom.d_gradQy, mom.d_gradQz);
  kernel_simple_add_phi_velocity_skew_component<<<gridFaces, block>>>(mesh.nInternalFaces, 0, dmesh.d_owner, dmesh.d_neigh, dmesh.d_ccx, dmesh.d_ccy, dmesh.d_ccz, dmesh.d_xfx, dmesh.d_xfy, dmesh.d_xfz, dmesh.d_nfx, dmesh.d_nfy, dmesh.d_nfz, dmesh.d_Af, mom.d_gradQx, mom.d_gradQy, mom.d_gradQz, rho, 1.0, d_phi);
  compute_lsq_gradient_gpu(gop, dmesh, dbcV, d_v, mom.d_gradQx, mom.d_gradQy, mom.d_gradQz);
  kernel_simple_add_phi_velocity_skew_component<<<gridFaces, block>>>(mesh.nInternalFaces, 1, dmesh.d_owner, dmesh.d_neigh, dmesh.d_ccx, dmesh.d_ccy, dmesh.d_ccz, dmesh.d_xfx, dmesh.d_xfy, dmesh.d_xfz, dmesh.d_nfx, dmesh.d_nfy, dmesh.d_nfz, dmesh.d_Af, mom.d_gradQx, mom.d_gradQy, mom.d_gradQz, rho, 1.0, d_phi);
  compute_lsq_gradient_gpu(gop, dmesh, dbcW, d_w, mom.d_gradQx, mom.d_gradQy, mom.d_gradQz);
  kernel_simple_add_phi_velocity_skew_component<<<gridFaces, block>>>(mesh.nInternalFaces, 2, dmesh.d_owner, dmesh.d_neigh, dmesh.d_ccx, dmesh.d_ccy, dmesh.d_ccz, dmesh.d_xfx, dmesh.d_xfy, dmesh.d_xfz, dmesh.d_nfx, dmesh.d_nfy, dmesh.d_nfz, dmesh.d_Af, mom.d_gradQx, mom.d_gradQy, mom.d_gradQz, rho, 1.0, d_phi);
  CUDA_CHECK_LAST();
}

static void add_simple_subtri_velocity_flux_correction(
    const DeviceMesh &dmesh,
    const Mesh &mesh,
    GPUMomentumAssembler &mom,
    const DeviceGradientOperator &gop,
    const DeviceBC &dbcU,
    const DeviceBC &dbcV,
    const DeviceBC &dbcW,
    const cfdScalar *d_u,
    const cfdScalar *d_v,
    const cfdScalar *d_w,
    double rho,
    double faceSkewScale,
    cfdScalar *d_phi)
{
  if(dmesh.nSubTris <= 0) return;
  const int block = 256;
  const int gridSub = (dmesh.nSubTris + block - 1) / block;
  compute_lsq_gradient_gpu(gop, dmesh, dbcU, d_u, mom.d_gradQx, mom.d_gradQy, mom.d_gradQz);
  kernel_simple_subtri_velocity_flux_correction_component<<<gridSub, block>>>(dmesh.nSubTris, mesh.nInternalFaces, 0, dmesh.d_subFace, dmesh.d_subx, dmesh.d_suby, dmesh.d_subz, dmesh.d_subsx, dmesh.d_subsy, dmesh.d_subsz, dmesh.d_owner, dmesh.d_neigh, dmesh.d_bPatch, dmesh.d_ccx, dmesh.d_ccy, dmesh.d_ccz, dmesh.d_xfx, dmesh.d_xfy, dmesh.d_xfz, d_u, mom.d_gradQx, mom.d_gradQy, mom.d_gradQz, dbcU.d_type, dbcU.d_faceValue, rho, faceSkewScale, d_phi);
  compute_lsq_gradient_gpu(gop, dmesh, dbcV, d_v, mom.d_gradQx, mom.d_gradQy, mom.d_gradQz);
  kernel_simple_subtri_velocity_flux_correction_component<<<gridSub, block>>>(dmesh.nSubTris, mesh.nInternalFaces, 1, dmesh.d_subFace, dmesh.d_subx, dmesh.d_suby, dmesh.d_subz, dmesh.d_subsx, dmesh.d_subsy, dmesh.d_subsz, dmesh.d_owner, dmesh.d_neigh, dmesh.d_bPatch, dmesh.d_ccx, dmesh.d_ccy, dmesh.d_ccz, dmesh.d_xfx, dmesh.d_xfy, dmesh.d_xfz, d_v, mom.d_gradQx, mom.d_gradQy, mom.d_gradQz, dbcV.d_type, dbcV.d_faceValue, rho, faceSkewScale, d_phi);
  compute_lsq_gradient_gpu(gop, dmesh, dbcW, d_w, mom.d_gradQx, mom.d_gradQy, mom.d_gradQz);
  kernel_simple_subtri_velocity_flux_correction_component<<<gridSub, block>>>(dmesh.nSubTris, mesh.nInternalFaces, 2, dmesh.d_subFace, dmesh.d_subx, dmesh.d_suby, dmesh.d_subz, dmesh.d_subsx, dmesh.d_subsy, dmesh.d_subsz, dmesh.d_owner, dmesh.d_neigh, dmesh.d_bPatch, dmesh.d_ccx, dmesh.d_ccy, dmesh.d_ccz, dmesh.d_xfx, dmesh.d_xfy, dmesh.d_xfz, d_w, mom.d_gradQx, mom.d_gradQy, mom.d_gradQz, dbcW.d_type, dbcW.d_faceValue, rho, faceSkewScale, d_phi);
  CUDA_CHECK_LAST();
}

static void apply_simple_phiStar_geometry_correction(
    const Params &par,
    const DeviceMesh &dmesh,
    const Mesh &mesh,
    GPUMomentumAssembler &mom,
    const DeviceGradientOperator &gop,
    const DeviceBC &dbcU,
    const DeviceBC &dbcV,
    const DeviceBC &dbcW,
    const cfdScalar *d_u,
    const cfdScalar *d_v,
    const cfdScalar *d_w,
    cfdScalar *d_phi)
{
  const int faceSkew = simple_env_or_param_int("ANABASIS_SIMPLE_FACE_SKEW_CORR", par.simpleFaceSkewCorr);
  const int subtriContinuity = simple_env_or_param_int("ANABASIS_SIMPLE_SUBTRI_CONTINUITY", par.simpleSubtriContinuity);
  static int printed = 0;
  if(!printed && (faceSkew || subtriContinuity)){
    int r = 0; MPI_Comm_rank(MPI_COMM_WORLD, &r);
    if(r == 0){
      std::printf("SIMPLE optional geometry flux correction: faceSkew=%d subtriContinuity=%d nSubTris=%d\n", faceSkew, subtriContinuity, dmesh.nSubTris);
    }
    printed = 1;
  }
  if(faceSkew){
    add_simple_parent_velocity_skew_to_phi(dmesh, mesh, mom, gop, dbcU, dbcV, dbcW, d_u, d_v, d_w, par.rho, d_phi);
  }
  if(subtriContinuity){
    add_simple_subtri_velocity_flux_correction(dmesh, mesh, mom, gop, dbcU, dbcV, dbcW, d_u, d_v, d_w, par.rho, faceSkew ? 1.0 : 0.0, d_phi);
  }
}

static void apply_simple_final_phi_geometry_correction_if_needed(
    const Params &par,
    const DeviceMesh &dmesh,
    const Mesh &mesh,
    GPUMomentumAssembler &mom,
    const DeviceGradientOperator &gop,
    const DeviceBC &dbcU,
    const DeviceBC &dbcV,
    const DeviceBC &dbcW,
    const cfdScalar *d_u,
    const cfdScalar *d_v,
    const cfdScalar *d_w,
    cfdScalar *d_phi)
{
  const int faceSkew = simple_env_or_param_int("ANABASIS_SIMPLE_FACE_SKEW_CORR", par.simpleFaceSkewCorr);
  const int subtriContinuity = simple_env_or_param_int("ANABASIS_SIMPLE_SUBTRI_CONTINUITY", par.simpleSubtriContinuity);
  const int subtriFluxRecon = simple_env_or_param_int("ANABASIS_SIMPLE_SUBTRI_FLUX_RECON", par.simpleSubtriFluxRecon);
  if(subtriContinuity || !subtriFluxRecon) return;
  static int printed = 0;
  if(!printed){
    int r = 0; MPI_Comm_rank(MPI_COMM_WORLD, &r);
    if(r == 0){
      std::printf("SIMPLE final phi uses subtriangle velocity-flux reconstruction nSubTris=%d\n", dmesh.nSubTris);
    }
    printed = 1;
  }
  add_simple_subtri_velocity_flux_correction(dmesh, mesh, mom, gop, dbcU, dbcV, dbcW, d_u, d_v, d_w, par.rho, faceSkew ? 1.0 : 0.0, d_phi);
}

static double maxabs_device(const cfdScalar *d_x, int n, cfdScalar *d_reduce, int reduceSize){
  const int block = 256;
  kernel_maxabs_reduce<<<reduceSize, block, block*sizeof(cfdScalar)>>>(n, d_x, d_reduce);
  CUDA_CHECK_LAST();

  std::vector<cfdScalar> h_reduce(reduceSize);
  CUDA_CALL(cudaMemcpy(h_reduce.data(), d_reduce, reduceSize*sizeof(cfdScalar), cudaMemcpyDeviceToHost));

  double m = 0.0;
  for(double v: h_reduce) m = std::max(m, std::fabs(v));
  return m;
}

static double relchg_device(const cfdScalar *d_a, const cfdScalar *d_b, int n, cfdScalar *d_numReduce, cfdScalar *d_denReduce, int reduceSize){
  const int block = 256;
  kernel_relchg_reduce<<<reduceSize, block, 2*block*sizeof(cfdScalar)>>>(n, d_a, d_b, d_numReduce, d_denReduce);
  CUDA_CHECK_LAST();

  std::vector<cfdScalar> h_num(reduceSize), h_den(reduceSize);
  CUDA_CALL(cudaMemcpy(h_num.data(), d_numReduce, reduceSize*sizeof(cfdScalar), cudaMemcpyDeviceToHost));
  CUDA_CALL(cudaMemcpy(h_den.data(), d_denReduce, reduceSize*sizeof(cfdScalar), cudaMemcpyDeviceToHost));

  double num = 0.0;
  double den = 1.0;
  for(int i=0;i<reduceSize;++i){
    num += h_num[i];
    den += h_den[i];
  }
  return std::sqrt(num/den);
}

static MatrixPattern build_momentum_pattern(const Mesh &mesh){
  MatrixPattern pat; pat.nRows=mesh.nCells; pat.rows.resize(mesh.nCells); pat.ncols.resize(mesh.nCells); pat.rowOffsets.resize(mesh.nCells+1); pat.diagPos.resize(mesh.nCells); pat.facePP.resize(mesh.nInternalFaces); pat.facePN.resize(mesh.nInternalFaces); pat.faceNP.resize(mesh.nInternalFaces); pat.faceNN.resize(mesh.nInternalFaces);
  std::vector<std::map<int,int>> pos(mesh.nCells);
  pat.rowOffsets[0]=0;
  for(int c=0;c<mesh.nCells;++c){
    pat.rows[c]=(HYPRE_BigInt)c;
    std::vector<int> cols = mesh.cellNbrs[c];
    cols.insert(cols.begin(), c);
    std::sort(cols.begin(), cols.end());
    cols.erase(std::unique(cols.begin(), cols.end()), cols.end());
    pat.ncols[c]=(int)cols.size();
    pat.rowOffsets[c+1]=pat.rowOffsets[c]+pat.ncols[c];
    for(int j=0;j<(int)cols.size();++j) pos[c][cols[j]] = pat.rowOffsets[c] + j;
  }
  pat.nnz = pat.rowOffsets.back();
  pat.cols.resize(pat.nnz);
  for(int c=0;c<mesh.nCells;++c){
    for(auto &kv : pos[c]) pat.cols[kv.second]=(HYPRE_BigInt)kv.first;
    pat.diagPos[c]=pos[c][c];
  }
  for(int f=0;f<mesh.nInternalFaces;++f){
    int P=mesh.owner[f], N=mesh.neigh[f];
    pat.facePP[f]=pos[P][P]; pat.facePN[f]=pos[P][N]; pat.faceNP[f]=pos[N][P]; pat.faceNN[f]=pos[N][N];
  }
  return pat;
}

static MatrixPattern build_pressure_pattern(const Mesh &mesh, int refCell, bool useRefAnchor){
  MatrixPattern pat; pat.nRows=mesh.nCells; pat.rows.resize(mesh.nCells); pat.ncols.resize(mesh.nCells); pat.rowOffsets.resize(mesh.nCells+1); pat.diagPos.resize(mesh.nCells); pat.facePP.resize(mesh.nInternalFaces); pat.facePN.resize(mesh.nInternalFaces); pat.faceNP.resize(mesh.nInternalFaces); pat.faceNN.resize(mesh.nInternalFaces);
  std::vector<std::map<int,int>> pos(mesh.nCells);
  pat.rowOffsets[0]=0;
  for(int c=0;c<mesh.nCells;++c){
    pat.rows[c]=(HYPRE_BigInt)c;
    std::vector<int> cols;
    cols.push_back(c);
    if(!useRefAnchor || c != refCell){
      for(int nb : mesh.cellNbrs[c]) if(!useRefAnchor || nb != refCell) cols.push_back(nb);
      std::sort(cols.begin(), cols.end());
      cols.erase(std::unique(cols.begin(), cols.end()), cols.end());
    }
    pat.ncols[c]=(int)cols.size();
    pat.rowOffsets[c+1]=pat.rowOffsets[c]+pat.ncols[c];
    for(int j=0;j<(int)cols.size();++j) pos[c][cols[j]] = pat.rowOffsets[c] + j;
  }
  pat.nnz = pat.rowOffsets.back();
  pat.cols.resize(pat.nnz);
  for(int c=0;c<mesh.nCells;++c){
    for(auto &kv : pos[c]) pat.cols[kv.second]=(HYPRE_BigInt)kv.first;
    pat.diagPos[c]=pos[c][c];
  }
  for(int f=0;f<mesh.nInternalFaces;++f){
    int P=mesh.owner[f], N=mesh.neigh[f];
    auto itPP=pos[P].find(P), itPN=pos[P].find(N), itNP=pos[N].find(P), itNN=pos[N].find(N);
    pat.facePP[f]=(itPP==pos[P].end())?-1:itPP->second;
    pat.facePN[f]=(itPN==pos[P].end())?-1:itPN->second;
    pat.faceNP[f]=(itNP==pos[N].end())?-1:itNP->second;
    pat.faceNN[f]=(itNN==pos[N].end())?-1:itNN->second;
  }
  return pat;
}

static void upload_pattern(MatrixPattern &pat, const char *memTag="pattern"){
  auto mark = [&](const char *name){
    char buf[192];
    std::snprintf(buf, sizeof(buf), "pattern_%s_%s", memTag, name);
    gpu_mem_mark0(buf);
  };

  if(g_memdiag_rank == 0){
    const double MiB = 1024.0 * 1024.0;
    const double b_ncols     = double(pat.ncols.size())     * sizeof(int);
    const double b_rowOff    = double(pat.rowOffsets.size())* sizeof(int);
    const double b_diag      = double(pat.diagPos.size())   * sizeof(int);
    const double b_rows      = double(pat.rows.size())      * sizeof(HYPRE_BigInt);
    const double b_cols      = double(pat.cols.size())      * sizeof(HYPRE_BigInt);
    const double b_facePP    = double(pat.facePP.size())    * sizeof(int);
    const double b_facePN    = double(pat.facePN.size())    * sizeof(int);
    const double b_faceNP    = double(pat.faceNP.size())    * sizeof(int);
    const double b_faceNN    = double(pat.faceNN.size())    * sizeof(int);
    const double b_vals      = double(pat.nnz)              * sizeof(HYPRE_Complex);
    const double b_total = b_ncols + b_rowOff + b_diag + b_rows + b_cols +
                           b_facePP + b_facePN + b_faceNP + b_faceNN + b_vals;

    std::printf(
      "PATTERN_MEM_EST tag=%s nRows=%d nnz=%d nFaceSlots=%zu "
      "MiB_total=%.1f ncols=%.1f rowOffsets=%.1f diagPos=%.1f rows=%.1f cols=%.1f "
      "facePP=%.1f facePN=%.1f faceNP=%.1f faceNN=%.1f vals=%.1f\n",
      memTag, pat.nRows, pat.nnz, pat.facePP.size(),
      b_total/MiB, b_ncols/MiB, b_rowOff/MiB, b_diag/MiB, b_rows/MiB, b_cols/MiB,
      b_facePP/MiB, b_facePN/MiB, b_faceNP/MiB, b_faceNN/MiB, b_vals/MiB);
    std::fflush(stdout);
  }

  device_alloc(pat.d_ncols, pat.ncols.size());
  copy_vec_to_device(pat.ncols, pat.d_ncols);
  mark("01_d_ncols");

  device_alloc(pat.d_rowOffsets, pat.rowOffsets.size());
  copy_vec_to_device(pat.rowOffsets, pat.d_rowOffsets);
  mark("02_d_rowOffsets");

  device_alloc(pat.d_diagPos, pat.diagPos.size());
  copy_vec_to_device(pat.diagPos, pat.d_diagPos);
  mark("03_d_diagPos");

  device_alloc(pat.d_rows, pat.rows.size());
  copy_vec_to_device(pat.rows, pat.d_rows);
  mark("04_d_rows");

  device_alloc(pat.d_cols, pat.cols.size());
  copy_vec_to_device(pat.cols, pat.d_cols);
  mark("05_d_cols");
  pat.d_facePP = nullptr;
  mark("06_d_facePP_skipped_diagPos");

  device_alloc(pat.d_facePN, pat.facePN.size());
  copy_vec_to_device(pat.facePN, pat.d_facePN);
  mark("07_d_facePN");

  device_alloc(pat.d_faceNP, pat.faceNP.size());
  copy_vec_to_device(pat.faceNP, pat.d_faceNP);
  mark("08_d_faceNP");
  pat.d_faceNN = nullptr;
  mark("09_d_faceNN_skipped_diagPos");

  device_alloc(pat.d_vals, pat.nnz);
  CUDA_CALL(cudaMemset(pat.d_vals, 0, pat.nnz*sizeof(HYPRE_Complex)));
  mark("10_d_vals");
}

static void destroy_pattern(MatrixPattern &pat){
  device_free(pat.d_ncols); device_free(pat.d_rowOffsets); device_free(pat.d_diagPos);
  device_free(pat.d_rows); device_free(pat.d_cols);
  device_free(pat.d_facePP); device_free(pat.d_facePN); device_free(pat.d_faceNP); device_free(pat.d_faceNN);
  device_free(pat.d_vals);
  pat = MatrixPattern{};
}

__global__ static void kernel_zero_scalar(cfdScalar *x, int n){
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if(i < n) x[i] = 0.0;
}

__global__ static void kernel_fill_scalar(cfdScalar *x, int n, double value){
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if(i < n) x[i] = value;
}



__global__ static void kernel_copy_scalar_to_hypre_complex(int n, const cfdScalar *src, HYPRE_Complex *dst){
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if(i < n) dst[i] = (HYPRE_Complex)src[i];
}

__global__ static void kernel_copy_hypre_complex_to_scalar(int n, const HYPRE_Complex *src, cfdScalar *dst){
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if(i < n) dst[i] = (cfdScalar)src[i];
}

static inline void copy_scalar_device_to_hypre_device(int n, const cfdScalar *src, HYPRE_Complex *dst){
  if(sizeof(HYPRE_Complex) == sizeof(cfdScalar)){
    CUDA_CALL(cudaMemcpy(dst, src, n * sizeof(HYPRE_Complex), cudaMemcpyDeviceToDevice));
  } else {
    const int block = 256;
    const int grid = (n + block - 1) / block;
    kernel_copy_scalar_to_hypre_complex<<<grid, block>>>(n, src, dst);
    CUDA_CHECK_LAST();
  }
}

static inline void copy_hypre_device_to_scalar_device(int n, const HYPRE_Complex *src, cfdScalar *dst){
  if(sizeof(HYPRE_Complex) == sizeof(cfdScalar)){
    CUDA_CALL(cudaMemcpy(dst, src, n * sizeof(HYPRE_Complex), cudaMemcpyDeviceToDevice));
  } else {
    const int block = 256;
    const int grid = (n + block - 1) / block;
    kernel_copy_hypre_complex_to_scalar<<<grid, block>>>(n, src, dst);
    CUDA_CHECK_LAST();
  }
}

__global__ static void kernel_apply_lsq_gradient(
    int nCells, const int *offsets, const int *src, const int *face,
    const cfdScalar *cx, const cfdScalar *cy, const cfdScalar *cz,
    const cfdScalar *phi, const int *bPatch, const int *bcType, const cfdScalar *bcFaceValue,
    cfdScalar *gx, cfdScalar *gy, cfdScalar *gz)
{
  int P = blockIdx.x*blockDim.x + threadIdx.x;
  if(P >= nCells) return;

  const double phiP = phi[P];
  double sx=0.0, sy=0.0, sz=0.0;

  const int beg = offsets[P];
  const int end = offsets[P+1];

  for(int t=beg; t<end; ++t){
    double val = phiP;

    const int s = src[t];
    if(s >= 0){
      val = phi[s];
    } else {
      const int f = face[t];
      const int patch = bPatch[f] - 1;
      if(patch >= 0 && bcType[patch] == 1)
        val = bcFaceValue[f];
    }

    const double dphi = val - phiP;
    sx += cx[t]*dphi;
    sy += cy[t]*dphi;
    sz += cz[t]*dphi;
  }

  gx[P]=sx;
  gy[P]=sy;
  gz[P]=sz;
}


__global__ static void kernel_apply_gauss_linear_gradient(
    int nFaces, int nInternalFaces,
    const int *owner, const int *neigh, const int *bPatch,
    const cfdScalar *ccx, const cfdScalar *ccy, const cfdScalar *ccz,
    const cfdScalar *xfx, const cfdScalar *xfy, const cfdScalar *xfz,
    const cfdScalar *sfx, const cfdScalar *sfy, const cfdScalar *sfz,
    const cfdScalar *vol,
    const cfdScalar *phi, const int *bcType, const cfdScalar *bcFaceValue,
    cfdScalar *gx, cfdScalar *gy, cfdScalar *gz)
{
  const int f = blockIdx.x*blockDim.x + threadIdx.x;
  if(f >= nFaces) return;

  const int P = owner[f];
  if(P < 0) return;

  double phiF = phi[P];
  if(f < nInternalFaces){
    const int N = neigh[f];
    if(N >= 0){
      const double dx = ccx[N] - ccx[P];
      const double dy = ccy[N] - ccy[P];
      const double dz = ccz[N] - ccz[P];
      const double d2 = dx*dx + dy*dy + dz*dz;
      double lam = 0.5;
      if(d2 > 1.0e-300){
        const double fx = xfx[f] - ccx[P];
        const double fy = xfy[f] - ccy[P];
        const double fz = xfz[f] - ccz[P];
        lam = (fx*dx + fy*dy + fz*dz)/d2;
        if(lam < 0.0) lam = 0.0;
        if(lam > 1.0) lam = 1.0;
      }
      phiF = (1.0 - lam)*phi[P] + lam*phi[N];
      const double invVP = (vol[P] > 1.0e-300) ? 1.0/vol[P] : 0.0;
      const double invVN = (vol[N] > 1.0e-300) ? 1.0/vol[N] : 0.0;
      atomicAdd(&gx[P],  sfx[f]*phiF*invVP);
      atomicAdd(&gy[P],  sfy[f]*phiF*invVP);
      atomicAdd(&gz[P],  sfz[f]*phiF*invVP);
      atomicAdd(&gx[N], -sfx[f]*phiF*invVN);
      atomicAdd(&gy[N], -sfy[f]*phiF*invVN);
      atomicAdd(&gz[N], -sfz[f]*phiF*invVN);
      return;
    }
  }

  const int patch = bPatch[f] - 1;
  if(patch >= 0 && bcType[patch] == 1) phiF = bcFaceValue[f];
  const double invVP = (vol[P] > 1.0e-300) ? 1.0/vol[P] : 0.0;
  atomicAdd(&gx[P], sfx[f]*phiF*invVP);
  atomicAdd(&gy[P], sfy[f]*phiF*invVP);
  atomicAdd(&gz[P], sfz[f]*phiF*invVP);
}

__global__ static void kernel_add_scaled_inplace(int n, cfdScalar *y, const cfdScalar *x, double a){
  int i = blockIdx.x*blockDim.x + threadIdx.x;
  if(i<n) y[i] += a*x[i];
}

__global__ static void kernel_update_pressure_relax(int n, cfdScalar *p, const cfdScalar *pcorr, double pRelax){
  int i = blockIdx.x*blockDim.x + threadIdx.x;
  if(i<n) p[i] += pRelax*pcorr[i];
}

__global__ static void kernel_subtract_scalar_inplace(int n, cfdScalar *x, double a){
  int i = blockIdx.x*blockDim.x + threadIdx.x;
  if(i<n) x[i] -= a;
}

__global__ static void kernel_relchg_reduce(int n, const cfdScalar *a, const cfdScalar *b, cfdScalar *blockNum, cfdScalar *blockDen){
  extern __shared__ cfdScalar sh[];
  cfdScalar *snum = sh;
  cfdScalar *sden = sh + blockDim.x;

  int tid = threadIdx.x;
  int i = blockIdx.x*blockDim.x + tid;

  double num = 0.0;
  double den = 0.0;
  if(i < n){
    const double av = a[i];
    const double d = av - b[i];
    num = d*d;
    den = av*av;
  }

  snum[tid] = num;
  sden[tid] = den;
  __syncthreads();

  for(int stride=blockDim.x/2; stride>0; stride>>=1){
    if(tid < stride){
      snum[tid] += snum[tid + stride];
      sden[tid] += sden[tid + stride];
    }
    __syncthreads();
  }

  if(tid == 0){
    blockNum[blockIdx.x] = snum[0];
    blockDen[blockIdx.x] = sden[0];
  }
}

__global__ static void kernel_maxabs_reduce(int n, const cfdScalar *x, cfdScalar *blockMax){
  extern __shared__ cfdScalar sdata[];

  int tid = threadIdx.x;
  int i = blockIdx.x*blockDim.x + tid;

  double v = 0.0;
  if(i<n) v = fabs(x[i]);

  sdata[tid] = v;
  __syncthreads();

  for(int stride=blockDim.x/2; stride>0; stride>>=1){
    if(tid < stride)
      sdata[tid] = fmax(sdata[tid], sdata[tid+stride]);
    __syncthreads();
  }

  if(tid==0)
    blockMax[blockIdx.x] = sdata[0];
}

__global__ static void kernel_build_rhiechow_predicted_flux_stokes_3d(
    int nFaces, int nInternalFaces,
    const int *owner, const int *neigh, const int *bPatch,
    const cfdScalar *ccx, const cfdScalar *ccy, const cfdScalar *ccz,
    const cfdScalar *xfx, const cfdScalar *xfy, const cfdScalar *xfz,
    const cfdScalar *nfx, const cfdScalar *nfy, const cfdScalar *nfz,
    const cfdScalar *Af,
    const cfdScalar *u, const cfdScalar *v, const cfdScalar *w,
    const cfdScalar *p,
    const cfdScalar *gradx, const cfdScalar *grady, const cfdScalar *gradz,
    const cfdScalar *rAU,
    const int *bcUType, const cfdScalar *uFaceBC,
    const int *bcVType, const cfdScalar *vFaceBC,
    const int *bcWType, const cfdScalar *wFaceBC,
    double rho,
    int rcMode, int hbyaBcMode,
    cfdScalar *phiStar)
{
  int f = blockIdx.x * blockDim.x + threadIdx.x;
  if(f >= nFaces) return;
  const int P = owner[f];
  if(f < nInternalFaces){
    const int N = neigh[f];
    const double dx = ccx[N] - ccx[P];
    const double dy = ccy[N] - ccy[P];
    const double dz = ccz[N] - ccz[P];
    const double dpn = nfx[f]*dx + nfy[f]*dy + nfz[f]*dz;
    const double denom = fmax(dx*dx + dy*dy + dz*dz, 1.0e-30);
    double lam = ((xfx[f]-ccx[P])*dx + (xfy[f]-ccy[P])*dy + (xfz[f]-ccz[P])*dz) / denom;
    lam = fmin(1.0, fmax(0.0, lam));
    const double ubar = (1.0-lam)*u[P] + lam*u[N];
    const double vbar = (1.0-lam)*v[P] + lam*v[N];
    const double wbar = (1.0-lam)*w[P] + lam*w[N];
    const double gpx = (1.0-lam)*gradx[P] + lam*gradx[N];
    const double gpy = (1.0-lam)*grady[P] + lam*grady[N];
    const double gpz = (1.0-lam)*gradz[P] + lam*gradz[N];
    const double rAUf = (1.0-lam)*rAU[P] + lam*rAU[N];
    const double phiInterp = rho * Af[f] * (ubar*nfx[f] + vbar*nfy[f] + wbar*nfz[f]);
    // rcMode 0: old explicit Rhie-Chow pressure-consistency term.
    // rcMode 1: OpenFOAM/Chalmers-style predictor phi = interpolate(U*) & Sf only.
    const double rc = (rcMode == 0)
      ? rho * Af[f] * rAUf * pressure_delta_coeff_runtime(dx, dy, dz, nfx[f], nfy[f], nfz[f]) *
          ((p[N] - p[P]) - (gpx*dx + gpy*dy + gpz*dz))
      : 0.0;
    phiStar[f] = phiInterp - rc;
  } else {
    const int patch = bPatch[f] - 1;
    const bool constrain = (hbyaBcMode != 0);
    const double uf = (constrain && bcUType[patch] == 1) ? uFaceBC[f] : u[P];
    const double vf = (constrain && bcVType[patch] == 1) ? vFaceBC[f] : v[P];
    const double wf = (constrain && bcWType[patch] == 1) ? wFaceBC[f] : w[P];
    phiStar[f] = rho * Af[f] * (uf*nfx[f] + vf*nfy[f] + wf*nfz[f]);
  }
}

__global__ static void kernel_continuity_residual_from_flux(
    int nFaces, int nInternalFaces,
    const int *owner, const int *neigh,
    const cfdScalar *phi, cfdScalar *divCell)
{
  int f = blockIdx.x * blockDim.x + threadIdx.x;
  if(f >= nFaces) return;
  int P = owner[f];
  atomicAdd(&divCell[P], phi[f]);
  if(f < nInternalFaces){
    int N = neigh[f];
    atomicAdd(&divCell[N], -phi[f]);
  }
}

__global__ static void kernel_pressure_nonorth_flux_and_divergence(
    int nFaces, int nInternalFaces,
    const int *owner, const int *neigh, const int *bPatch,
    const cfdScalar *ccx, const cfdScalar *ccy, const cfdScalar *ccz,
    const cfdScalar *xfx, const cfdScalar *xfy, const cfdScalar *xfz,
    const cfdScalar *nfx, const cfdScalar *nfy, const cfdScalar *nfz,
    const cfdScalar *sfx, const cfdScalar *sfy, const cfdScalar *sfz,
    const cfdScalar *Af, const cfdScalar *rAU, double rho,
    const int *bcPType, int skipBoundaryNonOrth,
    const cfdScalar *gradx, const cfdScalar *grady, const cfdScalar *gradz,
    cfdScalar *phiNonOrth, cfdScalar *divNonOrth)
{
  int f = blockIdx.x * blockDim.x + threadIdx.x;
  if(f >= nFaces) return;
  double flux = 0.0;
  if(f < nInternalFaces){
    int P = owner[f];
    int N = neigh[f];
    double dx = ccx[N] - ccx[P];
    double dy = ccy[N] - ccy[P];
    double dz = ccz[N] - ccz[P];
    double dpn = nfx[f]*dx + nfy[f]*dy + nfz[f]*dz;
    (void)dpn;
    {
      const double deltaCoeff = pressure_delta_coeff_runtime(dx, dy, dz, nfx[f], nfy[f], nfz[f]);
      double denom = fmax(dx*dx + dy*dy + dz*dz, 1.0e-30);
      double lam = ((xfx[f]-ccx[P])*dx + (xfy[f]-ccy[P])*dy + (xfz[f]-ccz[P])*dz) / denom;
      lam = fmin(1.0, fmax(0.0, lam));
      double rAUf = (1.0-lam)*rAU[P] + lam*rAU[N];
      double gx = (1.0-lam)*gradx[P] + lam*gradx[N];
      double gy = (1.0-lam)*grady[P] + lam*grady[N];
      double gz = (1.0-lam)*gradz[P] + lam*gradz[N];
      double tx = sfx[f] - (Af[f]*deltaCoeff)*dx;
      double ty = sfy[f] - (Af[f]*deltaCoeff)*dy;
      double tz = sfz[f] - (Af[f]*deltaCoeff)*dz;
      flux = rho * rAUf * (gx*tx + gy*ty + gz*tz);
      atomicAdd(&divNonOrth[P], flux);
      atomicAdd(&divNonOrth[N], -flux);
    }
  } else {
    if(skipBoundaryNonOrth){ phiNonOrth[f] = 0.0; return; }
    int patch = bPatch[f] - 1;
    if(patch >= 0 && bcPType[patch] == 1){
      int P = owner[f];
      double dx = xfx[f] - ccx[P];
      double dy = xfy[f] - ccy[P];
      double dz = xfz[f] - ccz[P];
      double dpn = nfx[f]*dx + nfy[f]*dy + nfz[f]*dz;
      (void)dpn;
      {
        const double deltaCoeff = pressure_delta_coeff_runtime(dx, dy, dz, nfx[f], nfy[f], nfz[f]);
        double tx = sfx[f] - (Af[f]*deltaCoeff)*dx;
        double ty = sfy[f] - (Af[f]*deltaCoeff)*dy;
        double tz = sfz[f] - (Af[f]*deltaCoeff)*dz;
        flux = rho * rAU[P] * (gradx[P]*tx + grady[P]*ty + gradz[P]*tz);
        atomicAdd(&divNonOrth[P], flux);
      }
    }
  }
  phiNonOrth[f] = flux;
}

__global__ static void kernel_build_pressure_rhs_from_divs(
    int nCells, const cfdScalar *divBase, const cfdScalar *divNonOrth, double nonOrthScale,
    int useAnchor, int refCell, HYPRE_Complex *rhs)
{
  int c = blockIdx.x * blockDim.x + threadIdx.x;
  if(c < nCells){
    rhs[c] = (useAnchor && c == refCell) ? (HYPRE_Complex)0.0 :
             (HYPRE_Complex)(-divBase[c] + nonOrthScale * divNonOrth[c]);
  }
}

__global__ static void kernel_build_pressure_rhs_minus_div(
    int nCells, const cfdScalar *divBase, int useAnchor, int refCell, HYPRE_Complex *rhs)
{
  int c = blockIdx.x * blockDim.x + threadIdx.x;
  if(c < nCells){
    rhs[c] = (useAnchor && c == refCell) ? (HYPRE_Complex)0.0 : (HYPRE_Complex)(-divBase[c]);
  }
}

__global__ static void kernel_correct_face_fluxes_simple_nonorth(
    int nFaces, int nInternalFaces,
    const int *owner, const int *neigh, const int *bPatch,
    const cfdScalar *ccx, const cfdScalar *ccy, const cfdScalar *ccz,
    const cfdScalar *xfx, const cfdScalar *xfy, const cfdScalar *xfz,
    const cfdScalar *nfx, const cfdScalar *nfy, const cfdScalar *nfz,
    const cfdScalar *Af, const cfdScalar *rAU, double rho,
    const int *bcPType, const cfdScalar *pFaceBC,
    const cfdScalar *phiStar, const cfdScalar *pCorr, const cfdScalar *phiNonOrth,
    double nonOrthScale, double pCoeffScale, int pFluxMode,
    int fixedVelocityBoundaryPressureEqn,
    cfdScalar *phi)
{
  int f = blockIdx.x * blockDim.x + threadIdx.x;
  if(f >= nFaces) return;
  (void)fixedVelocityBoundaryPressureEqn;
  if(f < nInternalFaces){
    int P = owner[f];
    int N = neigh[f];
    double dx = ccx[N] - ccx[P];
    double dy = ccy[N] - ccy[P];
    double dz = ccz[N] - ccz[P];
    double dpn = nfx[f]*dx + nfy[f]*dy + nfz[f]*dz;
    double denom = fmax(dx*dx + dy*dy + dz*dz, 1.0e-30);
    double lam = ((xfx[f]-ccx[P])*dx + (xfy[f]-ccy[P])*dy + (xfz[f]-ccz[P])*dz) / denom;
    lam = fmin(1.0, fmax(0.0, lam));
    double rAUf = (1.0-lam)*rAU[P] + lam*rAU[N];
    double coeff = pCoeffScale * rho * Af[f] * rAUf * pressure_delta_coeff_runtime(dx, dy, dz, nfx[f], nfy[f], nfz[f]);
    // Matrix-consistent pressure-equation flux.  pFluxMode is currently kept
    // as an explicit runtime switch for audit/testing; both modes use the
    // same sign convention as the pressure matrix for internal faces.
    double pEqnFlux = coeff*(pCorr[N] - pCorr[P]) + nonOrthScale * phiNonOrth[f];
    (void)pFluxMode;
    phi[f] = phiStar[f] - pEqnFlux;
  } else {
    int patch = bPatch[f] - 1;
    if(patch >= 0 && bcPType[patch] == 1){
      int P = owner[f];
      double dx = xfx[f] - ccx[P];
      double dy = xfy[f] - ccy[P];
      double dz = xfz[f] - ccz[P];
      double dpn = nfx[f]*dx + nfy[f]*dy + nfz[f]*dz;
      (void)dpn;
      double coeff = pCoeffScale * rho * Af[f] * rAU[P] * pressure_delta_coeff_runtime(dx, dy, dz, nfx[f], nfy[f], nfz[f]);
      // Matrix-consistent fixed-pressure boundary flux.  Zero-gradient
      // boundaries are handled by the else branch below and have pEqnFlux=0.
      double pEqnFlux = coeff*(pFaceBC[f] - pCorr[P]) + nonOrthScale * phiNonOrth[f];
      (void)pFluxMode;
      phi[f] = phiStar[f] - pEqnFlux;
    } else {
      phi[f] = phiStar[f];
    }
  }
}

__global__ static void kernel_correct_velocity_from_pcorr_grad(
    int nCells, const cfdScalar *uStar, const cfdScalar *vStar, const cfdScalar *wStar,
    const cfdScalar *rAU, const cfdScalar *gradx, const cfdScalar *grady, const cfdScalar *gradz,
    cfdScalar *u, cfdScalar *v, cfdScalar *w)
{
  int c = blockIdx.x * blockDim.x + threadIdx.x;
  if(c < nCells){
    u[c] = uStar[c] - rAU[c] * gradx[c];
    v[c] = vStar[c] - rAU[c] * grady[c];
    w[c] = wStar[c] - rAU[c] * gradz[c];
  }
}

__global__ static void kernel_add_rau_grad_to_velocity(
    int nCells, cfdScalar *u, cfdScalar *v, cfdScalar *w,
    const cfdScalar *rAU, const cfdScalar *gradx, const cfdScalar *grady, const cfdScalar *gradz)
{
  int c = blockIdx.x * blockDim.x + threadIdx.x;
  if(c < nCells){
    u[c] += rAU[c] * gradx[c];
    v[c] += rAU[c] * grady[c];
    w[c] += rAU[c] * gradz[c];
  }
}

__global__ static void kernel_update_pressure_absolute_relax(
    int n, cfdScalar *p, const cfdScalar *pAbs, double pRelax)
{
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if(i < n) p[i] = p[i] + pRelax * (pAbs[i] - p[i]);
}




static __host__ __device__ __forceinline__ void sine_mms_exact_fields(
    double x, double y, double z,
    double velScale, double pScale,
    double &u, double &v, double &w, double &p)
{
  const double k = 3.141592653589793238462643383279502884;
  const double sx = sin(k*x), sy = sin(k*y), sz = sin(k*z);
  const double tx = sin(2.0*k*x), ty = sin(2.0*k*y);
  const double cx = cos(2.0*k*x), cy = cos(2.0*k*y), cz = cos(2.0*k*z);
  const double Sx = sx*sx, Sy = sy*sy, Sz = sz*sz;

  u =  velScale * k * Sx * ty * Sz;
  v = -velScale * k * tx * Sy * Sz;
  w = 0.0;
  p = pScale * cx * cy * cz;
}

static __host__ __device__ __forceinline__ double sine_mms_source_component(
    double x, double y, double z,
    double rho, double mu,
    double velScale, double pScale,
    int comp)
{
  const double k = 3.141592653589793238462643383279502884;
  const double sx = sin(k*x), sy = sin(k*y), sz = sin(k*z);
  const double Sx = sx*sx, Sy = sy*sy, Sz = sz*sz;
  const double tx = sin(2.0*k*x), ty = sin(2.0*k*y), tz = sin(2.0*k*z);
  const double cx = cos(2.0*k*x), cy = cos(2.0*k*y), cz = cos(2.0*k*z);

  const double A = velScale;
  const double B = pScale;

  const double u  =  A*k*Sx*ty*Sz;
  const double v  = -A*k*tx*Sy*Sz;

  const double ux =  A*k*k*tx*ty*Sz;
  const double uy =  2.0*A*k*k*Sx*cy*Sz;
  const double vx = -2.0*A*k*k*cx*Sy*Sz;
  const double vy = -A*k*k*tx*ty*Sz;

  const double lapu = A*k*k*k*(2.0*cx*ty*Sz - 4.0*Sx*ty*Sz + 2.0*Sx*ty*cz);
  const double lapv = A*k*k*k*(4.0*tx*Sy*Sz - 2.0*tx*cy*Sz - 2.0*tx*Sy*cz);

  const double px = -2.0*B*k*tx*cy*cz;
  const double py = -2.0*B*k*cx*ty*cz;
  const double pz = -2.0*B*k*cx*cy*tz;

  if(comp == 0){
    const double convu = u*ux + v*uy;
    return rho*convu - mu*lapu + px;
  }
  if(comp == 1){
    const double convv = u*vx + v*vy;
    return rho*convv - mu*lapv + py;
  }
  return pz;
}

__global__ static void kernel_sine_mms_init_fields(
    int nCells,
    const cfdScalar *ccx, const cfdScalar *ccy, const cfdScalar *ccz,
    double velScale, double pScale,
    cfdScalar *u, cfdScalar *v, cfdScalar *w, cfdScalar *p)
{
  int c = blockIdx.x * blockDim.x + threadIdx.x;
  if(c >= nCells) return;

  double ue, ve, we, pe;
  sine_mms_exact_fields(ccx[c], ccy[c], ccz[c], velScale, pScale, ue, ve, we, pe);

  u[c] = (cfdScalar)ue;
  v[c] = (cfdScalar)ve;
  w[c] = (cfdScalar)we;
  p[c] = (cfdScalar)pe;
}

__global__ static void kernel_sine_mms_add_momentum_source(
    int nCells,
    const cfdScalar *ccx, const cfdScalar *ccy, const cfdScalar *ccz,
    const cfdScalar *vol,
    double rho, double mu,
    double velScale, double pScale,
    int comp,
    HYPRE_Complex *rhs)
{
  int c = blockIdx.x * blockDim.x + threadIdx.x;
  if(c >= nCells) return;

  const double src = sine_mms_source_component(
      ccx[c], ccy[c], ccz[c], rho, mu, velScale, pScale, comp);

  rhs[c] += (HYPRE_Complex)(src * vol[c]);
}

struct SineMmsErrorReport {
  double uRelL2=0.0, pRelL2=0.0, pRelL2MeanShift=0.0;
  double uAbsL2=0.0, pAbsL2=0.0, pAbsL2MeanShift=0.0;
  double pMeanError=0.0, maxUerr=0.0, maxPerr=0.0;
};

static SineMmsErrorReport compute_sine_mms_error_report(
    const Mesh &mesh,
    const std::vector<cfdScalar> &u,
    const std::vector<cfdScalar> &v,
    const std::vector<cfdScalar> &w,
    const std::vector<cfdScalar> &p,
    double velScale,
    double pScale)
{
  SineMmsErrorReport r;
  long double eu2=0.0L, uu2=0.0L, ep2=0.0L, pp2=0.0L;
  long double epMean=0.0L, volSum=0.0L;

  for(int c=0;c<mesh.nCells;++c){
    double ue, ve, we, pe;
    sine_mms_exact_fields(mesh.cc[c][0], mesh.cc[c][1], mesh.cc[c][2],
                          velScale, pScale, ue, ve, we, pe);

    const double V = mesh.vol[c];
    const double du = (double)u[c] - ue;
    const double dv = (double)v[c] - ve;
    const double dw = (double)w[c] - we;
    const double dp = (double)p[c] - pe;

    eu2 += (long double)V*(du*du + dv*dv + dw*dw);
    uu2 += (long double)V*(ue*ue + ve*ve + we*we);
    ep2 += (long double)V*dp*dp;
    pp2 += (long double)V*pe*pe;
    epMean += (long double)V*dp;
    volSum += (long double)V;

    r.maxUerr = std::max(r.maxUerr, std::sqrt(du*du + dv*dv + dw*dw));
    r.maxPerr = std::max(r.maxPerr, std::fabs(dp));
  }

  const double meanErr = (volSum > 0.0L) ? (double)(epMean/volSum) : 0.0;

  long double epShift2=0.0L;
  for(int c=0;c<mesh.nCells;++c){
    double ue, ve, we, pe;
    sine_mms_exact_fields(mesh.cc[c][0], mesh.cc[c][1], mesh.cc[c][2],
                          velScale, pScale, ue, ve, we, pe);

    const double V = mesh.vol[c];
    const double dp = ((double)p[c] - pe) - meanErr;
    epShift2 += (long double)V*dp*dp;
  }

  r.uAbsL2 = std::sqrt((double)eu2);
  r.pAbsL2 = std::sqrt((double)ep2);
  r.pAbsL2MeanShift = std::sqrt((double)epShift2);

  r.uRelL2 = r.uAbsL2 / std::max(std::sqrt((double)uu2), 1.0e-300);
  r.pRelL2 = r.pAbsL2 / std::max(std::sqrt((double)pp2), 1.0e-300);
  r.pRelL2MeanShift = r.pAbsL2MeanShift / std::max(std::sqrt((double)pp2), 1.0e-300);
  r.pMeanError = meanErr;

  return r;
}

__global__ static void kernel_zero_values(HYPRE_Complex *vals, int n){
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if(i < n) vals[i] = 0.0;
}

__global__ static void kernel_momentum_base_steady(int nCells, const int *diagPos, const cfdScalar *vol, const cfdScalar *gradPcomp, HYPRE_Complex *vals, HYPRE_Complex *rhs){
  int c = blockIdx.x * blockDim.x + threadIdx.x;
  if(c < nCells){
    vals[diagPos[c]] = 0.0;
    rhs[c] = -gradPcomp[c] * vol[c];
  }
}

__global__ static void kernel_relax_momentum_system(int nCells, const int *diagPos, HYPRE_Complex *vals, HYPRE_Complex *rhs, const cfdScalar *qOld, double uRelax){
  int c = blockIdx.x * blockDim.x + threadIdx.x;
  if(c < nCells){
    int diag = diagPos[c];
    double aP = (double)vals[diag];
    double invRelax = 1.0 / uRelax;
    vals[diag] = (HYPRE_Complex)(aP * invRelax);
    rhs[c] += (HYPRE_Complex)((invRelax - 1.0) * aP * qOld[c]);
  }
}

__global__ static void kernel_zero_rhs(HYPRE_Complex *rhs, int n){
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if(i < n) rhs[i] = 0.0;
}

__global__ static void kernel_momentum_base_rhs_only(int nCells, const cfdScalar *vol, const cfdScalar *gradPcomp, HYPRE_Complex *rhs){
  int c = blockIdx.x * blockDim.x + threadIdx.x;
  if(c < nCells) rhs[c] = -gradPcomp[c] * vol[c];
}

__global__ static void kernel_momentum_internal_faces_rhs_only(
    int nInternalFaces,
    const int *owner, const int *neigh,
    const cfdScalar *ccx, const cfdScalar *ccy, const cfdScalar *ccz,
    const cfdScalar *xfx, const cfdScalar *xfy, const cfdScalar *xfz,
    const cfdScalar *nfx, const cfdScalar *nfy, const cfdScalar *nfz,
    const cfdScalar *sfx, const cfdScalar *sfy, const cfdScalar *sfz,
    const cfdScalar *Af,
    const cfdScalar *gradQx, const cfdScalar *gradQy, const cfdScalar *gradQz,
    double mu, double corrPsi,
    HYPRE_Complex *rhs)
{
  int f = blockIdx.x * blockDim.x + threadIdx.x;
  if(f >= nInternalFaces) return;
  int P = owner[f];
  int N = neigh[f];
  double dx = ccx[N] - ccx[P];
  double dy = ccy[N] - ccy[P];
  double dz = ccz[N] - ccz[P];
  double nd = nfx[f]*dx + nfy[f]*dy + nfz[f]*dz;
  if(nd <= 1e-14) return;
  double af = Af[f];
  double tcorx = mu * (sfx[f] - (af/nd)*dx);
  double tcory = mu * (sfy[f] - (af/nd)*dy);
  double tcorz = mu * (sfz[f] - (af/nd)*dz);
  double denom = dx*dx + dy*dy + dz*dz;
  double lam = ((xfx[f]-ccx[P])*dx + (xfy[f]-ccy[P])*dy + (xfz[f]-ccz[P])*dz) / (denom > 1e-30 ? denom : 1e-30);
  lam = fmin(1.0, fmax(0.0, lam));
  double gradfx = (1.0-lam)*gradQx[P] + lam*gradQx[N];
  double gradfy = (1.0-lam)*gradQy[P] + lam*gradQy[N];
  double gradfz = (1.0-lam)*gradQz[P] + lam*gradQz[N];
  double corr = corrPsi * (gradfx*tcorx + gradfy*tcory + gradfz*tcorz);
  hypreAtomicAdd(&rhs[P], (HYPRE_Complex)corr);
  hypreAtomicAdd(&rhs[N], (HYPRE_Complex)(-corr));
}

__global__ static void kernel_momentum_boundary_faces_rhs_only(
    int nBoundaryFaces, int faceStart,
    const int *owner, const int *bPatch,
    const cfdScalar *ccx, const cfdScalar *ccy, const cfdScalar *ccz,
    const cfdScalar *xfx, const cfdScalar *xfy, const cfdScalar *xfz,
    const cfdScalar *nfx, const cfdScalar *nfy, const cfdScalar *nfz,
    const cfdScalar *sfx, const cfdScalar *sfy, const cfdScalar *sfz,
    const cfdScalar *Af,
    const cfdScalar *gradQx, const cfdScalar *gradQy, const cfdScalar *gradQz,
    const cfdScalar *uConv, const cfdScalar *vConv, const cfdScalar *wConv,
    const int *bcQType, const cfdScalar *bcQFaceVal,
    const int *bcUType, const cfdScalar *bcUFaceVal,
    const int *bcVType, const cfdScalar *bcVFaceVal,
    const int *bcWType, const cfdScalar *bcWFaceVal,
    double rho, double mu, double corrPsi,
    HYPRE_Complex *rhs)
{
  int ib = blockIdx.x * blockDim.x + threadIdx.x;
  if(ib >= nBoundaryFaces) return;
  int f = faceStart + ib;
  int P = owner[f];
  int patch = bPatch[f] - 1;
  double dx = xfx[f] - ccx[P];
  double dy = xfy[f] - ccy[P];
  double dz = xfz[f] - ccz[P];
  double nd = nfx[f]*dx + nfy[f]*dy + nfz[f]*dz;
  if(nd <= 1e-14) return;
  double af = Af[f];
  double alpha = mu * af / nd;
  double tcorx = mu * (sfx[f] - (af/nd)*dx);
  double tcory = mu * (sfy[f] - (af/nd)*dy);
  double tcorz = mu * (sfz[f] - (af/nd)*dz);
  if(bcQType[patch] == 1){
    double g = bcQFaceVal[f];
    double corr = corrPsi * (gradQx[P]*tcorx + gradQy[P]*tcory + gradQz[P]*tcorz);
    hypreAtomicAdd(&rhs[P], (HYPRE_Complex)(alpha*g + corr));
  }
  double ucb = (bcUType[patch] == 1) ? bcUFaceVal[f] : uConv[P];
  double vcb = (bcVType[patch] == 1) ? bcVFaceVal[f] : vConv[P];
  double wcb = (bcWType[patch] == 1) ? bcWFaceVal[f] : wConv[P];
  double F = rho * af * (ucb*nfx[f] + vcb*nfy[f] + wcb*nfz[f]);
  if(bcQType[patch] == 1) hypreAtomicAdd(&rhs[P], (HYPRE_Complex)(-F * bcQFaceVal[f]));
}

__global__ static void kernel_relax_momentum_rhs_only_from_relaxed_matrix(int nCells, const int *diagPos, const HYPRE_Complex *vals, HYPRE_Complex *rhs, const cfdScalar *qOld, double uRelax){
  int c = blockIdx.x * blockDim.x + threadIdx.x;
  if(c < nCells){
    double aP_old = ((double)vals[diagPos[c]]) * uRelax;
    double invRelax = 1.0 / uRelax;
    rhs[c] += (HYPRE_Complex)((invRelax - 1.0) * aP_old * qOld[c]);
  }
}

__global__ static void kernel_extract_vol_over_diag(int nCells, const int *diagPos, const HYPRE_Complex *vals, const cfdScalar *vol, cfdScalar *rAU, double diagScale, double rAUScale){
  int c = blockIdx.x * blockDim.x + threadIdx.x;
  if(c < nCells){
    // vals[diagPos[c]] is normally the matrix diagonal currently being solved.
    // After equation relaxation this is aP_relaxed = aP_raw/uRelax.
    // diagScale=1      -> V/aP_relaxed
    // diagScale=uRelax -> V/aP_raw
    double a = ((double) vals[diagPos[c]]) * diagScale;
    rAU[c] = (fabs(a) > 1e-30) ? rAUScale * vol[c] / a : 0.0;
  }
}

__global__ static void kernel_momentum_internal_faces(
    int nInternalFaces,
    const int *owner, const int *neigh,
    const cfdScalar *ccx, const cfdScalar *ccy, const cfdScalar *ccz,
    const cfdScalar *xfx, const cfdScalar *xfy, const cfdScalar *xfz,
    const cfdScalar *nfx, const cfdScalar *nfy, const cfdScalar *nfz,
    const cfdScalar *sfx, const cfdScalar *sfy, const cfdScalar *sfz,
    const cfdScalar *Af,
    const cfdScalar *gradQx, const cfdScalar *gradQy, const cfdScalar *gradQz,
    const cfdScalar *uConv, const cfdScalar *vConv, const cfdScalar *wConv,
    double rho, double mu, double corrPsi, int momentumConvectionScheme,
    const int *diagPos, const int *facePN, const int *faceNP,
    HYPRE_Complex *vals, HYPRE_Complex *rhs)
{
  int f = blockIdx.x * blockDim.x + threadIdx.x;
  if(f >= nInternalFaces) return;
  int P = owner[f];
  int N = neigh[f];
  double dx = ccx[N] - ccx[P];
  double dy = ccy[N] - ccy[P];
  double dz = ccz[N] - ccz[P];
  double nd = nfx[f]*dx + nfy[f]*dy + nfz[f]*dz;
  if(nd <= 1e-14) return;
  double af = Af[f];
  double alpha = mu * af / nd;
  double tcorx = mu * (sfx[f] - (af/nd)*dx);
  double tcory = mu * (sfy[f] - (af/nd)*dy);
  double tcorz = mu * (sfz[f] - (af/nd)*dz);
  double denom = dx*dx + dy*dy + dz*dz;
  double lam = ((xfx[f]-ccx[P])*dx + (xfy[f]-ccy[P])*dy + (xfz[f]-ccz[P])*dz) / (denom > 1e-30 ? denom : 1e-30);
  lam = fmin(1.0, fmax(0.0, lam));
  double gradfx = (1.0-lam)*gradQx[P] + lam*gradQx[N];
  double gradfy = (1.0-lam)*gradQy[P] + lam*gradQy[N];
  double gradfz = (1.0-lam)*gradQz[P] + lam*gradQz[N];
  double corr = corrPsi * (gradfx*tcorx + gradfy*tcory + gradfz*tcorz);
  double ucf = (1.0-lam)*uConv[P] + lam*uConv[N];
  double vcf = (1.0-lam)*vConv[P] + lam*vConv[N];
  double wcf = (1.0-lam)*wConv[P] + lam*wConv[N];
  double F = rho * af * (ucf*nfx[f] + vcf*nfy[f] + wcf*nfz[f]);
  if(momentumConvectionScheme == 2) F = 0.0;

  double cPP=0.0, cPN=0.0, cNP=0.0, cNN=0.0;
  if(momentumConvectionScheme == 1){
    // First-order upwind for div(phi,q), matching libscalar upwind algebra.
    // F is owner-outward. If F >= 0, owner value is upwind; otherwise neighbour.
    if(F >= 0.0){
      cPP += F;
      cNP -= F;
    } else {
      cPN += F;
      cNN -= F;
    }
  } else {
    // Central/linear interpolation to the face.
    cPP += F*(1.0-lam);
    cPN += F*lam;
    cNP -= F*(1.0-lam);
    cNN -= F*lam;
  }

  hypreAtomicAdd(&vals[diagPos[P]], (HYPRE_Complex)(alpha + cPP));
  hypreAtomicAdd(&vals[facePN[f]], (HYPRE_Complex)(-alpha + cPN));
  hypreAtomicAdd(&vals[faceNP[f]], (HYPRE_Complex)(-alpha + cNP));
  hypreAtomicAdd(&vals[diagPos[N]], (HYPRE_Complex)(alpha + cNN));
  hypreAtomicAdd(&rhs[P], (HYPRE_Complex)corr);
  hypreAtomicAdd(&rhs[N], (HYPRE_Complex)(-corr));
}

__global__ static void kernel_momentum_boundary_faces(
    int nBoundaryFaces, int faceStart,
    const int *owner, const int *bPatch,
    const cfdScalar *ccx, const cfdScalar *ccy, const cfdScalar *ccz,
    const cfdScalar *xfx, const cfdScalar *xfy, const cfdScalar *xfz,
    const cfdScalar *nfx, const cfdScalar *nfy, const cfdScalar *nfz,
    const cfdScalar *sfx, const cfdScalar *sfy, const cfdScalar *sfz,
    const cfdScalar *Af,
    const cfdScalar *gradQx, const cfdScalar *gradQy, const cfdScalar *gradQz,
    const cfdScalar *uConv, const cfdScalar *vConv, const cfdScalar *wConv,
    const int *bcQType, const cfdScalar *bcQFaceVal,
    const int *bcUType, const cfdScalar *bcUFaceVal,
    const int *bcVType, const cfdScalar *bcVFaceVal,
    const int *bcWType, const cfdScalar *bcWFaceVal,
    double rho, double mu, double corrPsi,
    const int *diagPos, HYPRE_Complex *vals, HYPRE_Complex *rhs)
{
  int ib = blockIdx.x * blockDim.x + threadIdx.x;
  if(ib >= nBoundaryFaces) return;
  int f = faceStart + ib;
  int P = owner[f];
  int patch = bPatch[f] - 1;
  double dx = xfx[f] - ccx[P];
  double dy = xfy[f] - ccy[P];
  double dz = xfz[f] - ccz[P];
  double nd = nfx[f]*dx + nfy[f]*dy + nfz[f]*dz;
  if(nd <= 1e-14) return;
  double af = Af[f];
  double alpha = mu * af / nd;
  double tcorx = mu * (sfx[f] - (af/nd)*dx);
  double tcory = mu * (sfy[f] - (af/nd)*dy);
  double tcorz = mu * (sfz[f] - (af/nd)*dz);
  int diag = diagPos[P];
  if(bcQType[patch] == 1){
    double g = bcQFaceVal[f];
    double corr = corrPsi * (gradQx[P]*tcorx + gradQy[P]*tcory + gradQz[P]*tcorz);
    hypreAtomicAdd(&vals[diag], (HYPRE_Complex)alpha);
    hypreAtomicAdd(&rhs[P], (HYPRE_Complex)(alpha*g + corr));
  }
  double ucb = (bcUType[patch] == 1) ? bcUFaceVal[f] : uConv[P];
  double vcb = (bcVType[patch] == 1) ? bcVFaceVal[f] : vConv[P];
  double wcb = (bcWType[patch] == 1) ? bcWFaceVal[f] : wConv[P];
  double F = rho * af * (ucb*nfx[f] + vcb*nfy[f] + wcb*nfz[f]);
  if(bcQType[patch] == 1) hypreAtomicAdd(&rhs[P], (HYPRE_Complex)(-F * bcQFaceVal[f]));
  else hypreAtomicAdd(&vals[diag], (HYPRE_Complex)F);
}

__global__ static void kernel_pressure_internal_faces_rau(
    int nInternalFaces,
    const int *owner, const int *neigh,
    const cfdScalar *ccx, const cfdScalar *ccy, const cfdScalar *ccz,
    const cfdScalar *xfx, const cfdScalar *xfy, const cfdScalar *xfz,
    const cfdScalar *nfx, const cfdScalar *nfy, const cfdScalar *nfz,
    const cfdScalar *Af, const cfdScalar *rAU, double rho, double pCoeffScale,
    const int *diagPos, const int *facePN, const int *faceNP,
    HYPRE_Complex *vals)
{
  int f = blockIdx.x * blockDim.x + threadIdx.x;
  if(f >= nInternalFaces) return;
  int P = owner[f];
  int N = neigh[f];
  double dx = ccx[N] - ccx[P];
  double dy = ccy[N] - ccy[P];
  double dz = ccz[N] - ccz[P];
  double dpn = nfx[f]*dx + nfy[f]*dy + nfz[f]*dz;
  (void)dpn;
  double denom = dx*dx + dy*dy + dz*dz;
  double lam = ((xfx[f]-ccx[P])*dx + (xfy[f]-ccy[P])*dy + (xfz[f]-ccz[P])*dz) / (denom > 1e-30 ? denom : 1e-30);
  lam = fmin(1.0, fmax(0.0, lam));
  double rAUf = (1.0-lam)*rAU[P] + lam*rAU[N];
  double coeff = pCoeffScale * rho * Af[f] * rAUf * pressure_delta_coeff_runtime(dx, dy, dz, nfx[f], nfy[f], nfz[f]);
  hypreAtomicAdd(&vals[diagPos[P]], (HYPRE_Complex)coeff);
  if(facePN[f] >= 0) hypreAtomicAdd(&vals[facePN[f]], (HYPRE_Complex)(-coeff));
  if(faceNP[f] >= 0) hypreAtomicAdd(&vals[faceNP[f]], (HYPRE_Complex)(-coeff));
  hypreAtomicAdd(&vals[diagPos[N]], (HYPRE_Complex)coeff);
}

__global__ static void kernel_pressure_boundary_faces_rau(
    int nBoundaryFaces, int faceStart,
    const int *owner, const int *bPatch,
    const cfdScalar *ccx, const cfdScalar *ccy, const cfdScalar *ccz,
    const cfdScalar *xfx, const cfdScalar *xfy, const cfdScalar *xfz,
    const cfdScalar *nfx, const cfdScalar *nfy, const cfdScalar *nfz,
    const cfdScalar *Af, const cfdScalar *rAU, double rho, double pCoeffScale,
    const int *bcPType, const int *diagPos, HYPRE_Complex *vals)
{
  int ib = blockIdx.x * blockDim.x + threadIdx.x;
  if(ib >= nBoundaryFaces) return;
  int f = faceStart + ib;
  int patch = bPatch[f] - 1;
  if(bcPType[patch] != 1) return;
  int P = owner[f];
  double dx = xfx[f] - ccx[P];
  double dy = xfy[f] - ccy[P];
  double dz = xfz[f] - ccz[P];
  double dpn = nfx[f]*dx + nfy[f]*dy + nfz[f]*dz;
  (void)dpn;
  double coeff = pCoeffScale * rho * Af[f] * rAU[P] * pressure_delta_coeff_runtime(dx, dy, dz, nfx[f], nfy[f], nfz[f]);
  hypreAtomicAdd(&vals[diagPos[P]], (HYPRE_Complex)coeff);
}

__global__ static void kernel_pressure_anchor(int refCell, const int *diagPos, HYPRE_Complex *vals){
  vals[diagPos[refCell]] = 1.0;
}

__global__ static void kernel_add_pressure_diag_penalty(
    int nCells,
    const int *diagPos,
    HYPRE_Complex *vals,
    double beta)
{
  const int c = blockIdx.x * blockDim.x + threadIdx.x;
  if(c < nCells && beta != 0.0){
    vals[diagPos[c]] = (HYPRE_Complex)((double)vals[diagPos[c]] + beta);
  }
}

__global__ static void kernel_add_simple_mms_pressure_penalty_rhs(
    int nCells,
    const double *ccx,
    const double *ccy,
    const double *ccz,
    double rho,
    double mu,
    double U0,
    double P0,
    int stokesOnly,
    double beta,
    HYPRE_Complex *rhs)
{
  const int c = blockIdx.x * blockDim.x + threadIdx.x;
  if(c >= nCells || beta == 0.0) return;

  const auto q = simple_mms_exact_value(
      ccx[c], ccy[c], ccz[c],
      rho, mu, U0, P0, stokesOnly);

  rhs[c] = (HYPRE_Complex)((double)rhs[c] + beta * q.p);
}

__global__ static void kernel_scatter_values_perm(int n, const HYPRE_Complex *src, const int *perm, HYPRE_Complex *dst){
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if(i < n) dst[perm[i]] = src[i];
}

__global__ static void kernel_remap_matrix_indices_inplace(int n, int *idx, const int *perm){
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if(i < n){
    const int old = idx[i];
    idx[i] = (old >= 0) ? perm[old] : -1;
  }
}


static void create_ij_matrix_from_pattern(const MatrixPattern &pat, HYPRE_IJMatrix &Aij){
  HYPRE_BigInt ilower=0, iupper=(HYPRE_BigInt)pat.nRows-1;
  HYPRE_CALL(HYPRE_IJMatrixCreate(MPI_COMM_WORLD, ilower, iupper, ilower, iupper, &Aij));
  HYPRE_CALL(HYPRE_IJMatrixSetObjectType(Aij, HYPRE_PARCSR));
  HYPRE_CALL(HYPRE_IJMatrixSetRowSizes(Aij, pat.ncols.data()));
}

static void build_diag_permutation_from_host_parcsr(GPULinearSystem &sys){
  hypre_CSRMatrix *diag = hypre_ParCSRMatrixDiag(sys.Apar);
  HYPRE_Int *I = hypre_CSRMatrixI(diag);
  HYPRE_Int *J = hypre_CSRMatrixJ(diag);
  if(!I || !J) throw std::runtime_error("Failed to access host ParCSR diag I/J arrays");
  sys.A_diag_perm_h.assign(sys.pat.nnz, -1);
  bool identity = true;
  for(int r=0; r<sys.pat.nRows; ++r){
    int p0 = sys.pat.rowOffsets[r], p1 = sys.pat.rowOffsets[r+1];
    int h0 = I[r], h1 = I[r+1];
    if((p1-p0) != (h1-h0)){
      std::ostringstream oss;
      oss << "Row nnz mismatch while building ParCSR permutation at row " << r
          << ": pattern=" << (p1-p0) << ", hypre=" << (h1-h0);
      throw std::runtime_error(oss.str());
    }
    for(int p=p0; p<p1; ++p){
      int col = (int)sys.pat.cols[p];
      int found = -1;
      for(int q=h0; q<h1; ++q){
        if(J[q] == col){ found = q; break; }
      }
      if(found < 0){
        std::ostringstream oss;
        oss << "Could not match ParCSR diag column for row " << r << ", col " << col;
        throw std::runtime_error(oss.str());
      }
      sys.A_diag_perm_h[p] = found;
      if(found != p) identity = false;
    }
  }
  sys.A_diag_identity_perm = identity;
  device_free(sys.A_diag_perm_d);
  if(!identity){
    device_alloc(sys.A_diag_perm_d, sys.pat.nnz);
    CUDA_CALL(cudaMemcpy(sys.A_diag_perm_d, sys.A_diag_perm_h.data(), sys.pat.nnz * sizeof(int), cudaMemcpyHostToDevice));
  }
}

static void update_ij_matrix_from_device(GPULinearSystem &sys){
  std::vector<HYPRE_Complex> h_vals(sys.pat.nnz);
  copy_device_to_vec(sys.pat.d_vals, h_vals);
  HYPRE_CALL(HYPRE_IJMatrixInitialize_v2(sys.Aij, HYPRE_MEMORY_HOST));
  HYPRE_CALL(HYPRE_IJMatrixSetValues(sys.Aij, sys.pat.nRows,
                                     const_cast<HYPRE_Int*>(sys.pat.ncols.data()),
                                     sys.pat.rows.data(), sys.pat.cols.data(), h_vals.data()));
  HYPRE_CALL(HYPRE_IJMatrixAssemble(sys.Aij));
  HYPRE_CALL(HYPRE_IJMatrixGetObject(sys.Aij, (void**)&sys.Apar));
  build_diag_permutation_from_host_parcsr(sys);
  HYPRE_CALL(HYPRE_IJMatrixMigrate(sys.Aij, HYPRE_MEMORY_DEVICE));
  HYPRE_CALL(HYPRE_IJMatrixGetObject(sys.Aij, (void**)&sys.Apar));
}

static void create_ij_vector_from_host(int n, const std::vector<HYPRE_BigInt> &idx, const std::vector<HYPRE_Complex> &vals, HYPRE_IJVector &vij, HYPRE_ParVector &vpar){
  HYPRE_BigInt ilower=0, iupper=(HYPRE_BigInt)n-1;
  HYPRE_CALL(HYPRE_IJVectorCreate(MPI_COMM_WORLD, ilower, iupper, &vij));
  HYPRE_CALL(HYPRE_IJVectorSetObjectType(vij, HYPRE_PARCSR));
  HYPRE_CALL(HYPRE_IJVectorInitialize_v2(vij, HYPRE_MEMORY_HOST));
  HYPRE_CALL(HYPRE_IJVectorSetValues(vij, n, const_cast<HYPRE_BigInt*>(idx.data()), const_cast<HYPRE_Complex*>(vals.data())));
  HYPRE_CALL(HYPRE_IJVectorAssemble(vij));
  HYPRE_CALL(HYPRE_IJVectorMigrate(vij, HYPRE_MEMORY_DEVICE));
  HYPRE_CALL(HYPRE_IJVectorGetObject(vij, (void**)&vpar));
}

static void destroy_ij_vector_pair(HYPRE_IJVector &vij, HYPRE_ParVector &vpar){
  if(vij) HYPRE_CALL(HYPRE_IJVectorDestroy(vij));
  vij = nullptr;
  vpar = nullptr;
}

static void get_ij_vector_to_host(HYPRE_IJVector vij, std::vector<cfdScalar> &x){
  HYPRE_CALL(HYPRE_IJVectorMigrate(vij, HYPRE_MEMORY_HOST));
  std::vector<HYPRE_BigInt> idx(x.size());
  for(std::size_t i=0;i<x.size();++i) idx[i]=(HYPRE_BigInt)i;
  std::vector<HYPRE_Complex> hx(x.size());
  HYPRE_CALL(HYPRE_IJVectorGetValues(vij, (HYPRE_Int)x.size(), idx.data(), hx.data()));
  for(std::size_t i=0;i<x.size();++i) x[i] = (cfdScalar)hx[i];
}

static void init_reusable_device_vectors(GPULinearSystem &sys){
  std::vector<HYPRE_Complex> h_zero(sys.n, 0.0);
  create_ij_vector_from_host(sys.n, sys.h_idx, h_zero, sys.bij, sys.bpar);
  create_ij_vector_from_host(sys.n, sys.h_idx, h_zero, sys.xij, sys.xpar);
  sys.b_data_dev = hypre_VectorData(hypre_ParVectorLocalVector(sys.bpar));
  sys.x_data_dev = hypre_VectorData(hypre_ParVectorLocalVector(sys.xpar));
  if(!sys.b_data_dev || !sys.x_data_dev){
    throw std::runtime_error("Failed to obtain hypre ParVector device data pointers");
  }
}

static void cache_parcsr_diag_pointer(GPULinearSystem &sys){
  hypre_CSRMatrix *diag = hypre_ParCSRMatrixDiag(sys.Apar);
  hypre_CSRMatrix *offd = hypre_ParCSRMatrixOffd(sys.Apar);
  HYPRE_Int diag_nnz = hypre_CSRMatrixNumNonzeros(diag);
  HYPRE_Int offd_nnz = offd ? hypre_CSRMatrixNumNonzeros(offd) : 0;
  if(diag_nnz != sys.pat.nnz || offd_nnz != 0){
    std::ostringstream oss;
    oss << "Unexpected ParCSR layout for 1-rank direct update path: diag nnz=" << diag_nnz
        << ", expected=" << sys.pat.nnz << ", offd nnz=" << offd_nnz;
    throw std::runtime_error(oss.str());
  }
  sys.A_diag_data_dev = hypre_CSRMatrixData(diag);
  if(!sys.A_diag_data_dev){
    throw std::runtime_error("Failed to obtain hypre ParCSR diag data pointer");
  }
}

static void copy_matrix_values_into_hypre(GPULinearSystem &sys){
  if(sys.mcgs_only) return;

  // Point-7 optimized path:
  // when enabled, assembly kernels already write directly into hypre's
  // ParCSR diag value array, so there is no pat.d_vals -> hypre copy.
  if(sys.direct_matrix_values) return;

  if(!sys.A_diag_data_dev){ cache_parcsr_diag_pointer(sys); }

  if(sys.A_diag_identity_perm){
    CUDA_CALL(cudaMemcpy(sys.A_diag_data_dev, sys.pat.d_vals,
                         sys.pat.nnz * sizeof(HYPRE_Complex),
                         cudaMemcpyDeviceToDevice));
  } else {
    int block = 256;
    int grid = (sys.pat.nnz + block - 1) / block;
    kernel_scatter_values_perm<<<grid, block>>>(
        sys.pat.nnz, sys.pat.d_vals, sys.A_diag_perm_d, sys.A_diag_data_dev);
    CUDA_CHECK_LAST();
  }
}

static inline HYPRE_Complex* matrix_values_ptr(GPULinearSystem &sys){
  return sys.direct_matrix_values ? sys.A_diag_data_dev : sys.pat.d_vals;
}

static void remap_device_index_array_inplace(int *d_idx, std::size_t n, const int *d_perm){
  if(!d_idx || n == 0) return;
  const int block = 256;
  const int grid = ((int)n + block - 1) / block;
  kernel_remap_matrix_indices_inplace<<<grid, block>>>((int)n, d_idx, d_perm);
  CUDA_CHECK_LAST();
}

static void enable_direct_hypre_matrix_updates(GPULinearSystem &sys){
  if(sys.direct_matrix_values) return;

  if(!sys.A_diag_data_dev){
    cache_parcsr_diag_pointer(sys);
  }

  // The CUDA assembly kernels index matrix entries using pat.d_diagPos and
  // pat.d_face**. These initially point into MatrixPattern order. If hypre's
  // ParCSR diag-data order differs, remap those device index arrays once.
  if(!sys.A_diag_identity_perm){
    if(!sys.A_diag_perm_d){
      throw std::runtime_error("Non-identity ParCSR permutation exists but A_diag_perm_d is null");
    }

    remap_device_index_array_inplace(sys.pat.d_diagPos, sys.pat.diagPos.size(), sys.A_diag_perm_d);
    remap_device_index_array_inplace(sys.pat.d_facePP,  sys.pat.facePP.size(),  sys.A_diag_perm_d);
    remap_device_index_array_inplace(sys.pat.d_facePN,  sys.pat.facePN.size(),  sys.A_diag_perm_d);
    remap_device_index_array_inplace(sys.pat.d_faceNP,  sys.pat.faceNP.size(),  sys.A_diag_perm_d);
    remap_device_index_array_inplace(sys.pat.d_faceNN,  sys.pat.faceNN.size(),  sys.A_diag_perm_d);
  }

  sys.direct_matrix_values = true;
}

static void copy_device_rhs_and_host_x0_into_hypre(GPULinearSystem &sys, const std::vector<cfdScalar> &x0){
  std::vector<HYPRE_Complex> h_x0(sys.n);
  for(int i=0;i<sys.n;++i) h_x0[i] = (HYPRE_Complex)x0[i];
  CUDA_CALL(cudaMemcpy(sys.b_data_dev, sys.d_rhs, sys.n * sizeof(HYPRE_Complex), cudaMemcpyDeviceToDevice));
  CUDA_CALL(cudaMemcpy(sys.x_data_dev, h_x0.data(), sys.n * sizeof(HYPRE_Complex), cudaMemcpyHostToDevice));
}

static void copy_device_rhs_and_device_x0_into_hypre(GPULinearSystem &sys, const cfdScalar *d_x0){
  CUDA_CALL(cudaMemcpy(sys.b_data_dev, sys.d_rhs, sys.n * sizeof(HYPRE_Complex), cudaMemcpyDeviceToDevice));
  copy_scalar_device_to_hypre_device(sys.n, d_x0, sys.x_data_dev);
}

static void copy_solution_from_hypre_to_device(GPULinearSystem &sys, cfdScalar *d_xout){
  copy_hypre_device_to_scalar_device(sys.n, sys.x_data_dev, d_xout);
}

static void copy_host_rhs_and_host_x0_into_hypre(GPULinearSystem &sys, const std::vector<cfdScalar> &rhs, const std::vector<cfdScalar> &x0){
  std::vector<HYPRE_Complex> h_rhs(sys.n), h_x0(sys.n);
  for(int i=0;i<sys.n;++i){ h_rhs[i] = (HYPRE_Complex)rhs[i]; h_x0[i] = (HYPRE_Complex)x0[i]; }
  CUDA_CALL(cudaMemcpy(sys.b_data_dev, h_rhs.data(), sys.n * sizeof(HYPRE_Complex), cudaMemcpyHostToDevice));
  CUDA_CALL(cudaMemcpy(sys.x_data_dev, h_x0.data(), sys.n * sizeof(HYPRE_Complex), cudaMemcpyHostToDevice));
}

static void copy_solution_from_hypre(GPULinearSystem &sys, std::vector<cfdScalar> &xout){
  std::vector<HYPRE_Complex> h_x(sys.n);
  CUDA_CALL(cudaMemcpy(h_x.data(), sys.x_data_dev, sys.n * sizeof(HYPRE_Complex), cudaMemcpyDeviceToHost));
  xout.assign(sys.n, 0.0);
  for(int i=0;i<sys.n;++i) xout[i] = (cfdScalar) h_x[i];
}

static void init_common_linear_storage(GPULinearSystem &sys, MatrixPattern pat, const char *memTag="linear"){
  sys.n = pat.nRows;
  sys.pat = std::move(pat);
  upload_pattern(sys.pat, memTag);
  sys.h_idx.resize(sys.n);
  for(int i=0;i<sys.n;++i) sys.h_idx[i]=(HYPRE_BigInt)i;
  device_alloc(sys.d_rhs, sys.n); CUDA_CALL(cudaMemset(sys.d_rhs, 0, sys.n*sizeof(HYPRE_Complex)));
  device_alloc(sys.d_x, sys.n); CUDA_CALL(cudaMemset(sys.d_x, 0, sys.n*sizeof(HYPRE_Complex)));
  create_ij_matrix_from_pattern(sys.pat, sys.Aij);
}

// Memory-capacity storage for fused MCGS momentum.
// This keeps the app-side CSR graph/values and one RHS vector, but does not
// create a HYPRE IJ/ParCSR matrix, BiCGSTAB solver, or HYPRE vectors.
// Pressure path remains unchanged.
static void init_common_linear_storage_mcgs_only(GPULinearSystem &sys, MatrixPattern pat, const char *memTag="mom_mcgs"){
  sys.n = pat.nRows;
  sys.pat = std::move(pat);
  upload_pattern(sys.pat, memTag);

  device_alloc(sys.d_rhs, sys.n);
  CUDA_CALL(cudaMemset(sys.d_rhs, 0, sys.n*sizeof(HYPRE_Complex)));

  sys.d_x = nullptr;
  sys.Aij = nullptr;
  sys.Apar = nullptr;
  sys.bij = nullptr;
  sys.xij = nullptr;
  sys.bpar = nullptr;
  sys.xpar = nullptr;
  sys.solver = nullptr;
  sys.prec = nullptr;
  sys.A_diag_data_dev = nullptr;
  sys.b_data_dev = nullptr;
  sys.x_data_dev = nullptr;
  sys.A_diag_perm_d = nullptr;
  sys.A_diag_identity_perm = true;
  sys.direct_matrix_values = false; // matrix_values_ptr() returns pat.d_vals
  sys.mcgs_only = true;
  sys.is_setup = false;
}

static void destroy_linear_storage(GPULinearSystem &sys){
  if(sys.prec) HYPRE_CALL(HYPRE_BoomerAMGDestroy(sys.prec));
  if(sys.solver){
    if(sys.isPCG) HYPRE_CALL(HYPRE_ParCSRPCGDestroy(sys.solver));
    else HYPRE_CALL(HYPRE_ParCSRBiCGSTABDestroy(sys.solver));
  }
  if(sys.bij) HYPRE_CALL(HYPRE_IJVectorDestroy(sys.bij));
  if(sys.xij) HYPRE_CALL(HYPRE_IJVectorDestroy(sys.xij));
  if(sys.Aij) HYPRE_CALL(HYPRE_IJMatrixDestroy(sys.Aij));
  device_free(sys.d_rhs); device_free(sys.d_x);
  device_free(sys.A_diag_perm_d);
  destroy_pattern(sys.pat);
  sys = GPULinearSystem{};
}

static void init_momentum_system(GPUMomentumAssembler &mom, const Mesh &mesh, const Params &par){
  const bool mcgsOnly = (par.velSolver == 3);

  if(mcgsOnly){
    init_common_linear_storage_mcgs_only(mom.lin, build_momentum_pattern(mesh), "mom_mcgs");
    if(g_memdiag_rank == 0){
      std::printf("MOMENTUM_STORAGE_MODE mcgs_fused_app_csr_no_hypre_velocity_matrix\n");
      std::fflush(stdout);
    }
  } else {
    init_common_linear_storage(mom.lin, build_momentum_pattern(mesh), "mom_hypre");
  }

  device_alloc(mom.d_qOld, mesh.nCells);
  device_alloc(mom.d_uConv, mesh.nCells);
  device_alloc(mom.d_vConv, mesh.nCells);
  device_alloc(mom.d_wConv, mesh.nCells);
  device_alloc(mom.d_gradQx, mesh.nCells);
  device_alloc(mom.d_gradQy, mesh.nCells);
  device_alloc(mom.d_gradQz, mesh.nCells);
  device_alloc(mom.d_gradPcomp, mesh.nCells);
  device_alloc(mom.d_rAU, mesh.nCells);

  mom.lin.isPCG = false;
  CUDA_CALL(cudaMemset(mom.lin.pat.d_vals, 0, mom.lin.pat.nnz * sizeof(HYPRE_Complex)));

  if(!mcgsOnly){
    HYPRE_CALL(HYPRE_ParCSRBiCGSTABCreate(MPI_COMM_WORLD, &mom.lin.solver));
    update_ij_matrix_from_device(mom.lin);
    cache_parcsr_diag_pointer(mom.lin);
    enable_direct_hypre_matrix_updates(mom.lin);
    init_reusable_device_vectors(mom.lin);
  }

  mom.lin.is_setup = false;
}

static void destroy_momentum_system(GPUMomentumAssembler &mom){
  device_free(mom.d_qOld); device_free(mom.d_uConv); device_free(mom.d_vConv); device_free(mom.d_wConv);
  device_free(mom.d_gradQx); device_free(mom.d_gradQy); device_free(mom.d_gradQz); device_free(mom.d_gradPcomp); device_free(mom.d_rAU);
  destroy_linear_storage(mom.lin);
  mom = GPUMomentumAssembler{};
}

static void init_pressure_system(GPULinearSystem &ps, const Mesh &mesh, const DeviceMesh &dm, const Params &par, int refCell, bool usePressureAnchor, double &tsetup){
  init_common_linear_storage(ps, build_pressure_pattern(mesh, refCell, usePressureAnchor), "pressure");
  int block=256;
  int gridVals=(ps.pat.nnz + block - 1)/block;
  double t0 = MPI_Wtime();
  kernel_zero_values<<<gridVals, block>>>(ps.pat.d_vals, ps.pat.nnz);
  kernel_pressure_anchor<<<1,1>>>(refCell, ps.pat.d_diagPos, ps.pat.d_vals);
  CUDA_CHECK_LAST();
  update_ij_matrix_from_device(ps);
  HYPRE_CALL(HYPRE_ParCSRPCGCreate(MPI_COMM_WORLD,&ps.solver));
  ps.isPCG = true;
  ps.prec = nullptr;
  HYPRE_CALL(HYPRE_ParCSRPCGSetTol(ps.solver, par.pRelTol));
  HYPRE_CALL(HYPRE_ParCSRPCGSetAbsoluteTol(ps.solver, par.pTol));
  HYPRE_CALL(HYPRE_ParCSRPCGSetMaxIter(ps.solver, par.pMaxit));
  HYPRE_CALL(HYPRE_ParCSRPCGSetTwoNorm(ps.solver, 1));
  HYPRE_CALL(HYPRE_ParCSRPCGSetPrintLevel(ps.solver, 0));
  HYPRE_CALL(HYPRE_ParCSRPCGSetLogging(ps.solver, 1));
  if(par.p_use_amg){
    HYPRE_CALL(HYPRE_BoomerAMGCreate(&ps.prec));
    HYPRE_CALL(HYPRE_BoomerAMGSetTol(ps.prec, 0.0));
    HYPRE_CALL(HYPRE_BoomerAMGSetMaxIter(ps.prec, par.pAmgMaxit));
    HYPRE_CALL(HYPRE_BoomerAMGSetPrintLevel(ps.prec, 0));
    HYPRE_CALL(HYPRE_BoomerAMGSetLogging(ps.prec, 0));
    HYPRE_CALL(HYPRE_BoomerAMGSetCoarsenType(ps.prec, par.pAmgCoarsenType));
    HYPRE_CALL(HYPRE_BoomerAMGSetInterpType(ps.prec, par.pAmgInterpType));
    HYPRE_CALL(HYPRE_BoomerAMGSetRelaxType(ps.prec, par.pAmgRelaxType));
    HYPRE_CALL(HYPRE_BoomerAMGSetNumSweeps(ps.prec, 1));
    HYPRE_CALL(HYPRE_BoomerAMGSetPMaxElmts(ps.prec, par.pAmgPmax));
    HYPRE_CALL(HYPRE_BoomerAMGSetTruncFactor(ps.prec, par.pAmgTruncFactor));
    HYPRE_CALL(HYPRE_BoomerAMGSetKeepTranspose(ps.prec, par.pAmgKeepTranspose));
    HYPRE_CALL(HYPRE_BoomerAMGSetRAP2(ps.prec, 0));
    if(par.pAmgAggLevels > 0){
      HYPRE_CALL(HYPRE_BoomerAMGSetAggNumLevels(ps.prec, par.pAmgAggLevels));
      HYPRE_CALL(HYPRE_BoomerAMGSetAggInterpType(ps.prec, par.pAmgAggInterpType));
    }
    HYPRE_CALL(HYPRE_ParCSRPCGSetPrecond(ps.solver, (HYPRE_PtrToParSolverFcn)HYPRE_BoomerAMGSolve, (HYPRE_PtrToParSolverFcn)HYPRE_BoomerAMGSetup, ps.prec));
  } else {
    HYPRE_CALL(HYPRE_ParCSRPCGSetPrecond(ps.solver, (HYPRE_PtrToParSolverFcn)HYPRE_ParCSRDiagScale, (HYPRE_PtrToParSolverFcn)HYPRE_ParCSRDiagScaleSetup, nullptr));
  }
  cache_parcsr_diag_pointer(ps);
  enable_direct_hypre_matrix_updates(ps);
  init_reusable_device_vectors(ps);
  ps.is_setup = false;
  // Do NOT call PCG/AMG setup here. At this point the pressure matrix is only a
  // dummy anchored shell with zero off-diagonals, which can make BoomerAMG/PCG
  // reject the operator in steady mode. The real setup is done after the first
  // rAU-based pressure matrix has been assembled in update_pressure_matrix_from_rAU().
  tsetup += MPI_Wtime() - t0;
}

static void assemble_momentum_on_gpu(
    const DeviceMesh &dm, const Mesh &mesh, GPUMomentumAssembler &mom,
    double rho, double mu, double /*unused_dt*/,
    const std::vector<cfdScalar> &qOld, const std::vector<cfdScalar> &uConv, const std::vector<cfdScalar> &vConv, const std::vector<cfdScalar> &wConv,
    const std::vector<cfdVector3> &gradQ, const std::vector<cfdScalar> &gradPcomp,
    const DeviceBC &bcQ, const DeviceBC &bcU, const DeviceBC &bcV, const DeviceBC &bcW,
    double corrPsi, int momentumConvectionScheme)
{
  std::vector<cfdScalar> gradQx(mesh.nCells), gradQy(mesh.nCells), gradQz(mesh.nCells);
  for(int c=0;c<mesh.nCells;++c){ gradQx[c]=gradQ[c][0]; gradQy[c]=gradQ[c][1]; gradQz[c]=gradQ[c][2]; }
  copy_vec_to_device(qOld, mom.d_qOld);
  copy_vec_to_device(uConv, mom.d_uConv);
  copy_vec_to_device(vConv, mom.d_vConv);
  copy_vec_to_device(wConv, mom.d_wConv);
  copy_vec_to_device(gradQx, mom.d_gradQx);
  copy_vec_to_device(gradQy, mom.d_gradQy);
  copy_vec_to_device(gradQz, mom.d_gradQz);
  copy_vec_to_device(gradPcomp, mom.d_gradPcomp);

  int block=256;
  int gridCells=(mesh.nCells + block - 1)/block;
  int gridVals=(mom.lin.pat.nnz + block - 1)/block;
  int gridFaces=(mesh.nInternalFaces + block - 1)/block;
  int nBoundaryFaces = mesh.nFaces - mesh.nInternalFaces;
  int gridBFaces=(nBoundaryFaces + block - 1)/block;
  HYPRE_Complex *Avals = matrix_values_ptr(mom.lin);

  kernel_zero_values<<<gridVals, block>>>(Avals, mom.lin.pat.nnz);
  kernel_momentum_base_steady<<<gridCells, block>>>(mesh.nCells, mom.lin.pat.d_diagPos, dm.d_vol, mom.d_gradPcomp, Avals, mom.lin.d_rhs);
  kernel_momentum_internal_faces<<<gridFaces, block>>>(mesh.nInternalFaces, dm.d_owner, dm.d_neigh, dm.d_ccx, dm.d_ccy, dm.d_ccz, dm.d_xfx, dm.d_xfy, dm.d_xfz, dm.d_nfx, dm.d_nfy, dm.d_nfz, dm.d_sfx, dm.d_sfy, dm.d_sfz, dm.d_Af, mom.d_gradQx, mom.d_gradQy, mom.d_gradQz, mom.d_uConv, mom.d_vConv, mom.d_wConv, rho, mu, corrPsi, momentumConvectionScheme, mom.lin.pat.d_diagPos,
      mom.lin.pat.d_facePN,
      mom.lin.pat.d_faceNP, Avals, mom.lin.d_rhs);
  kernel_momentum_boundary_faces<<<gridBFaces, block>>>(nBoundaryFaces, mesh.nInternalFaces, dm.d_owner, dm.d_bPatch, dm.d_ccx, dm.d_ccy, dm.d_ccz, dm.d_xfx, dm.d_xfy, dm.d_xfz, dm.d_nfx, dm.d_nfy, dm.d_nfz, dm.d_sfx, dm.d_sfy, dm.d_sfz, dm.d_Af, mom.d_gradQx, mom.d_gradQy, mom.d_gradQz, mom.d_uConv, mom.d_vConv, mom.d_wConv, bcQ.d_type, bcQ.d_faceValue, bcU.d_type, bcU.d_faceValue, bcV.d_type, bcV.d_faceValue, bcW.d_type, bcW.d_faceValue, rho, mu, corrPsi, mom.lin.pat.d_diagPos, Avals, mom.lin.d_rhs);
  CUDA_CHECK_LAST();
}

static void assemble_momentum_rhs_only_on_gpu(
    const DeviceMesh &dm, const Mesh &mesh, GPUMomentumAssembler &mom,
    double rho, double mu,
    const std::vector<cfdScalar> &qOld, const std::vector<cfdScalar> &uConv, const std::vector<cfdScalar> &vConv, const std::vector<cfdScalar> &wConv,
    const std::vector<cfdVector3> &gradQ, const std::vector<cfdScalar> &gradPcomp,
    const DeviceBC &bcQ, const DeviceBC &bcU, const DeviceBC &bcV, const DeviceBC &bcW,
    double corrPsi, double uRelax)
{
  std::vector<cfdScalar> gradQx(mesh.nCells), gradQy(mesh.nCells), gradQz(mesh.nCells);
  for(int c=0;c<mesh.nCells;++c){ gradQx[c]=gradQ[c][0]; gradQy[c]=gradQ[c][1]; gradQz[c]=gradQ[c][2]; }
  copy_vec_to_device(qOld, mom.d_qOld);
  copy_vec_to_device(uConv, mom.d_uConv);
  copy_vec_to_device(vConv, mom.d_vConv);
  copy_vec_to_device(wConv, mom.d_wConv);
  copy_vec_to_device(gradQx, mom.d_gradQx);
  copy_vec_to_device(gradQy, mom.d_gradQy);
  copy_vec_to_device(gradQz, mom.d_gradQz);
  copy_vec_to_device(gradPcomp, mom.d_gradPcomp);

  int block=256;
  int gridCells=(mesh.nCells + block - 1)/block;
  int gridFaces=(mesh.nInternalFaces + block - 1)/block;
  int nBoundaryFaces = mesh.nFaces - mesh.nInternalFaces;
  int gridBFaces=(nBoundaryFaces + block - 1)/block;
  HYPRE_Complex *Avals = matrix_values_ptr(mom.lin);

  kernel_zero_rhs<<<gridCells, block>>>(mom.lin.d_rhs, mesh.nCells);
  kernel_momentum_base_rhs_only<<<gridCells, block>>>(mesh.nCells, dm.d_vol, mom.d_gradPcomp, mom.lin.d_rhs);
  kernel_momentum_internal_faces_rhs_only<<<gridFaces, block>>>(mesh.nInternalFaces, dm.d_owner, dm.d_neigh, dm.d_ccx, dm.d_ccy, dm.d_ccz, dm.d_xfx, dm.d_xfy, dm.d_xfz, dm.d_nfx, dm.d_nfy, dm.d_nfz, dm.d_sfx, dm.d_sfy, dm.d_sfz, dm.d_Af, mom.d_gradQx, mom.d_gradQy, mom.d_gradQz, mu, corrPsi, mom.lin.d_rhs);
  kernel_momentum_boundary_faces_rhs_only<<<gridBFaces, block>>>(nBoundaryFaces, mesh.nInternalFaces, dm.d_owner, dm.d_bPatch, dm.d_ccx, dm.d_ccy, dm.d_ccz, dm.d_xfx, dm.d_xfy, dm.d_xfz, dm.d_nfx, dm.d_nfy, dm.d_nfz, dm.d_sfx, dm.d_sfy, dm.d_sfz, dm.d_Af, mom.d_gradQx, mom.d_gradQy, mom.d_gradQz, mom.d_uConv, mom.d_vConv, mom.d_wConv, bcQ.d_type, bcQ.d_faceValue, bcU.d_type, bcU.d_faceValue, bcV.d_type, bcV.d_faceValue, bcW.d_type, bcW.d_faceValue, rho, mu, corrPsi, mom.lin.d_rhs);
  if(uRelax < 0.999999){
    kernel_relax_momentum_rhs_only_from_relaxed_matrix<<<gridCells, block>>>(mesh.nCells, mom.lin.pat.d_diagPos, Avals, mom.lin.d_rhs, mom.d_qOld, uRelax);
  }
  CUDA_CHECK_LAST();
}


static void assemble_momentum_on_gpu_device_grad(
    const DeviceMesh &dm, const Mesh &mesh, GPUMomentumAssembler &mom,
    double rho, double mu, double /*unused_dt*/,
    const cfdScalar *d_qOld,
    const cfdScalar *d_uConv,
    const cfdScalar *d_vConv,
    const cfdScalar *d_wConv,
    const cfdScalar *d_gradQx,
    const cfdScalar *d_gradQy,
    const cfdScalar *d_gradQz,
    const cfdScalar *d_gradPcomp,
    const DeviceBC &bcQ,
    const DeviceBC &bcU,
    const DeviceBC &bcV,
    const DeviceBC &bcW,
    double corrPsi, int momentumConvectionScheme)
{
  (void)d_qOld; // used later by relaxation, not needed during unrelaxed matrix assembly

  int block=256;
  int gridCells=(mesh.nCells + block - 1)/block;
  int gridVals=(mom.lin.pat.nnz + block - 1)/block;
  int gridFaces=(mesh.nInternalFaces + block - 1)/block;
  int nBoundaryFaces = mesh.nFaces - mesh.nInternalFaces;
  int gridBFaces=(nBoundaryFaces + block - 1)/block;

  HYPRE_Complex *Avals = matrix_values_ptr(mom.lin);

  kernel_zero_values<<<gridVals, block>>>(Avals, mom.lin.pat.nnz);

  kernel_momentum_base_steady<<<gridCells, block>>>(
      mesh.nCells,
      mom.lin.pat.d_diagPos,
      dm.d_vol,
      d_gradPcomp,
      Avals,
      mom.lin.d_rhs);

  kernel_momentum_internal_faces<<<gridFaces, block>>>(
      mesh.nInternalFaces,
      dm.d_owner, dm.d_neigh,
      dm.d_ccx, dm.d_ccy, dm.d_ccz,
      dm.d_xfx, dm.d_xfy, dm.d_xfz,
      dm.d_nfx, dm.d_nfy, dm.d_nfz,
      dm.d_sfx, dm.d_sfy, dm.d_sfz,
      dm.d_Af,
      d_gradQx, d_gradQy, d_gradQz,
      d_uConv, d_vConv, d_wConv,
      rho, mu, corrPsi, momentumConvectionScheme,
      mom.lin.pat.d_diagPos,
      mom.lin.pat.d_facePN,
      mom.lin.pat.d_faceNP,
      Avals,
      mom.lin.d_rhs);

  kernel_momentum_boundary_faces<<<gridBFaces, block>>>(
      nBoundaryFaces,
      mesh.nInternalFaces,
      dm.d_owner,
      dm.d_bPatch,
      dm.d_ccx, dm.d_ccy, dm.d_ccz,
      dm.d_xfx, dm.d_xfy, dm.d_xfz,
      dm.d_nfx, dm.d_nfy, dm.d_nfz,
      dm.d_sfx, dm.d_sfy, dm.d_sfz,
      dm.d_Af,
      d_gradQx, d_gradQy, d_gradQz,
      d_uConv, d_vConv, d_wConv,
      bcQ.d_type, bcQ.d_faceValue,
      bcU.d_type, bcU.d_faceValue,
      bcV.d_type, bcV.d_faceValue,
      bcW.d_type, bcW.d_faceValue,
      rho, mu, corrPsi,
      mom.lin.pat.d_diagPos,
      Avals,
      mom.lin.d_rhs);

  CUDA_CHECK_LAST();
}

static void assemble_momentum_rhs_only_on_gpu_device_grad(
    const DeviceMesh &dm, const Mesh &mesh, GPUMomentumAssembler &mom,
    double rho, double mu,
    const cfdScalar *d_qOld,
    const cfdScalar *d_uConv,
    const cfdScalar *d_vConv,
    const cfdScalar *d_wConv,
    const cfdScalar *d_gradQx,
    const cfdScalar *d_gradQy,
    const cfdScalar *d_gradQz,
    const cfdScalar *d_gradPcomp,
    const DeviceBC &bcQ,
    const DeviceBC &bcU,
    const DeviceBC &bcV,
    const DeviceBC &bcW,
    double corrPsi,
    double uRelax)
{
  int block=256;
  int gridCells=(mesh.nCells + block - 1)/block;
  int gridFaces=(mesh.nInternalFaces + block - 1)/block;
  int nBoundaryFaces = mesh.nFaces - mesh.nInternalFaces;
  int gridBFaces=(nBoundaryFaces + block - 1)/block;

  HYPRE_Complex *Avals = matrix_values_ptr(mom.lin);

  kernel_zero_rhs<<<gridCells, block>>>(mom.lin.d_rhs, mesh.nCells);

  kernel_momentum_base_rhs_only<<<gridCells, block>>>(
      mesh.nCells,
      dm.d_vol,
      d_gradPcomp,
      mom.lin.d_rhs);

  kernel_momentum_internal_faces_rhs_only<<<gridFaces, block>>>(
      mesh.nInternalFaces,
      dm.d_owner, dm.d_neigh,
      dm.d_ccx, dm.d_ccy, dm.d_ccz,
      dm.d_xfx, dm.d_xfy, dm.d_xfz,
      dm.d_nfx, dm.d_nfy, dm.d_nfz,
      dm.d_sfx, dm.d_sfy, dm.d_sfz,
      dm.d_Af,
      d_gradQx, d_gradQy, d_gradQz,
      mu, corrPsi,
      mom.lin.d_rhs);

  kernel_momentum_boundary_faces_rhs_only<<<gridBFaces, block>>>(
      nBoundaryFaces,
      mesh.nInternalFaces,
      dm.d_owner,
      dm.d_bPatch,
      dm.d_ccx, dm.d_ccy, dm.d_ccz,
      dm.d_xfx, dm.d_xfy, dm.d_xfz,
      dm.d_nfx, dm.d_nfy, dm.d_nfz,
      dm.d_sfx, dm.d_sfy, dm.d_sfz,
      dm.d_Af,
      d_gradQx, d_gradQy, d_gradQz,
      d_uConv, d_vConv, d_wConv,
      bcQ.d_type, bcQ.d_faceValue,
      bcU.d_type, bcU.d_faceValue,
      bcV.d_type, bcV.d_faceValue,
      bcW.d_type, bcW.d_faceValue,
      rho, mu, corrPsi,
      mom.lin.d_rhs);

  if(uRelax < 0.999999){
    kernel_relax_momentum_rhs_only_from_relaxed_matrix<<<gridCells, block>>>(
        mesh.nCells,
        mom.lin.pat.d_diagPos,
        Avals,
        mom.lin.d_rhs,
        d_qOld,
        uRelax);
  }

  CUDA_CHECK_LAST();
}

static void extract_rAU_from_momentum_matrix(const Mesh &mesh, const DeviceMesh &dm, GPUMomentumAssembler &mom, const Params &par, std::vector<cfdScalar> &rAU_host){
  int block=256;
  int gridCells=(mesh.nCells + block - 1)/block;
  HYPRE_Complex *Avals = matrix_values_ptr(mom.lin);
  const double diagScale = (par.rAUMode == 0 && par.uRelax < 0.999999) ? par.uRelax : 1.0;
  kernel_extract_vol_over_diag<<<gridCells, block>>>(mesh.nCells, mom.lin.pat.d_diagPos, Avals, dm.d_vol, mom.d_rAU, diagScale, par.rAUScale);
  CUDA_CHECK_LAST();
  rAU_host.clear(); // device-resident rAU is used by pressure and Rhie-Chow; host copy is not needed in optimized path.
}

static void relax_momentum_system_on_gpu(const Mesh &mesh, GPUMomentumAssembler &mom, const cfdScalar *d_qOld, double uRelax){
  if(uRelax >= 0.999999) return;
  int block=256;
  int gridCells=(mesh.nCells + block - 1)/block;
  HYPRE_Complex *Avals = matrix_values_ptr(mom.lin);
  kernel_relax_momentum_system<<<gridCells, block>>>(mesh.nCells, mom.lin.pat.d_diagPos, Avals, mom.lin.d_rhs, d_qOld, uRelax);
  CUDA_CHECK_LAST();
}

static void pressure_solver_setup(GPULinearSystem &ps);

static void update_pressure_matrix_from_rAU(const Mesh &mesh, const DeviceMesh &dm, GPULinearSystem &ps, const DeviceBC &bcP, const cfdScalar *d_rAU, double rho, double pCoeffScale, int refCell, bool usePressureAnchor, bool skipPressureBoundaryCoeffs, double pressurePenaltyCoeff, bool doSetup, double &tsetup){
  int block=256;
  int gridVals=(ps.pat.nnz + block - 1)/block;
  int gridFaces=(mesh.nInternalFaces + block - 1)/block;
  int nBoundaryFaces = mesh.nFaces - mesh.nInternalFaces;
  int gridBFaces=(nBoundaryFaces + block - 1)/block;
  HYPRE_Complex *Avals = matrix_values_ptr(ps);
  double t0 = MPI_Wtime();
  kernel_zero_values<<<gridVals, block>>>(Avals, ps.pat.nnz);
  kernel_pressure_internal_faces_rau<<<gridFaces, block>>>(mesh.nInternalFaces, dm.d_owner, dm.d_neigh, dm.d_ccx, dm.d_ccy, dm.d_ccz, dm.d_xfx, dm.d_xfy, dm.d_xfz, dm.d_nfx, dm.d_nfy, dm.d_nfz, dm.d_Af, d_rAU, rho, pCoeffScale, ps.pat.d_diagPos,
      ps.pat.d_facePN,
      ps.pat.d_faceNP, Avals);
  if(!skipPressureBoundaryCoeffs){
    kernel_pressure_boundary_faces_rau<<<gridBFaces, block>>>(nBoundaryFaces, mesh.nInternalFaces, dm.d_owner, dm.d_bPatch, dm.d_ccx, dm.d_ccy, dm.d_ccz, dm.d_xfx, dm.d_xfy, dm.d_xfz, dm.d_nfx, dm.d_nfy, dm.d_nfz, dm.d_Af, d_rAU, rho, pCoeffScale, bcP.d_type, ps.pat.d_diagPos, Avals);
  }
  if(usePressureAnchor) kernel_pressure_anchor<<<1,1>>>(refCell, ps.pat.d_diagPos, Avals);
  if(pressurePenaltyCoeff != 0.0){
    kernel_add_pressure_diag_penalty<<<(mesh.nCells + block - 1)/block, block>>>(
        mesh.nCells, ps.pat.d_diagPos, Avals, pressurePenaltyCoeff);
  }
  CUDA_CHECK_LAST();
  if(doSetup){
    HYPRE_CALL(HYPRE_ParCSRPCGSetup(ps.solver, ps.Apar, ps.bpar, ps.xpar));
    ps.is_setup = true;
    CUDA_CALL(cudaDeviceSynchronize());
  }
  tsetup += MPI_Wtime() - t0;
}


__global__ static void kernel_copy_plus_hypre_solution(
    int n,
    cfdScalar *out,
    const cfdScalar *oldx,
    const HYPRE_Complex *dx)
{
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if(i >= n) return;
  out[i] = oldx[i] + (double)dx[i];
}

static void handle_velocity_bicgstab_status(
    const Params &par,
    HYPRE_Int solveErr,
    HYPRE_Int itsErr,
    HYPRE_Int relErr,
    HYPRE_Int &its,
    double &relres)
{
  if(solveErr || itsErr || relErr){
    if(solveErr == 256 || itsErr == 256 || relErr == 256){
      // Intentional inexact BiCGSTAB smoother mode: HYPRE reports max-iteration
      // nonconvergence as error 256. Accept it silently and report the requested
      // fixed iteration count when HYPRE refuses to return one.
      if(its <= 0) its = par.velMaxit;
      HYPRE_ClearAllErrors();
    } else {
      std::fprintf(stderr,
          "FATAL: velocity BiCGSTAB failed. "
          "solveErr=%d itsErr=%d relErr=%d its=%d finalRel=%.6e.\n",
          (int)solveErr, (int)itsErr, (int)relErr, (int)its, relres);
      MPI_Abort(MPI_COMM_WORLD, solveErr ? solveErr : (itsErr ? itsErr : relErr));
    }
  }
}

static void solve_momentum_gpu_device_x0_xout(
    GPUMomentumAssembler &mom,
    const Params &par,
    const cfdScalar *d_x0,
    cfdScalar *d_xout,
    HYPRE_Int &its,
    double &relres,
    double &tsetup,
    double &tsolve,
    bool doMatrixSetup)
{
  copy_device_rhs_and_device_x0_into_hypre(mom.lin, d_x0);

  HYPRE_CALL(HYPRE_ParCSRBiCGSTABSetTol(mom.lin.solver, par.velRelTol));
  HYPRE_CALL(HYPRE_ParCSRBiCGSTABSetAbsoluteTol(mom.lin.solver, par.velTol));
  HYPRE_CALL(HYPRE_ParCSRBiCGSTABSetMaxIter(mom.lin.solver, par.velMaxit));
  HYPRE_CALL(HYPRE_ParCSRBiCGSTABSetPrintLevel(mom.lin.solver, 0));
  HYPRE_CALL(HYPRE_ParCSRBiCGSTABSetLogging(mom.lin.solver, 1));

  if(doMatrixSetup || !mom.lin.is_setup){
    HYPRE_CALL(HYPRE_ParCSRBiCGSTABSetPrecond(mom.lin.solver,
        (HYPRE_PtrToParSolverFcn)HYPRE_ParCSRDiagScale,
        (HYPRE_PtrToParSolverFcn)HYPRE_ParCSRDiagScaleSetup,
        nullptr));
    copy_matrix_values_into_hypre(mom.lin);
    double t0=MPI_Wtime();
    HYPRE_CALL(HYPRE_ParCSRBiCGSTABSetup(mom.lin.solver, mom.lin.Apar, mom.lin.bpar, mom.lin.xpar));
    tsetup += MPI_Wtime()-t0;
    mom.lin.is_setup = true;
  }

  double t0=MPI_Wtime();
  HYPRE_Int solveErr = HYPRE_ParCSRBiCGSTABSolve(mom.lin.solver, mom.lin.Apar, mom.lin.bpar, mom.lin.xpar);
  tsolve += MPI_Wtime()-t0;

  its=-1; relres=0.0;
  HYPRE_Int itsTmp=-1; HYPRE_Real relTmp = (HYPRE_Real)(0.0);
  HYPRE_Int itsErr = HYPRE_ParCSRBiCGSTABGetNumIterations(mom.lin.solver,&itsTmp);
  HYPRE_Int relErr = HYPRE_ParCSRBiCGSTABGetFinalRelativeResidualNorm(mom.lin.solver,&relTmp);
  if(itsErr == 0) its = itsTmp;
  if(relErr == 0) relres = relTmp;
  handle_velocity_bicgstab_status(par, solveErr, itsErr, relErr, its, relres);
  copy_solution_from_hypre_to_device(mom.lin, d_xout);
}

static void solve_momentum_gpu_device_defect_x0_xout(
    GPUMomentumAssembler &mom,
    const Params &par,
    const cfdScalar *d_x0,
    cfdScalar *d_xout,
    HYPRE_Int &its,
    double &relres,
    double &tsetup,
    double &tsolve,
    bool doMatrixSetup)
{
  const int n = mom.lin.n;
  copy_matrix_values_into_hypre(mom.lin);

  CUDA_CALL(cudaMemcpy(mom.lin.b_data_dev, mom.lin.d_rhs, n*sizeof(HYPRE_Complex), cudaMemcpyDeviceToDevice));
  copy_scalar_device_to_hypre_device(n, d_x0, mom.lin.x_data_dev);
  HYPRE_CALL(HYPRE_ParCSRMatrixMatvec((HYPRE_Complex)-1.0, mom.lin.Apar, mom.lin.xpar, (HYPRE_Complex)1.0, mom.lin.bpar));
  CUDA_CALL(cudaMemset(mom.lin.x_data_dev, 0, n*sizeof(HYPRE_Complex)));

  HYPRE_CALL(HYPRE_ParCSRBiCGSTABSetTol(mom.lin.solver, par.velRelTol));
  HYPRE_CALL(HYPRE_ParCSRBiCGSTABSetAbsoluteTol(mom.lin.solver, par.velTol));
  HYPRE_CALL(HYPRE_ParCSRBiCGSTABSetMaxIter(mom.lin.solver, par.velMaxit));
  HYPRE_CALL(HYPRE_ParCSRBiCGSTABSetPrintLevel(mom.lin.solver, 0));
  HYPRE_CALL(HYPRE_ParCSRBiCGSTABSetLogging(mom.lin.solver, 1));

  if(doMatrixSetup || !mom.lin.is_setup){
    HYPRE_CALL(HYPRE_ParCSRBiCGSTABSetPrecond(mom.lin.solver,
        (HYPRE_PtrToParSolverFcn)HYPRE_ParCSRDiagScale,
        (HYPRE_PtrToParSolverFcn)HYPRE_ParCSRDiagScaleSetup,
        nullptr));
    double t0=MPI_Wtime();
    HYPRE_CALL(HYPRE_ParCSRBiCGSTABSetup(mom.lin.solver, mom.lin.Apar, mom.lin.bpar, mom.lin.xpar));
    tsetup += MPI_Wtime()-t0;
    mom.lin.is_setup = true;
  }

  double t0=MPI_Wtime();
  HYPRE_Int solveErr = HYPRE_ParCSRBiCGSTABSolve(mom.lin.solver, mom.lin.Apar, mom.lin.bpar, mom.lin.xpar);
  tsolve += MPI_Wtime()-t0;

  its=-1; relres=0.0;
  HYPRE_Int itsTmp=-1; HYPRE_Real relTmp = (HYPRE_Real)(0.0);
  HYPRE_Int itsErr = HYPRE_ParCSRBiCGSTABGetNumIterations(mom.lin.solver,&itsTmp);
  HYPRE_Int relErr = HYPRE_ParCSRBiCGSTABGetFinalRelativeResidualNorm(mom.lin.solver,&relTmp);
  if(itsErr == 0) its = itsTmp;
  if(relErr == 0) relres = relTmp;
  handle_velocity_bicgstab_status(par, solveErr, itsErr, relErr, its, relres);

  const int block=256;
  kernel_copy_plus_hypre_solution<<<(n + block - 1)/block, block>>>(n, d_xout, d_x0, mom.lin.x_data_dev);
  CUDA_CHECK_LAST();
}


struct MCGSColoring {
  bool built=false;
  int n=0;
  int nColors=0;
  std::vector<int> colorOffsets;
  std::vector<int> colorCells;
  int *d_colorOffsets=nullptr;
  int *d_colorCells=nullptr;
};

static MCGSColoring g_mcgs_coloring;

static void build_mcgs_coloring_once(const MatrixPattern &pat)
{
  if(g_mcgs_coloring.built && g_mcgs_coloring.n == pat.nRows) return;

  const int n = pat.nRows;
  std::vector<int> color(n, -1);
  std::vector<int> mark(64, -1);

  int nColors = 0;
  int tag = 1;

  for(int i=0; i<n; ++i){
    ++tag;
    if(tag == 0x3fffffff){
      std::fill(mark.begin(), mark.end(), -1);
      tag = 1;
    }

    const int p0 = pat.rowOffsets[i];
    const int p1 = pat.rowOffsets[i+1];

    for(int p=p0; p<p1; ++p){
      const int j = (int)pat.cols[p];
      if(j < 0 || j >= n || j == i) continue;
      const int cj = color[j];
      if(cj >= 0){
        if(cj >= (int)mark.size()) mark.resize(cj + 64, -1);
        mark[cj] = tag;
      }
    }

    int c = 0;
    while(c < nColors){
      if(c >= (int)mark.size()) mark.resize(c + 64, -1);
      if(mark[c] != tag) break;
      ++c;
    }

    color[i] = c;
    if(c == nColors) ++nColors;
  }

  long long conflicts = 0;
  long long graphEdges = 0;
  int maxDegree = 0;

  for(int i=0; i<n; ++i){
    const int p0 = pat.rowOffsets[i];
    const int p1 = pat.rowOffsets[i+1];
    int deg = 0;

    for(int p=p0; p<p1; ++p){
      const int j = (int)pat.cols[p];
      if(j < 0 || j >= n || j == i) continue;
      ++deg;
      if(color[i] == color[j]) ++conflicts;
    }

    maxDegree = std::max(maxDegree, deg);
    graphEdges += deg;
  }

  if(conflicts != 0){
    std::fprintf(stderr,
                 "ERROR: MCGS coloring has %lld same-color adjacency conflicts.\n",
                 conflicts);
    MPI_Abort(MPI_COMM_WORLD, 3);
  }

  std::vector<int> counts(nColors, 0);
  for(int i=0; i<n; ++i) counts[color[i]]++;

  std::vector<int> offsets(nColors + 1, 0);
  for(int c=0; c<nColors; ++c) offsets[c+1] = offsets[c] + counts[c];

  std::vector<int> cursor = offsets;
  std::vector<int> cells(n);
  for(int i=0; i<n; ++i){
    const int c = color[i];
    cells[cursor[c]++] = i;
  }

  int minCount = n;
  int maxCount = 0;
  for(int c=0; c<nColors; ++c){
    minCount = std::min(minCount, counts[c]);
    maxCount = std::max(maxCount, counts[c]);
  }

  int rank = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  if(rank == 0){
    std::printf("MCGS coloring: nColors=%d, minCells/color=%d, maxCells/color=%d, avgCells/color=%.1f, maxDegree=%d, directedEdges=%lld\n",
                nColors, minCount, maxCount, (double)n / std::max(1, nColors), maxDegree, graphEdges);
  }

  if(g_mcgs_coloring.d_colorOffsets) device_free(g_mcgs_coloring.d_colorOffsets);
  if(g_mcgs_coloring.d_colorCells)   device_free(g_mcgs_coloring.d_colorCells);

  g_mcgs_coloring.built = true;
  g_mcgs_coloring.n = n;
  g_mcgs_coloring.nColors = nColors;
  g_mcgs_coloring.colorOffsets = std::move(offsets);
  g_mcgs_coloring.colorCells = std::move(cells);

  device_alloc(g_mcgs_coloring.d_colorOffsets, g_mcgs_coloring.colorOffsets.size());
  copy_vec_to_device(g_mcgs_coloring.colorOffsets, g_mcgs_coloring.d_colorOffsets);

  device_alloc(g_mcgs_coloring.d_colorCells, g_mcgs_coloring.colorCells.size());
  copy_vec_to_device(g_mcgs_coloring.colorCells, g_mcgs_coloring.d_colorCells);
}

__global__ static void kernel_mcgs_color_sweep(
    const int color,
    const int *colorOffsets,
    const int *colorCells,
    const int *rowOffsets,
    const HYPRE_BigInt *cols,
    const HYPRE_Complex *Avals,
    const int *permPatternToHypre,
    int usePermutedHypreValues,
    const HYPRE_Complex *rhs,
    HYPRE_Complex *x,
    double omega)
{
  const int start = colorOffsets[color];
  const int end   = colorOffsets[color + 1];
  const int k = start + blockIdx.x * blockDim.x + threadIdx.x;

  if(k >= end) return;

  const int i = colorCells[k];
  const int p0 = rowOffsets[i];
  const int p1 = rowOffsets[i+1];

  double diag = 0.0;
  double sumOff = 0.0;

  for(int p=p0; p<p1; ++p){
    const int j = (int)cols[p];

    double a;
    if(usePermutedHypreValues){
      a = (double)Avals[permPatternToHypre[p]];
    } else {
      a = (double)Avals[p];
    }

    if(j == i){
      diag = a;
    } else {
      sumOff += a * (double)x[j];
    }
  }

  if(fabs(diag) > 1.0e-300){
    const double gsVal = ((double)rhs[i] - sumOff) / diag;
    x[i] = (HYPRE_Complex)((1.0 - omega) * (double)x[i] + omega * gsVal);
  }
}

static void solve_momentum_gpu_device_mcgs_defect_x0_xout(
    GPUMomentumAssembler &mom,
    const Params &par,
    const cfdScalar *d_x0,
    cfdScalar *d_xout,
    HYPRE_Int &its,
    double &relres,
    double &tsetup,
    double &tsolve,
    bool doMatrixSetup)
{
  (void)tsetup;

  const int n = mom.lin.n;
  const int block = 256;

  if(par.velSweeps < 0){
    std::fprintf(stderr, "ERROR: -vel-sweeps must be >= 0\n");
    MPI_Abort(MPI_COMM_WORLD, 1);
  }

  if(par.velSmootherOmega <= 0.0){
    std::fprintf(stderr, "ERROR: -vel-smoother-omega must be > 0\n");
    MPI_Abort(MPI_COMM_WORLD, 1);
  }

  build_mcgs_coloring_once(mom.lin.pat);

  double t0 = MPI_Wtime();

  if(doMatrixSetup || !mom.lin.direct_matrix_values){
    copy_matrix_values_into_hypre(mom.lin);
  }

  // Build defect RHS:
  //   bpar = b
  //   xpar = qOld
  //   bpar = bpar - A*xpar
  CUDA_CALL(cudaMemcpy(mom.lin.b_data_dev, mom.lin.d_rhs,
                       n * sizeof(HYPRE_Complex),
                       cudaMemcpyDeviceToDevice));
  copy_scalar_device_to_hypre_device(n, d_x0, mom.lin.x_data_dev);

  HYPRE_CALL(HYPRE_ParCSRMatrixMatvec(
      (HYPRE_Complex)-1.0,
      mom.lin.Apar,
      mom.lin.xpar,
      (HYPRE_Complex)1.0,
      mom.lin.bpar));

  // Correction starts from zero:
  //   A*dq = defect
  CUDA_CALL(cudaMemset(mom.lin.x_data_dev, 0, n * sizeof(HYPRE_Complex)));

  const bool usePermutedHypreValues =
      mom.lin.direct_matrix_values && !mom.lin.A_diag_identity_perm;

  const HYPRE_Complex *Avals =
      usePermutedHypreValues ? mom.lin.A_diag_data_dev : matrix_values_ptr(mom.lin);

  for(int sweep=0; sweep<par.velSweeps; ++sweep){
    for(int c=0; c<g_mcgs_coloring.nColors; ++c){
      const int start = g_mcgs_coloring.colorOffsets[c];
      const int end   = g_mcgs_coloring.colorOffsets[c+1];
      const int nThis = end - start;
      if(nThis <= 0) continue;

      const int grid = (nThis + block - 1) / block;
      kernel_mcgs_color_sweep<<<grid, block>>>(
          c,
          g_mcgs_coloring.d_colorOffsets,
          g_mcgs_coloring.d_colorCells,
          mom.lin.pat.d_rowOffsets,
          mom.lin.pat.d_cols,
          Avals,
          mom.lin.A_diag_perm_d,
          usePermutedHypreValues ? 1 : 0,
          mom.lin.b_data_dev,
          mom.lin.x_data_dev,
          par.velSmootherOmega);
      CUDA_CHECK_LAST();
    }

    if(par.velGsSymmetric){
      for(int c=g_mcgs_coloring.nColors-1; c>=0; --c){
        const int start = g_mcgs_coloring.colorOffsets[c];
        const int end   = g_mcgs_coloring.colorOffsets[c+1];
        const int nThis = end - start;
        if(nThis <= 0) continue;

        const int grid = (nThis + block - 1) / block;
        kernel_mcgs_color_sweep<<<grid, block>>>(
            c,
            g_mcgs_coloring.d_colorOffsets,
            g_mcgs_coloring.d_colorCells,
            mom.lin.pat.d_rowOffsets,
            mom.lin.pat.d_cols,
            Avals,
            mom.lin.A_diag_perm_d,
            usePermutedHypreValues ? 1 : 0,
            mom.lin.b_data_dev,
            mom.lin.x_data_dev,
            par.velSmootherOmega);
        CUDA_CHECK_LAST();
      }
    }
  }

  // qNew = qOld + dq
  kernel_copy_plus_hypre_solution<<<(n + block - 1) / block, block>>>(
      n,
      d_xout,
      d_x0,
      mom.lin.x_data_dev);
  CUDA_CHECK_LAST();

  tsolve += MPI_Wtime() - t0;

  its = (HYPRE_Int)par.velSweeps;
  relres = -1.0;
}



struct FusedMCGSScratch {
  int n=0;
  HYPRE_Complex *d_rhsU=nullptr, *d_rhsV=nullptr, *d_rhsW=nullptr;
  HYPRE_Complex *d_du=nullptr,   *d_dv=nullptr,   *d_dw=nullptr;
};

static FusedMCGSScratch g_fused_mcgs;

static void ensure_fused_mcgs_scratch(int n)
{
  if(g_fused_mcgs.n == n &&
     g_fused_mcgs.d_rhsU && g_fused_mcgs.d_rhsV && g_fused_mcgs.d_rhsW &&
     g_fused_mcgs.d_du   && g_fused_mcgs.d_dv   && g_fused_mcgs.d_dw) return;

  device_free(g_fused_mcgs.d_rhsU);
  device_free(g_fused_mcgs.d_rhsV);
  device_free(g_fused_mcgs.d_rhsW);
  device_free(g_fused_mcgs.d_du);
  device_free(g_fused_mcgs.d_dv);
  device_free(g_fused_mcgs.d_dw);

  g_fused_mcgs.n = n;
  device_alloc(g_fused_mcgs.d_rhsU, n);
  device_alloc(g_fused_mcgs.d_rhsV, n);
  device_alloc(g_fused_mcgs.d_rhsW, n);
  device_alloc(g_fused_mcgs.d_du,   n);
  device_alloc(g_fused_mcgs.d_dv,   n);
  device_alloc(g_fused_mcgs.d_dw,   n);
}

__global__ static void kernel_fused_defect_zero(
    int n,
    const int *rowOffsets,
    const HYPRE_BigInt *cols,
    const HYPRE_Complex *Avals,
    const int *permPatternToHypre,
    int usePermutedHypreValues,
    HYPRE_Complex *rhsU,
    HYPRE_Complex *rhsV,
    HYPRE_Complex *rhsW,
    const cfdScalar *uOld,
    const cfdScalar *vOld,
    const cfdScalar *wOld,
    HYPRE_Complex *du,
    HYPRE_Complex *dv,
    HYPRE_Complex *dw)
{
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if(i >= n) return;

  double Au = 0.0;
  double Av = 0.0;
  double Aw = 0.0;

  const int p0 = rowOffsets[i];
  const int p1 = rowOffsets[i+1];

  for(int p=p0; p<p1; ++p){
    const int j = (int)cols[p];
    double a;
    if(usePermutedHypreValues){
      a = (double)Avals[permPatternToHypre[p]];
    } else {
      a = (double)Avals[p];
    }

    Au += a * uOld[j];
    Av += a * vOld[j];
    Aw += a * wOld[j];
  }

  rhsU[i] = (HYPRE_Complex)((double)rhsU[i] - Au);
  rhsV[i] = (HYPRE_Complex)((double)rhsV[i] - Av);
  rhsW[i] = (HYPRE_Complex)((double)rhsW[i] - Aw);

  du[i] = 0.0;
  dv[i] = 0.0;
  dw[i] = 0.0;
}

__global__ static void kernel_mcgs_color_sweep_fused(
    const int color,
    const int *colorOffsets,
    const int *colorCells,
    const int *rowOffsets,
    const HYPRE_BigInt *cols,
    const HYPRE_Complex *Avals,
    const int *permPatternToHypre,
    int usePermutedHypreValues,
    const HYPRE_Complex *rhsU,
    const HYPRE_Complex *rhsV,
    const HYPRE_Complex *rhsW,
    HYPRE_Complex *du,
    HYPRE_Complex *dv,
    HYPRE_Complex *dw,
    double omega)
{
  const int start = colorOffsets[color];
  const int end   = colorOffsets[color + 1];

  const int k = start + blockIdx.x * blockDim.x + threadIdx.x;
  if(k >= end) return;

  const int i = colorCells[k];
  const int p0 = rowOffsets[i];
  const int p1 = rowOffsets[i+1];

  double diag = 0.0;
  double sumU = 0.0;
  double sumV = 0.0;
  double sumW = 0.0;

  for(int p=p0; p<p1; ++p){
    const int j = (int)cols[p];

    double a;
    if(usePermutedHypreValues){
      a = (double)Avals[permPatternToHypre[p]];
    } else {
      a = (double)Avals[p];
    }

    if(j == i){
      diag = a;
    } else {
      sumU += a * (double)du[j];
      sumV += a * (double)dv[j];
      sumW += a * (double)dw[j];
    }
  }

  if(fabs(diag) > 1.0e-300){
    const double newU = ((double)rhsU[i] - sumU) / diag;
    const double newV = ((double)rhsV[i] - sumV) / diag;
    const double newW = ((double)rhsW[i] - sumW) / diag;

    du[i] = (HYPRE_Complex)((1.0 - omega) * (double)du[i] + omega * newU);
    dv[i] = (HYPRE_Complex)((1.0 - omega) * (double)dv[i] + omega * newV);
    dw[i] = (HYPRE_Complex)((1.0 - omega) * (double)dw[i] + omega * newW);
  }
}

__global__ static void kernel_copy_plus3(
    int n,
    cfdScalar *uOut,
    cfdScalar *vOut,
    cfdScalar *wOut,
    const cfdScalar *uOld,
    const cfdScalar *vOld,
    const cfdScalar *wOld,
    const HYPRE_Complex *du,
    const HYPRE_Complex *dv,
    const HYPRE_Complex *dw)
{
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if(i >= n) return;

  uOut[i] = uOld[i] + (double)du[i];
  vOut[i] = vOld[i] + (double)dv[i];
  wOut[i] = wOld[i] + (double)dw[i];
}

static void solve_momentum_gpu_device_mcgs_fused_defect_x0_xout(
    GPUMomentumAssembler &mom,
    const Params &par,
    const cfdScalar *d_uOld,
    const cfdScalar *d_vOld,
    const cfdScalar *d_wOld,
    cfdScalar *d_uOut,
    cfdScalar *d_vOut,
    cfdScalar *d_wOut,
    HYPRE_Int &itsU,
    HYPRE_Int &itsV,
    HYPRE_Int &itsW,
    double &relU,
    double &relV,
    double &relW,
    double &tsetup,
    double &tsolve,
    bool doMatrixSetup)
{
  (void)tsetup;

  const int n = mom.lin.n;
  const int block = 256;
  const int gridCells = (n + block - 1) / block;

  ensure_fused_mcgs_scratch(n);
  build_mcgs_coloring_once(mom.lin.pat);

  double t0 = MPI_Wtime();

  if(doMatrixSetup || !mom.lin.direct_matrix_values){
    copy_matrix_values_into_hypre(mom.lin);
  }

  const bool usePermutedHypreValues =
      mom.lin.direct_matrix_values && !mom.lin.A_diag_identity_perm;

  const HYPRE_Complex *Avals =
      usePermutedHypreValues ? mom.lin.A_diag_data_dev : matrix_values_ptr(mom.lin);

  // Convert full RHS vectors into true defect RHS:
  //
  //   rhsU = bU - A*uOld
  //   rhsV = bV - A*vOld
  //   rhsW = bW - A*wOld
  //
  // Also zero correction vectors.
  kernel_fused_defect_zero<<<gridCells, block>>>(
      n,
      mom.lin.pat.d_rowOffsets,
      mom.lin.pat.d_cols,
      Avals,
      mom.lin.A_diag_perm_d,
      usePermutedHypreValues ? 1 : 0,
      g_fused_mcgs.d_rhsU,
      g_fused_mcgs.d_rhsV,
      g_fused_mcgs.d_rhsW,
      d_uOld,
      d_vOld,
      d_wOld,
      g_fused_mcgs.d_du,
      g_fused_mcgs.d_dv,
      g_fused_mcgs.d_dw);
  CUDA_CHECK_LAST();

  for(int sweep=0; sweep<par.velSweeps; ++sweep){
    for(int c=0; c<g_mcgs_coloring.nColors; ++c){
      const int start = g_mcgs_coloring.colorOffsets[c];
      const int end   = g_mcgs_coloring.colorOffsets[c+1];
      const int nThis = end - start;
      if(nThis <= 0) continue;

      const int grid = (nThis + block - 1) / block;
      kernel_mcgs_color_sweep_fused<<<grid, block>>>(
          c,
          g_mcgs_coloring.d_colorOffsets,
          g_mcgs_coloring.d_colorCells,
          mom.lin.pat.d_rowOffsets,
          mom.lin.pat.d_cols,
          Avals,
          mom.lin.A_diag_perm_d,
          usePermutedHypreValues ? 1 : 0,
          g_fused_mcgs.d_rhsU,
          g_fused_mcgs.d_rhsV,
          g_fused_mcgs.d_rhsW,
          g_fused_mcgs.d_du,
          g_fused_mcgs.d_dv,
          g_fused_mcgs.d_dw,
          par.velSmootherOmega);
      CUDA_CHECK_LAST();
    }

    if(par.velGsSymmetric){
      for(int c=g_mcgs_coloring.nColors-1; c>=0; --c){
        const int start = g_mcgs_coloring.colorOffsets[c];
        const int end   = g_mcgs_coloring.colorOffsets[c+1];
        const int nThis = end - start;
        if(nThis <= 0) continue;

        const int grid = (nThis + block - 1) / block;
        kernel_mcgs_color_sweep_fused<<<grid, block>>>(
            c,
            g_mcgs_coloring.d_colorOffsets,
            g_mcgs_coloring.d_colorCells,
            mom.lin.pat.d_rowOffsets,
            mom.lin.pat.d_cols,
            Avals,
            mom.lin.A_diag_perm_d,
            usePermutedHypreValues ? 1 : 0,
            g_fused_mcgs.d_rhsU,
            g_fused_mcgs.d_rhsV,
            g_fused_mcgs.d_rhsW,
            g_fused_mcgs.d_du,
            g_fused_mcgs.d_dv,
            g_fused_mcgs.d_dw,
            par.velSmootherOmega);
        CUDA_CHECK_LAST();
      }
    }
  }

  kernel_copy_plus3<<<gridCells, block>>>(
      n,
      d_uOut,
      d_vOut,
      d_wOut,
      d_uOld,
      d_vOld,
      d_wOld,
      g_fused_mcgs.d_du,
      g_fused_mcgs.d_dv,
      g_fused_mcgs.d_dw);
  CUDA_CHECK_LAST();

  tsolve += MPI_Wtime() - t0;

  itsU = (HYPRE_Int)par.velSweeps;
  itsV = (HYPRE_Int)par.velSweeps;
  itsW = (HYPRE_Int)par.velSweeps;
  relU = -1.0;
  relV = -1.0;
  relW = -1.0;
}



static void solve_momentum_gpu(GPUMomentumAssembler &mom, const Params &par, const std::vector<cfdScalar> &x0, std::vector<cfdScalar> &xout, HYPRE_Int &its, double &relres, double &tsetup, double &tsolve, bool doMatrixSetup){
  copy_device_rhs_and_host_x0_into_hypre(mom.lin, x0);

  HYPRE_CALL(HYPRE_ParCSRBiCGSTABSetTol(mom.lin.solver, par.velRelTol));
  HYPRE_CALL(HYPRE_ParCSRBiCGSTABSetAbsoluteTol(mom.lin.solver, par.velTol));
  HYPRE_CALL(HYPRE_ParCSRBiCGSTABSetMaxIter(mom.lin.solver, par.velMaxit));
  HYPRE_CALL(HYPRE_ParCSRBiCGSTABSetPrintLevel(mom.lin.solver, 0));
  HYPRE_CALL(HYPRE_ParCSRBiCGSTABSetLogging(mom.lin.solver, 1));
  if(doMatrixSetup || !mom.lin.is_setup){
    HYPRE_CALL(HYPRE_ParCSRBiCGSTABSetPrecond(mom.lin.solver, (HYPRE_PtrToParSolverFcn)HYPRE_ParCSRDiagScale, (HYPRE_PtrToParSolverFcn)HYPRE_ParCSRDiagScaleSetup, nullptr));
    copy_matrix_values_into_hypre(mom.lin);
    double t0=MPI_Wtime();
    HYPRE_CALL(HYPRE_ParCSRBiCGSTABSetup(mom.lin.solver, mom.lin.Apar, mom.lin.bpar, mom.lin.xpar));
    tsetup += MPI_Wtime()-t0;
    mom.lin.is_setup = true;
  }
  double t0=MPI_Wtime();
  HYPRE_Int solveErr = HYPRE_ParCSRBiCGSTABSolve(mom.lin.solver, mom.lin.Apar, mom.lin.bpar, mom.lin.xpar);
  tsolve += MPI_Wtime()-t0;

  its=-1; relres=0.0;
  HYPRE_Int itsTmp=-1; HYPRE_Real relTmp = (HYPRE_Real)(0.0);
  HYPRE_Int itsErr = HYPRE_ParCSRBiCGSTABGetNumIterations(mom.lin.solver,&itsTmp);
  HYPRE_Int relErr = HYPRE_ParCSRBiCGSTABGetFinalRelativeResidualNorm(mom.lin.solver,&relTmp);
  if(itsErr == 0) its = itsTmp;
  if(relErr == 0) relres = relTmp;
  handle_velocity_bicgstab_status(par, solveErr, itsErr, relErr, its, relres);
  copy_solution_from_hypre(mom.lin, xout);
}

static void solve_pressure_gpu(GPULinearSystem &ps, const std::vector<cfdScalar> &rhs, const std::vector<cfdScalar> &x0, double relTol, double absTol, int maxIter, std::vector<cfdScalar> &xout, HYPRE_Int &its, double &relres, double &tsolve){
  copy_host_rhs_and_host_x0_into_hypre(ps, rhs, x0);
  HYPRE_CALL(HYPRE_ParCSRPCGSetTol(ps.solver, relTol));
  HYPRE_CALL(HYPRE_ParCSRPCGSetAbsoluteTol(ps.solver, absTol));
  HYPRE_CALL(HYPRE_ParCSRPCGSetMaxIter(ps.solver, maxIter));
  double t0=MPI_Wtime();
  HYPRE_CALL(HYPRE_ParCSRPCGSolve(ps.solver, ps.Apar, ps.bpar, ps.xpar));
  tsolve += MPI_Wtime()-t0;
  its=0; relres=0.0;
  HYPRE_CALL(HYPRE_ParCSRPCGGetNumIterations(ps.solver,&its));
  do {
    HYPRE_Real relres_hypre_tmp = (HYPRE_Real)0.0;
    HYPRE_CALL(HYPRE_ParCSRPCGGetFinalRelativeResidualNorm(ps.solver, &relres_hypre_tmp));
    relres = (double)relres_hypre_tmp;
  } while(0);
  copy_solution_from_hypre(ps, xout);
}

static void solve_pressure_gpu_device_rhs(GPULinearSystem &ps, const std::vector<cfdScalar> &x0, double relTol, double absTol, int maxIter, std::vector<cfdScalar> &xout, HYPRE_Int &its, double &relres, double &tsolve){
  copy_device_rhs_and_host_x0_into_hypre(ps, x0);
  if(g_p_amg_setup_scope == 1){
    double ts0 = MPI_Wtime();
    pressure_solver_setup(ps);
    tsolve += MPI_Wtime() - ts0;
  }
  HYPRE_CALL(HYPRE_ParCSRPCGSetTol(ps.solver, relTol));
  HYPRE_CALL(HYPRE_ParCSRPCGSetAbsoluteTol(ps.solver, absTol));
  HYPRE_CALL(HYPRE_ParCSRPCGSetMaxIter(ps.solver, maxIter));
  double t0=MPI_Wtime();
  HYPRE_CALL(HYPRE_ParCSRPCGSolve(ps.solver, ps.Apar, ps.bpar, ps.xpar));
  tsolve += MPI_Wtime()-t0;
  its=0; relres=0.0;
  HYPRE_CALL(HYPRE_ParCSRPCGGetNumIterations(ps.solver,&its));
  do {
    HYPRE_Real relres_hypre_tmp = (HYPRE_Real)0.0;
    HYPRE_CALL(HYPRE_ParCSRPCGGetFinalRelativeResidualNorm(ps.solver, &relres_hypre_tmp));
    relres = (double)relres_hypre_tmp;
  } while(0);
  copy_solution_from_hypre(ps, xout);
}


static void pressure_solver_setup(GPULinearSystem &ps){
  HYPRE_CALL(HYPRE_ParCSRPCGSetup(ps.solver, ps.Apar, ps.bpar, ps.xpar));
  ps.is_setup = true;
}

static void solve_pressure_gpu_device_rhs_device_x0(
    GPULinearSystem &ps,
    const cfdScalar *d_x0,
    cfdScalar *d_xout,
    double relTol,
    double absTol,
    int maxIter,
    HYPRE_Int &its,
    double &relres,
    double &tsolve)
{
  copy_device_rhs_and_device_x0_into_hypre(ps, d_x0);

  if(g_p_amg_setup_scope == 1){
    double ts0 = MPI_Wtime();
    pressure_solver_setup(ps);
    tsolve += MPI_Wtime() - ts0;
  }

  HYPRE_CALL(HYPRE_ParCSRPCGSetTol(ps.solver, relTol));
  HYPRE_CALL(HYPRE_ParCSRPCGSetAbsoluteTol(ps.solver, absTol));
  HYPRE_CALL(HYPRE_ParCSRPCGSetMaxIter(ps.solver, maxIter));

  double t0 = MPI_Wtime();
  HYPRE_Int solveErr = HYPRE_ParCSRPCGSolve(ps.solver, ps.Apar, ps.bpar, ps.xpar);
  tsolve += MPI_Wtime() - t0;

  its = -1;
  relres = 0.0;

  HYPRE_Int itsTmp = -1;
  HYPRE_Real relTmp = (HYPRE_Real)0.0;

  HYPRE_Int itsErr = HYPRE_ParCSRPCGGetNumIterations(ps.solver, &itsTmp);
  HYPRE_Int relErr = HYPRE_ParCSRPCGGetFinalRelativeResidualNorm(ps.solver, &relTmp);

  if(itsErr == 0) its = itsTmp;
  if(relErr == 0) relres = relTmp;

  if(solveErr || itsErr || relErr){
    if(solveErr == 256 || itsErr == 256 || relErr == 256){
      std::fprintf(stderr,
          "WARNING: pressure PCG non-convergence/sticky HYPRE error. "
          "solveErr=%d itsErr=%d relErr=%d its=%d finalRel=%.6e. "
          "Continuing with approximate pCorr.\n",
          (int)solveErr, (int)itsErr, (int)relErr, (int)its, relres);

      HYPRE_ClearAllErrors();
    } else {
      std::fprintf(stderr,
          "FATAL: pressure PCG failed. "
          "solveErr=%d itsErr=%d relErr=%d its=%d finalRel=%.6e.\n",
          (int)solveErr, (int)itsErr, (int)relErr, (int)its, relres);

      MPI_Abort(MPI_COMM_WORLD, solveErr ? solveErr : (itsErr ? itsErr : relErr));
    }
  }

  copy_solution_from_hypre_to_device(ps, d_xout);
}


struct CylinderForceVectorReport {
  bool valid = false;
  int patchIndex = -1;
  std::string patchName;
  int nFaces = 0;
  double area = 0.0;

  std::array<double,3> Fp{{0.0,0.0,0.0}};
  std::array<double,3> Fv{{0.0,0.0,0.0}};
  std::array<double,3> F {{0.0,0.0,0.0}};
  std::array<double,3> C {{0.0,0.0,0.0}};

  double rho = 0.0;
  double mu = 0.0;
  double Ubar = 0.0;
  double D = 0.0;
  double H = 0.0;
  double coeffDenom = 0.0;

  double minWallDistance = 1.0e300;
  double maxWallDistance = 0.0;
  double maxUt = 0.0;
  double maxShearMag = 0.0;
};

// Orientation-independent cylinder force.
// mesh.nf[f] points from fluid cell toward boundary/outside.
// For cylinder obstacle, paper/body normal is solid -> fluid = -mesh.nf[f].
static CylinderForceVectorReport compute_cylinder_forces_vector_wall_shear(
    const Mesh& mesh,
    int cylinderPatch,
    const std::vector<cfdScalar>& u,
    const std::vector<cfdScalar>& v,
    const std::vector<cfdScalar>& w,
    const std::vector<cfdScalar>& p,
    double rho,
    double mu,
    double D,
    double H,
    double Ubar)
{
  CylinderForceVectorReport r;
  if(cylinderPatch < 0 || cylinderPatch >= (int)mesh.patchNames.size()) return r;

  r.valid = true;
  r.patchIndex = cylinderPatch;
  r.patchName = mesh.patchNames[cylinderPatch];
  r.rho = rho;
  r.mu = mu;
  r.D = D;
  r.H = H;
  r.Ubar = Ubar;
  r.coeffDenom = rho * Ubar * Ubar * D * H;

  const int f0 = mesh.patchStartFace[cylinderPatch];
  const int f1 = f0 + mesh.patchNFaces[cylinderPatch];

  for(int f = f0; f < f1; ++f){
    const int P = mesh.owner[f];
    const double A = mesh.Af[f];
    if(A <= 1.0e-300) continue;

    // Body/cylinder normal, from solid into fluid.
    std::array<double,3> n{{-mesh.nf[f][0], -mesh.nf[f][1], -mesh.nf[f][2]}};
    const double nmag = norm3(n);
    if(nmag <= 1.0e-300) continue;
    n = mul3(1.0/nmag, n);

    std::array<double,3> dx = sub3(mesh.cc[P], mesh.xf[f]);
    double dn = dot3(dx, n);
    if(dn <= 1.0e-14) dn = std::fabs(dn);
    if(dn <= 1.0e-14) dn = norm3(dx);
    if(dn <= 1.0e-14) continue;

    std::array<double,3> U{{u[P], v[P], w[P]}};
    const double Un = dot3(U, n);
    std::array<double,3> Ut = sub3(U, mul3(Un, n));

    const double Utmag = norm3(Ut);
    const double shearMag = mu * Utmag / dn;

    for(int d=0; d<3; ++d){
      const double fp = -p[P] * n[d] * A;
      const double fv =  mu * Ut[d] / dn * A;

      r.Fp[d] += fp;
      r.Fv[d] += fv;
      r.F[d]  += fp + fv;
    }

    r.area += A;
    r.nFaces += 1;
    r.minWallDistance = std::min(r.minWallDistance, dn);
    r.maxWallDistance = std::max(r.maxWallDistance, dn);
    r.maxUt = std::max(r.maxUt, Utmag);
    r.maxShearMag = std::max(r.maxShearMag, shearMag);
  }

  if(r.coeffDenom > 1.0e-300){
    for(int d=0; d<3; ++d){
      r.C[d] = 2.0 * r.F[d] / r.coeffDenom;
    }
  }

  if(r.minWallDistance == 1.0e300) r.minWallDistance = 0.0;
  return r;
}



struct PatchForceReport {
  bool requested = false;
  bool valid = false;

  int patchIndex = -1;
  std::string patchName;
  int nFaces = 0;
  double area = 0.0;

  int normalSign = -1;

  std::array<double,3> dragDir{{1.0,0.0,0.0}};
  std::array<double,3> liftDir{{0.0,1.0,0.0}};
  std::array<double,3> spanDir{{0.0,0.0,1.0}};

  std::array<double,3> Fp{{0.0,0.0,0.0}};
  std::array<double,3> Fv{{0.0,0.0,0.0}};
  std::array<double,3> F {{0.0,0.0,0.0}};

  double FpDrag = 0.0, FvDrag = 0.0, FDrag = 0.0;
  double FpLift = 0.0, FvLift = 0.0, FLift = 0.0;
  double FpSpan = 0.0, FvSpan = 0.0, FSpan = 0.0;

  double CDrag = 0.0;
  double CLift = 0.0;
  double CSpan = 0.0;

  double rho = 0.0;
  double mu = 0.0;
  double Uref = 0.0;
  double Aref = 0.0;
  double coeffDenom = 0.0;

  double minWallDistance = 1.0e300;
  double maxWallDistance = 0.0;
  double maxUt = 0.0;
  double maxShearMag = 0.0;
};

static PatchForceReport compute_patch_forces_wall_shear(
    const Mesh& mesh,
    int patchIndex,
    const std::vector<cfdScalar>& u,
    const std::vector<cfdScalar>& v,
    const std::vector<cfdScalar>& w,
    const std::vector<cfdScalar>& p,
    double rho,
    double mu,
    int normalSign,
    double Uref,
    double Aref,
    std::array<double,3> dragDir,
    std::array<double,3> liftDir,
    std::array<double,3> spanDir)
{
  PatchForceReport r;
  r.requested = true;

  if(patchIndex < 0 || patchIndex >= (int)mesh.patchNames.size()) return r;
  if(Uref <= 0.0 || Aref <= 0.0) return r;

  r.valid = true;
  r.patchIndex = patchIndex;
  r.patchName = mesh.patchNames[patchIndex];
  r.normalSign = normalSign;
  r.dragDir = normalized_vec3(dragDir);
  r.liftDir = normalized_vec3(liftDir);
  r.spanDir = normalized_vec3(spanDir);

  r.rho = rho;
  r.mu = mu;
  r.Uref = Uref;
  r.Aref = Aref;
  r.coeffDenom = rho * Uref * Uref * Aref;

  const int f0 = mesh.patchStartFace[patchIndex];
  const int f1 = f0 + mesh.patchNFaces[patchIndex];

  for(int f = f0; f < f1; ++f){
    const int P = mesh.owner[f];
    const double A = mesh.Af[f];
    if(A <= 1.0e-300) continue;

    std::array<double,3> n{{
        (double)normalSign * mesh.nf[f][0],
        (double)normalSign * mesh.nf[f][1],
        (double)normalSign * mesh.nf[f][2]}};

    const double nmag = norm3(n);
    if(nmag <= 1.0e-300) continue;
    n = mul3(1.0/nmag, n);

    const std::array<double,3> dx = sub3(mesh.cc[P], mesh.xf[f]);
    double dn = dot3(dx, n);
    if(dn <= 1.0e-14) dn = std::fabs(dn);
    if(dn <= 1.0e-14) dn = norm3(dx);
    if(dn <= 1.0e-14) continue;

    const std::array<double,3> U{{u[P], v[P], w[P]}};
    const double Un = dot3(U, n);
    const std::array<double,3> Ut = sub3(U, mul3(Un, n));

    const double Utmag = norm3(Ut);
    const double shearMag = mu * Utmag / dn;

    for(int d=0; d<3; ++d){
      const double fp = -p[P] * n[d] * A;
      const double fv =  mu * Ut[d] / dn * A;

      r.Fp[d] += fp;
      r.Fv[d] += fv;
      r.F[d]  += fp + fv;
    }

    r.area += A;
    r.nFaces += 1;
    r.minWallDistance = std::min(r.minWallDistance, dn);
    r.maxWallDistance = std::max(r.maxWallDistance, dn);
    r.maxUt = std::max(r.maxUt, Utmag);
    r.maxShearMag = std::max(r.maxShearMag, shearMag);
  }

  r.FpDrag = dot3(r.Fp, r.dragDir);
  r.FvDrag = dot3(r.Fv, r.dragDir);
  r.FDrag  = dot3(r.F,  r.dragDir);

  r.FpLift = dot3(r.Fp, r.liftDir);
  r.FvLift = dot3(r.Fv, r.liftDir);
  r.FLift  = dot3(r.F,  r.liftDir);

  r.FpSpan = dot3(r.Fp, r.spanDir);
  r.FvSpan = dot3(r.Fv, r.spanDir);
  r.FSpan  = dot3(r.F,  r.spanDir);

  if(r.coeffDenom > 1.0e-300){
    r.CDrag = 2.0 * r.FDrag / r.coeffDenom;
    r.CLift = 2.0 * r.FLift / r.coeffDenom;
    r.CSpan = 2.0 * r.FSpan / r.coeffDenom;
  }

  if(r.minWallDistance == 1.0e300) r.minWallDistance = 0.0;
  return r;
}



// -----------------------------------------------------------------------------
// Scalar transport coupling helpers (v1 modular path)
// -----------------------------------------------------------------------------
static std::string lower_copy_local(std::string v){
  for(char &c : v) c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
  return v;
}

static libscalar::DiffusionScheme scalar_diffusion_scheme_from_case(const std::string& raw){
  const std::string v = lower_copy_local(raw);
  if(v == "orth" || v == "orthogonal" || v == "uncorrected") return libscalar::DiffusionScheme::Orth;
  if(v == "nonorth" || v == "nonorthogonal" || v == "corrected") return libscalar::DiffusionScheme::NonOrth;
  throw std::runtime_error("Unknown scalarDiffusionScheme '" + raw + "'. Use orth or nonorth.");
}

static libscalar::ScalarBCSet read_scalar_bc_config(const Mesh& mesh, const std::string& path){
  libscalar::ScalarBCSet out;
  if(path.empty()) return out;

  std::ifstream in(path);
  if(!in) throw std::runtime_error("Could not open scalar BC config file: " + path);

  std::set<std::string> known(mesh.patchNames.begin(), mesh.patchNames.end());
  std::set<std::string> covered;

  std::string raw;
  int lineNo = 0;
  while(std::getline(in, raw)){
    ++lineNo;
    const std::string line = trim_case_line(raw);
    if(line.empty()) continue;
    const auto tok = tokenize_case_line(line);
    if(tok.empty()) continue;

    if(tok[0] != "scalar"){
      std::ostringstream oss;
      oss << "Scalar BC parse error in '" << path << "' at line " << lineNo
          << ": expected 'scalar <patch> <bcType> [value]'";
      throw std::runtime_error(oss.str());
    }
    if(tok.size() < 3){
      std::ostringstream oss;
      oss << "Scalar BC parse error in '" << path << "' at line " << lineNo
          << ": expected 'scalar <patch> <bcType> [value]'";
      throw std::runtime_error(oss.str());
    }

    const std::string patch = tok[1];
    if(!known.count(patch)){
      std::ostringstream oss;
      oss << "Scalar BC references unknown patch '" << patch << "' at line " << lineNo;
      throw std::runtime_error(oss.str());
    }

    const std::string typ = lower_copy_local(tok[2]);
    if(typ == "fixed_value" || typ == "fixedvalue" || typ == "dirichlet" || typ == "value"){
      if(tok.size() != 4){
        std::ostringstream oss;
        oss << "Scalar fixed_value BC for patch '" << patch << "' needs one value at line " << lineNo;
        throw std::runtime_error(oss.str());
      }
      out.patches.push_back(libscalar::make_dirichlet_constant_bc(patch, std::atof(tok[3].c_str())));
    } else if(typ == "zero_gradient" || typ == "zerogradient" || typ == "outlet"){
      if(tok.size() != 3){
        std::ostringstream oss;
        oss << "Scalar zero_gradient BC for patch '" << patch << "' takes no value at line " << lineNo;
        throw std::runtime_error(oss.str());
      }
      out.patches.push_back(libscalar::make_zero_gradient_patch_bc(patch));
    } else if(typ == "neumann" || typ == "neumann_gradient" || typ == "gradient"){
      if(tok.size() != 4){
        std::ostringstream oss;
        oss << "Scalar neumann_gradient BC for patch '" << patch << "' needs one gradient value at line " << lineNo;
        throw std::runtime_error(oss.str());
      }
      out.patches.push_back(libscalar::make_neumann_gradient_constant_bc(patch, std::atof(tok[3].c_str())));
    } else {
      std::ostringstream oss;
      oss << "Unknown scalar BC type '" << tok[2] << "' for patch '" << patch << "' at line " << lineNo
          << ". Use fixed_value, zero_gradient, or neumann_gradient.";
      throw std::runtime_error(oss.str());
    }
    covered.insert(patch);
  }

  for(const auto& name : mesh.patchNames){
    if(!covered.count(name)){
      throw std::runtime_error("No scalar BC provided for patch '" + name + "' in " + path);
    }
  }
  return out;
}

static bool scalar_solve_mode_is_after_flow(const std::string& raw){
  const std::string v = lower_copy_local(raw);
  return (v == "afterflow" || v == "after-flow" || v == "postflow" || v == "post-flow" || v == "post" || v == "once");
}

static std::vector<cfdScalar> solve_scalar_after_flow(
    const Params& par,
    const Mesh& mesh,
    const std::vector<cfdScalar>& facePhi,
    int rank)
{
  if(par.scalarBCConfigPath.empty()){
    throw std::runtime_error("scalarEnable=1 requires inline scalar BC lines or scalarBCConfig <file>");
  }
  if(!scalar_solve_mode_is_after_flow(par.scalarSolveMode)){
    throw std::runtime_error("Only scalarSolveMode afterFlow is implemented in simple_gpu v1.1/v1 modular build.");
  }

  libscalar::ScalarTransportInputs in;
  in.faceFlux.resize(mesh.nFaces, 0.0);
  const double rhoSafe = std::max(std::fabs(par.rho), 1.0e-300);
  for(int f=0; f<mesh.nFaces; ++f) in.faceFlux[f] = facePhi[f] / rhoSafe; // convert mass flux to volume flux when rho != 1
  in.gammaFace.assign(mesh.nFaces, par.scalarGamma);
  in.Su.assign(mesh.nCells, 0.0);
  in.Sp.assign(mesh.nCells, 0.0);

  libscalar::ScalarTransportOptions opt;
  opt.convectionScheme = libscalar::convection_scheme_from_string(par.scalarConvectionScheme);
  opt.diffusionScheme = scalar_diffusion_scheme_from_case(par.scalarDiffusionScheme);
  opt.nNonOrthCorr = par.scalarNonOrthCorr;
  opt.maxIter = par.scalarMaxit;
  opt.absTol = par.scalarTol;
  opt.relTol = par.scalarRelTol;
  opt.monitor = par.monitor ? 1 : 0;

  const auto bcSet = read_scalar_bc_config(mesh, par.scalarBCConfigPath);

  if(rank == 0){
    std::printf("\nScalar transport solve after flow:\n");
    std::printf("  scalarName             : %s\n", par.scalarName.c_str());
    std::printf("  scalarConvectionScheme : %s\n", libscalar::convection_scheme_name(opt.convectionScheme));
    std::printf("  scalarDiffusionScheme  : %s\n", opt.diffusionScheme == libscalar::DiffusionScheme::NonOrth ? "nonorth" : "orth");
    std::printf("  scalarGamma            : %.12e\n", par.scalarGamma);
    std::printf("  scalarNonOrthCorr      : %d\n", opt.nNonOrthCorr);
    std::printf("  scalarBCConfig         : %s\n", par.scalarBCConfigPath.c_str());
  }

  const double t0 = MPI_Wtime();
  const auto result = libscalar::solve_steady_scalar_transport(mesh, in, bcSet, opt);
  const double t1 = MPI_Wtime();

  if(rank == 0){
    double mn = 0.0, mx = 0.0;
    if(!result.phi.empty()){
      mn = *std::min_element(result.phi.begin(), result.phi.end());
      mx = *std::max_element(result.phi.begin(), result.phi.end());
    }
    std::printf("  scalar iterations      : %d\n", result.iterations);
    std::printf("  scalar final relres    : %.12e\n", result.finalRelRes);
    std::printf("  scalar min/max         : %.12e / %.12e\n", mn, mx);
    std::printf("  scalar wall time       : %.6e s\n", t1 - t0);
  }
  return to_cfd_scalar_vec(result.phi);
}


static void gpu_mem_mark(const char *tag, int rank, bool sync=true)
{
  if(sync){
    cudaError_t se = cudaDeviceSynchronize();
    if(se != cudaSuccess && rank == 0){
      std::fprintf(stderr, "GPU_MEM_MARK tag=%s cudaDeviceSynchronize_error=%s\n",
                   tag, cudaGetErrorString(se));
      std::fflush(stderr);
    }
  }

  size_t freeB = 0;
  size_t totalB = 0;
  cudaError_t e = cudaMemGetInfo(&freeB, &totalB);

  if(rank != 0) return;

  if(e != cudaSuccess){
    std::printf("GPU_MEM_MARK tag=%s cudaMemGetInfo_error=%s\n",
                tag, cudaGetErrorString(e));
    std::fflush(stdout);
    return;
  }

  constexpr double MiB = 1024.0 * 1024.0;
  const double freeMiB  = static_cast<double>(freeB) / MiB;
  const double totalMiB = static_cast<double>(totalB) / MiB;
  const double usedMiB  = totalMiB - freeMiB;

  static int haveBase = 0;
  static double baseMiB = 0.0;
  static double prevMiB = 0.0;
  static double peakSeenMiB = 0.0;

  if(!haveBase){
    haveBase = 1;
    baseMiB = usedMiB;
    prevMiB = usedMiB;
    peakSeenMiB = usedMiB;
  }

  if(usedMiB > peakSeenMiB) peakSeenMiB = usedMiB;

  std::printf(
      "GPU_MEM_MARK tag=%-48s used_MiB=%10.1f free_MiB=%10.1f total_MiB=%10.1f delta_prev_MiB=%+10.1f delta_base_MiB=%+10.1f peak_seen_MiB=%10.1f\n",
      tag, usedMiB, freeMiB, totalMiB,
      usedMiB - prevMiB,
      usedMiB - baseMiB,
      peakSeenMiB);

  std::fflush(stdout);
  prevMiB = usedMiB;
}

int main(int argc, char **argv){
  MPI_Init(&argc,&argv);
  int rank=0,size=1; MPI_Comm_rank(MPI_COMM_WORLD,&rank); MPI_Comm_size(MPI_COMM_WORLD,&size);
  if(size!=1){ if(rank==0) std::fprintf(stderr,"This driver supports exactly 1 MPI rank.\n"); MPI_Abort(MPI_COMM_WORLD,1); }

  Params par;
  auto expandedArgs = expand_case_config_args(argc, argv);
  std::vector<char*> expandedArgv;
  expandedArgv.reserve(expandedArgs.size());
  for(auto& a : expandedArgs) expandedArgv.push_back(a.data());
  parse_args((int)expandedArgv.size(), expandedArgv.data(), par);
  if(par.pSolveMode == 1) par.pMode = 1;
  g_profile_enabled = (par.profileSteps > 0);
  g_p_amg_setup_scope = par.pAmgSetupScope;
  CUDA_CALL(cudaSetDevice(par.device));

  g_pDeltaMode = par.pDeltaMode;
  g_pDeltaMinCos = par.pDeltaMinCos;
CUDA_CALL(cudaFree(0));
  gpu_mem_mark("after_cuda_context", rank);
  if(rank==0) print_device_info(par.device);

  Mesh mesh=read_openfoam_polymesh(par.polyMeshDir, par.geomMethod);
  double xmin=1e300,xmax=-1e300,ymin=1e300,ymax=-1e300,zmin=1e300,zmax=-1e300,vmin=1e300,vmax=-1e300;
  for(const auto &p:mesh.P){ xmin=std::min(xmin,p[0]); xmax=std::max(xmax,p[0]); ymin=std::min(ymin,p[1]); ymax=std::max(ymax,p[1]); zmin=std::min(zmin,p[2]); zmax=std::max(zmax,p[2]); }
  for(double v:mesh.vol){ vmin=std::min(vmin,v); vmax=std::max(vmax,v); }
  int wallPatch=-1, inletPatch=-1, outletPatch=-1;
  int cylinderPatch=-1; // optional separated cylinder wall patch, e.g. patch_3_0
  for(std::size_t k=0;k<mesh.patchNames.size();++k){
    if(mesh.patchNames[k]==par.wallPatchName) wallPatch=(int)k;
    if(mesh.patchNames[k]=="patch_3_0") cylinderPatch=(int)k;
    if(mesh.patchNames[k]==par.inletPatchName) inletPatch=(int)k;
    if(mesh.patchNames[k]==par.outletPatchName) outletPatch=(int)k;
  }
  if(wallPatch<0 || inletPatch<0 || outletPatch<0){
    if(par.simpleMmsEnable || par.sineMmsEnable){
      if(mesh.patchNames.empty()){
        if(rank==0) std::fprintf(stderr,"MMS run has no boundary patches.\n");
        MPI_Abort(MPI_COMM_WORLD,1);
      }
      const int fallbackPatch = 0;
      if(wallPatch < 0) wallPatch = fallbackPatch;
      if(inletPatch < 0) inletPatch = fallbackPatch;
      if(outletPatch < 0) outletPatch = fallbackPatch;
      if(rank==0){
        std::printf("MMS patch placeholder mode: missing wall/inlet/outlet names mapped to patch '%s' for legacy bookkeeping only.\n",
                    mesh.patchNames[fallbackPatch].c_str());
      }
    } else {
      if(rank==0) std::fprintf(stderr,"Could not find wall/inlet/outlet patch.\n");
      MPI_Abort(MPI_COMM_WORLD,1);
    }
  }
  double mu = par.muExplicit ? par.mu : (par.rho*par.Umean*par.pipeDiameter/par.Re);
  double hChar=1e300; for(double v:mesh.vol) hChar=std::min(hChar,std::cbrt(v));
  double dt0=par.CFL*hChar/std::max(par.Umean,1e-12);

  if(rank==0){
    std::printf("============================================================\n");
    std::printf("Anabasis simple_gpu v1.1b: OpenFOAM polyMesh SIMPLE solver\n");
#ifdef ANABASIS_FULL_SP_APP
    const char *precisionMode = "FULL_SP macro ON";
#else
    const char *precisionMode = "DP mode / FULL_SP macro OFF";
#endif
    std::printf("Precision      : %s; sizeof(cfdScalar)=%zu; sizeof(HYPRE_Complex)=%zu\n",
                precisionMode, sizeof(cfdScalar), sizeof(HYPRE_Complex));
    std::printf("GPU linear-system assembly + GPU solve; U/V/W matrix reuse; GPU pressure RHS/flux/div/vel-correct; direct hypre matrix writes; GPU old-field LSQ gradients\n");
    std::printf("============================================================\n");
    std::printf("polyMeshDir : %s\n", par.polyMeshDir.c_str());
    std::printf("outPrefix   : %s\n", par.outPrefix.c_str());
    std::printf("------------------------------------------------------------\n");
    std::printf("Points         : %d\n", (int)mesh.P.size());
    std::printf("Faces          : %d\n", mesh.nFaces);
    std::printf("Internal faces : %d\n", mesh.nInternalFaces);
    std::printf("Cells          : %d\n", mesh.nCells);
    std::printf("BBox           : [%g, %g] x [%g, %g] x [%g, %g]\n", xmin,xmax,ymin,ymax,zmin,zmax);
    std::printf("Volume min/max : %.8e / %.8e\n", vmin, vmax);
    std::printf("maxNonOrthDeg  : %.6f\n", mesh.maxNonOrthDeg);
    const char *velSolverName =
        par.velSolver == 3 ? "fused multi-color GS defect smoother" :
        (par.velSolver == 2 ? "multi-color GS defect smoother" : "ParCSR BiCGSTAB + DiagScale");
    std::printf("Momentum solve : %s, mode=%s, sweeps=%d, omega=%.3g, symmetricGS=%d\n",
                velSolverName,
                (par.velSolver == 2 || par.velSolver == 3 || par.velCorrectionSolve) ? "defect-correction" : "field",
                par.velSweeps, par.velSmootherOmega, par.velGsSymmetric);
    std::printf("Pressure solve : ParCSR PCG + %s\n", par.p_use_amg ? "BoomerAMG" : "DiagScale");
    std::printf("rho            : %.8g\n", par.rho);
    std::printf("rho in pEqn    : ON (pressure coeff/flux use rho*rAU)\n");
    std::printf("Re             : %.8g\n", par.Re);
    std::printf("Umean          : %.8g\n", par.Umean);
    std::printf("pipeDiameter   : %.8g\n", par.pipeDiameter);
    std::printf("pipeLength     : %.8g\n", par.pipeLength);
    std::printf("wall/in/out    : %s / %s / %s\n", par.wallPatchName.c_str(), par.inletPatchName.c_str(), par.outletPatchName.c_str());
    std::printf("mu             : %.8e\n", mu);
    std::printf("mu source      : %s\n", par.muExplicit ? "direct -mu / case mu" : "legacy rho*Umean*pipeD/Re");
    std::printf("steady mode    : ON (no pseudo-time mass term)\n");
    std::printf("nsteps         : %d\n", par.nsteps);
    std::printf("nVelNonOrthCorr: %d\n", par.nVelNonOrthCorr);
    std::printf("nNonOrthCorr   : %d\n", par.nNonOrthCorr);
    std::printf("nPressureCorr  : %d\n", par.nPressureCorr);
    std::printf("geomMethod     : %s\n", par.geomMethod == 0 ? "legacy/v1" : "robust");
    std::printf("LSQ stencil    : %s\n", par.lsqStencilMode == 0 ? "compact" : "extended-accepted-as-compact");
    std::printf("LSQ weight     : 1/|d|^%.6g\n", par.lsqWeightPower);
    std::printf("momNonOrthScale: %.8g\n", par.momNonOrthScale);
    std::printf("Momentum convection: %s\n", par.momentumConvectionScheme == 2 ? "none/stokes" : (par.momentumConvectionScheme == 1 ? "upwind" : "central"));
    std::printf("pNonOrthScale  : %.8g\n", par.pNonOrthScale);
    std::printf("pMode          : %s\n", par.pMode == 0 ? "pcorr" : "absolute");
    std::printf("pSolveMode     : %s\n", par.pSolveMode == 1 ? "ofAbsolute" : "correction-compatible");
    std::printf("pGradScheme    : %s\n", par.pGradScheme == 1 ? "Gauss linear" : "LSQ");
    std::printf("pCoeffScale    : %.8g\n", par.pCoeffScale);
    std::printf("rcMode         : %s\n", par.rcMode == 0 ? "old-explicit" : "oflike-no-explicit");
    std::printf("rAUMode        : %s\n", par.rAUMode == 0 ? "raw V/aP_raw" : "relaxed V/aP_relaxed");
    std::printf("pDeltaMode     : %s\n",
        par.pDeltaMode == 0 ? "legacy/v1 signed 1/(n.d)" :
        par.pDeltaMode == 1 ? "OF-stabilised" :
        par.pDeltaMode == 2 ? "abs-projected 1/|n.d|" : "distance 1/|d|");
    std::printf("pDeltaMinCos   : %.8g\n", par.pDeltaMinCos);
    std::printf("initMode       : %s\n", par.initMode == 1 ? "potential/projection" : "uniform inlet-average");
    if(par.initMode == 1){
      std::printf("potential init : rAU=%.8g maxit=%d tol=%.3e relTol=%.3e write=%d\n",
                  par.potentialInitRAU, par.potentialInitMaxit,
                  par.potentialInitTol, par.potentialInitRelTol, par.potentialInitWrite);
    }
    std::printf("badCellAudit   : every=%d top=%d start=%d onGrowth=%d growthFactor=%.3g massFloor=%.3e writeCsv=%d\n",
        par.badCellAuditEvery, par.badCellAuditTop, par.badCellAuditStart,
        par.badCellAuditOnGrowth, par.badCellAuditGrowthFactor,
        par.badCellAuditMassFloor, par.badCellAuditWriteCsv);
    std::printf("Poisson PDE opts: gradient=%s laplacian=%s nNonOrthCorr=%d\n",
        par.poissonGradientScheme.c_str(), par.poissonLaplacianScheme.c_str(), par.poissonNonOrthCorr);
    std::printf("Scalar PDE opts : enable=%d solveMode=%s name=%s phiScheme=%s diffusion=%s gamma=%.8g relax=%.8g nNonOrthCorr=%d maxit=%d tol=%.3e relTol=%.3e\n",
        par.scalarEnable, par.scalarSolveMode.c_str(), par.scalarName.c_str(),
        par.scalarConvectionScheme.c_str(), par.scalarDiffusionScheme.c_str(),
        par.scalarGamma, par.scalarRelax, par.scalarNonOrthCorr,
        par.scalarMaxit, par.scalarTol, par.scalarRelTol);
    std::printf("SIMPLE geom flux: faceSkew=%d subtriContinuity=%d subtriFluxRecon=%d\n",
        par.simpleFaceSkewCorr, par.simpleSubtriContinuity, par.simpleSubtriFluxRecon);
    std::printf("SIMPLE MMS     : enable=%d initExact=%d pressureDirichlet=%d velScale=%.8g pScale=%.8g fixedVelBoundaryPEqn=%d sourceQuad8=%d stokesOnly=%d\n",
        par.simpleMmsEnable, par.simpleMmsInitExact, par.simpleMmsPressureDirichlet,
        par.simpleMmsVelocityScale, par.simpleMmsPressureScale,
        par.simpleMmsFixedVelocityBoundaryPressureEqn, par.simpleMmsSourceQuad8, par.simpleMmsStokesOnly);
    std::printf("SIMPLE MMS DIAG: stages=%d exactPBeforeMom=%d exactUBeforeP=%d exactPAfterUpdate=%d pEqnExactResidual=%d writePErrorVtk=%d\n",
        par.simpleMmsDiagStages,
        par.simpleMmsForceExactPBeforeMomentum,
        par.simpleMmsForceExactUBeforePressure,
        par.simpleMmsForceExactPAfterUpdate,
        par.simpleMmsDiagPressureExactResidual,
        par.simpleMmsDiagWritePErrorVtk);
    std::printf("Sine MMS       : enable=%d initExact=%d velScale=%.8g pScale=%.8g\n",
        par.sineMmsEnable, par.sineMmsInitExact,
        par.sineMmsVelocityScale, par.sineMmsPressureScale);
    if(par.profileSteps>0) std::printf("profileSteps   : %d\n", par.profileSteps);
    std::printf("velTol / velRelTol : %.3e / %.3e\n", par.velTol, par.velRelTol);
    std::printf("pTol   / pRelTol   : %.3e / %.3e\n", par.pTol, par.pRelTol);
    std::printf("pAmgRebuildEvery   : %d\n", par.pAmgRebuildEvery);
    std::printf("------------------------------------------------------------\n");
  }

  // v1.1b default: standalone hypre 3.1 CUDA build with internal device SpGEMM.
  // This is the robust path for large A100 meshes where vendor/cuSPARSE SpGEMM
  // can fail during BoomerAMG setup with insufficient resources.
  HYPRE_CALL(HYPRE_Initialize());
  HYPRE_CALL(HYPRE_DeviceInitialize());
  {
    HYPRE_Int spgemm_status = HYPRE_SetSpGemmUseVendor(0);
    if(spgemm_status){
      if(rank==0) std::printf("WARNING: HYPRE_SetSpGemmUseVendor(0) returned %d; continuing with default SpGEMM backend.\n", (int)spgemm_status);
    } else {
      if(rank==0) std::printf("HYPRE SpGEMM backend switch: forced internal SpGEMM via HYPRE_SetSpGemmUseVendor(0) [v1.1b default]\n");
    }
  }
  HYPRE_CALL(HYPRE_SetMemoryLocation(HYPRE_MEMORY_DEVICE));
  HYPRE_CALL(HYPRE_SetExecutionPolicy(HYPRE_EXEC_DEVICE));
  gpu_mem_mark("after_hypre_initialize", rank);


  // Repair OpenFOAM patch range metadata if the v2 reader populated names
  // but forgot patchStartFace/patchNFaces.  The working v1 reader filled
  // these from boundary startFace/nFaces; here we reconstruct from bPatch.
  if (mesh.patchStartFace.size() != mesh.patchNames.size()
      || mesh.patchNFaces.size() != mesh.patchNames.size())
  {
    const int nPatches = static_cast<int>(mesh.patchNames.size());

    if (nPatches <= 0) {
      throw std::runtime_error("No boundary patches found in mesh.patchNames");
    }
    if (static_cast<int>(mesh.bPatch.size()) != mesh.nFaces) {
      throw std::runtime_error(
        "Cannot reconstruct patch ranges: mesh.bPatch size="
        + std::to_string(mesh.bPatch.size())
        + " but mesh.nFaces=" + std::to_string(mesh.nFaces));
    }

    int minRaw = 1000000000;
    int maxRaw = -1000000000;
    for (int f = mesh.nInternalFaces; f < mesh.nFaces; ++f) {
      minRaw = std::min(minRaw, mesh.bPatch[f]);
      maxRaw = std::max(maxRaw, mesh.bPatch[f]);
    }

    const bool oneBased  = (minRaw >= 1 && maxRaw <= nPatches);
    const bool zeroBased = (minRaw >= 0 && maxRaw <  nPatches);

    if (!oneBased && !zeroBased) {
      throw std::runtime_error(
        "Cannot reconstruct patch ranges: unexpected bPatch raw range ["
        + std::to_string(minRaw) + ", " + std::to_string(maxRaw)
        + "] for nPatches=" + std::to_string(nPatches));
    }

    mesh.patchStartFace.assign(nPatches, -1);
    mesh.patchNFaces.assign(nPatches, 0);

    for (int f = mesh.nInternalFaces; f < mesh.nFaces; ++f) {
      const int raw = mesh.bPatch[f];
      const int pidx = oneBased ? (raw - 1) : raw;

      if (pidx < 0 || pidx >= nPatches) {
        throw std::runtime_error(
          "Patch index out of range while reconstructing patch ranges: raw="
          + std::to_string(raw) + " pidx=" + std::to_string(pidx));
      }

      if (mesh.patchNFaces[pidx] == 0) mesh.patchStartFace[pidx] = f;
      ++mesh.patchNFaces[pidx];
    }

    int totalBoundaryFaces = 0;
    for (int pidx = 0; pidx < nPatches; ++pidx) {
      totalBoundaryFaces += mesh.patchNFaces[pidx];
      if (mesh.patchStartFace[pidx] < mesh.nInternalFaces || mesh.patchNFaces[pidx] <= 0) {
        throw std::runtime_error(
          "Bad reconstructed patch range for "
          + mesh.patchNames[pidx]
          + ": startFace=" + std::to_string(mesh.patchStartFace[pidx])
          + " nFaces=" + std::to_string(mesh.patchNFaces[pidx]));
      }
    }

    if (totalBoundaryFaces != mesh.nFaces - mesh.nInternalFaces) {
      throw std::runtime_error(
        "Bad reconstructed patch ranges: totalBoundaryFaces="
        + std::to_string(totalBoundaryFaces)
        + " expected=" + std::to_string(mesh.nFaces - mesh.nInternalFaces));
    }

    std::fprintf(stderr,
      "Reconstructed patch ranges from bPatch: nPatches=%d, rawRange=[%d,%d], base=%s\n",
      nPatches, minRaw, maxRaw, oneBased ? "one-based" : "zero-based");
  }

  pipebc::PatchGeometryInput patchGeomIn;
  patchGeomIn.nInternalFaces = mesh.nInternalFaces;
  patchGeomIn.nFaces = mesh.nFaces;
  patchGeomIn.xf = &mesh.xf;
  patchGeomIn.nf = &mesh.nf;
  patchGeomIn.Sf = &mesh.Sf;
  patchGeomIn.Af = &mesh.Af;
  patchGeomIn.patchNames = &mesh.patchNames;
  patchGeomIn.patchStartFace = &mesh.patchStartFace;
  patchGeomIn.patchNFaces = &mesh.patchNFaces;


  // Ensure PatchGeometryInput points at the actual mesh-owned arrays.
  patchGeomIn.patchNames = &mesh.patchNames;
  patchGeomIn.patchStartFace = &mesh.patchStartFace;
  patchGeomIn.patchNFaces = &mesh.patchNFaces;

  const auto patchGeometryTable = pipebc::build_patch_geometry_table(patchGeomIn);

  pipebc::LegacyBCMeshView legacyBCMesh;
  legacyBCMesh.nFaces = mesh.nFaces;
  legacyBCMesh.nInternalFaces = mesh.nInternalFaces;
  legacyBCMesh.patchNames = &mesh.patchNames;
  legacyBCMesh.patchStartFace = &mesh.patchStartFace;
  legacyBCMesh.patchNFaces = &mesh.patchNFaces;
  legacyBCMesh.xf = &mesh.xf;
  legacyBCMesh.nf = &mesh.nf;

  std::vector<pipebc::VelocityPatchBCSpec> velocityPatchSpecs;
  std::vector<pipebc::PressurePatchBCSpec> pressurePatchSpecs;

  // simple_gpu rule:
  // all physical BCs must come from the runtime/generated BC config.
  // This avoids silent fallback/default BCs on newly added patches.
  if(par.bcConfigPath.empty() && !par.simpleMmsEnable){
    if(rank==0){
      std::fprintf(stderr,
          "simple_gpu requires explicit runtime BCs.\n"
          "Add velocity/pressure lines to the case file, set bcConfig, or set simpleMmsEnable 1.\n");
    }
    MPI_Abort(MPI_COMM_WORLD, 1);
  }

  if(par.simpleMmsEnable){
    const double mmsRho = par.rho;
    const double mmsMu  = mu;
    const double mmsU0  = par.simpleMmsVelocityScale;
    const double mmsP0  = par.simpleMmsPressureScale;
    for(const auto& patchName : mesh.patchNames){
      velocityPatchSpecs.push_back(pipebc::make_fixed_vector_function_bc(
          patchName,
          [mmsRho, mmsMu, mmsU0, mmsP0](const std::array<double,3>& x, double){
            const auto q = simple_mms_exact_value(x[0], x[1], x[2], mmsRho, mmsMu, mmsU0, mmsP0);
            return std::array<double,3>{{q.u, q.v, q.w}};
          }));
      if(par.simpleMmsPressureDirichlet){
        pressurePatchSpecs.push_back(pipebc::make_pressure_fixed_value_function_bc(
            patchName,
            [mmsRho, mmsMu, mmsU0, mmsP0](const std::array<double,3>& x, double){
              return simple_mms_exact_value(x[0], x[1], x[2], mmsRho, mmsMu, mmsU0, mmsP0).p;
            }));
      } else {
        pressurePatchSpecs.push_back(pipebc::make_pressure_zero_gradient_bc(patchName));
      }
    }
    if(rank==0){
      std::printf("SIMPLE MMS BC mode: exact velocity on all patches; pressureDirichlet=%d\n",
                  par.simpleMmsPressureDirichlet);
    }
  } else if(!par.bcConfigPath.empty()){
    auto runtimeBC = pipebc::load_runtime_bc_config(par.bcConfigPath);
    pipebc::validate_runtime_bc_config_against_patches(runtimeBC, mesh.patchNames);
    velocityPatchSpecs = std::move(runtimeBC.velocityPatchSpecs);
    pressurePatchSpecs = std::move(runtimeBC.pressurePatchSpecs);
  } else {
    velocityPatchSpecs.push_back(pipebc::make_wall_noslip_bc(mesh.patchNames[wallPatch]));
    if(cylinderPatch >= 0){
      velocityPatchSpecs.push_back(pipebc::make_wall_noslip_bc(mesh.patchNames[cylinderPatch]));
    }
    // CYLINDER_BENCHMARK_PARABOLIC_INLET_PATCH
    // Benchmark 3D cylinder/channel inlet:
    //
    //   U = 16 Um y z (H-y)(H-z) / H^4
    //   V = W = 0
    //
    // Channel is x-streamwise, cross-section is y-z, H = 0.41.
    // We impose the scalar profile along the inward normal of the inlet patch.

    std::array<double,3> inletOutwardNormal{{0.0, 0.0, 0.0}};
    double inletAreaForNormal = 0.0;

    for(int f = mesh.nInternalFaces; f < mesh.nFaces; ++f){
      if(mesh.bPatch[f] - 1 == inletPatch){
        inletOutwardNormal[0] += mesh.nf[f][0] * mesh.Af[f];
        inletOutwardNormal[1] += mesh.nf[f][1] * mesh.Af[f];
        inletOutwardNormal[2] += mesh.nf[f][2] * mesh.Af[f];
        inletAreaForNormal += mesh.Af[f];
      }
    }

    const double inletNmag = std::sqrt(
        inletOutwardNormal[0]*inletOutwardNormal[0] +
        inletOutwardNormal[1]*inletOutwardNormal[1] +
        inletOutwardNormal[2]*inletOutwardNormal[2]);

    if(inletNmag < 1.0e-30 || inletAreaForNormal <= 0.0){
      if(rank == 0){
        std::fprintf(stderr, "Could not compute inlet patch normal for patch %s.\n",
                     mesh.patchNames[inletPatch].c_str());
      }
      MPI_Abort(MPI_COMM_WORLD, 1);
    }

    inletOutwardNormal[0] /= inletNmag;
    inletOutwardNormal[1] /= inletNmag;
    inletOutwardNormal[2] /= inletNmag;

    const std::array<double,3> inletDirection{{
        -inletOutwardNormal[0],
        -inletOutwardNormal[1],
        -inletOutwardNormal[2]}};

    const double benchmarkH  = 0.41;
    const double benchmarkUm = par.Umean;

    if(rank == 0){
      const double meanU = (4.0/9.0) * benchmarkUm;
      const double D = par.pipeDiameter;
      const double ReMean = par.rho * meanU * D / std::max(mu, 1.0e-300);

      std::printf("Cylinder benchmark inlet mode: ON\n");
      std::printf("  inlet patch          : %s\n", mesh.patchNames[inletPatch].c_str());
      std::printf("  inlet outward normal : [%.6e, %.6e, %.6e]\n",
                  inletOutwardNormal[0], inletOutwardNormal[1], inletOutwardNormal[2]);
      std::printf("  inlet direction      : [%.6e, %.6e, %.6e]\n",
                  inletDirection[0], inletDirection[1], inletDirection[2]);
      std::printf("  benchmark H          : %.12e\n", benchmarkH);
      std::printf("  benchmark Um         : %.12e\n", benchmarkUm);
      std::printf("  cross-section mean U : %.12e\n", meanU);
      std::printf("  Re(mean,D)           : %.12e\n", ReMean);
    }

    velocityPatchSpecs.push_back(pipebc::make_fixed_vector_function_bc(
        mesh.patchNames[inletPatch],
        [inletDirection, benchmarkH, benchmarkUm](const std::array<double,3>& x, double){
          const double y = std::max(0.0, std::min(benchmarkH, x[1]));
          const double z = std::max(0.0, std::min(benchmarkH, x[2]));

          const double H2 = benchmarkH * benchmarkH;
          const double H4 = H2 * H2;

          const double mag =
              16.0 * benchmarkUm * y * z * (benchmarkH - y) * (benchmarkH - z) / H4;

          return std::array<double,3>{{
              mag * inletDirection[0],
              mag * inletDirection[1],
              mag * inletDirection[2]}};
        }));
    velocityPatchSpecs.push_back(pipebc::make_zero_gradient_velocity_bc(mesh.patchNames[outletPatch]));

    pressurePatchSpecs.push_back(pipebc::make_pressure_zero_gradient_bc(mesh.patchNames[wallPatch]));
    if(cylinderPatch >= 0){
      pressurePatchSpecs.push_back(pipebc::make_pressure_zero_gradient_bc(mesh.patchNames[cylinderPatch]));
    }
    pressurePatchSpecs.push_back(pipebc::make_pressure_zero_gradient_bc(mesh.patchNames[inletPatch]));
    pressurePatchSpecs.push_back(pipebc::make_pressure_fixed_value_bc(mesh.patchNames[outletPatch], 0.0));
  }

  const auto dupVel = pipebc::duplicate_velocity_bc_patches(velocityPatchSpecs);
  if(!dupVel.empty()){
    if(rank==0){
      std::fprintf(stderr, "Duplicate velocity BC specs found:\n");
      for(const auto& name : dupVel) std::fprintf(stderr, "  %s\n", name.c_str());
    }
    MPI_Abort(MPI_COMM_WORLD, 1);
  }

  const auto dupP = pipebc::duplicate_pressure_bc_patches(pressurePatchSpecs);
  if(!dupP.empty()){
    if(rank==0){
      std::fprintf(stderr, "Duplicate pressure BC specs found:\n");
      for(const auto& name : dupP) std::fprintf(stderr, "  %s\n", name.c_str());
    }
    MPI_Abort(MPI_COMM_WORLD, 1);
  }


  if(rank==0){
    std::printf("------------------------------------------------------------\n");
    std::printf("All mesh patches detected in polyMesh:\n");
    for(const auto& pg : patchGeometryTable){
      print_patch_geometry_summary(pg);
    }

    std::printf("------------------------------------------------------------\n");
    std::printf("Runtime BC patch coverage:\n");
  }

  std::vector<std::string> missingVelocityBC;
  std::vector<std::string> missingPressureBC;

  for(const auto& patchName : mesh.patchNames){
    bool hasV = false;
    bool hasP = false;

    for(const auto& spec : velocityPatchSpecs){
      if(spec.patchName == patchName){
        hasV = true;
        break;
      }
    }

    for(const auto& spec : pressurePatchSpecs){
      if(spec.patchName == patchName){
        hasP = true;
        break;
      }
    }

    if(rank==0){
      std::printf("  %-32s velocity=%s pressure=%s\n",
                  patchName.c_str(),
                  hasV ? "yes" : "NO",
                  hasP ? "yes" : "NO");
    }

    if(!hasV) missingVelocityBC.push_back(patchName);
    if(!hasP) missingPressureBC.push_back(patchName);
  }

  if(!missingVelocityBC.empty() || !missingPressureBC.empty()){
    if(rank==0){
      std::fprintf(stderr, "\nERROR: incomplete runtime BC coverage.\n");

      if(!missingVelocityBC.empty()){
        std::fprintf(stderr, "Missing velocity BC for patches:\n");
        for(const auto& name : missingVelocityBC){
          std::fprintf(stderr, "  %s\n", name.c_str());
        }
      }

      if(!missingPressureBC.empty()){
        std::fprintf(stderr, "Missing pressure BC for patches:\n");
        for(const auto& name : missingPressureBC){
          std::fprintf(stderr, "  %s\n", name.c_str());
        }
      }

      std::fprintf(stderr,
          "\nEvery polyMesh boundary patch must have exactly one velocity BC and one pressure BC in simple_gpu.\n");
    }

    MPI_Abort(MPI_COMM_WORLD, 1);
  }

  const bool pressureReferenceNeeded = pipebc::pressure_reference_required(pressurePatchSpecs);

  const auto& wallPatchGeom   = pipebc::get_patch_geometry_or_throw(patchGeometryTable, mesh.patchNames[wallPatch]);
  const auto& inletPatchGeom  = pipebc::get_patch_geometry_or_throw(patchGeometryTable, mesh.patchNames[inletPatch]);
  const auto& outletPatchGeom = pipebc::get_patch_geometry_or_throw(patchGeometryTable, mesh.patchNames[outletPatch]);

  if(rank==0){
    print_patch_geometry_summary(wallPatchGeom);
    if(cylinderPatch >= 0){
      const auto& cylinderPatchGeom = pipebc::get_patch_geometry_or_throw(patchGeometryTable, mesh.patchNames[cylinderPatch]);
      print_patch_geometry_summary(cylinderPatchGeom);
    }
    print_patch_geometry_summary(inletPatchGeom);
    print_patch_geometry_summary(outletPatchGeom);
    std::printf("Pressure reference required: %s\n", pressureReferenceNeeded ? "yes" : "no");
    std::printf("BC config source         : %s\n", par.simpleMmsEnable ? "<SIMPLE MMS exact BCs>" : (par.bcConfigPath.empty() ? "<hardcoded-fallback>" : par.bcConfigPath.c_str()));
  }

  std::vector<std::string> bcPType(mesh.patchNames.size(),"Neumann");
  std::vector<std::string> bcUType(mesh.patchNames.size(),"Neumann");
  std::vector<std::string> bcVType(mesh.patchNames.size(),"Neumann");
  std::vector<std::string> bcWType(mesh.patchNames.size(),"Neumann");

  // Legacy runtime BC builder still fills host double arrays.
  // Immediately convert to cfdScalar below; device-side BC storage remains cfdScalar.
  std::vector<double> pFaceBC_legacyDouble(mesh.nFaces, 0.0);
  std::vector<double> uFaceBC_legacyDouble(mesh.nFaces, 0.0);
  std::vector<double> vFaceBC_legacyDouble(mesh.nFaces, 0.0);
  std::vector<double> wFaceBC_legacyDouble(mesh.nFaces, 0.0);

  double inletCx=0.0, inletCy=0.0;
  int inletFaceCount=0;
  for(int f=mesh.nInternalFaces; f<mesh.nFaces; ++f){
    if(mesh.bPatch[f]-1 == inletPatch){
      inletCx += mesh.xf[f][0];
      inletCy += mesh.xf[f][1];
      ++inletFaceCount;
    }
  }
  if(inletFaceCount <= 0){
    if(rank==0) std::fprintf(stderr,"Inlet patch has no boundary faces.\n");
    MPI_Abort(MPI_COMM_WORLD,1);
  }
  inletCx /= inletFaceCount;
  inletCy /= inletFaceCount;

  const double R = 0.5*par.pipeDiameter;
  const double Umax = 2.0*par.Umean;

  pipebc::apply_bc_specs_to_legacy_face_arrays(
      legacyBCMesh,
      patchGeometryTable,
      velocityPatchSpecs,
      pressurePatchSpecs,
      0.0,
      bcUType,
      bcVType,
      bcWType,
      bcPType,
      uFaceBC_legacyDouble,
      vFaceBC_legacyDouble,
      wFaceBC_legacyDouble,
      pFaceBC_legacyDouble);

  std::vector<cfdScalar> pFaceBC = to_cfd_scalar_vec(pFaceBC_legacyDouble);
  std::vector<cfdScalar> uFaceBC = to_cfd_scalar_vec(uFaceBC_legacyDouble);
  std::vector<cfdScalar> vFaceBC = to_cfd_scalar_vec(vFaceBC_legacyDouble);
  std::vector<cfdScalar> wFaceBC = to_cfd_scalar_vec(wFaceBC_legacyDouble);

  // SINE_MMS_BC_OVERRIDE:
  // Exact velocity is zero on all unit-cube boundaries, but evaluate it anyway.
  // Exact pressure has dp/dn=0 on all unit-cube boundaries, so keep pressure Neumann.
  if(par.sineMmsEnable){
    for(std::size_t ip=0; ip<mesh.patchNames.size(); ++ip){
      bcUType[ip] = "Dirichlet";
      bcVType[ip] = "Dirichlet";
      bcWType[ip] = "Dirichlet";
      bcPType[ip] = "Neumann";
    }

    for(int f=mesh.nInternalFaces; f<mesh.nFaces; ++f){
      double ue, ve, we, pe;
      sine_mms_exact_fields(mesh.xf[f][0], mesh.xf[f][1], mesh.xf[f][2],
                            par.sineMmsVelocityScale, par.sineMmsPressureScale,
                            ue, ve, we, pe);
      uFaceBC[f] = (cfdScalar)ue;
      vFaceBC[f] = (cfdScalar)ve;
      wFaceBC[f] = (cfdScalar)we;
      pFaceBC[f] = (cfdScalar)0.0;
    }

    if(rank == 0){
      std::printf("SINE_MMS_BC_OVERRIDE exact Dirichlet U and zeroGradient p on all boundary faces\n");
    }
  }

  std::vector<double>().swap(pFaceBC_legacyDouble);
  std::vector<double>().swap(uFaceBC_legacyDouble);
  std::vector<double>().swap(vFaceBC_legacyDouble);
  std::vector<double>().swap(wFaceBC_legacyDouble);

  {
    static bool wroteRuntimeVelocityBCPatch = false;
    if(!wroteRuntimeVelocityBCPatch){
      wroteRuntimeVelocityBCPatch = true;
      write_runtime_velocity_bc_patch_vtp(
          mesh,
          par.inletPatchName,
          bcUType, uFaceBC,
          bcVType, vFaceBC,
          bcWType, wFaceBC,
          par.rho,
          par.outPrefix + "_runtime_inlet_bc.vtp");
    }
  }


  const bool pressureEqnFixedVelocityBoundaryFlux =
      (par.simpleMmsEnable &&
       par.simpleMmsPressureDirichlet &&
       par.simpleMmsFixedVelocityBoundaryPressureEqn);

  const bool originalUsePressureAnchor = pressureReferenceNeeded || pressureEqnFixedVelocityBoundaryFlux || (par.sineMmsEnable != 0);
  const bool usePressureAnchor = false;

  if(rank==0){
    std::printf("Pressure anchor enabled : %s\n", usePressureAnchor ? "yes" : "no");
    std::printf("NOANCHOR_PREPENALTY_DIAG originalUsePressureAnchor=%s beta=%.12e\n",
        originalUsePressureAnchor ? "yes" : "no", par.simpleMmsPressurePenaltyCoeff);
    std::printf("Pressure-equation fixed-velocity boundary flux mode: %s\n",
        pressureEqnFixedVelocityBoundaryFlux ? "yes" : "no");
  }

  const bool needSimpleSubTris =
      (simple_env_or_param_int("ANABASIS_SIMPLE_SUBTRI_CONTINUITY", par.simpleSubtriContinuity) != 0) ||
      (simple_env_or_param_int("ANABASIS_SIMPLE_SUBTRI_FLUX_RECON", par.simpleSubtriFluxRecon) != 0);
  if(rank==0){
    std::printf("DeviceMesh subtriangle arrays: %s\n", needSimpleSubTris ? "allocated" : "skipped");
  }
  DeviceMesh dmesh; build_device_mesh(mesh, dmesh, needSimpleSubTris);
  DeviceBC dbcP = make_device_bc(mesh.nFaces, bcPType, pFaceBC);
  DeviceBC dbcU = make_device_bc(mesh.nFaces, bcUType, uFaceBC);
  DeviceBC dbcV = make_device_bc(mesh.nFaces, bcVType, vFaceBC);
  DeviceBC dbcW = make_device_bc(mesh.nFaces, bcWType, wFaceBC);
  gpu_mem_mark("after_device_mesh_and_bcs", rank);

  std::vector<cfdScalar> u(mesh.nCells,0.0), v(mesh.nCells,0.0), w(mesh.nCells,0.0), p(mesh.nCells,0.0);
  double initU = 0.0, initV = 0.0, initW = 0.0;
  int initCount = 0;
  for(int f=mesh.nInternalFaces; f<mesh.nFaces; ++f){
    if(mesh.bPatch[f]-1 == inletPatch){
      initU += static_cast<double>(uFaceBC[f]);
      initV += static_cast<double>(vFaceBC[f]);
      initW += static_cast<double>(wFaceBC[f]);
      ++initCount;
    }
  }
  if(initCount > 0){
    initU /= initCount;
    initV /= initCount;
    initW /= initCount;
  }
  for(int c=0; c<mesh.nCells; ++c){
    u[c] = initU;
    v[c] = initV;
    w[c] = initW;
  }
  std::vector<cfdScalar> uOld(mesh.nCells), vOld(mesh.nCells), wOld(mesh.nCells), pOld(mesh.nCells);
  std::vector<cfdScalar> uStar(mesh.nCells), vStar(mesh.nCells), wStar(mesh.nCells), pCorr(mesh.nCells,0.0);
  std::vector<cfdScalar> gradPcompX(mesh.nCells), gradPcompY(mesh.nCells), gradPcompZ(mesh.nCells);
  std::vector<cfdVector3> gradVec;
  std::vector<cfdScalar> phiStar, phi, divStar, divCorr(mesh.nCells, 0.0);
  int refCell=0;
  double totalAssemble=0.0, totalSetup=0.0, totalSolve=0.0;
  double pressureSetup=0.0, pressureSolve=0.0;

  GPUMomentumAssembler mom;
  init_momentum_system(mom, mesh, par);
  gpu_mem_mark("after_init_momentum_system", rank);
  GPULinearSystem pressureSys;
  init_pressure_system(pressureSys, mesh, dmesh, par, refCell, usePressureAnchor, pressureSetup);
  gpu_mem_mark("after_init_pressure_system_no_amg_setup", rank);
  GPUSimpleScratch ss;
  init_simple_scratch(ss, mesh);
  gpu_mem_mark("after_init_simple_scratch", rank);

  // SIMPLE_MMS_INIT_EXACT_FIELDS
  if(par.simpleMmsEnable && par.simpleMmsInitExact){
    simple_mms_fill_cell_fields(mesh, par.rho, mu,
        par.simpleMmsVelocityScale, par.simpleMmsPressureScale,
        u, v, w, p);
    copy_vec_to_device(u, ss.d_u);
    copy_vec_to_device(v, ss.d_v);
    copy_vec_to_device(w, ss.d_w);
    copy_vec_to_device(p, ss.d_p);
    if(rank==0) std::printf("SIMPLE MMS exact initial fields loaded at cell centres.\n");
  }
  // SINE_MMS_INIT_EXACT_FIELDS
  else if(par.sineMmsEnable && par.sineMmsInitExact){
    const int block = 256;
    kernel_sine_mms_init_fields<<<(mesh.nCells + block - 1)/block, block>>>(
        mesh.nCells, dmesh.d_ccx, dmesh.d_ccy, dmesh.d_ccz,
        par.sineMmsVelocityScale, par.sineMmsPressureScale,
        ss.d_u, ss.d_v, ss.d_w, ss.d_p);
    CUDA_CHECK_LAST();

    copy_device_to_vec(ss.d_u, u);
    copy_device_to_vec(ss.d_v, v);
    copy_device_to_vec(ss.d_w, w);
    copy_device_to_vec(ss.d_p, p);

    if(rank == 0) std::printf("SINE_MMS_INIT_EXACT_FIELDS done\n");
  } else {
    copy_vec_to_device(u, ss.d_u);
    copy_vec_to_device(v, ss.d_v);
    copy_vec_to_device(w, ss.d_w);
    copy_vec_to_device(p, ss.d_p);
  }
  DeviceGradientOperator gop;
  const bool needLsqGradientOperator = (par.pGradScheme != 1);
  if(needLsqGradientOperator){
    if(rank == 0) std::printf("LSQ gradient operator: built because pGradScheme=LSQ\n");
    build_lsq_gradient_operator(mesh, gop, par.lsqWeightPower);
  } else {
    if(rank == 0) std::printf("LSQ gradient operator: skipped because pGradScheme=Gauss\n");
  }

  gpu_mem_mark("after_gradient_operator_decision", rank);


  if(par.initMode == 1){
    const double tPot0 = MPI_Wtime();
    const int block = 256;
    const int gridCells = (mesh.nCells + block - 1) / block;
    const int gridFaces = (mesh.nFaces + block - 1) / block;

    // Potential/projection initialisation:
    //   1) start from zero cell velocity, but honour fixed velocity BCs in phiStar;
    //   2) solve L(potential) = -div(phiStar) with a constant rAU;
    //   3) set U = -rAU * grad(potential) and corrected phi.
    // This is intentionally not used as physical pressure; p remains zero.
    std::fill(u.begin(), u.end(), 0.0);
    std::fill(v.begin(), v.end(), 0.0);
    std::fill(w.begin(), w.end(), 0.0);
    std::fill(p.begin(), p.end(), 0.0);

    copy_vec_to_device(u, ss.d_u);
    copy_vec_to_device(v, ss.d_v);
    copy_vec_to_device(w, ss.d_w);
    copy_vec_to_device(p, ss.d_p);

    kernel_fill_scalar<<<gridCells, block>>>(mom.d_rAU, mesh.nCells, par.potentialInitRAU);
    kernel_zero_scalar<<<gridCells, block>>>(ss.d_gradx, mesh.nCells);
    kernel_zero_scalar<<<gridCells, block>>>(ss.d_grady, mesh.nCells);
    kernel_zero_scalar<<<gridCells, block>>>(ss.d_gradz, mesh.nCells);
    CUDA_CHECK_LAST();

    kernel_build_rhiechow_predicted_flux_stokes_3d<<<gridFaces, block>>>(
        mesh.nFaces, mesh.nInternalFaces,
        dmesh.d_owner, dmesh.d_neigh, dmesh.d_bPatch,
        dmesh.d_ccx, dmesh.d_ccy, dmesh.d_ccz,
        dmesh.d_xfx, dmesh.d_xfy, dmesh.d_xfz,
        dmesh.d_nfx, dmesh.d_nfy, dmesh.d_nfz,
        dmesh.d_Af,
        ss.d_u, ss.d_v, ss.d_w,
        ss.d_p,
        ss.d_gradx, ss.d_grady, ss.d_gradz,
        mom.d_rAU,
        dbcU.d_type, dbcU.d_faceValue,
        dbcV.d_type, dbcV.d_faceValue,
        dbcW.d_type, dbcW.d_faceValue,
        par.rho, 1, par.hbyaBcMode, ss.d_phiStar);
    CUDA_CHECK_LAST();

    continuity_residual_gpu(dmesh, ss.d_phiStar, ss.d_divStar);
    const double potMassBefore = maxabs_device(ss.d_divStar, mesh.nCells, ss.d_reduce, ss.reduceSize);

    double potSetup = 0.0;
    update_pressure_matrix_from_rAU(mesh, dmesh, pressureSys, dbcP, mom.d_rAU,
                                    par.rho, par.pCoeffScale, refCell, usePressureAnchor,
                                    pressureEqnFixedVelocityBoundaryFlux,
                                    par.simpleMmsPressurePenaltyCoeff,
                                    true, potSetup);

    kernel_zero_scalar<<<gridCells, block>>>(ss.d_pCorr, mesh.nCells);
    kernel_zero_scalar<<<gridCells, block>>>(ss.d_divNonOrth, mesh.nCells);
    kernel_zero_scalar<<<gridFaces, block>>>(ss.d_phiNonOrth, mesh.nFaces);
    CUDA_CHECK_LAST();

    kernel_build_pressure_rhs_from_divs<<<gridCells, block>>>(
        mesh.nCells, ss.d_divStar, ss.d_divNonOrth, 0.0,
        usePressureAnchor ? 1 : 0, refCell, pressureSys.d_rhs);
    CUDA_CHECK_LAST();
    if(par.simpleMmsEnable && par.simpleMmsPressurePenaltyCoeff != 0.0){
      kernel_add_simple_mms_pressure_penalty_rhs<<<gridCells, block>>>(
          mesh.nCells, dmesh.d_ccx, dmesh.d_ccy, dmesh.d_ccz,
          par.rho, mu, par.simpleMmsVelocityScale, par.simpleMmsPressureScale,
          par.simpleMmsStokesOnly, par.simpleMmsPressurePenaltyCoeff,
          pressureSys.d_rhs);
      CUDA_CHECK_LAST();
    }

    HYPRE_Int potIts = 0;
    double potRel = 0.0;
    double potSolve = 0.0;
    solve_pressure_gpu_device_rhs_device_x0(pressureSys, ss.d_pCorr, ss.d_pCorr,
        par.potentialInitRelTol, par.potentialInitTol,
        par.potentialInitMaxit, potIts, potRel, potSolve);

    compute_pressure_gradient_gpu(par, gop, dmesh, dbcP, ss.d_pCorr, ss.d_gradx, ss.d_grady, ss.d_gradz);

    kernel_correct_face_fluxes_simple_nonorth<<<gridFaces, block>>>(
        mesh.nFaces, mesh.nInternalFaces,
        dmesh.d_owner, dmesh.d_neigh, dmesh.d_bPatch,
        dmesh.d_ccx, dmesh.d_ccy, dmesh.d_ccz,
        dmesh.d_xfx, dmesh.d_xfy, dmesh.d_xfz,
        dmesh.d_nfx, dmesh.d_nfy, dmesh.d_nfz,
        dmesh.d_Af, mom.d_rAU, par.rho,
        dbcP.d_type, dbcP.d_faceValue,
        ss.d_phiStar, ss.d_pCorr, ss.d_phiNonOrth, 0.0,
        par.pCoeffScale, par.pFluxMode,
        pressureEqnFixedVelocityBoundaryFlux ? 1 : 0, ss.d_phi);
    CUDA_CHECK_LAST();

    kernel_correct_velocity_from_pcorr_grad<<<gridCells, block>>>(
        mesh.nCells, ss.d_u, ss.d_v, ss.d_w, mom.d_rAU,
        ss.d_gradx, ss.d_grady, ss.d_gradz,
        ss.d_u, ss.d_v, ss.d_w);
    CUDA_CHECK_LAST();

    continuity_residual_gpu(dmesh, ss.d_phi, ss.d_divCorr);
    const double potMassAfter = maxabs_device(ss.d_divCorr, mesh.nCells, ss.d_reduce, ss.reduceSize);

    copy_device_to_vec(ss.d_u, u);
    copy_device_to_vec(ss.d_v, v);
    copy_device_to_vec(ss.d_w, w);
    std::fill(p.begin(), p.end(), 0.0);

    if(rank == 0){
      std::printf("Potential init : mass before/after = %.3e / %.3e, pcgIts=%d, finalRel=%.3e, setup=%.3e s, solve=%.3e s, wall=%.3e s\n",
                  potMassBefore, potMassAfter, (int)potIts, potRel,
                  potSetup, potSolve, MPI_Wtime() - tPot0);
    }

    if(par.potentialInitWrite && par.write_vtu){
      std::vector<cfdVector3> Uvec(mesh.nCells);
      std::vector<cfdScalar> umag(mesh.nCells);
      std::vector<cfdScalar> divPot(mesh.nCells, 0.0);
      copy_device_to_vec(ss.d_divCorr, divPot);
      for(int c=0;c<mesh.nCells;++c){
        Uvec[c] = {u[c], v[c], w[c]};
        umag[c] = std::sqrt(u[c]*u[c] + v[c]*v[c] + w[c]*w[c]);
      }
      write_vtu_polyhedron_cell_data(par.outPrefix + "_potential_init.vtu", mesh,
          {"p","umag","cell_volume","divPotential"}, {p,umag,to_cfd_scalar_vec(mesh.vol),divPot}, "U", &Uvec);
    }
  }

  int stepConverged=0;
  double massRes=0.0, duRel=0.0, dvRel=0.0, dwRel=0.0, dpRel=0.0;
  HYPRE_Int lastItsU=0,lastItsV=0,lastItsW=0,lastItsP=0;
  double lastRelU=0.0,lastRelV=0.0,lastRelW=0.0,lastRelP=0.0;
  int corrUsedU=1,corrUsedV=1,corrUsedW=1;
  std::vector<cfdScalar> rAU;
  std::array<PhaseStats, PH_COUNT> prof{};
  int profStepsDone = 0;

  auto simple_mms_diag_stage = [&](const char* stage, int step,
                                   const cfdScalar *d_uCur,
                                   const cfdScalar *d_vCur,
                                   const cfdScalar *d_wCur,
                                   const cfdScalar *d_pCur)
  {
    if(rank != 0) return;
    if(!par.simpleMmsEnable || !par.simpleMmsDiagStages) return;

    std::vector<cfdScalar> uh(mesh.nCells), vh(mesh.nCells), wh(mesh.nCells), ph(mesh.nCells);
    copy_device_to_vec(d_uCur, uh);
    copy_device_to_vec(d_vCur, vh);
    copy_device_to_vec(d_wCur, wh);
    copy_device_to_vec(d_pCur, ph);

    double ua=0.0, ur=0.0, pa=0.0, pr=0.0, pm=0.0;
    simple_mms_report_error(mesh, uh, vh, wh, ph,
        par.rho, mu, par.simpleMmsVelocityScale, par.simpleMmsPressureScale,
        ua, ur, pa, pr, pm);

    std::printf("MMS_STAGE step=%04d stage=%s scope=all Urel=%.12e PrelMF=%.12e Pmean=%.12e\n",
                step, stage, ur, pr, pm);

    double volSumDiag = 0.0;
    for(int c=0; c<mesh.nCells; ++c) volSumDiag += mesh.vol[c];
    const double hDiag = std::cbrt(volSumDiag / std::max(1, mesh.nCells));

    for(int layer=1; layer<=3; ++layer){
      double uai=0.0, uri=0.0, pai=0.0, pri=0.0, pmi=0.0;
      int nUsed=0;
      simple_mms_accumulate_error_for_margin(
          mesh, uh, vh, wh, ph,
          par.rho, mu,
          par.simpleMmsVelocityScale,
          par.simpleMmsPressureScale,
          layer*hDiag,
          uai, uri, pai, pri, pmi, nUsed);

      std::printf("MMS_STAGE step=%04d stage=%s scope=layer%d nUsed=%d Urel=%.12e PrelMF=%.12e Pmean=%.12e\n",
                  step, stage, layer, nUsed, uri, pri, pmi);
    }
  };

  auto simple_mms_pressure_residual_for_vector_diag = [&](const char* tag, const char* stage, int step, const cfdScalar *d_x)
  {
    if(!par.simpleMmsEnable || !par.simpleMmsDiagPressureExactResidual) return;

    const int block = 256;
    HYPRE_Complex *Avals = matrix_values_ptr(pressureSys);

    kernel_mms_pressure_exact_residual_csr<<<(mesh.nCells + block - 1)/block, block>>>(
        mesh.nCells,
        pressureSys.pat.d_rowOffsets,
        pressureSys.pat.d_cols,
        Avals,
        d_x,
        pressureSys.d_rhs,
        ss.d_gradx);
    CUDA_CHECK_LAST();

    if(usePressureAnchor){
      kernel_mms_pressure_exact_residual_anchor_zero<<<1,1>>>(refCell, ss.d_gradx);
      CUDA_CHECK_LAST();
    }

    if(rank == 0){
      std::vector<cfdScalar> res(mesh.nCells);
      copy_device_to_vec(ss.d_gradx, res);

      double volSum = 0.0;
      for(int c = 0; c < mesh.nCells; ++c) volSum += mesh.vol[c];
      const double h = std::cbrt(volSum / std::max(1, mesh.nCells));

      auto norm_for_margin = [&](double margin, int &nUsed, double &l2, double &linf){
        double sum = 0.0;
        double wsum = 0.0;
        double mx = 0.0;
        nUsed = 0;

        for(int c = 0; c < mesh.nCells; ++c){
          const double x = mesh.cc[c][0];
          const double y = mesh.cc[c][1];
          const double z = mesh.cc[c][2];
          const double dmin = std::min(std::min(x, 1.0 - x), std::min(y, std::min(1.0 - y, std::min(z, 1.0 - z))));
          if(dmin + 1.0e-14 < margin) continue;

          const double r = (double)res[c];
          const double w = mesh.vol[c];
          sum += w * r * r;
          wsum += w;
          mx = std::max(mx, std::fabs(r));
          ++nUsed;
        }

        l2 = std::sqrt(sum / std::max(wsum, 1.0e-300));
        linf = mx;
      };

      int nAll = 0, n1 = 0, n2 = 0, n3 = 0;
      double l2All = 0.0, liAll = 0.0;
      double l21 = 0.0, li1 = 0.0;
      double l22 = 0.0, li2 = 0.0;
      double l23 = 0.0, li3 = 0.0;

      norm_for_margin(0.0, nAll, l2All, liAll);
      norm_for_margin(1.0*h, n1, l21, li1);
      norm_for_margin(2.0*h, n2, l22, li2);
      norm_for_margin(3.0*h, n3, l23, li3);

      std::printf("%s step=%04d stage=%s scope=all n=%d L2=%.12e Linf=%.12e\n",
                  tag, step, stage, nAll, l2All, liAll);
      std::printf("%s step=%04d stage=%s scope=layer1 n=%d L2=%.12e Linf=%.12e\n",
                  tag, step, stage, n1, l21, li1);
      std::printf("%s step=%04d stage=%s scope=layer2 n=%d L2=%.12e Linf=%.12e\n",
                  tag, step, stage, n2, l22, li2);
      std::printf("%s step=%04d stage=%s scope=layer3 n=%d L2=%.12e Linf=%.12e\n",
                  tag, step, stage, n3, l23, li3);
    }
  };

  auto simple_mms_pressure_exact_residual_diag = [&](const char* stage, int step)
  {
    if(!par.simpleMmsEnable || !par.simpleMmsDiagPressureExactResidual) return;

    const int block = 256;
    kernel_simple_mms_set_exact_p<<<(mesh.nCells + block - 1)/block, block>>>(
        mesh.nCells,
        dmesh.d_ccx, dmesh.d_ccy, dmesh.d_ccz,
        par.rho, mu,
        par.simpleMmsVelocityScale,
        par.simpleMmsPressureScale,
        par.simpleMmsStokesOnly,
        ss.d_pCorr);
    CUDA_CHECK_LAST();

    if(usePressureAnchor){
      cfdScalar pref = 0;
      CUDA_CALL(cudaMemcpy(&pref, ss.d_pCorr + refCell, sizeof(cfdScalar), cudaMemcpyDeviceToHost));
      kernel_simple_mms_subtract_ref_cell<<<(mesh.nCells + block - 1)/block, block>>>(
          mesh.nCells, ss.d_pCorr, pref);
      CUDA_CHECK_LAST();
    }

    HYPRE_Complex *Avals = matrix_values_ptr(pressureSys);

    kernel_mms_pressure_exact_residual_csr<<<(mesh.nCells + block - 1)/block, block>>>(
        mesh.nCells,
        pressureSys.pat.d_rowOffsets,
        pressureSys.pat.d_cols,
        Avals,
        ss.d_pCorr,
        pressureSys.d_rhs,
        ss.d_gradx);
    CUDA_CHECK_LAST();

    if(usePressureAnchor){
      kernel_mms_pressure_exact_residual_anchor_zero<<<1,1>>>(refCell, ss.d_gradx);
      CUDA_CHECK_LAST();
    }

    if(rank == 0){
      std::vector<cfdScalar> res(mesh.nCells);
      copy_device_to_vec(ss.d_gradx, res);

      double volSum = 0.0;
      for(int c = 0; c < mesh.nCells; ++c) volSum += mesh.vol[c];
      const double h = std::cbrt(volSum / std::max(1, mesh.nCells));

      auto norm_for_margin = [&](double margin, int &nUsed, double &l2, double &linf){
        double sum = 0.0;
        double wsum = 0.0;
        double mx = 0.0;
        nUsed = 0;

        for(int c = 0; c < mesh.nCells; ++c){
          const double x = mesh.cc[c][0];
          const double y = mesh.cc[c][1];
          const double z = mesh.cc[c][2];
          const double dmin = std::min(std::min(x, 1.0 - x), std::min(y, std::min(1.0 - y, std::min(z, 1.0 - z))));
          if(dmin + 1.0e-14 < margin) continue;

          const double r = (double)res[c];
          const double w = mesh.vol[c];
          sum += w * r * r;
          wsum += w;
          mx = std::max(mx, std::fabs(r));
          ++nUsed;
        }

        l2 = std::sqrt(sum / std::max(wsum, 1.0e-300));
        linf = mx;
      };

      int nAll = 0, n1 = 0, n2 = 0, n3 = 0;
      double l2All = 0.0, liAll = 0.0;
      double l21 = 0.0, li1 = 0.0;
      double l22 = 0.0, li2 = 0.0;
      double l23 = 0.0, li3 = 0.0;

      norm_for_margin(0.0, nAll, l2All, liAll);
      norm_for_margin(1.0*h, n1, l21, li1);
      norm_for_margin(2.0*h, n2, l22, li2);
      norm_for_margin(3.0*h, n3, l23, li3);

      std::printf("MMS_PEQN_EXACT_RES step=%04d stage=%s scope=all n=%d L2=%.12e Linf=%.12e\n",
                  step, stage, nAll, l2All, liAll);
      std::printf("MMS_PEQN_EXACT_RES step=%04d stage=%s scope=layer1 n=%d L2=%.12e Linf=%.12e\n",
                  step, stage, n1, l21, li1);
      std::printf("MMS_PEQN_EXACT_RES step=%04d stage=%s scope=layer2 n=%d L2=%.12e Linf=%.12e\n",
                  step, stage, n2, l22, li2);
      std::printf("MMS_PEQN_EXACT_RES step=%04d stage=%s scope=layer3 n=%d L2=%.12e Linf=%.12e\n",
                  step, stage, n3, l23, li3);
    }
  };

  auto simple_mms_pressure_delta_diag = [&](const char* stage, int step, const cfdScalar *d_pSolved)
  {
    if(!par.simpleMmsEnable || !par.simpleMmsDiagPressureExactResidual) return;

    const int block = 256;

    // Store exact pressure in ss.d_grady.
    kernel_simple_mms_set_exact_p<<<(mesh.nCells + block - 1)/block, block>>>(
        mesh.nCells,
        dmesh.d_ccx, dmesh.d_ccy, dmesh.d_ccz,
        par.rho, mu,
        par.simpleMmsVelocityScale,
        par.simpleMmsPressureScale,
        par.simpleMmsStokesOnly,
        ss.d_grady);
    CUDA_CHECK_LAST();

    if(usePressureAnchor){
      cfdScalar pref = 0;
      CUDA_CALL(cudaMemcpy(&pref, ss.d_grady + refCell, sizeof(cfdScalar), cudaMemcpyDeviceToHost));
      kernel_simple_mms_subtract_ref_cell<<<(mesh.nCells + block - 1)/block, block>>>(
          mesh.nCells, ss.d_grady, pref);
      CUDA_CHECK_LAST();
    }

    // delta = solved - exact, stored in ss.d_gradz.
    kernel_mms_pressure_delta<<<(mesh.nCells + block - 1)/block, block>>>(
        mesh.nCells, d_pSolved, ss.d_grady, ss.d_gradz);
    CUDA_CHECK_LAST();

    if(rank == 0){
      std::vector<cfdScalar> delta(mesh.nCells);
      copy_device_to_vec(ss.d_gradz, delta);

      double volSum = 0.0;
      for(int c = 0; c < mesh.nCells; ++c) volSum += mesh.vol[c];
      const double h = std::cbrt(volSum / std::max(1, mesh.nCells));

      auto norm_for_margin = [&](double margin, int &nUsed, double &mean, double &l2mf, double &linf){
        double sum = 0.0;
        double wsum = 0.0;
        mean = 0.0;
        nUsed = 0;

        for(int c = 0; c < mesh.nCells; ++c){
          const double x = mesh.cc[c][0];
          const double y = mesh.cc[c][1];
          const double z = mesh.cc[c][2];
          const double dmin = std::min(std::min(x, 1.0 - x), std::min(y, std::min(1.0 - y, std::min(z, 1.0 - z))));
          if(dmin + 1.0e-14 < margin) continue;

          const double w = mesh.vol[c];
          mean += w * (double)delta[c];
          wsum += w;
          ++nUsed;
        }

        mean /= std::max(wsum, 1.0e-300);

        linf = 0.0;
        for(int c = 0; c < mesh.nCells; ++c){
          const double x = mesh.cc[c][0];
          const double y = mesh.cc[c][1];
          const double z = mesh.cc[c][2];
          const double dmin = std::min(std::min(x, 1.0 - x), std::min(y, std::min(1.0 - y, std::min(z, 1.0 - z))));
          if(dmin + 1.0e-14 < margin) continue;

          const double r = (double)delta[c] - mean;
          const double w = mesh.vol[c];
          sum += w * r * r;
          linf = std::max(linf, std::fabs(r));
        }

        l2mf = std::sqrt(sum / std::max(wsum, 1.0e-300));
      };

      int nAll = 0, n1 = 0, n2 = 0, n3 = 0;
      double mAll = 0.0, l2All = 0.0, liAll = 0.0;
      double m1 = 0.0, l21 = 0.0, li1 = 0.0;
      double m2 = 0.0, l22 = 0.0, li2 = 0.0;
      double m3 = 0.0, l23 = 0.0, li3 = 0.0;

      norm_for_margin(0.0, nAll, mAll, l2All, liAll);
      norm_for_margin(1.0*h, n1, m1, l21, li1);
      norm_for_margin(2.0*h, n2, m2, l22, li2);
      norm_for_margin(3.0*h, n3, m3, l23, li3);

      std::printf("MMS_PDELTA step=%04d stage=%s scope=all n=%d mean=%.12e L2MF=%.12e LinfMF=%.12e\n",
                  step, stage, nAll, mAll, l2All, liAll);
      std::printf("MMS_PDELTA step=%04d stage=%s scope=layer1 n=%d mean=%.12e L2MF=%.12e LinfMF=%.12e\n",
                  step, stage, n1, m1, l21, li1);
      std::printf("MMS_PDELTA step=%04d stage=%s scope=layer2 n=%d mean=%.12e L2MF=%.12e LinfMF=%.12e\n",
                  step, stage, n2, m2, l22, li2);
      std::printf("MMS_PDELTA step=%04d stage=%s scope=layer3 n=%d mean=%.12e L2MF=%.12e LinfMF=%.12e\n",
                  step, stage, n3, m3, l23, li3);
    }

    // Now print A*delta residual using the generic vector residual diagnostic.
    simple_mms_pressure_residual_for_vector_diag("MMS_PDELTA_RES", stage, step, ss.d_gradz);
  };

  auto solve_scalar_component_with_nonorth = [&](
      const cfdScalar *d_qInitial,
      const cfdScalar *d_uConv,
      const cfdScalar *d_vConv,
      const cfdScalar *d_wConv,
      const cfdScalar *d_gradPcomp,
      const DeviceBC &dbcQ,
      int mmsComp,
      cfdScalar *d_qOut,
      HYPRE_Int &itsOut,
      double &relOut,
      int &corrUsedOut,
      bool rebuildMomentumMatrix,
      bool extractRAU,
      PhaseStats &gradStats,
      PhaseStats &asmStats,
      PhaseStats &solveStats)
  {
    const int nVelSolves = std::max(par.nVelNonOrthCorr, 0) + 1;
    corrUsedOut = 0;

    CUDA_CALL(cudaMemcpy(d_qOut, d_qInitial, mesh.nCells*sizeof(cfdScalar), cudaMemcpyDeviceToDevice));

    for(int it=1; it<=nVelSolves; ++it){
      corrUsedOut = it - 1;

      PhaseMark pm_grad = profile_begin();
      compute_pressure_gradient_gpu(par, gop, dmesh, dbcQ, d_qOut,
                               mom.d_gradQx, mom.d_gradQy, mom.d_gradQz);
      profile_record(gradStats, pm_grad);

      const bool doMatrixSetup = rebuildMomentumMatrix && (it == 1);
      PhaseMark pm_asm = profile_begin();
      if(doMatrixSetup){
        assemble_momentum_on_gpu_device_grad(
            dmesh, mesh, mom, par.rho, mu, 1.0,
            d_qOut, d_uConv, d_vConv, d_wConv,
            mom.d_gradQx, mom.d_gradQy, mom.d_gradQz, d_gradPcomp,
            dbcQ, dbcU, dbcV, dbcW,
            par.momNonOrthScale, par.momentumConvectionScheme);
        relax_momentum_system_on_gpu(mesh, mom, d_qOut, par.uRelax);
        if(extractRAU) extract_rAU_from_momentum_matrix(mesh, dmesh, mom, par, rAU);
      } else {
        assemble_momentum_rhs_only_on_gpu_device_grad(
            dmesh, mesh, mom, par.rho, mu,
            d_qOut, d_uConv, d_vConv, d_wConv,
            mom.d_gradQx, mom.d_gradQy, mom.d_gradQz, d_gradPcomp,
            dbcQ, dbcU, dbcV, dbcW,
            par.momNonOrthScale, par.uRelax);
      }
      if(par.simpleMmsEnable){
        add_simple_mms_source_to_momentum_rhs(par, dmesh, mesh, mom, par.rho, mu, mmsComp);
      }
      double dtAsm = MPI_Wtime()-pm_asm.t0;
      totalAssemble += dtAsm;
      profile_record(asmStats, pm_asm);

      PhaseMark pm_solve = profile_begin();
      if(par.velSolver == 2){
        solve_momentum_gpu_device_mcgs_defect_x0_xout(
            mom, par, d_qOut, d_qOut,
            itsOut, relOut, totalSetup, totalSolve, doMatrixSetup);
      } else if(par.velCorrectionSolve){
        solve_momentum_gpu_device_defect_x0_xout(
            mom, par, d_qOut, d_qOut,
            itsOut, relOut, totalSetup, totalSolve, doMatrixSetup);
      } else {
        solve_momentum_gpu_device_x0_xout(
            mom, par, d_qOut, d_qOut,
            itsOut, relOut, totalSetup, totalSolve, doMatrixSetup);
      }
      profile_record(solveStats, pm_solve);
    }
  };

  auto assemble_scalar_component_rhs_for_fused = [&](
      const cfdScalar *d_qInitial,
      const cfdScalar *d_uConv,
      const cfdScalar *d_vConv,
      const cfdScalar *d_wConv,
      const cfdScalar *d_gradPcomp,
      const DeviceBC &dbcQ,
      HYPRE_Complex *d_rhsSave,
      int mmsComp,
      bool rebuildMomentumMatrix,
      bool extractRAU,
      PhaseStats &gradStats,
      PhaseStats &asmStats)
  {
    if(std::max(par.nVelNonOrthCorr, 0) != 0){
      if(rank==0){
        std::fprintf(stderr,
            "ERROR: velSolver=mcgs-fused currently supports nVelNonOrthCorr=0 only. Use -nVelNonOrthCorr 0.\n");
      }
      MPI_Abort(MPI_COMM_WORLD, 4);
    }

    PhaseMark pm_grad = profile_begin();
    compute_pressure_gradient_gpu(par, gop, dmesh, dbcQ, d_qInitial,
                             mom.d_gradQx, mom.d_gradQy, mom.d_gradQz);
    profile_record(gradStats, pm_grad);

    PhaseMark pm_asm = profile_begin();
    if(rebuildMomentumMatrix){
      assemble_momentum_on_gpu_device_grad(
          dmesh, mesh, mom, par.rho, mu, 1.0,
          d_qInitial, d_uConv, d_vConv, d_wConv,
          mom.d_gradQx, mom.d_gradQy, mom.d_gradQz, d_gradPcomp,
          dbcQ, dbcU, dbcV, dbcW,
          par.momNonOrthScale, par.momentumConvectionScheme);
      relax_momentum_system_on_gpu(mesh, mom, d_qInitial, par.uRelax);
      if(extractRAU) extract_rAU_from_momentum_matrix(mesh, dmesh, mom, par, rAU);
    } else {
      assemble_momentum_rhs_only_on_gpu_device_grad(
          dmesh, mesh, mom, par.rho, mu,
          d_qInitial, d_uConv, d_vConv, d_wConv,
          mom.d_gradQx, mom.d_gradQy, mom.d_gradQz, d_gradPcomp,
          dbcQ, dbcU, dbcV, dbcW,
          par.momNonOrthScale, par.uRelax);
    }
    // SINE_MMS_ADD_SOURCE_FUSED
    if(par.sineMmsEnable){
      const int block = 256;
      kernel_sine_mms_add_momentum_source<<<(mesh.nCells + block - 1)/block, block>>>(
          mesh.nCells, dmesh.d_ccx, dmesh.d_ccy, dmesh.d_ccz, dmesh.d_vol,
          par.rho, mu, par.sineMmsVelocityScale, par.sineMmsPressureScale,
          mmsComp, mom.lin.d_rhs);
      CUDA_CHECK_LAST();
    }
    if(par.simpleMmsEnable){
      add_simple_mms_source_to_momentum_rhs(par, dmesh, mesh, mom, par.rho, mu, mmsComp);
    }

    double dtAsm = MPI_Wtime()-pm_asm.t0;
    totalAssemble += dtAsm;
    profile_record(asmStats, pm_asm);

    CUDA_CALL(cudaMemcpy(d_rhsSave, mom.lin.d_rhs,
                         mesh.nCells*sizeof(HYPRE_Complex),
                         cudaMemcpyDeviceToDevice));
  };

  double runStart = MPI_Wtime();
  gpu_mem_mark("before_solve_loop_after_all_setup", rank);
  double prevMassResForBadCellAudit = -1.0;
  int maxSteps = (par.profileSteps>0 ? std::min(par.nsteps, par.profileSteps) : par.nsteps);
  for(int step=1; step<=maxSteps; ++step){
    double iterStart = MPI_Wtime();
    std::fill(pCorr.begin(), pCorr.end(), 0.0);

    // Preserve old SIMPLE fields on device. Momentum predictors overwrite ss.d_u/v/w,
    // and pressure is updated in-place later in the iteration. Keep convergence checks
    // device-resident rather than copying full fields back to host every step.
    CUDA_CALL(cudaMemcpy(ss.d_uOld, ss.d_u, mesh.nCells*sizeof(cfdScalar), cudaMemcpyDeviceToDevice));
    CUDA_CALL(cudaMemcpy(ss.d_vOld, ss.d_v, mesh.nCells*sizeof(cfdScalar), cudaMemcpyDeviceToDevice));
    CUDA_CALL(cudaMemcpy(ss.d_wOld, ss.d_w, mesh.nCells*sizeof(cfdScalar), cudaMemcpyDeviceToDevice));
    CUDA_CALL(cudaMemcpy(ss.d_pOld, ss.d_p, mesh.nCells*sizeof(cfdScalar), cudaMemcpyDeviceToDevice));

    if(par.simpleMmsEnable && par.simpleMmsForceExactPBeforeMomentum){
      {
        const int block = 256;
        kernel_simple_mms_set_exact_p<<<(mesh.nCells + block - 1)/block, block>>>(
            mesh.nCells, dmesh.d_ccx, dmesh.d_ccy, dmesh.d_ccz,
            par.rho, mu, par.simpleMmsVelocityScale, par.simpleMmsPressureScale,
            par.simpleMmsStokesOnly,
            ss.d_p);
        CUDA_CHECK_LAST();

        if(usePressureAnchor){
          cfdScalar pref = 0;
          CUDA_CALL(cudaMemcpy(&pref, ss.d_p + refCell, sizeof(cfdScalar), cudaMemcpyDeviceToHost));
          kernel_simple_mms_subtract_ref_cell<<<(mesh.nCells + block - 1)/block, block>>>(
              mesh.nCells, ss.d_p, pref);
          CUDA_CHECK_LAST();
        }
      }
      simple_mms_diag_stage("forced_exact_p_before_momentum", step, ss.d_u, ss.d_v, ss.d_w, ss.d_p);
    }

    PhaseMark pm_pgrad = profile_begin();
    // ss.d_p already contains pOld from the previous pressure update / initial upload.
    // Compute the selected pressure gradient once and reuse it for both momentum RHS and HbyA.
    compute_pressure_gradient_gpu(par, gop, dmesh, dbcP, ss.d_p,
                                  ss.d_gradx, ss.d_grady, ss.d_gradz);
    profile_record(prof[PH_PGRAD], pm_pgrad);

    // The scalar momentum matrix is the same for Ux, Uy, Uz for this segregated equation.
    // Use device-resident old fields and convection fields; avoid old host<->device copies.
    if(par.velSolver == 3){
      ensure_fused_mcgs_scratch(mesh.nCells);
      if(step == 1) gpu_mem_mark("after_fused_mcgs_scratch_step1", rank);
      corrUsedU = corrUsedV = corrUsedW = 0;

      assemble_scalar_component_rhs_for_fused(
          ss.d_uOld, ss.d_uOld, ss.d_vOld, ss.d_wOld,
          ss.d_gradx, dbcU, g_fused_mcgs.d_rhsU,
          0, true, true, prof[PH_UGRAD], prof[PH_UASM]);
      if(step == 1) gpu_mem_mark("after_U_momentum_matrix_and_rhs_step1", rank);
      assemble_scalar_component_rhs_for_fused(
          ss.d_vOld, ss.d_uOld, ss.d_vOld, ss.d_wOld,
          ss.d_grady, dbcV, g_fused_mcgs.d_rhsV,
          1, false, false, prof[PH_VGRAD], prof[PH_VASM]);
      assemble_scalar_component_rhs_for_fused(
          ss.d_wOld, ss.d_uOld, ss.d_vOld, ss.d_wOld,
          ss.d_gradz, dbcW, g_fused_mcgs.d_rhsW,
          2, false, false, prof[PH_WGRAD], prof[PH_WASM]);
      if(step == 1) gpu_mem_mark("after_UVW_momentum_rhs_step1", rank);

      PhaseMark pm_solve = profile_begin();
      solve_momentum_gpu_device_mcgs_fused_defect_x0_xout(
          mom, par,
          ss.d_uOld, ss.d_vOld, ss.d_wOld,
          ss.d_u, ss.d_v, ss.d_w,
          lastItsU, lastItsV, lastItsW,
          lastRelU, lastRelV, lastRelW,
          totalSetup, totalSolve,
          true);
      profile_record(prof[PH_USOLVE], pm_solve);
    } else {
      solve_scalar_component_with_nonorth(ss.d_uOld, ss.d_uOld, ss.d_vOld, ss.d_wOld, ss.d_gradx, dbcU, 0, ss.d_u, lastItsU, lastRelU, corrUsedU, true,  true,  prof[PH_UGRAD], prof[PH_UASM], prof[PH_USOLVE]);
      solve_scalar_component_with_nonorth(ss.d_vOld, ss.d_uOld, ss.d_vOld, ss.d_wOld, ss.d_grady, dbcV, 1, ss.d_v, lastItsV, lastRelV, corrUsedV, false, false, prof[PH_VGRAD], prof[PH_VASM], prof[PH_VSOLVE]);
      solve_scalar_component_with_nonorth(ss.d_wOld, ss.d_uOld, ss.d_vOld, ss.d_wOld, ss.d_gradz, dbcW, 2, ss.d_w, lastItsW, lastRelW, corrUsedW, false, false, prof[PH_WGRAD], prof[PH_WASM], prof[PH_WSOLVE]);
    }

    PhaseMark pm_psetup = profile_begin();
    double ps0 = pressureSetup;
    const int rebuildEvery = std::max(par.pAmgRebuildEvery, 1);
    const bool doPressureSetup = (!pressureSys.is_setup) || (step == 1) || (((step - 1) % rebuildEvery) == 0);
    if(step == 1 || doPressureSetup) gpu_mem_mark("before_pressure_matrix_update_or_amg_setup_step1", rank);
    update_pressure_matrix_from_rAU(mesh, dmesh, pressureSys, dbcP, mom.d_rAU, par.rho, par.pCoeffScale, refCell, usePressureAnchor, pressureEqnFixedVelocityBoundaryFlux, par.simpleMmsPressurePenaltyCoeff, doPressureSetup, pressureSetup);
    if(step == 1 || doPressureSetup) gpu_mem_mark("after_pressure_matrix_update_or_amg_setup_step1", rank);
    profile_record(prof[PH_PSETUP], pm_psetup);

    // ss.d_u/v/w already contain the device-resident momentum predictor fields.
    // ss.d_p still contains pOld.
    if(par.pMode == 1 && par.pSolveMode == 1){
      // Reuse pOld gradient computed at the start of the SIMPLE iteration.
    } else {
      compute_pressure_gradient_gpu(par, gop, dmesh, dbcP, ss.d_p, ss.d_gradx, ss.d_grady, ss.d_gradz);
    }
    if(par.pMode == 1){
      const int block = 256;
      // Reconstruct OpenFOAM-like HbyA from predictor Ustar = HbyA - rAU*grad(pOld).
      kernel_add_rau_grad_to_velocity<<<(mesh.nCells + block - 1)/block, block>>>(
          mesh.nCells, ss.d_u, ss.d_v, ss.d_w, mom.d_rAU, ss.d_gradx, ss.d_grady, ss.d_gradz);
      CUDA_CHECK_LAST();
    }
    simple_mms_diag_stage("after_momentum_before_pressure", step, ss.d_u, ss.d_v, ss.d_w, ss.d_p);

    if(par.simpleMmsEnable && par.simpleMmsForceExactUBeforePressure){
      {
        const int block = 256;
        kernel_simple_mms_set_exact_u<<<(mesh.nCells + block - 1)/block, block>>>(
            mesh.nCells, dmesh.d_ccx, dmesh.d_ccy, dmesh.d_ccz,
            par.rho, mu, par.simpleMmsVelocityScale, par.simpleMmsPressureScale,
            par.simpleMmsStokesOnly,
            ss.d_u, ss.d_v, ss.d_w);
        CUDA_CHECK_LAST();
      }
      simple_mms_diag_stage("forced_exact_u_before_pressure", step, ss.d_u, ss.d_v, ss.d_w, ss.d_p);
    }

    PhaseMark pm_phi = profile_begin();
    {
      const int block = 256;
      kernel_build_rhiechow_predicted_flux_stokes_3d<<<(mesh.nFaces + block - 1)/block, block>>>(
          mesh.nFaces, mesh.nInternalFaces,
          dmesh.d_owner, dmesh.d_neigh, dmesh.d_bPatch,
          dmesh.d_ccx, dmesh.d_ccy, dmesh.d_ccz,
          dmesh.d_xfx, dmesh.d_xfy, dmesh.d_xfz,
          dmesh.d_nfx, dmesh.d_nfy, dmesh.d_nfz,
          dmesh.d_Af,
          ss.d_u, ss.d_v, ss.d_w,
          ss.d_p,
          ss.d_gradx, ss.d_grady, ss.d_gradz,
          mom.d_rAU,
          dbcU.d_type, dbcU.d_faceValue,
          dbcV.d_type, dbcV.d_faceValue,
          dbcW.d_type, dbcW.d_faceValue,
          par.rho, par.rcMode, par.hbyaBcMode, ss.d_phiStar);
      CUDA_CHECK_LAST();
      apply_simple_phiStar_geometry_correction(par, dmesh, mesh, mom, gop, dbcU, dbcV, dbcW,
                                               ss.d_u, ss.d_v, ss.d_w, ss.d_phiStar);
      CUDA_CALL(cudaMemcpy(ss.d_phi, ss.d_phiStar, mesh.nFaces*sizeof(cfdScalar), cudaMemcpyDeviceToDevice));
    }
    profile_record(prof[PH_PREDICTOR_PHI], pm_phi);
    PhaseMark pm_cont0 = profile_begin();
    continuity_residual_gpu(dmesh, ss.d_phi, ss.d_divCorr);
    CUDA_CALL(cudaMemcpy(ss.d_divStar, ss.d_divCorr, mesh.nCells*sizeof(cfdScalar), cudaMemcpyDeviceToDevice));
    profile_record(prof[PH_CONT_PRE_P], pm_cont0);
    const int nPressureSolves = std::max(par.nNonOrthCorr, 0) + 1;
    const int nExtraPressureCorr = std::max(par.nPressureCorr, 0);
    const int totalPlannedPressureSolves = nPressureSolves + nExtraPressureCorr;
    int pressureSolveCounter = 0;
    int pcgTotalIts = 0;
    std::vector<int> pcgPassIts(nPressureSolves, 0);
    { const int block = 256;
      if(par.pMode == 1){
        CUDA_CALL(cudaMemcpy(ss.d_pCorr, ss.d_p, mesh.nCells*sizeof(cfdScalar), cudaMemcpyDeviceToDevice));
      } else {
        kernel_zero_scalar<<<(mesh.nCells + block - 1)/block, block>>>(ss.d_pCorr, mesh.nCells);
        CUDA_CHECK_LAST();
      }
    }

    for(int pcorr=1; pcorr<=nPressureSolves; ++pcorr){
      PhaseMark pm_pcorrg_iter = profile_begin();
      compute_pressure_gradient_gpu(par, gop, dmesh, dbcP, ss.d_pCorr, ss.d_gradx, ss.d_grady, ss.d_gradz);
      profile_record(prof[PH_PCORR_GRAD], pm_pcorrg_iter);

      const int block = 256;
      kernel_zero_scalar<<<(mesh.nCells + block - 1)/block, block>>>(ss.d_divNonOrth, mesh.nCells);
      CUDA_CHECK_LAST();
      kernel_pressure_nonorth_flux_and_divergence<<<(mesh.nFaces + block - 1)/block, block>>>(
          mesh.nFaces, mesh.nInternalFaces,
          dmesh.d_owner, dmesh.d_neigh, dmesh.d_bPatch,
          dmesh.d_ccx, dmesh.d_ccy, dmesh.d_ccz,
          dmesh.d_xfx, dmesh.d_xfy, dmesh.d_xfz,
          dmesh.d_nfx, dmesh.d_nfy, dmesh.d_nfz,
          dmesh.d_sfx, dmesh.d_sfy, dmesh.d_sfz,
          dmesh.d_Af, mom.d_rAU, par.rho,
          dbcP.d_type, pressureEqnFixedVelocityBoundaryFlux ? 1 : 0,
          ss.d_gradx, ss.d_grady, ss.d_gradz,
          ss.d_phiNonOrth, ss.d_divNonOrth);
      CUDA_CHECK_LAST();
      kernel_build_pressure_rhs_from_divs<<<(mesh.nCells + block - 1)/block, block>>>(
          mesh.nCells, ss.d_divStar, ss.d_divNonOrth, par.pNonOrthScale, usePressureAnchor ? 1 : 0, refCell, pressureSys.d_rhs);
      CUDA_CHECK_LAST();
      if(par.simpleMmsEnable && par.simpleMmsPressurePenaltyCoeff != 0.0){
        kernel_add_simple_mms_pressure_penalty_rhs<<<(mesh.nCells + block - 1)/block, block>>>(
            mesh.nCells, dmesh.d_ccx, dmesh.d_ccy, dmesh.d_ccz,
            par.rho, mu, par.simpleMmsVelocityScale, par.simpleMmsPressureScale,
            par.simpleMmsStokesOnly, par.simpleMmsPressurePenaltyCoeff,
            pressureSys.d_rhs);
        CUDA_CHECK_LAST();
      }

      simple_mms_pressure_exact_residual_diag("after_pressure_rhs_before_solve", step);

      ++pressureSolveCounter;
      const bool isFinalPressureSolve = (pressureSolveCounter == totalPlannedPressureSolves);
      const double curPRelTol = isFinalPressureSolve ? 0.0 : par.pRelTol;
      PhaseMark pm_psolve = profile_begin();
      solve_pressure_gpu_device_rhs_device_x0(pressureSys, ss.d_pCorr, ss.d_pCorr, curPRelTol, par.pTol, par.pMaxit, lastItsP, lastRelP, pressureSolve);
      if(step == 1 && pcorr == 1) gpu_mem_mark("after_first_pressure_solve_step1", rank);
      profile_record(prof[PH_PSOLVE_LOOP], pm_psolve);
      pcgPassIts[pcorr-1] = (int)lastItsP;
      pcgTotalIts += (int)lastItsP;

      if(pcorr == nPressureSolves){
        PhaseMark pm_flux = profile_begin();
        compute_pressure_gradient_gpu(par, gop, dmesh, dbcP, ss.d_pCorr, ss.d_gradx, ss.d_grady, ss.d_gradz);
        kernel_zero_scalar<<<(mesh.nCells + block - 1)/block, block>>>(ss.d_divNonOrth, mesh.nCells);
        CUDA_CHECK_LAST();
        kernel_pressure_nonorth_flux_and_divergence<<<(mesh.nFaces + block - 1)/block, block>>>(
            mesh.nFaces, mesh.nInternalFaces,
            dmesh.d_owner, dmesh.d_neigh, dmesh.d_bPatch,
            dmesh.d_ccx, dmesh.d_ccy, dmesh.d_ccz,
            dmesh.d_xfx, dmesh.d_xfy, dmesh.d_xfz,
            dmesh.d_nfx, dmesh.d_nfy, dmesh.d_nfz,
            dmesh.d_sfx, dmesh.d_sfy, dmesh.d_sfz,
            dmesh.d_Af, mom.d_rAU, par.rho,
            dbcP.d_type,
            pressureEqnFixedVelocityBoundaryFlux ? 1 : 0,
          ss.d_gradx, ss.d_grady, ss.d_gradz,
            ss.d_phiNonOrth, ss.d_divNonOrth);
        CUDA_CHECK_LAST();
        kernel_correct_face_fluxes_simple_nonorth<<<(mesh.nFaces + block - 1)/block, block>>>(
            mesh.nFaces, mesh.nInternalFaces,
            dmesh.d_owner, dmesh.d_neigh, dmesh.d_bPatch,
            dmesh.d_ccx, dmesh.d_ccy, dmesh.d_ccz,
            dmesh.d_xfx, dmesh.d_xfy, dmesh.d_xfz,
            dmesh.d_nfx, dmesh.d_nfy, dmesh.d_nfz,
            dmesh.d_Af, mom.d_rAU, par.rho,
            dbcP.d_type, dbcP.d_faceValue,
            ss.d_phiStar, ss.d_pCorr, ss.d_phiNonOrth, par.pNonOrthScale, par.pCoeffScale, par.pFluxMode,
            pressureEqnFixedVelocityBoundaryFlux ? 1 : 0, ss.d_phi);
        CUDA_CHECK_LAST();
        if(pressureEqnFixedVelocityBoundaryFlux){
          kernel_mms_enforce_fixed_velocity_boundary_flux<<<(mesh.nFaces - mesh.nInternalFaces + block - 1)/block, block>>>(mesh.nFaces, mesh.nInternalFaces, ss.d_phiStar, ss.d_phi);
          CUDA_CHECK_LAST();
        }
        apply_simple_final_phi_geometry_correction_if_needed(par, dmesh, mesh, mom, gop, dbcU, dbcV, dbcW,
                                                             ss.d_u, ss.d_v, ss.d_w, ss.d_phi);
        profile_record(prof[PH_FLUX_CORR_LOOP], pm_flux);
        PhaseMark pm_contp = profile_begin();
        continuity_residual_gpu(dmesh, ss.d_phi, ss.d_divCorr);
        profile_record(prof[PH_CONT_IN_P_LOOP], pm_contp);
      }
    }

    for(int pcorr=1; pcorr<=std::max(par.nPressureCorr, 0); ++pcorr){
      const int block = 256;
      kernel_build_pressure_rhs_minus_div<<<(mesh.nCells + block - 1)/block, block>>>(
          mesh.nCells, ss.d_divCorr, usePressureAnchor ? 1 : 0, refCell, pressureSys.d_rhs);
      CUDA_CHECK_LAST();
      if(par.simpleMmsEnable && par.simpleMmsPressurePenaltyCoeff != 0.0){
        kernel_add_simple_mms_pressure_penalty_rhs<<<(mesh.nCells + block - 1)/block, block>>>(
            mesh.nCells, dmesh.d_ccx, dmesh.d_ccy, dmesh.d_ccz,
            par.rho, mu, par.simpleMmsVelocityScale, par.simpleMmsPressureScale,
            par.simpleMmsStokesOnly, par.simpleMmsPressurePenaltyCoeff,
            pressureSys.d_rhs);
        CUDA_CHECK_LAST();
      }
      kernel_zero_scalar<<<(mesh.nCells + block - 1)/block, block>>>(ss.d_pCorrDelta, mesh.nCells);
      CUDA_CHECK_LAST();
      ++pressureSolveCounter;
      const bool isFinalPressureSolve = (pressureSolveCounter == totalPlannedPressureSolves);
      const double curPRelTol = isFinalPressureSolve ? 0.0 : par.pRelTol;
      PhaseMark pm_psolve = profile_begin();
      solve_pressure_gpu_device_rhs_device_x0(pressureSys, ss.d_pCorrDelta, ss.d_pCorrDelta, curPRelTol, par.pTol, par.pMaxit, lastItsP, lastRelP, pressureSolve);
      profile_record(prof[PH_PSOLVE_LOOP], pm_psolve);
      pcgTotalIts += (int)lastItsP;
      kernel_add_scaled_inplace<<<(mesh.nCells + block - 1)/block, block>>>(mesh.nCells, ss.d_pCorr, ss.d_pCorrDelta, 1.0);
      CUDA_CHECK_LAST();
      PhaseMark pm_flux = profile_begin();
      compute_pressure_gradient_gpu(par, gop, dmesh, dbcP, ss.d_pCorr, ss.d_gradx, ss.d_grady, ss.d_gradz);
      kernel_zero_scalar<<<(mesh.nCells + block - 1)/block, block>>>(ss.d_divNonOrth, mesh.nCells);
      CUDA_CHECK_LAST();
      kernel_pressure_nonorth_flux_and_divergence<<<(mesh.nFaces + block - 1)/block, block>>>(
          mesh.nFaces, mesh.nInternalFaces,
          dmesh.d_owner, dmesh.d_neigh, dmesh.d_bPatch,
          dmesh.d_ccx, dmesh.d_ccy, dmesh.d_ccz,
          dmesh.d_xfx, dmesh.d_xfy, dmesh.d_xfz,
          dmesh.d_nfx, dmesh.d_nfy, dmesh.d_nfz,
          dmesh.d_sfx, dmesh.d_sfy, dmesh.d_sfz,
          dmesh.d_Af, mom.d_rAU, par.rho,
          dbcP.d_type, pressureEqnFixedVelocityBoundaryFlux ? 1 : 0,
          ss.d_gradx, ss.d_grady, ss.d_gradz,
          ss.d_phiNonOrth, ss.d_divNonOrth);
      CUDA_CHECK_LAST();
      kernel_correct_face_fluxes_simple_nonorth<<<(mesh.nFaces + block - 1)/block, block>>>(
          mesh.nFaces, mesh.nInternalFaces,
          dmesh.d_owner, dmesh.d_neigh, dmesh.d_bPatch,
          dmesh.d_ccx, dmesh.d_ccy, dmesh.d_ccz,
          dmesh.d_xfx, dmesh.d_xfy, dmesh.d_xfz,
          dmesh.d_nfx, dmesh.d_nfy, dmesh.d_nfz,
          dmesh.d_Af, mom.d_rAU, par.rho,
          dbcP.d_type, dbcP.d_faceValue,
          ss.d_phiStar, ss.d_pCorr, ss.d_phiNonOrth, par.pNonOrthScale, par.pCoeffScale, par.pFluxMode,
          pressureEqnFixedVelocityBoundaryFlux ? 1 : 0, ss.d_phi);
      CUDA_CHECK_LAST();
      if(pressureEqnFixedVelocityBoundaryFlux){
        kernel_mms_enforce_fixed_velocity_boundary_flux<<<(mesh.nFaces - mesh.nInternalFaces + block - 1)/block, block>>>(mesh.nFaces, mesh.nInternalFaces, ss.d_phiStar, ss.d_phi);
        CUDA_CHECK_LAST();
      }
      apply_simple_final_phi_geometry_correction_if_needed(par, dmesh, mesh, mom, gop, dbcU, dbcV, dbcW,
                                                           ss.d_u, ss.d_v, ss.d_w, ss.d_phi);
      profile_record(prof[PH_FLUX_CORR_LOOP], pm_flux);
      PhaseMark pm_contp = profile_begin();
      continuity_residual_gpu(dmesh, ss.d_phi, ss.d_divCorr);
      profile_record(prof[PH_CONT_IN_P_LOOP], pm_contp);
    }
    simple_mms_pressure_residual_for_vector_diag("MMS_PEQN_SOLVED_RES", "after_pressure_solve", step, ss.d_pCorr);
    simple_mms_pressure_delta_diag("after_pressure_solve", step, ss.d_pCorr);

    simple_mms_diag_stage("after_pressure_solve_before_update", step, ss.d_u, ss.d_v, ss.d_w, ss.d_pCorr);

    massRes = maxabs_device(ss.d_divCorr, mesh.nCells, ss.d_reduce, ss.reduceSize);
    {
      const int block = 256;
      if(par.pMode == 1){
        kernel_update_pressure_absolute_relax<<<(mesh.nCells + block - 1)/block, block>>>(mesh.nCells, ss.d_p, ss.d_pCorr, par.pRelax);
      } else {
        kernel_update_pressure_relax<<<(mesh.nCells + block - 1)/block, block>>>(mesh.nCells, ss.d_p, ss.d_pCorr, par.pRelax);
      }
      CUDA_CHECK_LAST();
      if(usePressureAnchor){
        double pref = 0.0;
        CUDA_CALL(cudaMemcpy(&pref, ss.d_p + refCell, sizeof(cfdScalar), cudaMemcpyDeviceToHost));
        kernel_subtract_scalar_inplace<<<(mesh.nCells + block - 1)/block, block>>>(mesh.nCells, ss.d_p, pref);
        CUDA_CHECK_LAST();
      }
    }

    simple_mms_diag_stage("after_pressure_update_before_velcorr", step, ss.d_u, ss.d_v, ss.d_w, ss.d_p);

    if(par.simpleMmsEnable && par.simpleMmsForceExactPAfterUpdate){
      {
        const int block = 256;
        kernel_simple_mms_set_exact_p<<<(mesh.nCells + block - 1)/block, block>>>(
            mesh.nCells, dmesh.d_ccx, dmesh.d_ccy, dmesh.d_ccz,
            par.rho, mu, par.simpleMmsVelocityScale, par.simpleMmsPressureScale,
            par.simpleMmsStokesOnly,
            ss.d_p);
        CUDA_CHECK_LAST();

        if(usePressureAnchor){
          cfdScalar pref = 0;
          CUDA_CALL(cudaMemcpy(&pref, ss.d_p + refCell, sizeof(cfdScalar), cudaMemcpyDeviceToHost));
          kernel_simple_mms_subtract_ref_cell<<<(mesh.nCells + block - 1)/block, block>>>(
              mesh.nCells, ss.d_p, pref);
          CUDA_CHECK_LAST();
        }
      }
      simple_mms_diag_stage("forced_exact_p_after_update_before_velcorr", step, ss.d_u, ss.d_v, ss.d_w, ss.d_p);
    }

    PhaseMark pm_pcorrg = profile_begin();
    if(par.pMode == 1){
      if(par.pSolveMode == 1) compute_pressure_gradient_gpu(par, gop, dmesh, dbcP, ss.d_p, ss.d_gradx, ss.d_grady, ss.d_gradz);
      else compute_pressure_gradient_gpu(par, gop, dmesh, dbcP, ss.d_p, ss.d_gradx, ss.d_grady, ss.d_gradz);
    } else {
      compute_pressure_gradient_gpu(par, gop, dmesh, dbcP, ss.d_pCorr, ss.d_gradx, ss.d_grady, ss.d_gradz);
    }
    profile_record(prof[PH_PCORR_GRAD], pm_pcorrg);
    PhaseMark pm_velcorr = profile_begin();
    {
      const int block = 256;
      kernel_correct_velocity_from_pcorr_grad<<<(mesh.nCells + block - 1)/block, block>>>(
          mesh.nCells, ss.d_u, ss.d_v, ss.d_w, mom.d_rAU,
          ss.d_gradx, ss.d_grady, ss.d_gradz,
          ss.d_u, ss.d_v, ss.d_w);
      CUDA_CHECK_LAST();
    }
    profile_record(prof[PH_VEL_CORRECT], pm_velcorr);

    simple_mms_diag_stage("after_velocity_correction", step, ss.d_u, ss.d_v, ss.d_w, ss.d_p);

    duRel = relchg_device(ss.d_u, ss.d_uOld, mesh.nCells, ss.d_reduce, ss.d_reduce2, ss.reduceSize);
    dvRel = relchg_device(ss.d_v, ss.d_vOld, mesh.nCells, ss.d_reduce, ss.d_reduce2, ss.reduceSize);
    dwRel = relchg_device(ss.d_w, ss.d_wOld, mesh.nCells, ss.d_reduce, ss.d_reduce2, ss.reduceSize);
    dpRel = relchg_device(ss.d_p, ss.d_pOld, mesh.nCells, ss.d_reduce, ss.d_reduce2, ss.reduceSize);

    if(rank == 0){
      bool doBadCellAudit = false;
      const char* badCellAuditReason = "periodic";
      if(par.badCellAuditEvery > 0 && step >= par.badCellAuditStart && (step % par.badCellAuditEvery) == 0){
        doBadCellAudit = true;
        badCellAuditReason = "periodic";
      }
      if(par.badCellAuditOnGrowth != 0 && step >= par.badCellAuditStart && prevMassResForBadCellAudit > 0.0){
        const double growthFactor = std::max(par.badCellAuditGrowthFactor, 1.0);
        if(massRes > growthFactor * prevMassResForBadCellAudit && massRes >= par.badCellAuditMassFloor){
          doBadCellAudit = true;
          badCellAuditReason = "mass-growth";
        }
      }
      if(doBadCellAudit){
        std::vector<cfdScalar> gradxH(mesh.nCells), gradyH(mesh.nCells), gradzH(mesh.nCells);
        std::vector<cfdScalar> rAUH(mesh.nCells), divCorrH(mesh.nCells);
        std::vector<cfdScalar> phiStarH(mesh.nFaces), phiH(mesh.nFaces), phiNonOrthH(mesh.nFaces);
        copy_device_to_vec(ss.d_gradx, gradxH);
        copy_device_to_vec(ss.d_grady, gradyH);
        copy_device_to_vec(ss.d_gradz, gradzH);
        copy_device_to_vec(mom.d_rAU, rAUH);
        copy_device_to_vec(ss.d_divCorr, divCorrH);
        copy_device_to_vec(ss.d_phiStar, phiStarH);
        copy_device_to_vec(ss.d_phi, phiH);
        copy_device_to_vec(ss.d_phiNonOrth, phiNonOrthH);
        copy_device_to_vec(ss.d_pCorr, pCorr);
        copy_device_to_vec(ss.d_u, u);
        copy_device_to_vec(ss.d_v, v);
        copy_device_to_vec(ss.d_w, w);
        uStar = u; vStar = v; wStar = w;
        run_bad_cell_audit(par, mesh, bcPType, step, badCellAuditReason,
            massRes, duRel, dvRel, dwRel, dpRel,
            pCorr, gradxH, gradyH, gradzH, rAUH, divCorrH,
            uStar, vStar, wStar, u, v, w, phiStarH, phiH, phiNonOrthH);
      }
      prevMassResForBadCellAudit = massRes;
    }

    double iterWall = MPI_Wtime() - iterStart;
    double totalWall = MPI_Wtime() - runStart;

    if(rank==0 && (step==1 || (par.printEvery>0 && step%par.printEvery==0))){
      std::printf("iter %4d : massRes = %.3e, duRel = %.3e, dvRel = %.3e, dwRel = %.3e, dpRel = %.3e, velIts=[%d %d %d], pcgLast=%d, pcgTot=%d, iterWall = %.3e s, totalWall = %.3e s\n",
                  step, massRes, duRel, dvRel, dwRel, dpRel,
                  (int)lastItsU, (int)lastItsV, (int)lastItsW, (int)lastItsP, pcgTotalIts, iterWall, totalWall);
    }

    profStepsDone = step;

    if(par.writeEvery>0 && step%par.writeEvery==0 && par.write_vtu){
      copy_device_to_vec(ss.d_p, p);
      copy_device_to_vec(ss.d_u, u);
      copy_device_to_vec(ss.d_v, v);
      copy_device_to_vec(ss.d_w, w);
      copy_device_to_vec(ss.d_divCorr, divCorr);
      std::vector<cfdVector3> Uvec(mesh.nCells);
      std::vector<cfdScalar> umag(mesh.nCells);
      for(int c=0;c<mesh.nCells;++c){ Uvec[c]={u[c],v[c],w[c]}; umag[c]=std::sqrt(u[c]*u[c]+v[c]*v[c]+w[c]*w[c]); }
      std::ostringstream oss; oss<<par.outPrefix<<"_iter"<<std::setw(6)<<std::setfill('0')<<step<<".vtu";
      write_vtu_polyhedron_cell_data(oss.str(), mesh, {"p","umag","cell_volume","divCorr"}, {p,umag,to_cfd_scalar_vec(mesh.vol),divCorr}, "U", &Uvec);
    }

    if(!std::isfinite(massRes) || massRes > 1e50){
      if(rank==0) std::fprintf(stderr, "SIMPLE-like steady solve diverged.\n");
      break;
    }
    if(massRes < par.tolMass && std::max({duRel,dvRel,dwRel}) < par.tolVel){
      stepConverged = step;
      if(rank==0) std::printf("Converged at iteration %d : massRes = %.3e\n", step, massRes);
      break;
    }
    stepConverged = step;
  }

  double solveLoopWall = MPI_Wtime() - runStart;
  gpu_mem_mark("after_solve_loop_before_host_copies", rank);

  // Bring final fields back to host once for final summaries, force postprocessing, scalar coupling, and VTU output.
  copy_device_to_vec(ss.d_p, p);
  copy_device_to_vec(ss.d_u, u);
  copy_device_to_vec(ss.d_v, v);
  copy_device_to_vec(ss.d_w, w);
  copy_device_to_vec(ss.d_divCorr, divCorr);
  copy_device_to_vec(ss.d_phi, phi);

  std::vector<cfdVector3> Uvec(mesh.nCells);
  std::vector<cfdScalar> umag(mesh.nCells);
  double umax=0.0,vmaxf=0.0,wmaxf=0.0,pmax=0.0;
  for(int c=0;c<mesh.nCells;++c){
    Uvec[c]={u[c],v[c],w[c]};
    umag[c]=std::sqrt(u[c]*u[c]+v[c]*v[c]+w[c]*w[c]);
    umax  = std::max(umax,  std::fabs(static_cast<double>(u[c])));
    vmaxf = std::max(vmaxf, std::fabs(static_cast<double>(v[c])));
    wmaxf = std::max(wmaxf, std::fabs(static_cast<double>(w[c])));
    pmax  = std::max(pmax,  std::fabs(static_cast<double>(p[c])));
  }

  double simpleMmsUAbsL2 = 0.0, simpleMmsURelL2 = 0.0;
  double simpleMmsPAbsL2 = 0.0, simpleMmsPRelL2 = 0.0, simpleMmsPMeanOffset = 0.0;
  if(par.simpleMmsEnable){
    simple_mms_report_error(mesh, u, v, w, p, par.rho, mu,
        par.simpleMmsVelocityScale, par.simpleMmsPressureScale,
        simpleMmsUAbsL2, simpleMmsURelL2,
        simpleMmsPAbsL2, simpleMmsPRelL2, simpleMmsPMeanOffset);
    if(rank == 0){
      std::printf("------------------------------------------------------------\n");
      std::printf("SIMPLE_MMS U abs L2  = %.12e\n", simpleMmsUAbsL2);
      std::printf("SIMPLE_MMS U rel L2  = %.12e\n", simpleMmsURelL2);
      std::printf("SIMPLE_MMS p abs L2  = %.12e  (mean-free)\n", simpleMmsPAbsL2);
      std::printf("SIMPLE_MMS p rel L2  = %.12e  (mean-free)\n", simpleMmsPRelL2);
      std::printf("SIMPLE_MMS p mean offset = %.12e\n", simpleMmsPMeanOffset);

      double simpleMmsVolSum = 0.0;
      for(int c = 0; c < mesh.nCells; ++c) simpleMmsVolSum += mesh.vol[c];
      const double simpleMmsH = std::cbrt(simpleMmsVolSum / std::max(1, mesh.nCells));

      for(int layer = 1; layer <= 3; ++layer){
        double uAbsI = 0.0, uRelI = 0.0;
        double pAbsI = 0.0, pRelI = 0.0, pMeanI = 0.0;
        int nUsedI = 0;
        simple_mms_accumulate_error_for_margin(
            mesh, u, v, w, p,
            par.rho, mu,
            par.simpleMmsVelocityScale,
            par.simpleMmsPressureScale,
            layer * simpleMmsH,
            uAbsI, uRelI,
            pAbsI, pRelI,
            pMeanI,
            nUsedI);

        std::printf("SIMPLE_MMS_INTERIOR layer=%d margin=%.12e nUsed=%d UrelL2=%.12e PrelL2MeanFree=%.12e pMeanOffset=%.12e\n",
                    layer, layer * simpleMmsH, nUsedI, uRelI, pRelI, pMeanI);
      }


      const double fixedMargins[] = {0.05, 0.10, 0.15, 0.20};
      for(double marginPhys : fixedMargins){
        double uAbsI = 0.0, uRelI = 0.0;
        double pAbsI = 0.0, pRelI = 0.0, pMeanI = 0.0;
        int nUsedI = 0;

        simple_mms_accumulate_error_for_margin(
            mesh, u, v, w, p,
            par.rho, mu,
            par.simpleMmsVelocityScale,
            par.simpleMmsPressureScale,
            marginPhys,
            uAbsI, uRelI,
            pAbsI, pRelI,
            pMeanI,
            nUsedI);

        std::printf("SIMPLE_MMS_FIXED_INTERIOR margin=%.8e nUsed=%d UrelL2=%.12e PrelL2MeanFree=%.12e pMeanOffset=%.12e\n",
                    marginPhys, nUsedI, uRelI, pRelI, pMeanI);
      }

      if(par.simpleMmsDiagWritePErrorVtk){
        std::string vtkPath = par.outPrefix + "_mms_p_error_points.vtk";
        simple_mms_write_p_error_point_vtk(
            mesh, p,
            par.rho, mu,
            par.simpleMmsVelocityScale,
            par.simpleMmsPressureScale,
            par.simpleMmsStokesOnly,
            vtkPath);
      }

      std::printf("------------------------------------------------------------\n");
    }
  }

  // SINE_MMS_ERROR_REPORT
  if(par.sineMmsEnable){
    SineMmsErrorReport er = compute_sine_mms_error_report(
        mesh, u, v, w, p,
        par.sineMmsVelocityScale, par.sineMmsPressureScale);

    if(rank == 0){
      std::printf("------------------------------------------------------------\n");
      std::printf("SINE_MMS U rel L2           = %.12e\n", er.uRelL2);
      std::printf("SINE_MMS p rel L2 raw       = %.12e\n", er.pRelL2);
      std::printf("SINE_MMS p rel L2 meanshift = %.12e\n", er.pRelL2MeanShift);
      std::printf("SINE_MMS U abs L2           = %.12e\n", er.uAbsL2);
      std::printf("SINE_MMS p abs L2 raw       = %.12e\n", er.pAbsL2);
      std::printf("SINE_MMS p mean error       = %.12e\n", er.pMeanError);
      std::printf("SINE_MMS max U error        = %.12e\n", er.maxUerr);
      std::printf("SINE_MMS max p error        = %.12e\n", er.maxPerr);
      std::printf("------------------------------------------------------------\n");
    }
  }

  std::vector<cfdScalar> scalarField;
  if(par.scalarEnable != 0){
    scalarField = solve_scalar_after_flow(par, mesh, phi, rank);
  }

  const double forceBenchmarkH = 0.41;
  const double forceUbar = (4.0/9.0) * par.Umean;
  CylinderForceReport cylForce;
  CylinderForceVectorReport cylForceVec;
  PatchForceReport patchForce;

  if(par.forceEnable && cylinderPatch >= 0){
    cylForce = compute_cylinder_forces_paper(
        mesh, cylinderPatch, u, v, w, p,
        par.rho, mu, par.pipeDiameter, forceBenchmarkH, forceUbar);

    cylForceVec = compute_cylinder_forces_vector_wall_shear(
        mesh, cylinderPatch, u, v, w, p,
        par.rho, mu, par.pipeDiameter, forceBenchmarkH, forceUbar);
  }

  if(par.forceEnable){
    const int forcePatch = find_patch_index_local(mesh, par.forcePatchName);
    patchForce = compute_patch_forces_wall_shear(
        mesh, forcePatch, u, v, w, p,
        par.rho, mu,
        par.forceNormalSign,
        par.forceUref,
        par.forceAreaRef,
        par.forceDragDir,
        par.forceLiftDir,
        par.forceSpanDir);
  }

  if(rank==0){
    std::printf("\nFinal summary:\n");
    std::printf("------------------------------------------------------------\n");
    std::printf("Iterations    = %d\n", stepConverged);
    std::printf("massRes       = %.8e\n", massRes);
    std::printf("max|u|        = %.8e\n", umax);
    std::printf("max|v|        = %.8e\n", vmaxf);
    std::printf("max|w|        = %.8e\n", wmaxf);
    std::printf("max|p|        = %.8e\n", pmax);
    if(cylForce.valid){
      std::printf("------------------------------------------------------------\n");
      std::printf("Cylinder force postprocess, paper formula\n");
      std::printf("force patch   = %s\n", cylForce.patchName.c_str());
      std::printf("force faces   = %d\n", cylForce.nFaces);
      std::printf("force area    = %.12e\n", cylForce.area);
      std::printf("rho, mu       = %.12e  %.12e\n", cylForce.rho, cylForce.mu);
      std::printf("Ubar, D, H    = %.12e  %.12e  %.12e\n", cylForce.Ubar, cylForce.D, cylForce.H);
      std::printf("coeff denom   = rho*Ubar^2*D*H = %.12e\n", cylForce.coeffDenom);
      std::printf("FD pressure   = %.12e\n", cylForce.FD_pressure);
      std::printf("FD viscous    = %.12e\n", cylForce.FD_viscous);
      std::printf("FD total      = %.12e\n", cylForce.FD_total);
      std::printf("FL pressure   = %.12e\n", cylForce.FL_pressure);
      std::printf("FL viscous    = %.12e\n", cylForce.FL_viscous);
      std::printf("FL total      = %.12e\n", cylForce.FL_total);
      std::printf("CD            = %.12e\n", cylForce.CD);
      std::printf("CL            = %.12e\n", cylForce.CL);
      std::printf("wall dn min/max = %.12e / %.12e\n", cylForce.minWallDistance, cylForce.maxWallDistance);
      std::printf("max|dvt/dn|   = %.12e\n", cylForce.maxAbsDvtDn);
    }

    if(cylForceVec.valid){
      std::printf("------------------------------------------------------------\n");
      std::printf("Cylinder force postprocess, vector wall-shear traction\n");
      std::printf("force patch   = %s\n", cylForceVec.patchName.c_str());
      std::printf("force faces   = %d\n", cylForceVec.nFaces);
      std::printf("force area    = %.12e\n", cylForceVec.area);
      std::printf("rho, mu       = %.12e  %.12e\n", cylForceVec.rho, cylForceVec.mu);
      std::printf("Ubar, D, H    = %.12e  %.12e  %.12e\n", cylForceVec.Ubar, cylForceVec.D, cylForceVec.H);
      std::printf("coeff denom   = rho*Ubar^2*D*H = %.12e\n", cylForceVec.coeffDenom);

      std::printf("Fp vector     = [%.12e, %.12e, %.12e]\n", cylForceVec.Fp[0], cylForceVec.Fp[1], cylForceVec.Fp[2]);
      std::printf("Fv vector     = [%.12e, %.12e, %.12e]\n", cylForceVec.Fv[0], cylForceVec.Fv[1], cylForceVec.Fv[2]);
      std::printf("F  vector     = [%.12e, %.12e, %.12e]\n", cylForceVec.F[0],  cylForceVec.F[1],  cylForceVec.F[2]);
      std::printf("C  vector     = [%.12e, %.12e, %.12e]\n", cylForceVec.C[0],  cylForceVec.C[1],  cylForceVec.C[2]);

      std::printf("CD_vector     = %.12e\n", cylForceVec.C[0]);
      std::printf("CL_y_vector   = %.12e\n", cylForceVec.C[1]);
      std::printf("CL_z_vector   = %.12e\n", cylForceVec.C[2]);
      std::printf("wall dn min/max = %.12e / %.12e\n", cylForceVec.minWallDistance, cylForceVec.maxWallDistance);
      std::printf("max|Ut|       = %.12e\n", cylForceVec.maxUt);
      std::printf("max shear mag = %.12e\n", cylForceVec.maxShearMag);
    }

    if(par.forceEnable){
      std::printf("------------------------------------------------------------\n");
      if(patchForce.valid){
        std::printf("Patch force postprocess, generic wall-shear traction\n");
        std::printf("force patch      = %s\n", patchForce.patchName.c_str());
        std::printf("force faces      = %d\n", patchForce.nFaces);
        std::printf("force area       = %.12e\n", patchForce.area);
        std::printf("normalSign       = %d\n", patchForce.normalSign);
        std::printf("rho, mu          = %.12e  %.12e\n", patchForce.rho, patchForce.mu);
        std::printf("Uref, Aref       = %.12e  %.12e\n", patchForce.Uref, patchForce.Aref);
        std::printf("coeff denom      = rho*Uref^2*Aref = %.12e\n", patchForce.coeffDenom);
        std::printf("dragDir          = [%.12e, %.12e, %.12e]\n", patchForce.dragDir[0], patchForce.dragDir[1], patchForce.dragDir[2]);
        std::printf("liftDir          = [%.12e, %.12e, %.12e]\n", patchForce.liftDir[0], patchForce.liftDir[1], patchForce.liftDir[2]);
        std::printf("spanDir          = [%.12e, %.12e, %.12e]\n", patchForce.spanDir[0], patchForce.spanDir[1], patchForce.spanDir[2]);

        std::printf("Fp vector        = [%.12e, %.12e, %.12e]\n", patchForce.Fp[0], patchForce.Fp[1], patchForce.Fp[2]);
        std::printf("Fv vector        = [%.12e, %.12e, %.12e]\n", patchForce.Fv[0], patchForce.Fv[1], patchForce.Fv[2]);
        std::printf("F  vector        = [%.12e, %.12e, %.12e]\n", patchForce.F[0],  patchForce.F[1],  patchForce.F[2]);

        std::printf("F_drag pressure  = %.12e\n", patchForce.FpDrag);
        std::printf("F_drag viscous   = %.12e\n", patchForce.FvDrag);
        std::printf("F_drag total     = %.12e\n", patchForce.FDrag);
        std::printf("F_lift pressure  = %.12e\n", patchForce.FpLift);
        std::printf("F_lift viscous   = %.12e\n", patchForce.FvLift);
        std::printf("F_lift total     = %.12e\n", patchForce.FLift);
        std::printf("F_span pressure  = %.12e\n", patchForce.FpSpan);
        std::printf("F_span viscous   = %.12e\n", patchForce.FvSpan);
        std::printf("F_span total     = %.12e\n", patchForce.FSpan);

        std::printf("C_drag           = %.12e\n", patchForce.CDrag);
        std::printf("C_lift           = %.12e\n", patchForce.CLift);
        std::printf("C_span           = %.12e\n", patchForce.CSpan);
        std::printf("wall dn min/max  = %.12e / %.12e\n", patchForce.minWallDistance, patchForce.maxWallDistance);
        std::printf("max|Ut|          = %.12e\n", patchForce.maxUt);
        std::printf("max shear mag    = %.12e\n", patchForce.maxShearMag);
      } else {
        std::printf("Patch force postprocess requested but skipped. Check forcePatch, forceUref, forceAreaRef.\n");
        std::printf("requested patch  = %s\n", par.forcePatchName.c_str());
        std::printf("forceUref        = %.12e\n", par.forceUref);
        std::printf("forceAreaRef     = %.12e\n", par.forceAreaRef);
      }
    }

    std::printf("last velocity it = [%d %d %d]\n", (int)lastItsU, (int)lastItsV, (int)lastItsW);
    std::printf("last pcg it   = %d\n", (int)lastItsP);
    std::printf("assemble time : %.6e s\n", totalAssemble);
    std::printf("mom setup time: %.6e s\n", totalSetup);
    std::printf("mom solve time: %.6e s\n", totalSolve);
    std::printf("p setup time  : %.6e s\n", pressureSetup);
    std::printf("p solve time  : %.6e s\n", pressureSolve);
    double lightPhaseSum = 0.0;
    for(int i=0; i<PH_COUNT; ++i) lightPhaseSum += prof[i].time_sum;

    const double solverAccounted =
        totalAssemble + totalSetup + totalSolve + pressureSetup + pressureSolve;

    const double denomWall = (solveLoopWall > 1.0e-30 ? solveLoopWall : 1.0);
    const double solverOther = solveLoopWall - solverAccounted;
    const double phaseOther  = solveLoopWall - lightPhaseSum;

    std::printf("solve loop wall: %.6e s\n", solveLoopWall);
    std::printf("solver-accounted total : %.6e s  (%6.2f %%)\n",
                solverAccounted, 100.0*solverAccounted/denomWall);
    std::printf("solver-unaccounted gap : %.6e s  (%6.2f %%)\n",
                solverOther, 100.0*solverOther/denomWall);
    std::printf("light phase sum        : %.6e s  (%6.2f %%)\n",
                lightPhaseSum, 100.0*lightPhaseSum/denomWall);
    std::printf("light phase remainder  : %.6e s  (%6.2f %%)\n",
                phaseOther, 100.0*phaseOther/denomWall);

    std::printf("------------------------------------------------------------\n");
    std::printf("Lightweight timing breakdown, no CUDA sync when -profile-steps 0:\n");
    std::printf("%-28s %14s %12s %12s\n", "phase", "total[s]", "avg/iter[s]", "%wall");
    for(int i=0; i<PH_COUNT; ++i){
      if(prof[i].time_sum > 1.0e-9){
        std::printf("%-28s %14.6e %12.6e %11.2f\n",
                    kProfilePhaseNames[i],
                    prof[i].time_sum,
                    prof[i].time_sum / std::max(stepConverged,1),
                    100.0*prof[i].time_sum/denomWall);
      }
    }
    std::printf("------------------------------------------------------------\n");
    if(par.profileSteps>0 && profStepsDone>0){
      int hottest=0, highestGpuAfter=0, highestGpuDelta=0;
      for(int i=1;i<PH_COUNT;++i){
        if(prof[i].time_sum > prof[hottest].time_sum) hottest=i;
        if(prof[i].max_gpu_after_mb > prof[highestGpuAfter].max_gpu_after_mb) highestGpuAfter=i;
        if(prof[i].max_gpu_delta_mb > prof[highestGpuDelta].max_gpu_delta_mb) highestGpuDelta=i;
      }
      std::printf("Detailed profiling averages over %d steps:\n", profStepsDone);
      std::printf("%-24s %11s %11s %11s %11s %11s %11s %11s\n", "phase", "avg_t[s]", "avgG0", "avgG1", "avgdG", "maxG1", "maxdG", "maxdCPU");
      for(int i=0;i<PH_COUNT;++i){
        std::printf("%-24s %11.4f %11.1f %11.1f %11.1f %11.1f %11.1f %11.1f\n",
                    kProfilePhaseNames[i],
                    prof[i].time_sum / profStepsDone,
                    prof[i].gpu_before_sum / profStepsDone,
                    prof[i].gpu_after_sum / profStepsDone,
                    prof[i].gpu_delta_sum / profStepsDone,
                    prof[i].max_gpu_after_mb,
                    prof[i].max_gpu_delta_mb,
                    prof[i].max_cpu_delta_mb);
      }
      std::printf("------------------------------------------------------------\n");
      std::printf("Hottest phase by avg time  : %s (%.4f s/step)\n", kProfilePhaseNames[hottest], prof[hottest].time_sum / profStepsDone);
      std::printf("Highest GPU resident after : %s (%.1f MB)\n", kProfilePhaseNames[highestGpuAfter], prof[highestGpuAfter].max_gpu_after_mb);
      std::printf("Largest GPU memory jump    : %s (%.1f MB)\n", kProfilePhaseNames[highestGpuDelta], prof[highestGpuDelta].max_gpu_delta_mb);
      std::printf("%-24s %14.3f\n", "final CPU RSS[MB]", get_cpu_rss_mb());
      std::printf("%-24s %14.3f\n", "final GPU used[MB]", get_gpu_used_mb());
      std::printf("------------------------------------------------------------\n");
    }
  }

  if(par.write_vtu){
    std::string vtuFile = par.outPrefix + "_final.vtu";
    std::vector<std::string> scalarNames = {"p","umag","cell_volume","divCorr"};
    std::vector<std::vector<cfdScalar>> scalarData = {p,umag,to_cfd_scalar_vec(mesh.vol),divCorr};
    if(!scalarField.empty()){
      scalarNames.push_back(par.scalarName);
      scalarData.push_back(scalarField);
    }
    write_vtu_polyhedron_cell_data(vtuFile, mesh, scalarNames, scalarData, "U", &Uvec);
    if(rank==0) std::printf("Wrote VTU         : %s\n", vtuFile.c_str());
  }

  std::ofstream sout((par.outPrefix+"_summary.txt").c_str());
  if(par.simpleMmsEnable){
    sout << "simpleMmsUAbsL2 " << simpleMmsUAbsL2 << "\n";
    sout << "simpleMmsURelL2 " << simpleMmsURelL2 << "\n";
    sout << "simpleMmsPAbsL2MeanFree " << simpleMmsPAbsL2 << "\n";
    sout << "simpleMmsPRelL2MeanFree " << simpleMmsPRelL2 << "\n";
    sout << "simpleMmsPMeanOffset " << simpleMmsPMeanOffset << "\n";
  }
  sout << std::setprecision(16);
  sout << "points " << mesh.P.size() << "\nfaces " << mesh.nFaces << "\ninternalFaces " << mesh.nInternalFaces << "\ncells " << mesh.nCells << "\n";
  sout << "maxNonOrthDeg " << mesh.maxNonOrthDeg << "\nmu " << mu << "\n";
  sout << "steps " << stepConverged << "\nmassRes " << massRes << "\n";
  sout << "maxAbsU " << umax << "\nmaxAbsV " << vmaxf << "\nmaxAbsW " << wmaxf << "\nmaxAbsP " << pmax << "\n";
  sout << "lastBiCGSTABU " << lastItsU << "\nlastBiCGSTABV " << lastItsV << "\nlastBiCGSTABW " << lastItsW << "\nlastPCG " << lastItsP << "\n";
  if(par.sineMmsEnable){
    SineMmsErrorReport er = compute_sine_mms_error_report(
        mesh, u, v, w, p,
        par.sineMmsVelocityScale, par.sineMmsPressureScale);
    sout << "sineMmsUrelL2 " << er.uRelL2 << "\n";
    sout << "sineMmsPrelL2Raw " << er.pRelL2 << "\n";
    sout << "sineMmsPrelL2MeanShift " << er.pRelL2MeanShift << "\n";
    sout << "sineMmsUabsL2 " << er.uAbsL2 << "\n";
    sout << "sineMmsPabsL2Raw " << er.pAbsL2 << "\n";
    sout << "sineMmsPMeanError " << er.pMeanError << "\n";
  }
  if(!scalarField.empty()){
    auto mnmx = std::minmax_element(scalarField.begin(), scalarField.end());
    sout << "scalarName " << par.scalarName << "\n";
    sout << "scalarMin " << *mnmx.first << "\n";
    sout << "scalarMax " << *mnmx.second << "\n";
    sout << "scalarConvectionScheme " << par.scalarConvectionScheme << "\n";
    sout << "scalarGamma " << par.scalarGamma << "\n";
  }
  if(cylForce.valid){
    sout << "cylinderForcePatch " << cylForce.patchName << "\n";
    sout << "cylinderForceFaces " << cylForce.nFaces << "\n";
    sout << "cylinderForceArea " << cylForce.area << "\n";
    sout << "forceUbar " << cylForce.Ubar << "\n";
    sout << "forceD " << cylForce.D << "\n";
    sout << "forceH " << cylForce.H << "\n";
    sout << "forceCoeffDenom " << cylForce.coeffDenom << "\n";
    sout << "FDPressure " << cylForce.FD_pressure << "\n";
    sout << "FDViscous " << cylForce.FD_viscous << "\n";
    sout << "FDTotal " << cylForce.FD_total << "\n";
    sout << "FLPressure " << cylForce.FL_pressure << "\n";
    sout << "FLViscous " << cylForce.FL_viscous << "\n";
    sout << "FLTotal " << cylForce.FL_total << "\n";
    sout << "CD " << cylForce.CD << "\n";
    sout << "CL " << cylForce.CL << "\n";
    sout << "wallDnMin " << cylForce.minWallDistance << "\n";
    sout << "wallDnMax " << cylForce.maxWallDistance << "\n";
    sout << "maxAbsDvtDn " << cylForce.maxAbsDvtDn << "\n";
  }
  if(cylForceVec.valid){
    sout << "CDVector " << cylForceVec.C[0] << "\n";
    sout << "CLyVector " << cylForceVec.C[1] << "\n";
    sout << "CLzVector " << cylForceVec.C[2] << "\n";
    sout << "FxPressureVector " << cylForceVec.Fp[0] << "\n";
    sout << "FyPressureVector " << cylForceVec.Fp[1] << "\n";
    sout << "FzPressureVector " << cylForceVec.Fp[2] << "\n";
    sout << "FxViscousVector " << cylForceVec.Fv[0] << "\n";
    sout << "FyViscousVector " << cylForceVec.Fv[1] << "\n";
    sout << "FzViscousVector " << cylForceVec.Fv[2] << "\n";
    sout << "FxTotalVector " << cylForceVec.F[0] << "\n";
    sout << "FyTotalVector " << cylForceVec.F[1] << "\n";
    sout << "FzTotalVector " << cylForceVec.F[2] << "\n";
  }
  if(patchForce.valid){
    sout << "genericForcePatch " << patchForce.patchName << "\n";
    sout << "genericForceFaces " << patchForce.nFaces << "\n";
    sout << "genericForceArea " << patchForce.area << "\n";
    sout << "genericForceNormalSign " << patchForce.normalSign << "\n";
    sout << "genericForceUref " << patchForce.Uref << "\n";
    sout << "genericForceAref " << patchForce.Aref << "\n";
    sout << "genericForceCoeffDenom " << patchForce.coeffDenom << "\n";
    sout << "genericFxPressure " << patchForce.Fp[0] << "\n";
    sout << "genericFyPressure " << patchForce.Fp[1] << "\n";
    sout << "genericFzPressure " << patchForce.Fp[2] << "\n";
    sout << "genericFxViscous " << patchForce.Fv[0] << "\n";
    sout << "genericFyViscous " << patchForce.Fv[1] << "\n";
    sout << "genericFzViscous " << patchForce.Fv[2] << "\n";
    sout << "genericFxTotal " << patchForce.F[0] << "\n";
    sout << "genericFyTotal " << patchForce.F[1] << "\n";
    sout << "genericFzTotal " << patchForce.F[2] << "\n";
    sout << "genericCDrag " << patchForce.CDrag << "\n";
    sout << "genericCLift " << patchForce.CLift << "\n";
    sout << "genericCSpan " << patchForce.CSpan << "\n";
  }
  sout.close();

    gpu_mem_mark("before_destroy_systems", rank);
destroy_momentum_system(mom);
  destroy_simple_scratch(ss);
  if(needLsqGradientOperator) destroy_lsq_gradient_operator(gop);
  destroy_linear_storage(pressureSys);
  destroy_device_bc(dbcP); destroy_device_bc(dbcU); destroy_device_bc(dbcV); destroy_device_bc(dbcW);
  destroy_device_mesh(dmesh);
    gpu_mem_mark("after_destroy_systems_before_hypre_finalize", rank);
  HYPRE_CALL(HYPRE_Finalize());
  MPI_Finalize();
  return 0;
}
