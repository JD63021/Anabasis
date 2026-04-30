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
#include <cuda_runtime.h>
#include "patch_geometry.h"
#include "bc_specs.h"
#include "velocity_bc_eval.h"
#include "bc_runtime_config.h"
extern "C" {
#include "HYPRE.h"
#include "HYPRE_IJ_mv.h"
#include "HYPRE_parcsr_ls.h"
#include "HYPRE_krylov.h"
#include "_hypre_parcsr_mv.h"
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
  atomicAdd(reinterpret_cast<double*>(addr), static_cast<double>(val));
}

struct PatchInfo { std::string name; int nFaces=0; int startFace=0; std::string type; };
struct Mesh {
  std::vector<std::array<double,3>> P;
  std::vector<std::vector<int>> faces;
  std::vector<int> owner, neigh, bPatch;
  std::vector<std::string> patchNames;
  std::vector<int> patchStartFace, patchNFaces;
  int nFaces=0, nInternalFaces=0, nCells=0;
  std::vector<std::array<double,3>> cc, xf, nf, Sf;
  std::vector<double> vol, Af;
  std::vector<std::vector<int>> cellNbrs, cellBFace, cellFaces, cellOrient;
  double maxNonOrthDeg=0.0;
};
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
};

static inline std::array<double,3> add3(const std::array<double,3>& a,const std::array<double,3>& b){return {a[0]+b[0],a[1]+b[1],a[2]+b[2]};}
static inline std::array<double,3> sub3(const std::array<double,3>& a,const std::array<double,3>& b){return {a[0]-b[0],a[1]-b[1],a[2]-b[2]};}
static inline std::array<double,3> mul3(double s,const std::array<double,3>& a){return {s*a[0],s*a[1],s*a[2]};}
static inline double dot3(const std::array<double,3>& a,const std::array<double,3>& b){return a[0]*b[0]+a[1]*b[1]+a[2]*b[2];}
static inline std::array<double,3> cross3(const std::array<double,3>& a,const std::array<double,3>& b){return {a[1]*b[2]-a[2]*b[1],a[2]*b[0]-a[0]*b[2],a[0]*b[1]-a[1]*b[0]};}
static inline double norm3(const std::array<double,3>& a){return std::sqrt(dot3(a,a));}

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
    {"assemblyBackend", "-assembly-backend"}
  };

  std::ifstream in(casePath);
  if(!in){
    throw std::runtime_error("Could not open case config file: " + casePath);
  }

  std::vector<std::string> out;
  out.emplace_back(argv[0]);

  std::vector<std::string> bcLines;
  bool explicitBCConfig = false;

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
    else if(!std::strcmp(argv[i],"-vel-maxit")){need(argv[i]); par.velMaxit=std::atoi(argv[++i]);}
    else if(!std::strcmp(argv[i],"-vel-tol")){need(argv[i]); par.velTol=std::atof(argv[++i]);}
    else if(!std::strcmp(argv[i],"-vel-reltol")){need(argv[i]); par.velRelTol=std::atof(argv[++i]);}
    else if(!std::strcmp(argv[i],"-nVelNonOrthCorr")){need(argv[i]); par.nVelNonOrthCorr=std::atoi(argv[++i]);}
    else if(!std::strcmp(argv[i],"-nNonOrthCorr")){need(argv[i]); par.nNonOrthCorr=std::atoi(argv[++i]);}
    else if(!std::strcmp(argv[i],"-nPressureCorr")){need(argv[i]); par.nPressureCorr=std::atoi(argv[++i]);}
    else if(!std::strcmp(argv[i],"-u-relax")){need(argv[i]); par.uRelax=std::atof(argv[++i]);}
    else if(!std::strcmp(argv[i],"-p-relax")){need(argv[i]); par.pRelax=std::atof(argv[++i]);}
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

static void print_device_info(int device){
  cudaDeviceProp prop;
  CUDA_CALL(cudaGetDeviceProperties(&prop,device));
  std::printf("Running on \"%s\", major %d, minor %d, total memory %.2f GiB\n",prop.name,prop.major,prop.minor,(double)prop.totalGlobalMem/(1024.0*1024.0*1024.0));
}


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
static Mesh read_openfoam_polymesh(const std::string &polyMeshDir){
  Mesh mesh; auto patches=read_foam_boundary(polyMeshDir+"/boundary"); mesh.P=read_foam_points(polyMeshDir+"/points"); mesh.faces=read_foam_faces(polyMeshDir+"/faces"); std::vector<int> owner0=read_foam_labels(polyMeshDir+"/owner"); std::vector<int> neigh0=read_foam_labels(polyMeshDir+"/neighbour");
  mesh.nFaces=(int)mesh.faces.size(); mesh.nInternalFaces=(int)neigh0.size(); mesh.nCells=0;
  for(int v:owner0) mesh.nCells=std::max(mesh.nCells,v+1); for(int v:neigh0) mesh.nCells=std::max(mesh.nCells,v+1);
  mesh.owner.resize(mesh.nFaces); mesh.neigh.assign(mesh.nInternalFaces,0); for(int i=0;i<mesh.nFaces;++i) mesh.owner[i]=owner0[i]; for(int i=0;i<mesh.nInternalFaces;++i) mesh.neigh[i]=neigh0[i];
  mesh.bPatch.assign(mesh.nFaces,0); mesh.patchNames.resize(patches.size()); mesh.patchStartFace.resize(patches.size()); mesh.patchNFaces.resize(patches.size()); for(std::size_t k=0;k<patches.size();++k){ mesh.patchNames[k]=patches[k].name; mesh.patchStartFace[k]=patches[k].startFace; mesh.patchNFaces[k]=patches[k].nFaces; for(int f=patches[k].startFace; f<patches[k].startFace+patches[k].nFaces; ++f) mesh.bPatch[f]=(int)k+1; }
  mesh.cellFaces.assign(mesh.nCells,{}); mesh.cellOrient.assign(mesh.nCells,{});
  for(int f=0;f<mesh.nFaces;++f){ int P=mesh.owner[f]; mesh.cellFaces[P].push_back(f); mesh.cellOrient[P].push_back(+1); }
  for(int f=0;f<mesh.nInternalFaces;++f){ int N=mesh.neigh[f]; mesh.cellFaces[N].push_back(f); mesh.cellOrient[N].push_back(-1); }
  mesh.cc.assign(mesh.nCells,{0,0,0}); mesh.vol.assign(mesh.nCells,0.0);
  for(int c=0;c<mesh.nCells;++c){ std::set<int> vertsSet; for(int f:mesh.cellFaces[c]) for(int v:mesh.faces[f]) vertsSet.insert(v); std::array<double,3> c0{0,0,0}; for(int v:vertsSet) c0=add3(c0,mesh.P[v]); c0=mul3(1.0/std::max((int)vertsSet.size(),1),c0); double V=0.0; std::array<double,3> M{0,0,0}; for(std::size_t j=0;j<mesh.cellFaces[c].size();++j){ int f=mesh.cellFaces[c][j]; int ori=mesh.cellOrient[c][j]; std::vector<int> fv=mesh.faces[f]; if(ori<0) std::reverse(fv.begin(),fv.end()); auto a=mesh.P[fv[0]]; for(std::size_t i=1;i+1<fv.size();++i){ auto b=mesh.P[fv[i]], d=mesh.P[fv[i+1]]; double vTet=dot3(sub3(a,c0),cross3(sub3(b,c0),sub3(d,c0)))/6.0; auto cTet=mul3(0.25,add3(add3(c0,a),add3(b,d))); V += vTet; M = add3(M,mul3(vTet,cTet)); } } if(V<=0) throw std::runtime_error("Non-positive cell volume at cell "+std::to_string(c)); mesh.vol[c]=V; mesh.cc[c]=mul3(1.0/V,M); }
  mesh.xf.assign(mesh.nFaces,{0,0,0}); mesh.Af.assign(mesh.nFaces,0.0); mesh.nf.assign(mesh.nFaces,{0,0,0}); mesh.Sf.assign(mesh.nFaces,{0,0,0});
  for(int f=0;f<mesh.nFaces;++f){ std::array<double,3> xfc{0,0,0}; for(int v:mesh.faces[f]) xfc=add3(xfc,mesh.P[v]); xfc=mul3(1.0/std::max((int)mesh.faces[f].size(),1),xfc); mesh.xf[f]=xfc; auto a=mesh.P[mesh.faces[f][0]]; std::array<double,3> areaVec{0,0,0}; for(std::size_t i=1;i+1<mesh.faces[f].size();++i){ auto b=mesh.P[mesh.faces[f][i]], d=mesh.P[mesh.faces[f][i+1]]; areaVec=add3(areaVec,mul3(0.5,cross3(sub3(b,a),sub3(d,a)))); } double areaMag=norm3(areaVec); if(areaMag<=1e-30) throw std::runtime_error("Degenerate face area at face "+std::to_string(f)); auto nloc=mul3(1.0/areaMag,areaVec); int P=mesh.owner[f]; std::array<double,3> dtest; if(f<mesh.nInternalFaces) dtest=sub3(mesh.cc[mesh.neigh[f]],mesh.cc[P]); else dtest=sub3(mesh.xf[f],mesh.cc[P]); if(dot3(nloc,dtest)<0) nloc=mul3(-1.0,nloc); mesh.Af[f]=areaMag; mesh.nf[f]=nloc; mesh.Sf[f]=mul3(areaMag,nloc); }
  mesh.cellNbrs.assign(mesh.nCells,{}); mesh.cellBFace.assign(mesh.nCells,{}); for(int f=0;f<mesh.nFaces;++f){ int P=mesh.owner[f]; if(f<mesh.nInternalFaces){ int N=mesh.neigh[f]; mesh.cellNbrs[P].push_back(N); mesh.cellNbrs[N].push_back(P); } else mesh.cellBFace[P].push_back(f); }
  for(int c=0;c<mesh.nCells;++c){ std::sort(mesh.cellNbrs[c].begin(),mesh.cellNbrs[c].end()); mesh.cellNbrs[c].erase(std::unique(mesh.cellNbrs[c].begin(),mesh.cellNbrs[c].end()),mesh.cellNbrs[c].end()); }
  mesh.maxNonOrthDeg=0.0; for(int f=0;f<mesh.nInternalFaces;++f){ auto d=sub3(mesh.cc[mesh.neigh[f]],mesh.cc[mesh.owner[f]]); double cosang=std::fabs(dot3(d,mesh.nf[f]))/std::max(norm3(d),1e-30); cosang=std::min(1.0,std::max(0.0,cosang)); mesh.maxNonOrthDeg=std::max(mesh.maxNonOrthDeg, std::acos(cosang)*180.0/M_PI); }
  return mesh;
}

static void compute_lsq_gradient(const Mesh &mesh,const std::vector<double> &phi,const std::vector<std::string> &bcType,const std::vector<double> &bcFaceValue,std::vector<std::array<double,3>> &grad){
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
      grad[P] = {inv[0][0]*rhs[0]+inv[0][1]*rhs[1]+inv[0][2]*rhs[2], inv[1][0]*rhs[0]+inv[1][1]*rhs[1]+inv[1][2]*rhs[2], inv[2][0]*rhs[0]+inv[2][1]*rhs[1]+inv[2][2]*rhs[2]};
    }
  }
}

static void predictor_face_fluxes(const Mesh &mesh,const std::vector<double> &u,const std::vector<double> &v,const std::vector<double> &w,const std::vector<std::string> &patchUType,const std::vector<double> &uFaceBC,const std::vector<std::string> &patchVType,const std::vector<double> &vFaceBC,const std::vector<std::string> &patchWType,const std::vector<double> &wFaceBC,double rho,std::vector<double> &phi){
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
    const std::vector<double> &u,
    const std::vector<double> &v,
    const std::vector<double> &w,
    const std::vector<double> &p,
    const std::vector<std::array<double,3>> &gradP,
    const std::vector<double> &rAU,
    const std::vector<std::string> &bcUType,
    const std::vector<double> &uFaceBC,
    const std::vector<std::string> &bcVType,
    const std::vector<double> &vFaceBC,
    const std::vector<std::string> &bcWType,
    const std::vector<double> &wFaceBC,
    double rho,
    std::vector<double> &phiStar)
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

static void continuity_residual(const Mesh &mesh,const std::vector<double> &phi,std::vector<double> &divCell){
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
static void correct_face_fluxes_simple(const Mesh &mesh,const std::vector<double> &rAU,const std::vector<double> &phiStar,const std::vector<double> &pCorr,std::vector<double> &phi){
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
                                                 const std::vector<double> &rAU,
                                                 const std::vector<std::string> &bcPType,
                                                 const std::vector<std::array<double,3>> &gradP,
                                                 std::vector<double> &phiNonOrth,
                                                 std::vector<double> &divNonOrth)
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
    double flux = rAU[P] * dot3(gradP[P], t);
    phiNonOrth[f] = flux;
    divNonOrth[P] += flux;
  }
}

static void correct_face_fluxes_simple_nonorth(const Mesh &mesh,
                                               const std::vector<double> &rAU,
                                               const std::vector<std::string> &bcPType,
                                               const std::vector<double> &pFaceBC,
                                               const std::vector<double> &phiStar,
                                               const std::vector<double> &pCorr,
                                               const std::vector<std::array<double,3>> &gradP,
                                               std::vector<double> &phi)
{
  std::vector<double> phiNonOrth, divDummy;
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
    const std::vector<double>& u,
    const std::vector<double>& v,
    const std::vector<double>& w,
    const std::vector<double>& p,
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

static void write_vtu_polyhedron_cell_data(const std::string &filename,const Mesh &mesh,const std::vector<std::string> &scalarNames,const std::vector<std::vector<double>> &scalarData,const std::string &vecName,const std::vector<std::array<double,3>> *vecData){
  int nPts=(int)mesh.P.size(), nCells=mesh.nCells; std::vector<long long> connectivity, offsets(nCells), facesStream, faceoffsets(nCells); std::vector<int> types(nCells,42); long long connCount=0, faceCount=0;
  for(int c=0;c<nCells;++c){ const auto &fids=mesh.cellFaces[c]; const auto &oris=mesh.cellOrient[c]; std::vector<std::vector<int>> cellFacePts(fids.size()); std::vector<int> allPts; for(std::size_t j=0;j<fids.size();++j){ int f=fids[j]; cellFacePts[j]=mesh.faces[f]; if(oris[j]<0) std::reverse(cellFacePts[j].begin(),cellFacePts[j].end()); allPts.insert(allPts.end(),cellFacePts[j].begin(),cellFacePts[j].end()); } std::vector<int> uniqPts; std::set<int> seen; for(int p:allPts) if(seen.insert(p).second) uniqPts.push_back(p); for(int p:uniqPts) connectivity.push_back((long long)p); connCount += (long long)uniqPts.size(); offsets[c]=connCount; facesStream.push_back((long long)fids.size()); for(const auto &fv:cellFacePts){ facesStream.push_back((long long)fv.size()); for(int p:fv) facesStream.push_back((long long)p); } faceCount += 1; for(const auto &fv:cellFacePts) faceCount += 1 + (long long)fv.size(); faceoffsets[c]=faceCount; }
  std::ofstream fid(filename.c_str()); if(!fid) throw std::runtime_error("Could not open "+filename+" for writing");
  fid << "<?xml version=\"1.0\"?>\n<VTKFile type=\"UnstructuredGrid\" version=\"1.0\" byte_order=\"LittleEndian\">\n  <UnstructuredGrid>\n    <Piece NumberOfPoints=\""<<nPts<<"\" NumberOfCells=\""<<nCells<<"\">\n      <Points>\n        <DataArray type=\"Float64\" NumberOfComponents=\"3\" format=\"ascii\">\n"; for(const auto &p:mesh.P) fid<<"          "<<std::setprecision(15)<<p[0]<<" "<<p[1]<<" "<<p[2]<<"\n"; fid<<"        </DataArray>\n      </Points>\n      <Cells>\n        <DataArray type=\"Int64\" Name=\"connectivity\" format=\"ascii\">\n          "; for(auto v:connectivity) fid<<v<<" "; fid<<"\n        </DataArray>\n        <DataArray type=\"Int64\" Name=\"offsets\" format=\"ascii\">\n          "; for(auto v:offsets) fid<<v<<" "; fid<<"\n        </DataArray>\n        <DataArray type=\"UInt8\" Name=\"types\" format=\"ascii\">\n          "; for(auto v:types) fid<<v<<" "; fid<<"\n        </DataArray>\n        <DataArray type=\"Int64\" Name=\"faces\" format=\"ascii\">\n          "; for(auto v:facesStream) fid<<v<<" "; fid<<"\n        </DataArray>\n        <DataArray type=\"Int64\" Name=\"faceoffsets\" format=\"ascii\">\n          "; for(auto v:faceoffsets) fid<<v<<" "; fid<<"\n        </DataArray>\n      </Cells>\n      <CellData>\n";
  if(vecData){ fid<<"        <DataArray type=\"Float64\" Name=\""<<vecName<<"\" NumberOfComponents=\"3\" format=\"ascii\">\n"; for(const auto &v:*vecData) fid<<"          "<<std::setprecision(15)<<v[0]<<" "<<v[1]<<" "<<v[2]<<"\n"; fid<<"        </DataArray>\n"; }
  for(std::size_t k=0;k<scalarNames.size();++k){ fid<<"        <DataArray type=\"Float64\" Name=\""<<scalarNames[k]<<"\" NumberOfComponents=\"1\" format=\"ascii\">\n"; for(double v:scalarData[k]) fid<<"          "<<std::setprecision(15)<<v<<"\n"; fid<<"        </DataArray>\n"; }
  fid<<"      </CellData>\n    </Piece>\n  </UnstructuredGrid>\n</VTKFile>\n";
}

struct DeviceMesh {
  int nCells=0, nFaces=0, nInternalFaces=0;
  int *d_owner=nullptr, *d_neigh=nullptr, *d_bPatch=nullptr;
  double *d_ccx=nullptr, *d_ccy=nullptr, *d_ccz=nullptr;
  double *d_xfx=nullptr, *d_xfy=nullptr, *d_xfz=nullptr;
  double *d_nfx=nullptr, *d_nfy=nullptr, *d_nfz=nullptr;
  double *d_sfx=nullptr, *d_sfy=nullptr, *d_sfz=nullptr;
  double *d_Af=nullptr, *d_vol=nullptr;
};

struct DeviceGradientOperator {
  int nCells=0, nTerms=0;
  int *d_offsets=nullptr;
  int *d_src=nullptr;
  int *d_face=nullptr;
  double *d_cx=nullptr, *d_cy=nullptr, *d_cz=nullptr;
};

struct DeviceBC {
  int nPatch=0;
  std::vector<int> type;
  std::vector<double> faceValue;
  int *d_type=nullptr;
  double *d_faceValue=nullptr;
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
  bool is_setup=false;
};

struct GPUMomentumAssembler {
  GPULinearSystem lin;
  double *d_qOld=nullptr, *d_uConv=nullptr, *d_vConv=nullptr, *d_wConv=nullptr;
  double *d_gradQx=nullptr, *d_gradQy=nullptr, *d_gradQz=nullptr;
  double *d_gradPcomp=nullptr;
  double *d_rAU=nullptr;
};

struct GPUSimpleScratch {
  int nCells=0, nFaces=0;
  double *d_u=nullptr, *d_v=nullptr, *d_w=nullptr, *d_p=nullptr, *d_pCorr=nullptr;
  double *d_gradx=nullptr, *d_grady=nullptr, *d_gradz=nullptr;
  double *d_phiStar=nullptr, *d_phi=nullptr, *d_phiNonOrth=nullptr;
  double *d_divStar=nullptr, *d_divCorr=nullptr, *d_divNonOrth=nullptr;
  double *d_pCorrDelta=nullptr;
  double *d_reduce=nullptr;
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

static void build_device_mesh(const Mesh &mesh, DeviceMesh &dm){
  dm.nCells=mesh.nCells; dm.nFaces=mesh.nFaces; dm.nInternalFaces=mesh.nInternalFaces;
  std::vector<double> ccx(mesh.nCells), ccy(mesh.nCells), ccz(mesh.nCells), vol(mesh.nCells);
  std::vector<double> xfx(mesh.nFaces), xfy(mesh.nFaces), xfz(mesh.nFaces), nfx(mesh.nFaces), nfy(mesh.nFaces), nfz(mesh.nFaces), sfx(mesh.nFaces), sfy(mesh.nFaces), sfz(mesh.nFaces), Af(mesh.nFaces);
  for(int c=0;c<mesh.nCells;++c){ ccx[c]=mesh.cc[c][0]; ccy[c]=mesh.cc[c][1]; ccz[c]=mesh.cc[c][2]; vol[c]=mesh.vol[c]; }
  for(int f=0;f<mesh.nFaces;++f){ xfx[f]=mesh.xf[f][0]; xfy[f]=mesh.xf[f][1]; xfz[f]=mesh.xf[f][2]; nfx[f]=mesh.nf[f][0]; nfy[f]=mesh.nf[f][1]; nfz[f]=mesh.nf[f][2]; sfx[f]=mesh.Sf[f][0]; sfy[f]=mesh.Sf[f][1]; sfz[f]=mesh.Sf[f][2]; Af[f]=mesh.Af[f]; }
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
  device_free(dm.d_owner); device_free(dm.d_neigh); device_free(dm.d_bPatch);
  device_free(dm.d_ccx); device_free(dm.d_ccy); device_free(dm.d_ccz);
  device_free(dm.d_xfx); device_free(dm.d_xfy); device_free(dm.d_xfz);
  device_free(dm.d_nfx); device_free(dm.d_nfy); device_free(dm.d_nfz);
  device_free(dm.d_sfx); device_free(dm.d_sfy); device_free(dm.d_sfz);
  device_free(dm.d_Af); device_free(dm.d_vol);
  dm = DeviceMesh{};
}

static DeviceBC make_device_bc(int nFaces, const std::vector<std::string> &bcType, const std::vector<double> &bcFaceValue){
  DeviceBC bc; bc.nPatch=(int)bcType.size(); bc.type.resize(bc.nPatch); bc.faceValue = bcFaceValue;
  if((int)bc.faceValue.size() != nFaces) throw std::runtime_error("BC face-value size mismatch");
  for(int i=0;i<bc.nPatch;++i) bc.type[i]=bc_to_int(bcType[i]);
  device_alloc(bc.d_type, bc.type.size()); copy_vec_to_device(bc.type, bc.d_type);
  device_alloc(bc.d_faceValue, bc.faceValue.size()); copy_vec_to_device(bc.faceValue, bc.d_faceValue);
  return bc;
}
static void destroy_device_bc(DeviceBC &bc){ device_free(bc.d_type); device_free(bc.d_faceValue); bc=DeviceBC{}; }

static void init_simple_scratch(GPUSimpleScratch &ss, const Mesh &mesh){
  ss.nCells = mesh.nCells;
  ss.nFaces = mesh.nFaces;
  device_alloc(ss.d_u, mesh.nCells); device_alloc(ss.d_v, mesh.nCells); device_alloc(ss.d_w, mesh.nCells);
  device_alloc(ss.d_p, mesh.nCells); device_alloc(ss.d_pCorr, mesh.nCells);
  device_alloc(ss.d_gradx, mesh.nCells); device_alloc(ss.d_grady, mesh.nCells); device_alloc(ss.d_gradz, mesh.nCells);
  device_alloc(ss.d_phiStar, mesh.nFaces); device_alloc(ss.d_phi, mesh.nFaces); device_alloc(ss.d_phiNonOrth, mesh.nFaces);
  device_alloc(ss.d_divStar, mesh.nCells); device_alloc(ss.d_divCorr, mesh.nCells); device_alloc(ss.d_divNonOrth, mesh.nCells);
  device_alloc(ss.d_pCorrDelta, mesh.nCells);
  ss.reduceSize = std::max((mesh.nCells + 255)/256, 1);
  device_alloc(ss.d_reduce, ss.reduceSize);
}

static void destroy_simple_scratch(GPUSimpleScratch &ss){
  device_free(ss.d_u); device_free(ss.d_v); device_free(ss.d_w); device_free(ss.d_p); device_free(ss.d_pCorr);
  device_free(ss.d_gradx); device_free(ss.d_grady); device_free(ss.d_gradz);
  device_free(ss.d_phiStar); device_free(ss.d_phi); device_free(ss.d_phiNonOrth);
  device_free(ss.d_divStar); device_free(ss.d_divCorr); device_free(ss.d_divNonOrth);
  device_free(ss.d_pCorrDelta); device_free(ss.d_reduce);
  ss = GPUSimpleScratch{};
}

static void upload_gradient_to_device(const std::vector<std::array<double,3>> &grad,
                                      std::vector<double> &bufX, std::vector<double> &bufY, std::vector<double> &bufZ,
                                      double *d_gx, double *d_gy, double *d_gz){
  const int n = (int)grad.size();
  if((int)bufX.size() != n){ bufX.resize(n); bufY.resize(n); bufZ.resize(n); }
  for(int c=0;c<n;++c){ bufX[c]=grad[c][0]; bufY[c]=grad[c][1]; bufZ[c]=grad[c][2]; }
  copy_vec_to_device(bufX, d_gx);
  copy_vec_to_device(bufY, d_gy);
  copy_vec_to_device(bufZ, d_gz);
}

// Forward declarations for CUDA kernels used before their definitions.
__global__ static void kernel_zero_double(double *x, int n);

__global__ static void kernel_continuity_residual_from_flux(
    int nFaces, int nInternalFaces,
    const int *owner, const int *neigh,
    const double *phi, double *divCell);


static void build_lsq_gradient_operator(const Mesh &mesh, DeviceGradientOperator &op){
  op.nCells = mesh.nCells;
  std::vector<int> offsets(mesh.nCells + 1, 0), src, face;
  std::vector<double> cx, cy, cz;
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
      double w = 1.0/std::max(dot3(r,r),1e-30);
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
      double w = 1.0/std::max(dot3(r,r),1e-30);
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
__global__ static void kernel_zero_double(double *x, int n);

__global__ static void kernel_continuity_residual_from_flux(
    int nFaces, int nInternalFaces,
    const int *owner, const int *neigh,
    const double *phi, double *divCell);

__global__ static void kernel_apply_lsq_gradient(
    int nCells, const int *offsets, const int *src, const int *face,
    const double *cx, const double *cy, const double *cz,
    const double *phi, const int *bPatch, const int *bcType, const double *bcFaceValue,
    double *gx, double *gy, double *gz);

__global__ static void kernel_add_scaled_inplace(int n, double *y, const double *x, double a);
__global__ static void kernel_update_pressure_relax(int n, double *p, const double *pcorr, double pRelax);
__global__ static void kernel_maxabs_reduce(int n, const double *x, double *blockMax);

static void continuity_residual_gpu(const DeviceMesh &dm, const double *d_phi, double *d_div){
  const int block = 256;
  kernel_zero_double<<<(dm.nCells + block - 1)/block, block>>>(d_div, dm.nCells);
  CUDA_CHECK_LAST();
  kernel_continuity_residual_from_flux<<<(dm.nFaces + block - 1)/block, block>>>(
      dm.nFaces, dm.nInternalFaces, dm.d_owner, dm.d_neigh, d_phi, d_div);
  CUDA_CHECK_LAST();
}


static void compute_lsq_gradient_gpu(const DeviceGradientOperator &gop, const DeviceMesh &dm, const DeviceBC &bc,
                                     const double *d_phi, double *d_gx, double *d_gy, double *d_gz){
  const int block = 256;
  kernel_apply_lsq_gradient<<<(gop.nCells + block - 1)/block, block>>>(
      gop.nCells, gop.d_offsets, gop.d_src, gop.d_face,
      gop.d_cx, gop.d_cy, gop.d_cz,
      d_phi, dm.d_bPatch, bc.d_type, bc.d_faceValue,
      d_gx, d_gy, d_gz);
  CUDA_CHECK_LAST();
}

static double maxabs_device(const double *d_x, int n, double *d_reduce, int reduceSize){
  const int block = 256;
  kernel_maxabs_reduce<<<reduceSize, block, block*sizeof(double)>>>(n, d_x, d_reduce);
  CUDA_CHECK_LAST();

  std::vector<double> h_reduce(reduceSize);
  CUDA_CALL(cudaMemcpy(h_reduce.data(), d_reduce, reduceSize*sizeof(double), cudaMemcpyDeviceToHost));

  double m = 0.0;
  for(double v: h_reduce) m = std::max(m, std::fabs(v));
  return m;
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

static void upload_pattern(MatrixPattern &pat){
  device_alloc(pat.d_ncols, pat.ncols.size()); copy_vec_to_device(pat.ncols, pat.d_ncols);
  device_alloc(pat.d_rowOffsets, pat.rowOffsets.size()); copy_vec_to_device(pat.rowOffsets, pat.d_rowOffsets);
  device_alloc(pat.d_diagPos, pat.diagPos.size()); copy_vec_to_device(pat.diagPos, pat.d_diagPos);
  device_alloc(pat.d_rows, pat.rows.size()); copy_vec_to_device(pat.rows, pat.d_rows);
  device_alloc(pat.d_cols, pat.cols.size()); copy_vec_to_device(pat.cols, pat.d_cols);
  device_alloc(pat.d_facePP, pat.facePP.size()); copy_vec_to_device(pat.facePP, pat.d_facePP);
  device_alloc(pat.d_facePN, pat.facePN.size()); copy_vec_to_device(pat.facePN, pat.d_facePN);
  device_alloc(pat.d_faceNP, pat.faceNP.size()); copy_vec_to_device(pat.faceNP, pat.d_faceNP);
  device_alloc(pat.d_faceNN, pat.faceNN.size()); copy_vec_to_device(pat.faceNN, pat.d_faceNN);
  device_alloc(pat.d_vals, pat.nnz); CUDA_CALL(cudaMemset(pat.d_vals, 0, pat.nnz*sizeof(HYPRE_Complex)));
}

static void destroy_pattern(MatrixPattern &pat){
  device_free(pat.d_ncols); device_free(pat.d_rowOffsets); device_free(pat.d_diagPos);
  device_free(pat.d_rows); device_free(pat.d_cols);
  device_free(pat.d_facePP); device_free(pat.d_facePN); device_free(pat.d_faceNP); device_free(pat.d_faceNN);
  device_free(pat.d_vals);
  pat = MatrixPattern{};
}

__global__ static void kernel_zero_double(double *x, int n){
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if(i < n) x[i] = 0.0;
}


__global__ static void kernel_apply_lsq_gradient(
    int nCells, const int *offsets, const int *src, const int *face,
    const double *cx, const double *cy, const double *cz,
    const double *phi, const int *bPatch, const int *bcType, const double *bcFaceValue,
    double *gx, double *gy, double *gz)
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

__global__ static void kernel_add_scaled_inplace(int n, double *y, const double *x, double a){
  int i = blockIdx.x*blockDim.x + threadIdx.x;
  if(i<n) y[i] += a*x[i];
}

__global__ static void kernel_update_pressure_relax(int n, double *p, const double *pcorr, double pRelax){
  int i = blockIdx.x*blockDim.x + threadIdx.x;
  if(i<n) p[i] += pRelax*pcorr[i];
}

__global__ static void kernel_maxabs_reduce(int n, const double *x, double *blockMax){
  extern __shared__ double sdata[];

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
    const double *ccx, const double *ccy, const double *ccz,
    const double *xfx, const double *xfy, const double *xfz,
    const double *nfx, const double *nfy, const double *nfz,
    const double *Af,
    const double *u, const double *v, const double *w,
    const double *p,
    const double *gradx, const double *grady, const double *gradz,
    const double *rAU,
    const int *bcUType, const double *uFaceBC,
    const int *bcVType, const double *vFaceBC,
    const int *bcWType, const double *wFaceBC,
    double rho,
    double *phiStar)
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
    const double rc = rho * Af[f] * rAUf / fmax(fabs(dpn), 1.0e-30) *
                      ((p[N] - p[P]) - (gpx*dx + gpy*dy + gpz*dz));
    phiStar[f] = phiInterp - rc;
  } else {
    const int patch = bPatch[f] - 1;
    const double uf = (bcUType[patch] == 1) ? uFaceBC[f] : u[P];
    const double vf = (bcVType[patch] == 1) ? vFaceBC[f] : v[P];
    const double wf = (bcWType[patch] == 1) ? wFaceBC[f] : w[P];
    phiStar[f] = rho * Af[f] * (uf*nfx[f] + vf*nfy[f] + wf*nfz[f]);
  }
}

__global__ static void kernel_continuity_residual_from_flux(
    int nFaces, int nInternalFaces,
    const int *owner, const int *neigh,
    const double *phi, double *divCell)
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
    const double *ccx, const double *ccy, const double *ccz,
    const double *xfx, const double *xfy, const double *xfz,
    const double *nfx, const double *nfy, const double *nfz,
    const double *sfx, const double *sfy, const double *sfz,
    const double *Af, const double *rAU,
    const int *bcPType,
    const double *gradx, const double *grady, const double *gradz,
    double *phiNonOrth, double *divNonOrth)
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
    if(fabs(dpn) > 1.0e-30){
      double denom = fmax(dx*dx + dy*dy + dz*dz, 1.0e-30);
      double lam = ((xfx[f]-ccx[P])*dx + (xfy[f]-ccy[P])*dy + (xfz[f]-ccz[P])*dz) / denom;
      lam = fmin(1.0, fmax(0.0, lam));
      double rAUf = (1.0-lam)*rAU[P] + lam*rAU[N];
      double gx = (1.0-lam)*gradx[P] + lam*gradx[N];
      double gy = (1.0-lam)*grady[P] + lam*grady[N];
      double gz = (1.0-lam)*gradz[P] + lam*gradz[N];
      double tx = sfx[f] - (Af[f]/dpn)*dx;
      double ty = sfy[f] - (Af[f]/dpn)*dy;
      double tz = sfz[f] - (Af[f]/dpn)*dz;
      flux = rAUf * (gx*tx + gy*ty + gz*tz);
      atomicAdd(&divNonOrth[P], flux);
      atomicAdd(&divNonOrth[N], -flux);
    }
  } else {
    int patch = bPatch[f] - 1;
    if(patch >= 0 && bcPType[patch] == 1){
      int P = owner[f];
      double dx = xfx[f] - ccx[P];
      double dy = xfy[f] - ccy[P];
      double dz = xfz[f] - ccz[P];
      double dpn = nfx[f]*dx + nfy[f]*dy + nfz[f]*dz;
      if(fabs(dpn) > 1.0e-30){
        double tx = sfx[f] - (Af[f]/dpn)*dx;
        double ty = sfy[f] - (Af[f]/dpn)*dy;
        double tz = sfz[f] - (Af[f]/dpn)*dz;
        flux = rAU[P] * (gradx[P]*tx + grady[P]*ty + gradz[P]*tz);
        atomicAdd(&divNonOrth[P], flux);
      }
    }
  }
  phiNonOrth[f] = flux;
}

__global__ static void kernel_build_pressure_rhs_from_divs(
    int nCells, const double *divBase, const double *divNonOrth,
    int useAnchor, int refCell, HYPRE_Complex *rhs)
{
  int c = blockIdx.x * blockDim.x + threadIdx.x;
  if(c < nCells){
    rhs[c] = (useAnchor && c == refCell) ? (HYPRE_Complex)0.0 :
             (HYPRE_Complex)(-divBase[c] + divNonOrth[c]);
  }
}

__global__ static void kernel_build_pressure_rhs_minus_div(
    int nCells, const double *divBase, int useAnchor, int refCell, HYPRE_Complex *rhs)
{
  int c = blockIdx.x * blockDim.x + threadIdx.x;
  if(c < nCells){
    rhs[c] = (useAnchor && c == refCell) ? (HYPRE_Complex)0.0 : (HYPRE_Complex)(-divBase[c]);
  }
}

__global__ static void kernel_correct_face_fluxes_simple_nonorth(
    int nFaces, int nInternalFaces,
    const int *owner, const int *neigh, const int *bPatch,
    const double *ccx, const double *ccy, const double *ccz,
    const double *xfx, const double *xfy, const double *xfz,
    const double *nfx, const double *nfy, const double *nfz,
    const double *Af, const double *rAU,
    const int *bcPType, const double *pFaceBC,
    const double *phiStar, const double *pCorr, const double *phiNonOrth, double *phi)
{
  int f = blockIdx.x * blockDim.x + threadIdx.x;
  if(f >= nFaces) return;
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
    double coeff = Af[f] * rAUf / dpn;
    phi[f] = phiStar[f] - coeff*(pCorr[N] - pCorr[P]) - phiNonOrth[f];
  } else {
    int patch = bPatch[f] - 1;
    if(patch >= 0 && bcPType[patch] == 1){
      int P = owner[f];
      double dx = xfx[f] - ccx[P];
      double dy = xfy[f] - ccy[P];
      double dz = xfz[f] - ccz[P];
      double dpn = nfx[f]*dx + nfy[f]*dy + nfz[f]*dz;
      if(fabs(dpn) <= 1.0e-30){
        phi[f] = phiStar[f];
        return;
      }
      double coeff = Af[f] * rAU[P] / dpn;
      phi[f] = phiStar[f] - coeff*(pFaceBC[f] - pCorr[P]) - phiNonOrth[f];
    } else {
      phi[f] = phiStar[f];
    }
  }
}

__global__ static void kernel_correct_velocity_from_pcorr_grad(
    int nCells, const double *uStar, const double *vStar, const double *wStar,
    const double *rAU, const double *gradx, const double *grady, const double *gradz,
    double *u, double *v, double *w)
{
  int c = blockIdx.x * blockDim.x + threadIdx.x;
  if(c < nCells){
    u[c] = uStar[c] - rAU[c] * gradx[c];
    v[c] = vStar[c] - rAU[c] * grady[c];
    w[c] = wStar[c] - rAU[c] * gradz[c];
  }
}

__global__ static void kernel_zero_values(HYPRE_Complex *vals, int n){
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if(i < n) vals[i] = 0.0;
}

__global__ static void kernel_momentum_base_steady(int nCells, const int *diagPos, const double *vol, const double *gradPcomp, HYPRE_Complex *vals, HYPRE_Complex *rhs){
  int c = blockIdx.x * blockDim.x + threadIdx.x;
  if(c < nCells){
    vals[diagPos[c]] = 0.0;
    rhs[c] = -gradPcomp[c] * vol[c];
  }
}

__global__ static void kernel_relax_momentum_system(int nCells, const int *diagPos, HYPRE_Complex *vals, HYPRE_Complex *rhs, const double *qOld, double uRelax){
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

__global__ static void kernel_momentum_base_rhs_only(int nCells, const double *vol, const double *gradPcomp, HYPRE_Complex *rhs){
  int c = blockIdx.x * blockDim.x + threadIdx.x;
  if(c < nCells) rhs[c] = -gradPcomp[c] * vol[c];
}

__global__ static void kernel_momentum_internal_faces_rhs_only(
    int nInternalFaces,
    const int *owner, const int *neigh,
    const double *ccx, const double *ccy, const double *ccz,
    const double *xfx, const double *xfy, const double *xfz,
    const double *nfx, const double *nfy, const double *nfz,
    const double *sfx, const double *sfy, const double *sfz,
    const double *Af,
    const double *gradQx, const double *gradQy, const double *gradQz,
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
    const double *ccx, const double *ccy, const double *ccz,
    const double *xfx, const double *xfy, const double *xfz,
    const double *nfx, const double *nfy, const double *nfz,
    const double *sfx, const double *sfy, const double *sfz,
    const double *Af,
    const double *gradQx, const double *gradQy, const double *gradQz,
    const double *uConv, const double *vConv, const double *wConv,
    const int *bcQType, const double *bcQFaceVal,
    const int *bcUType, const double *bcUFaceVal,
    const int *bcVType, const double *bcVFaceVal,
    const int *bcWType, const double *bcWFaceVal,
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

__global__ static void kernel_relax_momentum_rhs_only_from_relaxed_matrix(int nCells, const int *diagPos, const HYPRE_Complex *vals, HYPRE_Complex *rhs, const double *qOld, double uRelax){
  int c = blockIdx.x * blockDim.x + threadIdx.x;
  if(c < nCells){
    double aP_old = ((double)vals[diagPos[c]]) * uRelax;
    double invRelax = 1.0 / uRelax;
    rhs[c] += (HYPRE_Complex)((invRelax - 1.0) * aP_old * qOld[c]);
  }
}

__global__ static void kernel_extract_vol_over_diag(int nCells, const int *diagPos, const HYPRE_Complex *vals, const double *vol, double *rAU){
  int c = blockIdx.x * blockDim.x + threadIdx.x;
  if(c < nCells){
    double a = (double) vals[diagPos[c]];
    rAU[c] = (fabs(a) > 1e-30) ? vol[c] / a : 0.0;
  }
}

__global__ static void kernel_momentum_internal_faces(
    int nInternalFaces,
    const int *owner, const int *neigh,
    const double *ccx, const double *ccy, const double *ccz,
    const double *xfx, const double *xfy, const double *xfz,
    const double *nfx, const double *nfy, const double *nfz,
    const double *sfx, const double *sfy, const double *sfz,
    const double *Af,
    const double *gradQx, const double *gradQy, const double *gradQz,
    const double *uConv, const double *vConv, const double *wConv,
    double rho, double mu, double corrPsi,
    const int *facePP, const int *facePN, const int *faceNP, const int *faceNN,
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
  hypreAtomicAdd(&vals[facePP[f]], (HYPRE_Complex)(alpha + F*(1.0-lam)));
  hypreAtomicAdd(&vals[facePN[f]], (HYPRE_Complex)(-alpha + F*lam));
  hypreAtomicAdd(&vals[faceNP[f]], (HYPRE_Complex)(-alpha - F*(1.0-lam)));
  hypreAtomicAdd(&vals[faceNN[f]], (HYPRE_Complex)(alpha - F*lam));
  hypreAtomicAdd(&rhs[P], (HYPRE_Complex)corr);
  hypreAtomicAdd(&rhs[N], (HYPRE_Complex)(-corr));
}

__global__ static void kernel_momentum_boundary_faces(
    int nBoundaryFaces, int faceStart,
    const int *owner, const int *bPatch,
    const double *ccx, const double *ccy, const double *ccz,
    const double *xfx, const double *xfy, const double *xfz,
    const double *nfx, const double *nfy, const double *nfz,
    const double *sfx, const double *sfy, const double *sfz,
    const double *Af,
    const double *gradQx, const double *gradQy, const double *gradQz,
    const double *uConv, const double *vConv, const double *wConv,
    const int *bcQType, const double *bcQFaceVal,
    const int *bcUType, const double *bcUFaceVal,
    const int *bcVType, const double *bcVFaceVal,
    const int *bcWType, const double *bcWFaceVal,
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
    const double *ccx, const double *ccy, const double *ccz,
    const double *xfx, const double *xfy, const double *xfz,
    const double *nfx, const double *nfy, const double *nfz,
    const double *Af, const double *rAU,
    const int *facePP, const int *facePN, const int *faceNP, const int *faceNN,
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
  if(dpn <= 1e-14) return;
  double denom = dx*dx + dy*dy + dz*dz;
  double lam = ((xfx[f]-ccx[P])*dx + (xfy[f]-ccy[P])*dy + (xfz[f]-ccz[P])*dz) / (denom > 1e-30 ? denom : 1e-30);
  lam = fmin(1.0, fmax(0.0, lam));
  double rAUf = (1.0-lam)*rAU[P] + lam*rAU[N];
  double coeff = Af[f] * rAUf / dpn;
  if(facePP[f] >= 0) hypreAtomicAdd(&vals[facePP[f]], (HYPRE_Complex)coeff);
  if(facePN[f] >= 0) hypreAtomicAdd(&vals[facePN[f]], (HYPRE_Complex)(-coeff));
  if(faceNP[f] >= 0) hypreAtomicAdd(&vals[faceNP[f]], (HYPRE_Complex)(-coeff));
  if(faceNN[f] >= 0) hypreAtomicAdd(&vals[faceNN[f]], (HYPRE_Complex)coeff);
}

__global__ static void kernel_pressure_boundary_faces_rau(
    int nBoundaryFaces, int faceStart,
    const int *owner, const int *bPatch,
    const double *ccx, const double *ccy, const double *ccz,
    const double *xfx, const double *xfy, const double *xfz,
    const double *nfx, const double *nfy, const double *nfz,
    const double *Af, const double *rAU,
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
  if(dpn <= 1e-14) return;
  double coeff = Af[f] * rAU[P] / dpn;
  hypreAtomicAdd(&vals[diagPos[P]], (HYPRE_Complex)coeff);
}

__global__ static void kernel_pressure_anchor(int refCell, const int *diagPos, HYPRE_Complex *vals){
  vals[diagPos[refCell]] = 1.0;
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

static void get_ij_vector_to_host(HYPRE_IJVector vij, std::vector<double> &x){
  HYPRE_CALL(HYPRE_IJVectorMigrate(vij, HYPRE_MEMORY_HOST));
  std::vector<HYPRE_BigInt> idx(x.size());
  for(std::size_t i=0;i<x.size();++i) idx[i]=(HYPRE_BigInt)i;
  HYPRE_CALL(HYPRE_IJVectorGetValues(vij, (HYPRE_Int)x.size(), idx.data(), x.data()));
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

static void copy_device_rhs_and_host_x0_into_hypre(GPULinearSystem &sys, const std::vector<double> &x0){
  std::vector<HYPRE_Complex> h_x0(sys.n);
  for(int i=0;i<sys.n;++i) h_x0[i] = (HYPRE_Complex)x0[i];
  CUDA_CALL(cudaMemcpy(sys.b_data_dev, sys.d_rhs, sys.n * sizeof(HYPRE_Complex), cudaMemcpyDeviceToDevice));
  CUDA_CALL(cudaMemcpy(sys.x_data_dev, h_x0.data(), sys.n * sizeof(HYPRE_Complex), cudaMemcpyHostToDevice));
}

static void copy_device_rhs_and_device_x0_into_hypre(GPULinearSystem &sys, const double *d_x0){
  CUDA_CALL(cudaMemcpy(sys.b_data_dev, sys.d_rhs, sys.n * sizeof(HYPRE_Complex), cudaMemcpyDeviceToDevice));
  CUDA_CALL(cudaMemcpy(sys.x_data_dev, d_x0, sys.n * sizeof(HYPRE_Complex), cudaMemcpyDeviceToDevice));
}

static void copy_solution_from_hypre_to_device(GPULinearSystem &sys, double *d_xout){
  CUDA_CALL(cudaMemcpy(d_xout, sys.x_data_dev, sys.n * sizeof(HYPRE_Complex), cudaMemcpyDeviceToDevice));
}

static void copy_host_rhs_and_host_x0_into_hypre(GPULinearSystem &sys, const std::vector<double> &rhs, const std::vector<double> &x0){
  std::vector<HYPRE_Complex> h_rhs(sys.n), h_x0(sys.n);
  for(int i=0;i<sys.n;++i){ h_rhs[i] = (HYPRE_Complex)rhs[i]; h_x0[i] = (HYPRE_Complex)x0[i]; }
  CUDA_CALL(cudaMemcpy(sys.b_data_dev, h_rhs.data(), sys.n * sizeof(HYPRE_Complex), cudaMemcpyHostToDevice));
  CUDA_CALL(cudaMemcpy(sys.x_data_dev, h_x0.data(), sys.n * sizeof(HYPRE_Complex), cudaMemcpyHostToDevice));
}

static void copy_solution_from_hypre(GPULinearSystem &sys, std::vector<double> &xout){
  std::vector<HYPRE_Complex> h_x(sys.n);
  CUDA_CALL(cudaMemcpy(h_x.data(), sys.x_data_dev, sys.n * sizeof(HYPRE_Complex), cudaMemcpyDeviceToHost));
  xout.assign(sys.n, 0.0);
  for(int i=0;i<sys.n;++i) xout[i] = (double) h_x[i];
}

static void init_common_linear_storage(GPULinearSystem &sys, MatrixPattern pat){
  sys.n = pat.nRows;
  sys.pat = std::move(pat);
  upload_pattern(sys.pat);
  sys.h_idx.resize(sys.n);
  for(int i=0;i<sys.n;++i) sys.h_idx[i]=(HYPRE_BigInt)i;
  device_alloc(sys.d_rhs, sys.n); CUDA_CALL(cudaMemset(sys.d_rhs, 0, sys.n*sizeof(HYPRE_Complex)));
  device_alloc(sys.d_x, sys.n); CUDA_CALL(cudaMemset(sys.d_x, 0, sys.n*sizeof(HYPRE_Complex)));
  create_ij_matrix_from_pattern(sys.pat, sys.Aij);
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

static void init_momentum_system(GPUMomentumAssembler &mom, const Mesh &mesh){
  init_common_linear_storage(mom.lin, build_momentum_pattern(mesh));
  device_alloc(mom.d_qOld, mesh.nCells);
  device_alloc(mom.d_uConv, mesh.nCells);
  device_alloc(mom.d_vConv, mesh.nCells);
  device_alloc(mom.d_wConv, mesh.nCells);
  device_alloc(mom.d_gradQx, mesh.nCells);
  device_alloc(mom.d_gradQy, mesh.nCells);
  device_alloc(mom.d_gradQz, mesh.nCells);
  device_alloc(mom.d_gradPcomp, mesh.nCells);
  device_alloc(mom.d_rAU, mesh.nCells);
  HYPRE_CALL(HYPRE_ParCSRBiCGSTABCreate(MPI_COMM_WORLD, &mom.lin.solver));
  mom.lin.isPCG = false;
  CUDA_CALL(cudaMemset(mom.lin.pat.d_vals, 0, mom.lin.pat.nnz * sizeof(HYPRE_Complex)));
  update_ij_matrix_from_device(mom.lin);
  cache_parcsr_diag_pointer(mom.lin);
  enable_direct_hypre_matrix_updates(mom.lin);
  init_reusable_device_vectors(mom.lin);
  mom.lin.is_setup = false;
}

static void destroy_momentum_system(GPUMomentumAssembler &mom){
  device_free(mom.d_qOld); device_free(mom.d_uConv); device_free(mom.d_vConv); device_free(mom.d_wConv);
  device_free(mom.d_gradQx); device_free(mom.d_gradQy); device_free(mom.d_gradQz); device_free(mom.d_gradPcomp); device_free(mom.d_rAU);
  destroy_linear_storage(mom.lin);
  mom = GPUMomentumAssembler{};
}

static void init_pressure_system(GPULinearSystem &ps, const Mesh &mesh, const DeviceMesh &dm, const Params &par, int refCell, bool usePressureAnchor, double &tsetup){
  init_common_linear_storage(ps, build_pressure_pattern(mesh, refCell, usePressureAnchor));
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
    const std::vector<double> &qOld, const std::vector<double> &uConv, const std::vector<double> &vConv, const std::vector<double> &wConv,
    const std::vector<std::array<double,3>> &gradQ, const std::vector<double> &gradPcomp,
    const DeviceBC &bcQ, const DeviceBC &bcU, const DeviceBC &bcV, const DeviceBC &bcW,
    double corrPsi)
{
  std::vector<double> gradQx(mesh.nCells), gradQy(mesh.nCells), gradQz(mesh.nCells);
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
  kernel_momentum_internal_faces<<<gridFaces, block>>>(mesh.nInternalFaces, dm.d_owner, dm.d_neigh, dm.d_ccx, dm.d_ccy, dm.d_ccz, dm.d_xfx, dm.d_xfy, dm.d_xfz, dm.d_nfx, dm.d_nfy, dm.d_nfz, dm.d_sfx, dm.d_sfy, dm.d_sfz, dm.d_Af, mom.d_gradQx, mom.d_gradQy, mom.d_gradQz, mom.d_uConv, mom.d_vConv, mom.d_wConv, rho, mu, corrPsi, mom.lin.pat.d_facePP, mom.lin.pat.d_facePN, mom.lin.pat.d_faceNP, mom.lin.pat.d_faceNN, Avals, mom.lin.d_rhs);
  kernel_momentum_boundary_faces<<<gridBFaces, block>>>(nBoundaryFaces, mesh.nInternalFaces, dm.d_owner, dm.d_bPatch, dm.d_ccx, dm.d_ccy, dm.d_ccz, dm.d_xfx, dm.d_xfy, dm.d_xfz, dm.d_nfx, dm.d_nfy, dm.d_nfz, dm.d_sfx, dm.d_sfy, dm.d_sfz, dm.d_Af, mom.d_gradQx, mom.d_gradQy, mom.d_gradQz, mom.d_uConv, mom.d_vConv, mom.d_wConv, bcQ.d_type, bcQ.d_faceValue, bcU.d_type, bcU.d_faceValue, bcV.d_type, bcV.d_faceValue, bcW.d_type, bcW.d_faceValue, rho, mu, corrPsi, mom.lin.pat.d_diagPos, Avals, mom.lin.d_rhs);
  CUDA_CHECK_LAST();
}

static void assemble_momentum_rhs_only_on_gpu(
    const DeviceMesh &dm, const Mesh &mesh, GPUMomentumAssembler &mom,
    double rho, double mu,
    const std::vector<double> &qOld, const std::vector<double> &uConv, const std::vector<double> &vConv, const std::vector<double> &wConv,
    const std::vector<std::array<double,3>> &gradQ, const std::vector<double> &gradPcomp,
    const DeviceBC &bcQ, const DeviceBC &bcU, const DeviceBC &bcV, const DeviceBC &bcW,
    double corrPsi, double uRelax)
{
  std::vector<double> gradQx(mesh.nCells), gradQy(mesh.nCells), gradQz(mesh.nCells);
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
    const std::vector<double> &qOld,
    const std::vector<double> &uConv,
    const std::vector<double> &vConv,
    const std::vector<double> &wConv,
    const double *d_gradQx,
    const double *d_gradQy,
    const double *d_gradQz,
    const double *d_gradPcomp,
    const DeviceBC &bcQ,
    const DeviceBC &bcU,
    const DeviceBC &bcV,
    const DeviceBC &bcW,
    double corrPsi)
{
  copy_vec_to_device(qOld, mom.d_qOld);
  copy_vec_to_device(uConv, mom.d_uConv);
  copy_vec_to_device(vConv, mom.d_vConv);
  copy_vec_to_device(wConv, mom.d_wConv);

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
      mom.d_uConv, mom.d_vConv, mom.d_wConv,
      rho, mu, corrPsi,
      mom.lin.pat.d_facePP,
      mom.lin.pat.d_facePN,
      mom.lin.pat.d_faceNP,
      mom.lin.pat.d_faceNN,
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
      mom.d_uConv, mom.d_vConv, mom.d_wConv,
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
    const std::vector<double> &qOld,
    const std::vector<double> &uConv,
    const std::vector<double> &vConv,
    const std::vector<double> &wConv,
    const double *d_gradQx,
    const double *d_gradQy,
    const double *d_gradQz,
    const double *d_gradPcomp,
    const DeviceBC &bcQ,
    const DeviceBC &bcU,
    const DeviceBC &bcV,
    const DeviceBC &bcW,
    double corrPsi,
    double uRelax)
{
  copy_vec_to_device(qOld, mom.d_qOld);
  copy_vec_to_device(uConv, mom.d_uConv);
  copy_vec_to_device(vConv, mom.d_vConv);
  copy_vec_to_device(wConv, mom.d_wConv);

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
      mom.d_uConv, mom.d_vConv, mom.d_wConv,
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
        mom.d_qOld,
        uRelax);
  }

  CUDA_CHECK_LAST();
}

static void extract_rAU_from_momentum_matrix(const Mesh &mesh, const DeviceMesh &dm, GPUMomentumAssembler &mom, std::vector<double> &rAU_host){
  int block=256;
  int gridCells=(mesh.nCells + block - 1)/block;
  HYPRE_Complex *Avals = matrix_values_ptr(mom.lin);
  kernel_extract_vol_over_diag<<<gridCells, block>>>(mesh.nCells, mom.lin.pat.d_diagPos, Avals, dm.d_vol, mom.d_rAU);
  CUDA_CHECK_LAST();
  rAU_host.clear(); // device-resident rAU is used by pressure and Rhie-Chow; host copy is not needed in optimized path.
}

static void relax_momentum_system_on_gpu(const Mesh &mesh, GPUMomentumAssembler &mom, double uRelax){
  if(uRelax >= 0.999999) return;
  int block=256;
  int gridCells=(mesh.nCells + block - 1)/block;
  HYPRE_Complex *Avals = matrix_values_ptr(mom.lin);
  kernel_relax_momentum_system<<<gridCells, block>>>(mesh.nCells, mom.lin.pat.d_diagPos, Avals, mom.lin.d_rhs, mom.d_qOld, uRelax);
  CUDA_CHECK_LAST();
}

static void pressure_solver_setup(GPULinearSystem &ps);

static void update_pressure_matrix_from_rAU(const Mesh &mesh, const DeviceMesh &dm, GPULinearSystem &ps, const DeviceBC &bcP, const double *d_rAU, int refCell, bool usePressureAnchor, bool doSetup, double &tsetup){
  int block=256;
  int gridVals=(ps.pat.nnz + block - 1)/block;
  int gridFaces=(mesh.nInternalFaces + block - 1)/block;
  int nBoundaryFaces = mesh.nFaces - mesh.nInternalFaces;
  int gridBFaces=(nBoundaryFaces + block - 1)/block;
  HYPRE_Complex *Avals = matrix_values_ptr(ps);
  double t0 = MPI_Wtime();
  kernel_zero_values<<<gridVals, block>>>(Avals, ps.pat.nnz);
  kernel_pressure_internal_faces_rau<<<gridFaces, block>>>(mesh.nInternalFaces, dm.d_owner, dm.d_neigh, dm.d_ccx, dm.d_ccy, dm.d_ccz, dm.d_xfx, dm.d_xfy, dm.d_xfz, dm.d_nfx, dm.d_nfy, dm.d_nfz, dm.d_Af, d_rAU, ps.pat.d_facePP, ps.pat.d_facePN, ps.pat.d_faceNP, ps.pat.d_faceNN, Avals);
  kernel_pressure_boundary_faces_rau<<<gridBFaces, block>>>(nBoundaryFaces, mesh.nInternalFaces, dm.d_owner, dm.d_bPatch, dm.d_ccx, dm.d_ccy, dm.d_ccz, dm.d_xfx, dm.d_xfy, dm.d_xfz, dm.d_nfx, dm.d_nfy, dm.d_nfz, dm.d_Af, d_rAU, bcP.d_type, ps.pat.d_diagPos, Avals);
  if(usePressureAnchor) kernel_pressure_anchor<<<1,1>>>(refCell, ps.pat.d_diagPos, Avals);
  CUDA_CHECK_LAST();
  if(doSetup){
    HYPRE_CALL(HYPRE_ParCSRPCGSetup(ps.solver, ps.Apar, ps.bpar, ps.xpar));
    ps.is_setup = true;
    CUDA_CALL(cudaDeviceSynchronize());
  }
  tsetup += MPI_Wtime() - t0;
}

static void solve_momentum_gpu(GPUMomentumAssembler &mom, const Params &par, const std::vector<double> &x0, std::vector<double> &xout, HYPRE_Int &its, double &relres, double &tsetup, double &tsolve, bool doMatrixSetup){
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
  HYPRE_CALL(HYPRE_ParCSRBiCGSTABSolve(mom.lin.solver, mom.lin.Apar, mom.lin.bpar, mom.lin.xpar));
  tsolve += MPI_Wtime()-t0;
  its=0; relres=0.0;
  HYPRE_CALL(HYPRE_ParCSRBiCGSTABGetNumIterations(mom.lin.solver,&its));
  HYPRE_CALL(HYPRE_ParCSRBiCGSTABGetFinalRelativeResidualNorm(mom.lin.solver,&relres));
  copy_solution_from_hypre(mom.lin, xout);
}

static void solve_pressure_gpu(GPULinearSystem &ps, const std::vector<double> &rhs, const std::vector<double> &x0, double relTol, double absTol, int maxIter, std::vector<double> &xout, HYPRE_Int &its, double &relres, double &tsolve){
  copy_host_rhs_and_host_x0_into_hypre(ps, rhs, x0);
  HYPRE_CALL(HYPRE_ParCSRPCGSetTol(ps.solver, relTol));
  HYPRE_CALL(HYPRE_ParCSRPCGSetAbsoluteTol(ps.solver, absTol));
  HYPRE_CALL(HYPRE_ParCSRPCGSetMaxIter(ps.solver, maxIter));
  double t0=MPI_Wtime();
  HYPRE_CALL(HYPRE_ParCSRPCGSolve(ps.solver, ps.Apar, ps.bpar, ps.xpar));
  tsolve += MPI_Wtime()-t0;
  its=0; relres=0.0;
  HYPRE_CALL(HYPRE_ParCSRPCGGetNumIterations(ps.solver,&its));
  HYPRE_CALL(HYPRE_ParCSRPCGGetFinalRelativeResidualNorm(ps.solver,&relres));
  copy_solution_from_hypre(ps, xout);
}

static void solve_pressure_gpu_device_rhs(GPULinearSystem &ps, const std::vector<double> &x0, double relTol, double absTol, int maxIter, std::vector<double> &xout, HYPRE_Int &its, double &relres, double &tsolve){
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
  HYPRE_CALL(HYPRE_ParCSRPCGGetFinalRelativeResidualNorm(ps.solver,&relres));
  copy_solution_from_hypre(ps, xout);
}


static void pressure_solver_setup(GPULinearSystem &ps){
  HYPRE_CALL(HYPRE_ParCSRPCGSetup(ps.solver, ps.Apar, ps.bpar, ps.xpar));
  ps.is_setup = true;
}

static void solve_pressure_gpu_device_rhs_device_x0(
    GPULinearSystem &ps,
    const double *d_x0,
    double *d_xout,
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
  double relTmp = 0.0;

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
    const std::vector<double>& u,
    const std::vector<double>& v,
    const std::vector<double>& w,
    const std::vector<double>& p,
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
    const std::vector<double>& u,
    const std::vector<double>& v,
    const std::vector<double>& w,
    const std::vector<double>& p,
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
  g_profile_enabled = (par.profileSteps > 0);
  g_p_amg_setup_scope = par.pAmgSetupScope;
  CUDA_CALL(cudaSetDevice(par.device));
  CUDA_CALL(cudaFree(0));
  if(rank==0) print_device_info(par.device);

  Mesh mesh=read_openfoam_polymesh(par.polyMeshDir);
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
  if(wallPatch<0 || inletPatch<0 || outletPatch<0){ if(rank==0) std::fprintf(stderr,"Could not find wall/inlet/outlet patch.\n"); MPI_Abort(MPI_COMM_WORLD,1); }
  double mu = par.muExplicit ? par.mu : (par.rho*par.Umean*par.pipeDiameter/par.Re);
  double hChar=1e300; for(double v:mesh.vol) hChar=std::min(hChar,std::cbrt(v));
  double dt0=par.CFL*hChar/std::max(par.Umean,1e-12);

  if(rank==0){
    std::printf("============================================================\n");
    std::printf("generic_simple_v1: OpenFOAM polyMesh SIMPLE solver\n");
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
    std::printf("Momentum solve : ParCSR BiCGSTAB + DiagScale\n");
    std::printf("Pressure solve : ParCSR PCG + %s\n", par.p_use_amg ? "BoomerAMG" : "DiagScale");
    std::printf("rho            : %.8g\n", par.rho);
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
    if(par.profileSteps>0) std::printf("profileSteps   : %d\n", par.profileSteps);
    std::printf("velTol / velRelTol : %.3e / %.3e\n", par.velTol, par.velRelTol);
    std::printf("pTol   / pRelTol   : %.3e / %.3e\n", par.pTol, par.pRelTol);
    std::printf("pAmgRebuildEvery   : %d\n", par.pAmgRebuildEvery);
    std::printf("------------------------------------------------------------\n");
  }

  HYPRE_CALL(HYPRE_Initialize());
  HYPRE_CALL(HYPRE_SetMemoryLocation(HYPRE_MEMORY_DEVICE));
  HYPRE_CALL(HYPRE_SetExecutionPolicy(HYPRE_EXEC_DEVICE));

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

  // generic_simple_v1 rule:
  // all physical BCs must come from the runtime/generated BC config.
  // This avoids silent fallback/default BCs on newly added patches.
  if(par.bcConfigPath.empty()){
    if(rank==0){
      std::fprintf(stderr,
          "generic_simple_v1 requires explicit runtime BCs.\n"
          "Add velocity/pressure lines to the case file, or set bcConfig.\n");
    }
    MPI_Abort(MPI_COMM_WORLD, 1);
  }

  if(!par.bcConfigPath.empty()){
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
          "\nEvery polyMesh boundary patch must have exactly one velocity BC and one pressure BC in generic_simple_v1.\n");
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
    std::printf("BC config source         : %s\n", par.bcConfigPath.empty() ? "<hardcoded-fallback>" : par.bcConfigPath.c_str());
  }

  std::vector<std::string> bcPType(mesh.patchNames.size(),"Neumann");
  std::vector<std::string> bcUType(mesh.patchNames.size(),"Neumann");
  std::vector<std::string> bcVType(mesh.patchNames.size(),"Neumann");
  std::vector<std::string> bcWType(mesh.patchNames.size(),"Neumann");

  std::vector<double> pFaceBC(mesh.nFaces, 0.0), uFaceBC(mesh.nFaces, 0.0), vFaceBC(mesh.nFaces, 0.0), wFaceBC(mesh.nFaces, 0.0);

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
      uFaceBC,
      vFaceBC,
      wFaceBC,
      pFaceBC);

  const bool usePressureAnchor = false;

  DeviceMesh dmesh; build_device_mesh(mesh, dmesh);
  DeviceBC dbcP = make_device_bc(mesh.nFaces, bcPType, pFaceBC);
  DeviceBC dbcU = make_device_bc(mesh.nFaces, bcUType, uFaceBC);
  DeviceBC dbcV = make_device_bc(mesh.nFaces, bcVType, vFaceBC);
  DeviceBC dbcW = make_device_bc(mesh.nFaces, bcWType, wFaceBC);

  std::vector<double> u(mesh.nCells,0.0), v(mesh.nCells,0.0), w(mesh.nCells,0.0), p(mesh.nCells,0.0);
  double initU = 0.0, initV = 0.0, initW = 0.0;
  int initCount = 0;
  for(int f=mesh.nInternalFaces; f<mesh.nFaces; ++f){
    if(mesh.bPatch[f]-1 == inletPatch){
      initU += uFaceBC[f];
      initV += vFaceBC[f];
      initW += wFaceBC[f];
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
  std::vector<double> uOld(mesh.nCells), vOld(mesh.nCells), wOld(mesh.nCells), pOld(mesh.nCells);
  std::vector<double> uStar(mesh.nCells), vStar(mesh.nCells), wStar(mesh.nCells), pCorr(mesh.nCells,0.0);
  std::vector<double> gradPcompX(mesh.nCells), gradPcompY(mesh.nCells), gradPcompZ(mesh.nCells);
  std::vector<std::array<double,3>> gradVec;
  std::vector<double> phiStar, phi, divStar, divCorr(mesh.nCells, 0.0);
  int refCell=0;
  double totalAssemble=0.0, totalSetup=0.0, totalSolve=0.0;
  double pressureSetup=0.0, pressureSolve=0.0;
  
  GPUMomentumAssembler mom;
  init_momentum_system(mom, mesh);
  GPULinearSystem pressureSys;
  init_pressure_system(pressureSys, mesh, dmesh, par, refCell, usePressureAnchor, pressureSetup);
  GPUSimpleScratch ss;
  init_simple_scratch(ss, mesh);
  DeviceGradientOperator gop;
  build_lsq_gradient_operator(mesh, gop);

  int stepConverged=0;
  double massRes=0.0, duRel=0.0, dvRel=0.0, dwRel=0.0, dpRel=0.0;
  HYPRE_Int lastItsU=0,lastItsV=0,lastItsW=0,lastItsP=0;
  double lastRelU=0.0,lastRelV=0.0,lastRelW=0.0,lastRelP=0.0;
  int corrUsedU=1,corrUsedV=1,corrUsedW=1;
  std::vector<double> rAU;
  std::array<PhaseStats, PH_COUNT> prof{};
  int profStepsDone = 0;

  auto solve_scalar_component_with_nonorth = [&](const std::vector<double> &qOld, const std::vector<double> &uConv, const std::vector<double> &vConv, const std::vector<double> &wConv, const double *d_gradPcomp, const std::vector<std::string> &bcQType, const std::vector<double> &bcQFaceVal, const DeviceBC &dbcQ, std::vector<double> &qOut, HYPRE_Int &itsOut, double &relOut, int &corrUsedOut, bool rebuildMomentumMatrix, bool extractRAU, PhaseStats &gradStats, PhaseStats &asmStats, PhaseStats &solveStats){
    std::vector<double> qIter = qOld, qNew = qOld;
    std::vector<std::array<double,3>> gradQ;
    const int nVelSolves = std::max(par.nVelNonOrthCorr, 0) + 1;
    corrUsedOut = 0;
    for(int it=1; it<=nVelSolves; ++it){
      corrUsedOut = it - 1;
      PhaseMark pm_grad = profile_begin();
      copy_vec_to_device(qIter, mom.d_qOld);
      compute_lsq_gradient_gpu(gop, dmesh, dbcQ, mom.d_qOld,
                               mom.d_gradQx, mom.d_gradQy, mom.d_gradQz);
      profile_record(gradStats, pm_grad);
      const bool doMatrixSetup = rebuildMomentumMatrix && (it == 1);
      PhaseMark pm_asm = profile_begin();
      if(doMatrixSetup){
        assemble_momentum_on_gpu_device_grad(dmesh, mesh, mom, par.rho, mu, 1.0, qIter, uConv, vConv, wConv, mom.d_gradQx, mom.d_gradQy, mom.d_gradQz, d_gradPcomp, dbcQ, dbcU, dbcV, dbcW, 1.0);
        relax_momentum_system_on_gpu(mesh, mom, par.uRelax);
        if(extractRAU) extract_rAU_from_momentum_matrix(mesh, dmesh, mom, rAU);
      } else {
        assemble_momentum_rhs_only_on_gpu_device_grad(dmesh, mesh, mom, par.rho, mu, qIter, uConv, vConv, wConv, mom.d_gradQx, mom.d_gradQy, mom.d_gradQz, d_gradPcomp, dbcQ, dbcU, dbcV, dbcW, 1.0, par.uRelax);
      }
      double dtAsm = MPI_Wtime()-pm_asm.t0;
      totalAssemble += dtAsm;
      profile_record(asmStats, pm_asm);
      PhaseMark pm_solve = profile_begin();
      solve_momentum_gpu(mom, par, qIter, qNew, itsOut, relOut, totalSetup, totalSolve, doMatrixSetup);
      profile_record(solveStats, pm_solve);
      double num=0.0, den=1.0;
      for(int c=0;c<mesh.nCells;++c){ double d=qNew[c]-qIter[c]; num += d*d; den += qNew[c]*qNew[c]; }
      double relchg = std::sqrt(num/den);
      qIter.swap(qNew);
      if(relchg < par.corrTol) break;
    }
    qOut = qIter;
  };

  double runStart = MPI_Wtime();
  int maxSteps = (par.profileSteps>0 ? std::min(par.nsteps, par.profileSteps) : par.nsteps);
  for(int step=1; step<=maxSteps; ++step){
    double iterStart = MPI_Wtime();
    uOld=u; vOld=v; wOld=w; pOld=p;
    std::fill(pCorr.begin(), pCorr.end(), 0.0);

    PhaseMark pm_pgrad = profile_begin();
    copy_vec_to_device(pOld, ss.d_p);
    compute_lsq_gradient_gpu(gop, dmesh, dbcP, ss.d_p,
                             ss.d_gradx, ss.d_grady, ss.d_gradz);
    profile_record(prof[PH_PGRAD], pm_pgrad);

    auto uConv = uOld, vConv = vOld, wConv = wOld;

    // The scalar momentum matrix is the same for Ux, Uy, Uz for this segregated equation.
    // Build/copy/setup it once on Ux, then rebuild only the RHS for Uy/Uz and reuse the same BiCGSTAB setup.
    solve_scalar_component_with_nonorth(uOld, uConv, vConv, wConv, ss.d_gradx, bcUType, uFaceBC, dbcU, uStar, lastItsU, lastRelU, corrUsedU, true,  true,  prof[PH_UGRAD], prof[PH_UASM], prof[PH_USOLVE]);
    solve_scalar_component_with_nonorth(vOld, uConv, vConv, wConv, ss.d_grady, bcVType, vFaceBC, dbcV, vStar, lastItsV, lastRelV, corrUsedV, false, false, prof[PH_VGRAD], prof[PH_VASM], prof[PH_VSOLVE]);
    solve_scalar_component_with_nonorth(wOld, uConv, vConv, wConv, ss.d_gradz, bcWType, wFaceBC, dbcW, wStar, lastItsW, lastRelW, corrUsedW, false, false, prof[PH_WGRAD], prof[PH_WASM], prof[PH_WSOLVE]);

    PhaseMark pm_psetup = profile_begin();
    double ps0 = pressureSetup;
    const int rebuildEvery = std::max(par.pAmgRebuildEvery, 1);
    const bool doPressureSetup = (!pressureSys.is_setup) || (step == 1) || (((step - 1) % rebuildEvery) == 0);
    update_pressure_matrix_from_rAU(mesh, dmesh, pressureSys, dbcP, mom.d_rAU, refCell, usePressureAnchor, doPressureSetup, pressureSetup);
    profile_record(prof[PH_PSETUP], pm_psetup);

    u = uStar; v = vStar; w = wStar; p = pOld;
    PhaseMark pm_phi = profile_begin();
    copy_vec_to_device(uStar, ss.d_u);
    copy_vec_to_device(vStar, ss.d_v);
    copy_vec_to_device(wStar, ss.d_w);
    copy_vec_to_device(p, ss.d_p);
    compute_lsq_gradient_gpu(gop, dmesh, dbcP, ss.d_p, ss.d_gradx, ss.d_grady, ss.d_gradz);
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
          par.rho, ss.d_phiStar);
      CUDA_CHECK_LAST();
      CUDA_CALL(cudaMemcpy(ss.d_phi, ss.d_phiStar, mesh.nFaces*sizeof(double), cudaMemcpyDeviceToDevice));
    }
    profile_record(prof[PH_PREDICTOR_PHI], pm_phi);
    PhaseMark pm_cont0 = profile_begin();
    continuity_residual_gpu(dmesh, ss.d_phi, ss.d_divCorr);
    CUDA_CALL(cudaMemcpy(ss.d_divStar, ss.d_divCorr, mesh.nCells*sizeof(double), cudaMemcpyDeviceToDevice));
    profile_record(prof[PH_CONT_PRE_P], pm_cont0);
    const int nPressureSolves = std::max(par.nNonOrthCorr, 0) + 1;
    const int nExtraPressureCorr = std::max(par.nPressureCorr, 0);
    const int totalPlannedPressureSolves = nPressureSolves + nExtraPressureCorr;
    int pressureSolveCounter = 0;
    int pcgTotalIts = 0;
    std::vector<int> pcgPassIts(nPressureSolves, 0);
    { const int block = 256; kernel_zero_double<<<(mesh.nCells + block - 1)/block, block>>>(ss.d_pCorr, mesh.nCells); CUDA_CHECK_LAST(); }

    for(int pcorr=1; pcorr<=nPressureSolves; ++pcorr){
      PhaseMark pm_pcorrg_iter = profile_begin();
      compute_lsq_gradient_gpu(gop, dmesh, dbcP, ss.d_pCorr, ss.d_gradx, ss.d_grady, ss.d_gradz);
      profile_record(prof[PH_PCORR_GRAD], pm_pcorrg_iter);

      const int block = 256;
      kernel_zero_double<<<(mesh.nCells + block - 1)/block, block>>>(ss.d_divNonOrth, mesh.nCells);
      CUDA_CHECK_LAST();
      kernel_pressure_nonorth_flux_and_divergence<<<(mesh.nFaces + block - 1)/block, block>>>(
          mesh.nFaces, mesh.nInternalFaces,
          dmesh.d_owner, dmesh.d_neigh, dmesh.d_bPatch,
          dmesh.d_ccx, dmesh.d_ccy, dmesh.d_ccz,
          dmesh.d_xfx, dmesh.d_xfy, dmesh.d_xfz,
          dmesh.d_nfx, dmesh.d_nfy, dmesh.d_nfz,
          dmesh.d_sfx, dmesh.d_sfy, dmesh.d_sfz,
          dmesh.d_Af, mom.d_rAU,
          dbcP.d_type,
          ss.d_gradx, ss.d_grady, ss.d_gradz,
          ss.d_phiNonOrth, ss.d_divNonOrth);
      CUDA_CHECK_LAST();
      kernel_build_pressure_rhs_from_divs<<<(mesh.nCells + block - 1)/block, block>>>(
          mesh.nCells, ss.d_divStar, ss.d_divNonOrth, usePressureAnchor ? 1 : 0, refCell, pressureSys.d_rhs);
      CUDA_CHECK_LAST();

      ++pressureSolveCounter;
      const bool isFinalPressureSolve = (pressureSolveCounter == totalPlannedPressureSolves);
      const double curPRelTol = isFinalPressureSolve ? 0.0 : par.pRelTol;
      PhaseMark pm_psolve = profile_begin();
      solve_pressure_gpu_device_rhs_device_x0(pressureSys, ss.d_pCorr, ss.d_pCorr, curPRelTol, par.pTol, par.pMaxit, lastItsP, lastRelP, pressureSolve);
      profile_record(prof[PH_PSOLVE_LOOP], pm_psolve);
      pcgPassIts[pcorr-1] = (int)lastItsP;
      pcgTotalIts += (int)lastItsP;

      if(pcorr == nPressureSolves){
        PhaseMark pm_flux = profile_begin();
        compute_lsq_gradient_gpu(gop, dmesh, dbcP, ss.d_pCorr, ss.d_gradx, ss.d_grady, ss.d_gradz);
        kernel_zero_double<<<(mesh.nCells + block - 1)/block, block>>>(ss.d_divNonOrth, mesh.nCells);
        CUDA_CHECK_LAST();
        kernel_pressure_nonorth_flux_and_divergence<<<(mesh.nFaces + block - 1)/block, block>>>(
            mesh.nFaces, mesh.nInternalFaces,
            dmesh.d_owner, dmesh.d_neigh, dmesh.d_bPatch,
            dmesh.d_ccx, dmesh.d_ccy, dmesh.d_ccz,
            dmesh.d_xfx, dmesh.d_xfy, dmesh.d_xfz,
            dmesh.d_nfx, dmesh.d_nfy, dmesh.d_nfz,
            dmesh.d_sfx, dmesh.d_sfy, dmesh.d_sfz,
            dmesh.d_Af, mom.d_rAU,
            dbcP.d_type,
            ss.d_gradx, ss.d_grady, ss.d_gradz,
            ss.d_phiNonOrth, ss.d_divNonOrth);
        CUDA_CHECK_LAST();
        kernel_correct_face_fluxes_simple_nonorth<<<(mesh.nFaces + block - 1)/block, block>>>(
            mesh.nFaces, mesh.nInternalFaces,
            dmesh.d_owner, dmesh.d_neigh, dmesh.d_bPatch,
            dmesh.d_ccx, dmesh.d_ccy, dmesh.d_ccz,
            dmesh.d_xfx, dmesh.d_xfy, dmesh.d_xfz,
            dmesh.d_nfx, dmesh.d_nfy, dmesh.d_nfz,
            dmesh.d_Af, mom.d_rAU,
            dbcP.d_type, dbcP.d_faceValue,
            ss.d_phiStar, ss.d_pCorr, ss.d_phiNonOrth, ss.d_phi);
        CUDA_CHECK_LAST();
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
      kernel_zero_double<<<(mesh.nCells + block - 1)/block, block>>>(ss.d_pCorrDelta, mesh.nCells);
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
      compute_lsq_gradient_gpu(gop, dmesh, dbcP, ss.d_pCorr, ss.d_gradx, ss.d_grady, ss.d_gradz);
      kernel_zero_double<<<(mesh.nCells + block - 1)/block, block>>>(ss.d_divNonOrth, mesh.nCells);
      CUDA_CHECK_LAST();
      kernel_pressure_nonorth_flux_and_divergence<<<(mesh.nFaces + block - 1)/block, block>>>(
          mesh.nFaces, mesh.nInternalFaces,
          dmesh.d_owner, dmesh.d_neigh, dmesh.d_bPatch,
          dmesh.d_ccx, dmesh.d_ccy, dmesh.d_ccz,
          dmesh.d_xfx, dmesh.d_xfy, dmesh.d_xfz,
          dmesh.d_nfx, dmesh.d_nfy, dmesh.d_nfz,
          dmesh.d_sfx, dmesh.d_sfy, dmesh.d_sfz,
          dmesh.d_Af, mom.d_rAU,
          dbcP.d_type,
          ss.d_gradx, ss.d_grady, ss.d_gradz,
          ss.d_phiNonOrth, ss.d_divNonOrth);
      CUDA_CHECK_LAST();
      kernel_correct_face_fluxes_simple_nonorth<<<(mesh.nFaces + block - 1)/block, block>>>(
          mesh.nFaces, mesh.nInternalFaces,
          dmesh.d_owner, dmesh.d_neigh, dmesh.d_bPatch,
          dmesh.d_ccx, dmesh.d_ccy, dmesh.d_ccz,
          dmesh.d_xfx, dmesh.d_xfy, dmesh.d_xfz,
          dmesh.d_nfx, dmesh.d_nfy, dmesh.d_nfz,
          dmesh.d_Af, mom.d_rAU,
          dbcP.d_type, dbcP.d_faceValue,
          ss.d_phiStar, ss.d_pCorr, ss.d_phiNonOrth, ss.d_phi);
      CUDA_CHECK_LAST();
      profile_record(prof[PH_FLUX_CORR_LOOP], pm_flux);
      PhaseMark pm_contp = profile_begin();
      continuity_residual_gpu(dmesh, ss.d_phi, ss.d_divCorr);
      profile_record(prof[PH_CONT_IN_P_LOOP], pm_contp);
    }
    massRes = maxabs_device(ss.d_divCorr, mesh.nCells, ss.d_reduce, ss.reduceSize);
    copy_device_to_vec(ss.d_pCorr, pCorr);
    {
      const int block = 256;
      kernel_update_pressure_relax<<<(mesh.nCells + block - 1)/block, block>>>(mesh.nCells, ss.d_p, ss.d_pCorr, par.pRelax);
      CUDA_CHECK_LAST();
      copy_device_to_vec(ss.d_p, p);
    }
    if(usePressureAnchor){ double pref = p[refCell]; for(double &pv : p) pv -= pref; }

    PhaseMark pm_pcorrg = profile_begin();
    compute_lsq_gradient_gpu(gop, dmesh, dbcP, ss.d_pCorr, ss.d_gradx, ss.d_grady, ss.d_gradz);
    profile_record(prof[PH_PCORR_GRAD], pm_pcorrg);
    PhaseMark pm_velcorr = profile_begin();
    {
      const int block = 256;
      kernel_correct_velocity_from_pcorr_grad<<<(mesh.nCells + block - 1)/block, block>>>(
          mesh.nCells, ss.d_u, ss.d_v, ss.d_w, mom.d_rAU,
          ss.d_gradx, ss.d_grady, ss.d_gradz,
          ss.d_u, ss.d_v, ss.d_w);
      CUDA_CHECK_LAST();
      copy_device_to_vec(ss.d_u, u);
      copy_device_to_vec(ss.d_v, v);
      copy_device_to_vec(ss.d_w, w);
    }
    profile_record(prof[PH_VEL_CORRECT], pm_velcorr);

    auto relchg_field = [&](const std::vector<double>& a, const std::vector<double>& b){
      double num=0.0, den=1.0;
      for(std::size_t i=0;i<a.size();++i){ double d=a[i]-b[i]; num+=d*d; den+=a[i]*a[i];}
      return std::sqrt(num/den);
    };
    duRel = relchg_field(u, uOld);
    dvRel = relchg_field(v, vOld);
    dwRel = relchg_field(w, wOld);
    dpRel = relchg_field(p, pOld);

    double iterWall = MPI_Wtime() - iterStart;
    double totalWall = MPI_Wtime() - runStart;

    if(rank==0 && (step==1 || (par.printEvery>0 && step%par.printEvery==0))){
      std::printf("iter %4d : massRes = %.3e, duRel = %.3e, dvRel = %.3e, dwRel = %.3e, dpRel = %.3e, bicgstab=[%d %d %d], pcgLast=%d, pcgTot=%d, iterWall = %.3e s, totalWall = %.3e s\n",
                  step, massRes, duRel, dvRel, dwRel, dpRel,
                  (int)lastItsU, (int)lastItsV, (int)lastItsW, (int)lastItsP, pcgTotalIts, iterWall, totalWall);
    }

    profStepsDone = step;

    if(par.writeEvery>0 && step%par.writeEvery==0 && par.write_vtu){
      std::vector<std::array<double,3>> Uvec(mesh.nCells);
      std::vector<double> umag(mesh.nCells);
      for(int c=0;c<mesh.nCells;++c){ Uvec[c]={u[c],v[c],w[c]}; umag[c]=std::sqrt(u[c]*u[c]+v[c]*v[c]+w[c]*w[c]); }
      std::ostringstream oss; oss<<par.outPrefix<<"_iter"<<std::setw(6)<<std::setfill('0')<<step<<".vtu";
      write_vtu_polyhedron_cell_data(oss.str(), mesh, {"p","umag","cell_volume","divCorr"}, {p,umag,mesh.vol,divCorr}, "U", &Uvec);
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

  std::vector<std::array<double,3>> Uvec(mesh.nCells);
  std::vector<double> umag(mesh.nCells);
  double umax=0.0,vmaxf=0.0,wmaxf=0.0,pmax=0.0;
  for(int c=0;c<mesh.nCells;++c){
    Uvec[c]={u[c],v[c],w[c]};
    umag[c]=std::sqrt(u[c]*u[c]+v[c]*v[c]+w[c]*w[c]);
    umax=std::max(umax,std::fabs(u[c])); vmaxf=std::max(vmaxf,std::fabs(v[c])); wmaxf=std::max(wmaxf,std::fabs(w[c])); pmax=std::max(pmax,std::fabs(p[c]));
  }

  const double forceBenchmarkH = 0.41;
  const double forceUbar = (4.0/9.0) * par.Umean;
  CylinderForceReport cylForce;
  CylinderForceVectorReport cylForceVec;
  PatchForceReport patchForce;

  if(cylinderPatch >= 0){
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
    } else {
      std::printf("Cylinder force postprocess skipped: patch_3_0 not found.\n");
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

    std::printf("last bicgstab it = [%d %d %d]\n", (int)lastItsU, (int)lastItsV, (int)lastItsW);
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
    write_vtu_polyhedron_cell_data(vtuFile, mesh, {"p","umag","cell_volume","divCorr"}, {p,umag,mesh.vol,divCorr}, "U", &Uvec);
    if(rank==0) std::printf("Wrote VTU         : %s\n", vtuFile.c_str());
  }

  std::ofstream sout((par.outPrefix+"_summary.txt").c_str());
  sout << std::setprecision(16);
  sout << "points " << mesh.P.size() << "\nfaces " << mesh.nFaces << "\ninternalFaces " << mesh.nInternalFaces << "\ncells " << mesh.nCells << "\n";
  sout << "maxNonOrthDeg " << mesh.maxNonOrthDeg << "\nmu " << mu << "\n";
  sout << "steps " << stepConverged << "\nmassRes " << massRes << "\n";
  sout << "maxAbsU " << umax << "\nmaxAbsV " << vmaxf << "\nmaxAbsW " << wmaxf << "\nmaxAbsP " << pmax << "\n";
  sout << "lastBiCGSTABU " << lastItsU << "\nlastBiCGSTABV " << lastItsV << "\nlastBiCGSTABW " << lastItsW << "\nlastPCG " << lastItsP << "\n";
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

  destroy_momentum_system(mom);
  destroy_simple_scratch(ss);
  destroy_lsq_gradient_operator(gop);
  destroy_linear_storage(pressureSys);
  destroy_device_bc(dbcP); destroy_device_bc(dbcU); destroy_device_bc(dbcV); destroy_device_bc(dbcW);
  destroy_device_mesh(dmesh);
  HYPRE_CALL(HYPRE_Finalize());
  MPI_Finalize();
  return 0;
}
