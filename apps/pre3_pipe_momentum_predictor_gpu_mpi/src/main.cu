#include <mpi.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <map>
#include <regex>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "mesh.h"
#include "mms.h"
#include "hypre_backend.h"

struct ProcPatch {
  std::string name;
  int nFaces = 0;
  int startFace = 0;
  int myProcNo = -1;
  int neighbProcNo = -1;
};

static std::string read_file(const std::string &path)
{
  std::ifstream in(path);
  if(!in) throw std::runtime_error("Could not open " + path);
  std::ostringstream ss;
  ss << in.rdbuf();
  return ss.str();
}

static int find_int_entry(const std::string &body, const std::string &key, int def = -1)
{
  std::regex re("\\b" + key + R"(\s+([-+]?[0-9]+)\s*;)");
  std::smatch m;
  if(std::regex_search(body, m, re)) return std::stoi(m[1].str());
  return def;
}

static std::string find_word_entry(const std::string &body, const std::string &key)
{
  std::regex re("\\b" + key + R"(\s+([A-Za-z0-9_]+)\s*;)");
  std::smatch m;
  if(std::regex_search(body, m, re)) return m[1].str();
  return "";
}

static std::vector<ProcPatch> read_processor_patches(const std::string &boundaryPath)
{
  const std::string txt = read_file(boundaryPath);

  std::vector<ProcPatch> out;

  std::regex blockRe(R"(([A-Za-z0-9_]+)\s*\{([^{}]*)\})");
  auto begin = std::sregex_iterator(txt.begin(), txt.end(), blockRe);
  auto end = std::sregex_iterator();

  for(auto it = begin; it != end; ++it) {
    const std::string name = (*it)[1].str();
    const std::string body = (*it)[2].str();

    if(find_word_entry(body, "type") != "processor") continue;

    ProcPatch p;
    p.name = name;
    p.nFaces = find_int_entry(body, "nFaces");
    p.startFace = find_int_entry(body, "startFace");
    p.myProcNo = find_int_entry(body, "myProcNo");
    p.neighbProcNo = find_int_entry(body, "neighbProcNo");

    if(p.nFaces < 0 || p.startFace < 0 || p.myProcNo < 0 || p.neighbProcNo < 0) {
      throw std::runtime_error("Incomplete processor patch in " + boundaryPath);
    }

    out.push_back(p);
  }

  return out;
}

static void add_col(std::vector<std::map<HYPRE_BigInt,int>> &pos,
                    int localRow,
                    HYPRE_BigInt col)
{
  if(pos[localRow].find(col) == pos[localRow].end()) {
    int dummy = -1;
    pos[localRow][col] = dummy;
  }
}

__device__ static double d_phi_exact_xyz(double x, double y, double z)
{
  const double pi = 3.141592653589793238462643383279502884;
  return sin(pi*x) * sin(pi*y) * sin(pi*z);
}

__device__ static double d_rhs_exact_xyz(double x, double y, double z)
{
  const double pi = 3.141592653589793238462643383279502884;
  return 3.0 * pi * pi * d_phi_exact_xyz(x, y, z);
}

__global__ static void k_zero_values_rhs(int nnz, int nCells,
                                         HYPRE_Complex *vals,
                                         HYPRE_Complex *rhs)
{
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if(i < nnz) vals[i] = (HYPRE_Complex)0.0;
  if(i < nCells) rhs[i] = (HYPRE_Complex)0.0;
}

__global__ static void k_source_rhs(int nCells,
                                    const double *ccx,
                                    const double *ccy,
                                    const double *ccz,
                                    const double *vol,
                                    HYPRE_Complex *rhs)
{
  int c = blockIdx.x * blockDim.x + threadIdx.x;
  if(c >= nCells) return;

  const double f = d_rhs_exact_xyz(ccx[c], ccy[c], ccz[c]);
  rhs[c] += (HYPRE_Complex)(f * vol[c]);
}

__global__ static void k_internal_geom_poisson(
    int nFaces,
    const int *owner,
    const int *neigh,
    const double *ccx,
    const double *ccy,
    const double *ccz,
    const double *Sfx,
    const double *Sfy,
    const double *Sfz,
    const int *pp,
    const int *pn,
    const int *np,
    const int *nn,
    HYPRE_Complex *vals)
{
  int f = blockIdx.x * blockDim.x + threadIdx.x;
  if(f >= nFaces) return;

  const int P = owner[f];
  const int N = neigh[f];

  const double dx = ccx[N] - ccx[P];
  const double dy = ccy[N] - ccy[P];
  const double dz = ccz[N] - ccz[P];

  const double sx = Sfx[f];
  const double sy = Sfy[f];
  const double sz = Sfz[f];

  const double ss = sx*sx + sy*sy + sz*sz;
  double dDotS = dx*sx + dy*sy + dz*sz;
  if(dDotS < 1e-300) dDotS = 1e-300;

  const double D = ss / dDotS;

  atomicAdd(&vals[pp[f]], (HYPRE_Complex)(+D));
  atomicAdd(&vals[pn[f]], (HYPRE_Complex)(-D));
  atomicAdd(&vals[np[f]], (HYPRE_Complex)(-D));
  atomicAdd(&vals[nn[f]], (HYPRE_Complex)(+D));
}

__global__ static void k_boundary_dirichlet_geom_poisson(
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
  // PRE3F0 momentum-solve skeleton:
  // use W = 0 as a simple physical-boundary anchor.
  // The predictor mass imbalance enters only through div(phiStar).
  atomicAdd(&vals[bDiag[i]], (HYPRE_Complex)D);
}

__global__ static void k_processor_geom_poisson(
    int nPfaces,
    const int *pFace,
    const int *pOwner,
    const int *pDiag,
    const int *pOff,
    const double *remoteCCx,
    const double *remoteCCy,
    const double *remoteCCz,
    const double *ccx,
    const double *ccy,
    const double *ccz,
    const double *Sfx,
    const double *Sfy,
    const double *Sfz,
    HYPRE_Complex *vals)
{
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if(i >= nPfaces) return;

  const int f = pFace[i];
  const int P = pOwner[i];

  const double dx = remoteCCx[i] - ccx[P];
  const double dy = remoteCCy[i] - ccy[P];
  const double dz = remoteCCz[i] - ccz[P];

  const double sx = Sfx[f];
  const double sy = Sfy[f];
  const double sz = Sfz[f];

  const double ss = sx*sx + sy*sy + sz*sz;
  double dDotS = dx*sx + dy*sy + dz*sz;
  if(dDotS < 1e-300) dDotS = 1e-300;

  const double D = ss / dDotS;

  atomicAdd(&vals[pDiag[i]], (HYPRE_Complex)(+D));
  atomicAdd(&vals[pOff[i]],  (HYPRE_Complex)(-D));
}


// -----------------------------------------------------------------------------
// PRE3F0 bridge kernels:
// Use the solved local W/pressure field on GPU, exchange processor-patch
// halo values through host-staged MPI, then consume the remote halo on GPU.
// This is not full SIMPLE correction yet; it proves the pressure-solve-to-halo-
// to-GPU-correction path.
// -----------------------------------------------------------------------------
__global__ static void k_pack_proc_owner_scalar(
    int nPfaces,
    const int *pOwner,
    const HYPRE_Complex *localPhi,
    double *sendPhi)
{
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if(i >= nPfaces) return;
  sendPhi[i] = (double)localPhi[pOwner[i]];
}

__global__ static void k_proc_pcorr_bridge_diagnostic(
    int nPfaces,
    const int *pFace,
    const int *pOwner,
    const HYPRE_Complex *localPhi,
    const double *remotePhi,
    const double *remoteCCx,
    const double *remoteCCy,
    const double *remoteCCz,
    const double *ccx,
    const double *ccy,
    const double *ccz,
    const double *Sfx,
    const double *Sfy,
    const double *Sfz,
    double *faceCorr)
{
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if(i >= nPfaces) return;

  const int f = pFace[i];
  const int P = pOwner[i];

  const double phiP = (double)localPhi[P];
  const double phiN = remotePhi[i];

  const double dx = remoteCCx[i] - ccx[P];
  const double dy = remoteCCy[i] - ccy[P];
  const double dz = remoteCCz[i] - ccz[P];

  const double sx = Sfx[f];
  const double sy = Sfy[f];
  const double sz = Sfz[f];

  const double ss = sx*sx + sy*sy + sz*sz;
  double dDotS = dx*sx + dy*sy + dz*sz;
  if(dDotS < 1e-300) dDotS = 1e-300;

  const double D = ss / dDotS;

  // This is the momentum-solve-like face contribution across a processor
  // face. In SIMPLE this kind of data feeds flux/velocity correction.
  faceCorr[i] = D * (phiN - phiP);
}


// -----------------------------------------------------------------------------
// PRE3F0 velocity bridge kernels.
// These are diagnostic/skeleton kernels, not final SIMPLE discretization.
// They prove U/V/W halo exchange and W-driven GPU velocity update plumbing.
// -----------------------------------------------------------------------------
__global__ static void k_init_dummy_velocity(
    int nCells,
    const double *ccx,
    const double *ccy,
    const double *ccz,
    double *u,
    double *v,
    double *w)
{
  int c = blockIdx.x * blockDim.x + threadIdx.x;
  if(c >= nCells) return;

  const double x = ccx[c];
  const double y = ccy[c];
  const double z = ccz[c];

  u[c] = 1.0 + 0.10 * sin(7.0*x) + 0.03 * y;
  v[c] = 0.5 + 0.05 * cos(5.0*y) + 0.02 * z;
  w[c] = 0.25 + 0.04 * sin(3.0*z) + 0.01 * x;
}

__global__ static void k_pack_proc_owner_vector(
    int nPfaces,
    const int *pOwner,
    const double *u,
    const double *v,
    const double *w,
    double *sendVec)
{
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if(i >= nPfaces) return;

  const int P = pOwner[i];
  sendVec[3*i + 0] = u[P];
  sendVec[3*i + 1] = v[P];
  sendVec[3*i + 2] = w[P];
}

__global__ static void k_vector_halo_diagnostic(
    int nPfaces,
    const int *pOwner,
    const double *u,
    const double *v,
    const double *w,
    const double *recvVec,
    double *jumpMag)
{
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if(i >= nPfaces) return;

  const int P = pOwner[i];

  const double du = recvVec[3*i + 0] - u[P];
  const double dv = recvVec[3*i + 1] - v[P];
  const double dw = recvVec[3*i + 2] - w[P];

  jumpMag[i] = sqrt(du*du + dv*dv + dw*dw);
}

__global__ static void k_pcorr_velocity_bridge_update(
    int nPfaces,
    const int *pFace,
    const int *pOwner,
    const HYPRE_Complex *localPhi,
    const double *remotePhi,
    const double *remoteCCx,
    const double *remoteCCy,
    const double *remoteCCz,
    const double *ccx,
    const double *ccy,
    const double *ccz,
    double *u,
    double *v,
    double *w,
    double *velCorrMag)
{
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if(i >= nPfaces) return;

  const int P = pOwner[i];

  const double phiP = (double)localPhi[P];
  const double phiN = remotePhi[i];

  const double dx = remoteCCx[i] - ccx[P];
  const double dy = remoteCCy[i] - ccy[P];
  const double dz = remoteCCz[i] - ccz[P];

  double d2 = dx*dx + dy*dy + dz*dz;
  if(d2 < 1e-300) d2 = 1e-300;

  const double dpdd = (phiN - phiP) / d2;

  // rAU-style dummy scale for the bridge. Final SIMPLE will use real rAU.
  const double rAU = 1.0;

  const double du = -rAU * dpdd * dx;
  const double dv = -rAU * dpdd * dy;
  const double dw = -rAU * dpdd * dz;

  atomicAdd(&u[P], du);
  atomicAdd(&v[P], dv);
  atomicAdd(&w[P], dw);

  velCorrMag[i] = sqrt(du*du + dv*dv + dw*dw);
}



// -----------------------------------------------------------------------------
// PRE3F0 axial momentum predictor.
// Applies the same geometric FV operator on GPU after solving:
//   residual = A*x - rhs
// This is the momentum-solve consistency check before full SIMPLE.
// -----------------------------------------------------------------------------
__global__ static void k_init_residual_from_rhs(
    int nCells,
    const HYPRE_Complex *rhs,
    double *res)
{
  int c = blockIdx.x * blockDim.x + threadIdx.x;
  if(c >= nCells) return;
  res[c] = -(double)rhs[c];
}

__global__ static void k_internal_apply_A_residual(
    int nFaces,
    const int *owner,
    const int *neigh,
    const double *ccx,
    const double *ccy,
    const double *ccz,
    const double *Sfx,
    const double *Sfy,
    const double *Sfz,
    const HYPRE_Complex *x,
    double *res)
{
  int f = blockIdx.x * blockDim.x + threadIdx.x;
  if(f >= nFaces) return;

  const int P = owner[f];
  const int N = neigh[f];

  const double dx = ccx[N] - ccx[P];
  const double dy = ccy[N] - ccy[P];
  const double dz = ccz[N] - ccz[P];

  const double sx = Sfx[f];
  const double sy = Sfy[f];
  const double sz = Sfz[f];

  const double ss = sx*sx + sy*sy + sz*sz;
  double dDotS = dx*sx + dy*sy + dz*sz;
  if(dDotS < 1e-300) dDotS = 1e-300;

  const double D = ss / dDotS;

  const double xP = (double)x[P];
  const double xN = (double)x[N];

  const double fluxPN = D * (xP - xN);

  atomicAdd(&res[P], +fluxPN);
  atomicAdd(&res[N], -fluxPN);
}

__global__ static void k_boundary_apply_A_residual(
    int nB,
    const int *bFace,
    const int *bOwner,
    const double *ccx,
    const double *ccy,
    const double *ccz,
    const double *xfx,
    const double *xfy,
    const double *xfz,
    const double *Sfx,
    const double *Sfy,
    const double *Sfz,
    const HYPRE_Complex *x,
    double *res)
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

  const double xP = (double)x[P];
  const double phiB = d_phi_exact_xyz(xfx[f], xfy[f], xfz[f]);

  // Matrix/RHS form had: A[P,P]+=D, rhs[P]+=D*phiB.
  // Since res starts as -rhs, add D*xP here.
  atomicAdd(&res[P], D * xP);
}

__global__ static void k_processor_apply_A_residual(
    int nPfaces,
    const int *pFace,
    const int *pOwner,
    const HYPRE_Complex *localPhi,
    const double *remotePhi,
    const double *remoteCCx,
    const double *remoteCCy,
    const double *remoteCCz,
    const double *ccx,
    const double *ccy,
    const double *ccz,
    const double *Sfx,
    const double *Sfy,
    const double *Sfz,
    double *res)
{
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if(i >= nPfaces) return;

  const int f = pFace[i];
  const int P = pOwner[i];

  const double dx = remoteCCx[i] - ccx[P];
  const double dy = remoteCCy[i] - ccy[P];
  const double dz = remoteCCz[i] - ccz[P];

  const double sx = Sfx[f];
  const double sy = Sfy[f];
  const double sz = Sfz[f];

  const double ss = sx*sx + sy*sy + sz*sz;
  double dDotS = dx*sx + dy*sy + dz*sz;
  if(dDotS < 1e-300) dDotS = 1e-300;

  const double D = ss / dDotS;

  const double xP = (double)localPhi[P];
  const double xN = remotePhi[i];

  atomicAdd(&res[P], D * (xP - xN));
}



// -----------------------------------------------------------------------------
// PRE3F0 pipe momentum-solve kernels.
// This is not full SIMPLE yet. It creates a predictor flux from a dummy axial
// velocity field, assembles div(phiStar) as momentum-solve RHS, solves W,
// and checks whether A*W removes that algebraic mass imbalance.
// -----------------------------------------------------------------------------
__global__ static void k_init_pipe_predictor_velocity(
    int nCells,
    const double *ccx,
    const double *ccy,
    const double *ccz,
    double uMean,
    double *u,
    double *v,
    double *w)
{
  int c = blockIdx.x * blockDim.x + threadIdx.x;
  if(c >= nCells) return;

  const double x = ccx[c];
  const double y = ccy[c];
  const double r2 = x*x + y*y;
  const double R = 0.025;
  double prof = 1.0 - r2 / (R*R);
  if(prof < 0.0) prof = 0.0;

  // Axial dummy predictor. Final SIMPLE will come from momentum solve.
  u[c] = 0.0;
  v[c] = 0.0;
  w[c] = uMean * prof;
}

__global__ static void k_internal_flux_divergence(
    int nFaces,
    const int *owner,
    const int *neigh,
    const double *u,
    const double *v,
    const double *w,
    const double *Sfx,
    const double *Sfy,
    const double *Sfz,
    HYPRE_Complex *rhs)
{
  int f = blockIdx.x * blockDim.x + threadIdx.x;
  if(f >= nFaces) return;

  const int P = owner[f];
  const int N = neigh[f];

  const double uf = 0.5 * (u[P] + u[N]);
  const double vf = 0.5 * (v[P] + v[N]);
  const double wf = 0.5 * (w[P] + w[N]);

  const double phi = uf*Sfx[f] + vf*Sfy[f] + wf*Sfz[f];

  // rhs = div(phiStar), using owner-oriented face flux.
  atomicAdd(&rhs[P], (HYPRE_Complex)(+phi));
  atomicAdd(&rhs[N], (HYPRE_Complex)(-phi));
}

__global__ static void k_processor_flux_divergence(
    int nPfaces,
    const int *pFace,
    const int *pOwner,
    const double *u,
    const double *v,
    const double *w,
    const double *remoteVec,
    const double *Sfx,
    const double *Sfy,
    const double *Sfz,
    HYPRE_Complex *rhs)
{
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if(i >= nPfaces) return;

  const int f = pFace[i];
  const int P = pOwner[i];

  const double uN = remoteVec[3*i + 0];
  const double vN = remoteVec[3*i + 1];
  const double wN = remoteVec[3*i + 2];

  const double uf = 0.5 * (u[P] + uN);
  const double vf = 0.5 * (v[P] + vN);
  const double wf = 0.5 * (w[P] + wN);

  const double phi = uf*Sfx[f] + vf*Sfy[f] + wf*Sfz[f];

  // Only local owner row receives this side of processor-face divergence.
  atomicAdd(&rhs[P], (HYPRE_Complex)(+phi));
}

__global__ static void k_boundary_flux_divergence(
    int nB,
    const int *bFace,
    const int *bOwner,
    const int *bType,
    double uMean,
    const double *Sfx,
    const double *Sfy,
    const double *Sfz,
    HYPRE_Complex *rhs)
{
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if(i >= nB) return;

  const int f = bFace[i];
  const int P = bOwner[i];
  const int typ = bType[i];

  double phi = 0.0;

  // bType:
  // 0 = wall: no penetration
  // 1 = inlet patch_2_0, z=-0.25
  // 2 = outlet patch_1_0, z=+0.25
  if(typ == 1 || typ == 2) {
    // Prescribed axial velocity +z. OpenFOAM boundary Sf is outward,
    // so inlet at z=-0.25 naturally gives negative flux.
    phi = uMean * Sfz[f];
  }

  atomicAdd(&rhs[P], (HYPRE_Complex)(+phi));
}

__global__ static void k_scale_rhs_zero_mean(
    int nCells,
    HYPRE_Complex *rhs,
    double shift)
{
  int c = blockIdx.x * blockDim.x + threadIdx.x;
  if(c >= nCells) return;
  rhs[c] -= (HYPRE_Complex)shift;
}



// -----------------------------------------------------------------------------
// PRE3F0 explicit flux correction report.
// Compute momentum-solve flux magnitudes on GPU for internal, boundary,
// and processor faces. This makes the SIMPLE momentum-solve action visible.
// -----------------------------------------------------------------------------
__global__ static void k_internal_pcorr_flux_correction_mag(
    int nFaces,
    const int *owner,
    const int *neigh,
    const double *ccx,
    const double *ccy,
    const double *ccz,
    const double *Sfx,
    const double *Sfy,
    const double *Sfz,
    const HYPRE_Complex *W,
    double *corrMag)
{
  int f = blockIdx.x * blockDim.x + threadIdx.x;
  if(f >= nFaces) return;

  const int P = owner[f];
  const int N = neigh[f];

  const double dx = ccx[N] - ccx[P];
  const double dy = ccy[N] - ccy[P];
  const double dz = ccz[N] - ccz[P];

  const double sx = Sfx[f];
  const double sy = Sfy[f];
  const double sz = Sfz[f];

  const double ss = sx*sx + sy*sy + sz*sz;
  double dDotS = dx*sx + dy*sy + dz*sz;
  if(dDotS < 1e-300) dDotS = 1e-300;

  const double D = ss / dDotS;
  const double pP = (double)W[P];
  const double pN = (double)W[N];

  corrMag[f] = fabs(D * (pN - pP));
}

__global__ static void k_boundary_pcorr_flux_correction_mag(
    int nB,
    const int *bFace,
    const int *bOwner,
    const double *ccx,
    const double *ccy,
    const double *ccz,
    const double *xfx,
    const double *xfy,
    const double *xfz,
    const double *Sfx,
    const double *Sfy,
    const double *Sfz,
    const HYPRE_Complex *W,
    double *corrMag)
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
  const double pP = (double)W[P];

  // PRE3E momentum-solve skeleton uses boundary W=0 anchor.
  corrMag[i] = fabs(D * (0.0 - pP));
}

__global__ static void k_processor_pcorr_flux_correction_mag(
    int nPfaces,
    const int *pFace,
    const int *pOwner,
    const HYPRE_Complex *localPhi,
    const double *remotePhi,
    const double *remoteCCx,
    const double *remoteCCy,
    const double *remoteCCz,
    const double *ccx,
    const double *ccy,
    const double *ccz,
    const double *Sfx,
    const double *Sfy,
    const double *Sfz,
    double *corrMag)
{
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if(i >= nPfaces) return;

  const int f = pFace[i];
  const int P = pOwner[i];

  const double dx = remoteCCx[i] - ccx[P];
  const double dy = remoteCCy[i] - ccy[P];
  const double dz = remoteCCz[i] - ccz[P];

  const double sx = Sfx[f];
  const double sy = Sfy[f];
  const double sz = Sfz[f];

  const double ss = sx*sx + sy*sy + sz*sz;
  double dDotS = dx*sx + dy*sy + dz*sz;
  if(dDotS < 1e-300) dDotS = 1e-300;

  const double D = ss / dDotS;
  const double pP = (double)localPhi[P];
  const double pN = remotePhi[i];

  corrMag[i] = fabs(D * (pN - pP));
}



// -----------------------------------------------------------------------------
// PRE3F0 axial momentum boundary contribution.
// Reuses the same geometric stencil. Wall/inlet are Dirichlet; outlet is
// zero-gradient and receives no boundary contribution.
// -----------------------------------------------------------------------------
__global__ static void k_boundary_axial_momentum_bc(
    int nB,
    const int *bFace,
    const int *bOwner,
    const int *bDiag,
    const int *bType,
    double uMean,
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
  const int typ = bType[i];

  // typ:
  // 0 wall/default: W=0 Dirichlet
  // 1 inlet patch_2_0: W=uMean Dirichlet
  // 2 outlet patch_1_0: zeroGradient
  if(typ == 2) {
    return;
  }

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

  double wB = 0.0;
  if(typ == 1) {
    wB = uMean;
  }

  atomicAdd(&vals[bDiag[i]], (HYPRE_Complex)D);
  atomicAdd(&rhs[P], (HYPRE_Complex)(D * wB));
}


static void copy_mesh_arrays_to_device(
    const Mesh &mesh,
    int **d_owner,
    int **d_neigh,
    double **d_ccx,
    double **d_ccy,
    double **d_ccz,
    double **d_xfx,
    double **d_xfy,
    double **d_xfz,
    double **d_Sfx,
    double **d_Sfy,
    double **d_Sfz,
    double **d_vol)
{
  std::vector<double> ccx(mesh.nCells), ccy(mesh.nCells), ccz(mesh.nCells), vol(mesh.nCells);
  for(int c = 0; c < mesh.nCells; ++c) {
    ccx[c] = mesh.cc[c][0];
    ccy[c] = mesh.cc[c][1];
    ccz[c] = mesh.cc[c][2];
    vol[c] = mesh.vol[c];
  }

  std::vector<double> xfx(mesh.nFaces), xfy(mesh.nFaces), xfz(mesh.nFaces);
  std::vector<double> Sfx(mesh.nFaces), Sfy(mesh.nFaces), Sfz(mesh.nFaces);
  for(int f = 0; f < mesh.nFaces; ++f) {
    xfx[f] = mesh.xf[f][0];
    xfy[f] = mesh.xf[f][1];
    xfz[f] = mesh.xf[f][2];
    Sfx[f] = mesh.Sf[f][0];
    Sfy[f] = mesh.Sf[f][1];
    Sfz[f] = mesh.Sf[f][2];
  }

  CUDA_CALL(cudaMalloc((void**)d_owner, sizeof(int) * mesh.nFaces));
  CUDA_CALL(cudaMemcpy(*d_owner, mesh.owner.data(), sizeof(int) * mesh.nFaces, cudaMemcpyHostToDevice));

  CUDA_CALL(cudaMalloc((void**)d_neigh, sizeof(int) * mesh.nInternalFaces));
  CUDA_CALL(cudaMemcpy(*d_neigh, mesh.neigh.data(), sizeof(int) * mesh.nInternalFaces, cudaMemcpyHostToDevice));

  CUDA_CALL(cudaMalloc((void**)d_ccx, sizeof(double) * mesh.nCells));
  CUDA_CALL(cudaMalloc((void**)d_ccy, sizeof(double) * mesh.nCells));
  CUDA_CALL(cudaMalloc((void**)d_ccz, sizeof(double) * mesh.nCells));
  CUDA_CALL(cudaMalloc((void**)d_vol, sizeof(double) * mesh.nCells));

  CUDA_CALL(cudaMemcpy(*d_ccx, ccx.data(), sizeof(double) * mesh.nCells, cudaMemcpyHostToDevice));
  CUDA_CALL(cudaMemcpy(*d_ccy, ccy.data(), sizeof(double) * mesh.nCells, cudaMemcpyHostToDevice));
  CUDA_CALL(cudaMemcpy(*d_ccz, ccz.data(), sizeof(double) * mesh.nCells, cudaMemcpyHostToDevice));
  CUDA_CALL(cudaMemcpy(*d_vol, vol.data(), sizeof(double) * mesh.nCells, cudaMemcpyHostToDevice));

  CUDA_CALL(cudaMalloc((void**)d_xfx, sizeof(double) * mesh.nFaces));
  CUDA_CALL(cudaMalloc((void**)d_xfy, sizeof(double) * mesh.nFaces));
  CUDA_CALL(cudaMalloc((void**)d_xfz, sizeof(double) * mesh.nFaces));
  CUDA_CALL(cudaMalloc((void**)d_Sfx, sizeof(double) * mesh.nFaces));
  CUDA_CALL(cudaMalloc((void**)d_Sfy, sizeof(double) * mesh.nFaces));
  CUDA_CALL(cudaMalloc((void**)d_Sfz, sizeof(double) * mesh.nFaces));

  CUDA_CALL(cudaMemcpy(*d_xfx, xfx.data(), sizeof(double) * mesh.nFaces, cudaMemcpyHostToDevice));
  CUDA_CALL(cudaMemcpy(*d_xfy, xfy.data(), sizeof(double) * mesh.nFaces, cudaMemcpyHostToDevice));
  CUDA_CALL(cudaMemcpy(*d_xfz, xfz.data(), sizeof(double) * mesh.nFaces, cudaMemcpyHostToDevice));
  CUDA_CALL(cudaMemcpy(*d_Sfx, Sfx.data(), sizeof(double) * mesh.nFaces, cudaMemcpyHostToDevice));
  CUDA_CALL(cudaMemcpy(*d_Sfy, Sfy.data(), sizeof(double) * mesh.nFaces, cudaMemcpyHostToDevice));
  CUDA_CALL(cudaMemcpy(*d_Sfz, Sfz.data(), sizeof(double) * mesh.nFaces, cudaMemcpyHostToDevice));
}

template<class T>
static T* copy_vec_to_device(const std::vector<T> &h)
{
  T *d = nullptr;
  if(!h.empty()) {
    CUDA_CALL(cudaMalloc((void**)&d, sizeof(T) * h.size()));
    CUDA_CALL(cudaMemcpy(d, h.data(), sizeof(T) * h.size(), cudaMemcpyHostToDevice));
  }
  return d;
}

int main(int argc, char **argv)
{
  MPI_Init(&argc, &argv);

  int rank = 0, size = 1;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  try {
    std::string caseRoot = "/tmp/case";
    int maxit = 500;
    double tol = 1e-7;
    double uMean = 1.0;
    double mu = 0.05;
    int device = rank;

    for(int i = 1; i < argc; ++i) {
      std::string a = argv[i];
      auto need = [&](const char *key){
        if(i + 1 >= argc) {
          std::fprintf(stderr, "Missing value after %s\n", key);
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
      } else if(a == "-uMean") {
        need("-uMean");
        uMean = std::atof(argv[++i]);
      } else if(a == "-mu") {
        need("-mu");
        mu = std::atof(argv[++i]);
      } else if(a == "-device") {
        need("-device");
        device = std::atoi(argv[++i]);
      }
    }

    int devCount = 0;
    CUDA_CALL(cudaGetDeviceCount(&devCount));
    if(devCount > 0) {
      CUDA_CALL(cudaSetDevice(rank % devCount));
    }

    const std::string polyMeshDir =
      caseRoot + "/processor" + std::to_string(rank) + "/constant/polyMesh";

    Mesh mesh = read_openfoam_polymesh(polyMeshDir);
    const auto procPatches = read_processor_patches(polyMeshDir + "/boundary");

    const int nLocal = mesh.nCells;

    std::vector<int> counts(size, 0);
    MPI_Allgather(&nLocal, 1, MPI_INT, counts.data(), 1, MPI_INT, MPI_COMM_WORLD);

    std::vector<int> offsets(size + 1, 0);
    for(int r = 0; r < size; ++r) offsets[r+1] = offsets[r] + counts[r];

    const HYPRE_BigInt ilower = offsets[rank];
    const HYPRE_BigInt iupper = offsets[rank] + nLocal - 1;
    const HYPRE_BigInt globalN = offsets[size];

    auto local_row = [&](int c) -> HYPRE_BigInt {
      return HYPRE_BigInt(offsets[rank] + c);
    };

    // Exchange remote row IDs and remote owner cell centers on processor patches.
    std::vector<HYPRE_BigInt> remoteRowForFace(mesh.nFaces, -1);
    std::vector<std::array<double,3>> remoteCCForFace(mesh.nFaces, {0.0, 0.0, 0.0});

    for(const auto &pp : procPatches) {
      const int nbr = pp.neighbProcNo;

      std::vector<long long> sendRows(pp.nFaces, -1);
      std::vector<double> sendCC(3 * pp.nFaces, 0.0);

      for(int i = 0; i < pp.nFaces; ++i) {
        const int f = pp.startFace + i;
        const int P = mesh.owner[f];

        sendRows[i] = (long long)local_row(P);
        sendCC[3*i + 0] = mesh.cc[P][0];
        sendCC[3*i + 1] = mesh.cc[P][1];
        sendCC[3*i + 2] = mesh.cc[P][2];
      }

      int recvN = 0;
      MPI_Sendrecv(&pp.nFaces, 1, MPI_INT, nbr, 100,
                   &recvN, 1, MPI_INT, nbr, 100,
                   MPI_COMM_WORLD, MPI_STATUS_IGNORE);

      if(recvN != pp.nFaces) {
        throw std::runtime_error("processor patch face-count mismatch");
      }

      std::vector<long long> recvRows(recvN, -1);
      std::vector<double> recvCC(3 * recvN, 0.0);

      MPI_Sendrecv(sendRows.data(), pp.nFaces, MPI_LONG_LONG, nbr, 101,
                   recvRows.data(), recvN, MPI_LONG_LONG, nbr, 101,
                   MPI_COMM_WORLD, MPI_STATUS_IGNORE);

      MPI_Sendrecv(sendCC.data(), 3 * pp.nFaces, MPI_DOUBLE, nbr, 102,
                   recvCC.data(), 3 * recvN, MPI_DOUBLE, nbr, 102,
                   MPI_COMM_WORLD, MPI_STATUS_IGNORE);

      for(int i = 0; i < pp.nFaces; ++i) {
        const int f = pp.startFace + i;
        remoteRowForFace[f] = HYPRE_BigInt(recvRows[i]);
        remoteCCForFace[f] = {recvCC[3*i + 0], recvCC[3*i + 1], recvCC[3*i + 2]};
      }
    }

    std::vector<char> isProcFace(mesh.nFaces, 0);
    for(const auto &pp : procPatches) {
      for(int i = 0; i < pp.nFaces; ++i) {
        isProcFace[pp.startFace + i] = 1;
      }
    }

    // Build static CSR pattern on host.
    std::vector<std::map<HYPRE_BigInt,int>> pos(nLocal);

    for(int c = 0; c < nLocal; ++c) {
      add_col(pos, c, local_row(c));
    }

    for(int f = 0; f < mesh.nInternalFaces; ++f) {
      const int P = mesh.owner[f];
      const int N = mesh.neigh[f];
      add_col(pos, P, local_row(N));
      add_col(pos, N, local_row(P));
    }

    for(int f = mesh.nInternalFaces; f < mesh.nFaces; ++f) {
      if(isProcFace[f]) {
        const int P = mesh.owner[f];
        add_col(pos, P, remoteRowForFace[f]);
      }
    }

    CSRPattern pat;
    pat.nRows = nLocal;
    pat.rows.resize(nLocal);
    pat.ncols.resize(nLocal);
    pat.rowOffsets.resize(nLocal + 1);
    pat.diagPos.resize(nLocal);

    pat.rowOffsets[0] = 0;
    for(int c = 0; c < nLocal; ++c) {
      pat.rows[c] = local_row(c);
      int j = 0;
      for(auto &kv : pos[c]) {
        kv.second = pat.rowOffsets[c] + j;
        ++j;
      }
      pat.ncols[c] = HYPRE_Int(pos[c].size());
      pat.rowOffsets[c + 1] = pat.rowOffsets[c] + int(pos[c].size());
    }

    pat.nnz = pat.rowOffsets[nLocal];
    pat.cols.resize(pat.nnz);
    for(int c = 0; c < nLocal; ++c) {
      const HYPRE_BigInt diag = local_row(c);
      for(const auto &kv : pos[c]) {
        pat.cols[kv.second] = kv.first;
        if(kv.first == diag) pat.diagPos[c] = kv.second;
      }
    }

    // Per-face sparse positions.
    std::vector<int> h_pp(mesh.nInternalFaces), h_pn(mesh.nInternalFaces);
    std::vector<int> h_np(mesh.nInternalFaces), h_nn(mesh.nInternalFaces);

    for(int f = 0; f < mesh.nInternalFaces; ++f) {
      const int P = mesh.owner[f];
      const int N = mesh.neigh[f];
      h_pp[f] = pos[P][local_row(P)];
      h_pn[f] = pos[P][local_row(N)];
      h_np[f] = pos[N][local_row(P)];
      h_nn[f] = pos[N][local_row(N)];
    }

    std::vector<int> bFace, bOwner, bDiag, bType;
    std::vector<int> pFace, pOwner, pDiag, pOff;
    std::vector<double> pRemoteX, pRemoteY, pRemoteZ;

    for(int f = mesh.nInternalFaces; f < mesh.nFaces; ++f) {
      const int P = mesh.owner[f];
      if(isProcFace[f]) {
        pFace.push_back(f);
        pOwner.push_back(P);
        pDiag.push_back(pos[P][local_row(P)]);
        pOff.push_back(pos[P][remoteRowForFace[f]]);
        pRemoteX.push_back(remoteCCForFace[f][0]);
        pRemoteY.push_back(remoteCCForFace[f][1]);
        pRemoteZ.push_back(remoteCCForFace[f][2]);
      } else {
        bFace.push_back(f);
        bOwner.push_back(P);
        bDiag.push_back(pos[P][local_row(P)]);

        int typ = 0;
        const int pidx = mesh.bPatch[f] - 1;
        const std::string pname =
          (pidx >= 0 && pidx < (int)mesh.patchNames.size()) ? mesh.patchNames[pidx] : "";

        if(pname == "patch_2_0") {
          typ = 1; // inlet at z=-0.25
        } else if(pname == "patch_1_0") {
          typ = 2; // outlet at z=+0.25
        } else {
          typ = 0; // wall/default
        }

        bType.push_back(typ);
      }
    }

    if(rank == 0) {
      long long globalNnz = 0;
      long long localNnz = pat.nnz;
      MPI_Allreduce(&localNnz, &globalNnz, 1, MPI_LONG_LONG, MPI_SUM, MPI_COMM_WORLD);
      std::printf("PRE3F0 axial momentum predictor setup: ranks=%d globalRows=%lld globalNnz=%lld\n",
                  size, (long long)globalN, globalNnz);
    } else {
      long long globalNnz = 0;
      long long localNnz = pat.nnz;
      MPI_Allreduce(&localNnz, &globalNnz, 1, MPI_LONG_LONG, MPI_SUM, MPI_COMM_WORLD);
    }

    std::printf("rank %d/%d rows=[%lld,%lld] nLocal=%d nnz=%d internalFaces=%d procFaces=%zu boundaryFaces=%zu\n",
                rank, size, (long long)ilower, (long long)iupper, nLocal, pat.nnz,
                mesh.nInternalFaces, pFace.size(), bFace.size());
    std::fflush(stdout);

    // Device arrays.
    int *d_owner = nullptr, *d_neigh = nullptr;
    double *d_ccx = nullptr, *d_ccy = nullptr, *d_ccz = nullptr;
    double *d_xfx = nullptr, *d_xfy = nullptr, *d_xfz = nullptr;
    double *d_Sfx = nullptr, *d_Sfy = nullptr, *d_Sfz = nullptr;
    double *d_vol = nullptr;

    copy_mesh_arrays_to_device(mesh,
                               &d_owner, &d_neigh,
                               &d_ccx, &d_ccy, &d_ccz,
                               &d_xfx, &d_xfy, &d_xfz,
                               &d_Sfx, &d_Sfy, &d_Sfz,
                               &d_vol);

    int *d_pp = copy_vec_to_device(h_pp);
    int *d_pn = copy_vec_to_device(h_pn);
    int *d_np = copy_vec_to_device(h_np);
    int *d_nn = copy_vec_to_device(h_nn);

    int *d_bFace = copy_vec_to_device(bFace);
    int *d_bOwner = copy_vec_to_device(bOwner);
    int *d_bDiag = copy_vec_to_device(bDiag);
    int *d_bType = copy_vec_to_device(bType);

    int *d_pFace = copy_vec_to_device(pFace);
    int *d_pOwner = copy_vec_to_device(pOwner);
    int *d_pDiag = copy_vec_to_device(pDiag);
    int *d_pOff = copy_vec_to_device(pOff);
    double *d_pRemoteX = copy_vec_to_device(pRemoteX);
    double *d_pRemoteY = copy_vec_to_device(pRemoteY);
    double *d_pRemoteZ = copy_vec_to_device(pRemoteZ);

    HYPRE_Complex *d_values = nullptr;
    HYPRE_Complex *d_rhs = nullptr;
    HYPRE_Complex *d_x0 = nullptr;
    HYPRE_Complex *d_x = nullptr;

    CUDA_CALL(cudaMalloc((void**)&d_values, sizeof(HYPRE_Complex) * pat.nnz));
    CUDA_CALL(cudaMalloc((void**)&d_rhs, sizeof(HYPRE_Complex) * nLocal));
    CUDA_CALL(cudaMalloc((void**)&d_x0, sizeof(HYPRE_Complex) * nLocal));
    CUDA_CALL(cudaMalloc((void**)&d_x, sizeof(HYPRE_Complex) * nLocal));
    CUDA_CALL(cudaMemset(d_x0, 0, sizeof(HYPRE_Complex) * nLocal));

    const int block = 256;

    double *d_u = nullptr;
    double *d_v = nullptr;
    double *d_w = nullptr;

    CUDA_CALL(cudaMalloc((void**)&d_u, sizeof(double) * nLocal));
    CUDA_CALL(cudaMalloc((void**)&d_v, sizeof(double) * nLocal));
    CUDA_CALL(cudaMalloc((void**)&d_w, sizeof(double) * nLocal));

    k_init_pipe_predictor_velocity<<<(nLocal + block - 1)/block, block>>>(
      nLocal,
      d_ccx,
      d_ccy,
      d_ccz,
      uMean,
      d_u,
      d_v,
      d_w);
    CUDA_CALL(cudaGetLastError());
    CUDA_CALL(cudaDeviceSynchronize());

    k_zero_values_rhs<<<(std::max(pat.nnz, nLocal) + block - 1)/block, block>>>(
      pat.nnz, nLocal, d_values, d_rhs);
    CUDA_CALL(cudaGetLastError());

    // PRE3F0 axial W momentum predictor:
    // use the same geometric diffusion operator as the pressure skeleton.
    // Internal/proc coefficients are assembled below. Physical BCs are:
    //   wall patch_0_0     : W = 0
    //   inlet patch_2_0    : W = uMean
    //   outlet patch_1_0   : zeroGradient, implemented by omitting boundary contribution
    // For this first predictor, no convection and no pressure-gradient source.
    k_internal_geom_poisson<<<(mesh.nInternalFaces + block - 1)/block, block>>>(
      mesh.nInternalFaces,
      d_owner, d_neigh,
      d_ccx, d_ccy, d_ccz,
      d_Sfx, d_Sfy, d_Sfz,
      d_pp, d_pn, d_np, d_nn,
      d_values);
    CUDA_CALL(cudaGetLastError());

    k_boundary_axial_momentum_bc<<<((int)bFace.size() + block - 1)/block, block>>>(
      (int)bFace.size(),
      d_bFace,
      d_bOwner,
      d_bDiag,
      d_bType,
      uMean,
      d_ccx,
      d_ccy,
      d_ccz,
      d_xfx,
      d_xfy,
      d_xfz,
      d_Sfx,
      d_Sfy,
      d_Sfz,
      d_values,
      d_rhs);
    CUDA_CALL(cudaGetLastError());

    k_processor_geom_poisson<<<((int)pFace.size() + block - 1)/block, block>>>(
      (int)pFace.size(),
      d_pFace, d_pOwner, d_pDiag, d_pOff,
      d_pRemoteX, d_pRemoteY, d_pRemoteZ,
      d_ccx, d_ccy, d_ccz,
      d_Sfx, d_Sfy, d_Sfz,
      d_values);
    CUDA_CALL(cudaGetLastError());

    CUDA_CALL(cudaDeviceSynchronize());

    // HYPRE device IJ setup/solve.
    HYPRE_CALL(HYPRE_Initialize());
#if defined(HYPRE_USING_GPU)
    HYPRE_CALL(HYPRE_DeviceInitialize());
    HYPRE_CALL(HYPRE_SetMemoryLocation(HYPRE_MEMORY_DEVICE));
    HYPRE_CALL(HYPRE_SetExecutionPolicy(HYPRE_EXEC_DEVICE));
#endif

    HYPRE_BigInt *d_rows = copy_vec_to_device(pat.rows);
    HYPRE_Int *d_ncols = copy_vec_to_device(pat.ncols);
    HYPRE_BigInt *d_cols = copy_vec_to_device(pat.cols);

    HYPRE_IJMatrix Aij = nullptr;
    HYPRE_ParCSRMatrix A = nullptr;
    HYPRE_IJVector bij = nullptr, xij = nullptr;
    HYPRE_ParVector bpar = nullptr, xpar = nullptr;
    HYPRE_Solver solver = nullptr, prec = nullptr;

    HYPRE_CALL(HYPRE_IJMatrixCreate(MPI_COMM_WORLD, ilower, iupper, ilower, iupper, &Aij));
    HYPRE_CALL(HYPRE_IJMatrixSetObjectType(Aij, HYPRE_PARCSR));
    HYPRE_CALL(HYPRE_IJMatrixSetRowSizes(Aij, pat.ncols.data()));
    HYPRE_CALL(HYPRE_IJMatrixInitialize_v2(Aij, HYPRE_MEMORY_DEVICE));
    HYPRE_CALL(HYPRE_IJMatrixSetValues(Aij, pat.nRows, d_ncols, d_rows, d_cols, d_values));
    HYPRE_CALL(HYPRE_IJMatrixAssemble(Aij));
    HYPRE_CALL(HYPRE_IJMatrixGetObject(Aij, (void**)&A));

    HYPRE_CALL(HYPRE_IJVectorCreate(MPI_COMM_WORLD, ilower, iupper, &bij));
    HYPRE_CALL(HYPRE_IJVectorSetObjectType(bij, HYPRE_PARCSR));
    HYPRE_CALL(HYPRE_IJVectorInitialize_v2(bij, HYPRE_MEMORY_DEVICE));
    HYPRE_CALL(HYPRE_IJVectorSetValues(bij, nLocal, d_rows, d_rhs));
    HYPRE_CALL(HYPRE_IJVectorAssemble(bij));
    HYPRE_CALL(HYPRE_IJVectorGetObject(bij, (void**)&bpar));

    HYPRE_CALL(HYPRE_IJVectorCreate(MPI_COMM_WORLD, ilower, iupper, &xij));
    HYPRE_CALL(HYPRE_IJVectorSetObjectType(xij, HYPRE_PARCSR));
    HYPRE_CALL(HYPRE_IJVectorInitialize_v2(xij, HYPRE_MEMORY_DEVICE));
    HYPRE_CALL(HYPRE_IJVectorSetValues(xij, nLocal, d_rows, d_x0));
    HYPRE_CALL(HYPRE_IJVectorAssemble(xij));
    HYPRE_CALL(HYPRE_IJVectorGetObject(xij, (void**)&xpar));

    HYPRE_CALL(HYPRE_ParCSRPCGCreate(MPI_COMM_WORLD, &solver));
    HYPRE_CALL(HYPRE_PCGSetMaxIter(solver, maxit));
    HYPRE_CALL(HYPRE_PCGSetTol(solver, 0.0));
    HYPRE_CALL(HYPRE_PCGSetAbsoluteTol(solver, tol));
    HYPRE_CALL(HYPRE_PCGSetTwoNorm(solver, 1));
    HYPRE_CALL(HYPRE_PCGSetLogging(solver, 1));
    HYPRE_CALL(HYPRE_PCGSetPrintLevel(solver, 0));

    HYPRE_CALL(HYPRE_BoomerAMGCreate(&prec));
    HYPRE_CALL(HYPRE_BoomerAMGSetPrintLevel(prec, 0));
    HYPRE_CALL(HYPRE_BoomerAMGSetMaxIter(prec, 1));
    HYPRE_CALL(HYPRE_BoomerAMGSetTol(prec, 0.0));
    HYPRE_CALL(HYPRE_BoomerAMGSetRelaxType(prec, 18));
    HYPRE_CALL(HYPRE_BoomerAMGSetCoarsenType(prec, 8));
    HYPRE_CALL(HYPRE_BoomerAMGSetInterpType(prec, 6));
    HYPRE_CALL(HYPRE_BoomerAMGSetNumSweeps(prec, 1));
    HYPRE_CALL(HYPRE_BoomerAMGSetPMaxElmts(prec, 4));
    HYPRE_CALL(HYPRE_BoomerAMGSetKeepTranspose(prec, 1));

    HYPRE_CALL(HYPRE_PCGSetPrecond(
      solver,
      (HYPRE_PtrToSolverFcn)HYPRE_BoomerAMGSolve,
      (HYPRE_PtrToSolverFcn)HYPRE_BoomerAMGSetup,
      prec));

    MPI_Barrier(MPI_COMM_WORLD);
    const double t0 = MPI_Wtime();

    HYPRE_CALL(HYPRE_ParCSRPCGSetup(solver, A, bpar, xpar));
    HYPRE_Int solveErr = HYPRE_ParCSRPCGSolve(solver, A, bpar, xpar);

    MPI_Barrier(MPI_COMM_WORLD);
    const double t1 = MPI_Wtime();

    HYPRE_Int its = 0;
    HYPRE_Real rel = 0.0;
    HYPRE_CALL(HYPRE_PCGGetNumIterations(solver, &its));
    HYPRE_CALL(HYPRE_PCGGetFinalRelativeResidualNorm(solver, &rel));

    HYPRE_CALL(HYPRE_IJVectorGetValues(xij, nLocal, d_rows, d_x));
    std::vector<HYPRE_Complex> xhost(nLocal);
    CUDA_CALL(cudaMemcpy(xhost.data(), d_x, sizeof(HYPRE_Complex) * nLocal, cudaMemcpyDeviceToHost));

    // -----------------------------------------------------------------------
    // PRE3F0: W/pressure halo exchange bridge.
    // Pack local owner values on GPU -> host-staged MPI exchange -> copy remote
    // halo back to GPU -> consume remote halo on GPU in a correction diagnostic.
    // -----------------------------------------------------------------------
    double *d_sendPhi = nullptr;
    double *d_recvPhi = nullptr;
    double *d_faceCorr = nullptr;

    if(!pFace.empty()) {
      CUDA_CALL(cudaMalloc((void**)&d_sendPhi, sizeof(double) * pFace.size()));
      CUDA_CALL(cudaMalloc((void**)&d_recvPhi, sizeof(double) * pFace.size()));
      CUDA_CALL(cudaMalloc((void**)&d_faceCorr, sizeof(double) * pFace.size()));

      k_pack_proc_owner_scalar<<<((int)pFace.size() + block - 1)/block, block>>>(
        (int)pFace.size(),
        d_pOwner,
        d_x,
        d_sendPhi);
      CUDA_CALL(cudaGetLastError());
      CUDA_CALL(cudaDeviceSynchronize());

      std::vector<double> h_sendPhi(pFace.size(), 0.0);
      std::vector<double> h_recvPhi(pFace.size(), 0.0);

      CUDA_CALL(cudaMemcpy(h_sendPhi.data(), d_sendPhi,
                           sizeof(double) * pFace.size(),
                           cudaMemcpyDeviceToHost));

      // Current pre3 decomposed cube has one processor patch. Keep this simple
      // first. Multi-neighbor exchange will use per-patch offsets next.
      if(procPatches.size() != 1) {
        throw std::runtime_error("PRE3F0 currently expects exactly one processor patch");
      }

      const int nbr = procPatches[0].neighbProcNo;
      int sendN = (int)pFace.size();
      int recvN = 0;

      MPI_Sendrecv(&sendN, 1, MPI_INT, nbr, 201,
                   &recvN, 1, MPI_INT, nbr, 201,
                   MPI_COMM_WORLD, MPI_STATUS_IGNORE);

      if(recvN != sendN) {
        throw std::runtime_error("PRE3F0 W halo count mismatch");
      }

      MPI_Sendrecv(h_sendPhi.data(), sendN, MPI_DOUBLE, nbr, 202,
                   h_recvPhi.data(), recvN, MPI_DOUBLE, nbr, 202,
                   MPI_COMM_WORLD, MPI_STATUS_IGNORE);

      CUDA_CALL(cudaMemcpy(d_recvPhi, h_recvPhi.data(),
                           sizeof(double) * pFace.size(),
                           cudaMemcpyHostToDevice));

      // PRE3F0: vector U/V/W halo exchange, host staged first.
      double *d_sendVec = nullptr;
      double *d_recvVec = nullptr;
      double *d_vecJump = nullptr;
      double *d_velCorrMag = nullptr;

      CUDA_CALL(cudaMalloc((void**)&d_sendVec, sizeof(double) * 3 * pFace.size()));
      CUDA_CALL(cudaMalloc((void**)&d_recvVec, sizeof(double) * 3 * pFace.size()));
      CUDA_CALL(cudaMalloc((void**)&d_vecJump, sizeof(double) * pFace.size()));
      CUDA_CALL(cudaMalloc((void**)&d_velCorrMag, sizeof(double) * pFace.size()));

      k_pack_proc_owner_vector<<<((int)pFace.size() + block - 1)/block, block>>>(
        (int)pFace.size(),
        d_pOwner,
        d_u,
        d_v,
        d_w,
        d_sendVec);
      CUDA_CALL(cudaGetLastError());
      CUDA_CALL(cudaDeviceSynchronize());

      std::vector<double> h_sendVec(3 * pFace.size(), 0.0);
      std::vector<double> h_recvVec(3 * pFace.size(), 0.0);

      CUDA_CALL(cudaMemcpy(h_sendVec.data(), d_sendVec,
                           sizeof(double) * 3 * pFace.size(),
                           cudaMemcpyDeviceToHost));

      MPI_Sendrecv(h_sendVec.data(), 3 * sendN, MPI_DOUBLE, nbr, 203,
                   h_recvVec.data(), 3 * recvN, MPI_DOUBLE, nbr, 203,
                   MPI_COMM_WORLD, MPI_STATUS_IGNORE);

      CUDA_CALL(cudaMemcpy(d_recvVec, h_recvVec.data(),
                           sizeof(double) * 3 * pFace.size(),
                           cudaMemcpyHostToDevice));

      k_vector_halo_diagnostic<<<((int)pFace.size() + block - 1)/block, block>>>(
        (int)pFace.size(),
        d_pOwner,
        d_u,
        d_v,
        d_w,
        d_recvVec,
        d_vecJump);
      CUDA_CALL(cudaGetLastError());

      k_pcorr_velocity_bridge_update<<<((int)pFace.size() + block - 1)/block, block>>>(
        (int)pFace.size(),
        d_pFace,
        d_pOwner,
        d_x,
        d_recvPhi,
        d_pRemoteX,
        d_pRemoteY,
        d_pRemoteZ,
        d_ccx,
        d_ccy,
        d_ccz,
        d_u,
        d_v,
        d_w,
        d_velCorrMag);
      CUDA_CALL(cudaGetLastError());
      CUDA_CALL(cudaDeviceSynchronize());

      std::vector<double> h_vecJump(pFace.size(), 0.0);
      std::vector<double> h_velCorrMag(pFace.size(), 0.0);

      CUDA_CALL(cudaMemcpy(h_vecJump.data(), d_vecJump,
                           sizeof(double) * pFace.size(),
                           cudaMemcpyDeviceToHost));
      CUDA_CALL(cudaMemcpy(h_velCorrMag.data(), d_velCorrMag,
                           sizeof(double) * pFace.size(),
                           cudaMemcpyDeviceToHost));

      double localMaxVecJump = 0.0;
      double localMaxVelCorr = 0.0;
      double localSumVelCorr = 0.0;

      for(size_t i = 0; i < pFace.size(); ++i) {
        localMaxVecJump = std::max(localMaxVecJump, h_vecJump[i]);
        localMaxVelCorr = std::max(localMaxVelCorr, h_velCorrMag[i]);
        localSumVelCorr += h_velCorrMag[i];
      }

      double globalMaxVecJump = 0.0;
      double globalMaxVelCorr = 0.0;
      double globalSumVelCorr = 0.0;

      MPI_Allreduce(&localMaxVecJump, &globalMaxVecJump, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
      MPI_Allreduce(&localMaxVelCorr, &globalMaxVelCorr, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
      MPI_Allreduce(&localSumVelCorr, &globalSumVelCorr, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

      if(rank == 0) {
        std::printf("PRE3F0 vector halo / velocity correction bridge: globalMaxVecJump=%.12e globalMaxVelCorr=%.12e globalSumVelCorr=%.12e\n",
                    globalMaxVecJump, globalMaxVelCorr, globalSumVelCorr);
      }

      k_proc_pcorr_bridge_diagnostic<<<((int)pFace.size() + block - 1)/block, block>>>(
        (int)pFace.size(),
        d_pFace,
        d_pOwner,
        d_x,
        d_recvPhi,
        d_pRemoteX,
        d_pRemoteY,
        d_pRemoteZ,
        d_ccx,
        d_ccy,
        d_ccz,
        d_Sfx,
        d_Sfy,
        d_Sfz,
        d_faceCorr);
      CUDA_CALL(cudaGetLastError());
      CUDA_CALL(cudaDeviceSynchronize());

      std::vector<double> h_faceCorr(pFace.size(), 0.0);
      CUDA_CALL(cudaMemcpy(h_faceCorr.data(), d_faceCorr,
                           sizeof(double) * pFace.size(),
                           cudaMemcpyDeviceToHost));

      double localMaxHaloJump = 0.0;
      double localMaxFaceCorr = 0.0;
      double localSumAbsFaceCorr = 0.0;

      for(size_t i = 0; i < pFace.size(); ++i) {
        const double jump = std::abs(h_recvPhi[i] - h_sendPhi[i]);
        localMaxHaloJump = std::max(localMaxHaloJump, jump);
        localMaxFaceCorr = std::max(localMaxFaceCorr, std::abs(h_faceCorr[i]));
        localSumAbsFaceCorr += std::abs(h_faceCorr[i]);
      }

      double globalMaxHaloJump = 0.0;
      double globalMaxFaceCorr = 0.0;
      double globalSumAbsFaceCorr = 0.0;

      MPI_Allreduce(&localMaxHaloJump, &globalMaxHaloJump, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
      MPI_Allreduce(&localMaxFaceCorr, &globalMaxFaceCorr, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
      MPI_Allreduce(&localSumAbsFaceCorr, &globalSumAbsFaceCorr, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

      if(rank == 0) {
        std::printf("PRE3F0 W halo/correction bridge: nProcFacesLocal=%zu globalMaxHaloJump=%.12e globalMaxFaceCorr=%.12e globalSumAbsFaceCorr=%.12e\n",
                    pFace.size(), globalMaxHaloJump, globalMaxFaceCorr, globalSumAbsFaceCorr);
      }
      // PRE3F0 explicit momentum-solve face-flux report.
      double *d_intCorrMag = nullptr;
      double *d_bndCorrMag = nullptr;
      double *d_procCorrMag = nullptr;

      CUDA_CALL(cudaMalloc((void**)&d_intCorrMag, sizeof(double) * mesh.nInternalFaces));
      CUDA_CALL(cudaMalloc((void**)&d_bndCorrMag, sizeof(double) * bFace.size()));
      CUDA_CALL(cudaMalloc((void**)&d_procCorrMag, sizeof(double) * pFace.size()));

      k_internal_pcorr_flux_correction_mag<<<(mesh.nInternalFaces + block - 1)/block, block>>>(
        mesh.nInternalFaces,
        d_owner,
        d_neigh,
        d_ccx,
        d_ccy,
        d_ccz,
        d_Sfx,
        d_Sfy,
        d_Sfz,
        d_x,
        d_intCorrMag);
      CUDA_CALL(cudaGetLastError());

      k_boundary_pcorr_flux_correction_mag<<<((int)bFace.size() + block - 1)/block, block>>>(
        (int)bFace.size(),
        d_bFace,
        d_bOwner,
        d_ccx,
        d_ccy,
        d_ccz,
        d_xfx,
        d_xfy,
        d_xfz,
        d_Sfx,
        d_Sfy,
        d_Sfz,
        d_x,
        d_bndCorrMag);
      CUDA_CALL(cudaGetLastError());

      k_processor_pcorr_flux_correction_mag<<<((int)pFace.size() + block - 1)/block, block>>>(
        (int)pFace.size(),
        d_pFace,
        d_pOwner,
        d_x,
        d_recvPhi,
        d_pRemoteX,
        d_pRemoteY,
        d_pRemoteZ,
        d_ccx,
        d_ccy,
        d_ccz,
        d_Sfx,
        d_Sfy,
        d_Sfz,
        d_procCorrMag);
      CUDA_CALL(cudaGetLastError());
      CUDA_CALL(cudaDeviceSynchronize());

      std::vector<double> h_intCorrMag(mesh.nInternalFaces, 0.0);
      std::vector<double> h_bndCorrMag(bFace.size(), 0.0);
      std::vector<double> h_procCorrMag(pFace.size(), 0.0);

      CUDA_CALL(cudaMemcpy(h_intCorrMag.data(), d_intCorrMag,
                           sizeof(double) * mesh.nInternalFaces,
                           cudaMemcpyDeviceToHost));
      CUDA_CALL(cudaMemcpy(h_bndCorrMag.data(), d_bndCorrMag,
                           sizeof(double) * bFace.size(),
                           cudaMemcpyDeviceToHost));
      CUDA_CALL(cudaMemcpy(h_procCorrMag.data(), d_procCorrMag,
                           sizeof(double) * pFace.size(),
                           cudaMemcpyDeviceToHost));

      double localMaxIntCorr = 0.0, localSumIntCorr = 0.0;
      double localMaxBndCorr = 0.0, localSumBndCorr = 0.0;
      double localMaxProcCorr = 0.0, localSumProcCorr = 0.0;

      for(double v : h_intCorrMag) {
        localMaxIntCorr = std::max(localMaxIntCorr, v);
        localSumIntCorr += v;
      }
      for(double v : h_bndCorrMag) {
        localMaxBndCorr = std::max(localMaxBndCorr, v);
        localSumBndCorr += v;
      }
      for(double v : h_procCorrMag) {
        localMaxProcCorr = std::max(localMaxProcCorr, v);
        localSumProcCorr += v;
      }

      double globalMaxIntCorr = 0.0, globalSumIntCorr = 0.0;
      double globalMaxBndCorr = 0.0, globalSumBndCorr = 0.0;
      double globalMaxProcCorr = 0.0, globalSumProcCorr = 0.0;

      MPI_Allreduce(&localMaxIntCorr, &globalMaxIntCorr, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
      MPI_Allreduce(&localSumIntCorr, &globalSumIntCorr, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
      MPI_Allreduce(&localMaxBndCorr, &globalMaxBndCorr, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
      MPI_Allreduce(&localSumBndCorr, &globalSumBndCorr, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
      MPI_Allreduce(&localMaxProcCorr, &globalMaxProcCorr, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
      MPI_Allreduce(&localSumProcCorr, &globalSumProcCorr, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

      if(rank == 0) {
        std::printf("PRE3F0 explicit flux correction report: maxInternal=%.12e sumInternal=%.12e maxBoundary=%.12e sumBoundary=%.12e maxProcessor=%.12e sumProcessor=%.12e\n",
                    globalMaxIntCorr,
                    globalSumIntCorr,
                    globalMaxBndCorr,
                    globalSumBndCorr,
                    globalMaxProcCorr,
                    globalSumProcCorr);
      }



      // PRE3F0 momentum-solve residual: apply A*x-rhs on GPU.
      double *d_res = nullptr;
      CUDA_CALL(cudaMalloc((void**)&d_res, sizeof(double) * nLocal));

      k_init_residual_from_rhs<<<(nLocal + block - 1)/block, block>>>(
        nLocal,
        d_rhs,
        d_res);
      CUDA_CALL(cudaGetLastError());

      k_internal_apply_A_residual<<<(mesh.nInternalFaces + block - 1)/block, block>>>(
        mesh.nInternalFaces,
        d_owner,
        d_neigh,
        d_ccx,
        d_ccy,
        d_ccz,
        d_Sfx,
        d_Sfy,
        d_Sfz,
        d_x,
        d_res);
      CUDA_CALL(cudaGetLastError());

      k_boundary_apply_A_residual<<<((int)bFace.size() + block - 1)/block, block>>>(
        (int)bFace.size(),
        d_bFace,
        d_bOwner,
        d_ccx,
        d_ccy,
        d_ccz,
        d_xfx,
        d_xfy,
        d_xfz,
        d_Sfx,
        d_Sfy,
        d_Sfz,
        d_x,
        d_res);
      CUDA_CALL(cudaGetLastError());

      k_processor_apply_A_residual<<<((int)pFace.size() + block - 1)/block, block>>>(
        (int)pFace.size(),
        d_pFace,
        d_pOwner,
        d_x,
        d_recvPhi,
        d_pRemoteX,
        d_pRemoteY,
        d_pRemoteZ,
        d_ccx,
        d_ccy,
        d_ccz,
        d_Sfx,
        d_Sfy,
        d_Sfz,
        d_res);
      CUDA_CALL(cudaGetLastError());
      CUDA_CALL(cudaDeviceSynchronize());

      std::vector<double> h_res(nLocal, 0.0);
      CUDA_CALL(cudaMemcpy(h_res.data(), d_res,
                           sizeof(double) * nLocal,
                           cudaMemcpyDeviceToHost));

      double localResInf = 0.0;
      double localResL2 = 0.0;
      double localRhsInf = 0.0;

      std::vector<HYPRE_Complex> h_rhs(nLocal);
      CUDA_CALL(cudaMemcpy(h_rhs.data(), d_rhs,
                           sizeof(HYPRE_Complex) * nLocal,
                           cudaMemcpyDeviceToHost));

      for(int c = 0; c < nLocal; ++c) {
        localResInf = std::max(localResInf, std::abs(h_res[c]));
        localResL2 += h_res[c] * h_res[c];
        localRhsInf = std::max(localRhsInf, std::abs((double)h_rhs[c]));
      }

      double globalResInf = 0.0;
      double globalResL2 = 0.0;
      double globalRhsInf = 0.0;

      MPI_Allreduce(&localResInf, &globalResInf, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
      MPI_Allreduce(&localResL2, &globalResL2, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
      MPI_Allreduce(&localRhsInf, &globalRhsInf, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);

      globalResL2 = std::sqrt(globalResL2 / (double)globalN);

      if(rank == 0) {
        const double massBefore = globalRhsInf;
        const double massAfter  = globalResInf;
        const double reduction  = massAfter / std::max(massBefore, 1e-300);
        const double improvement = massBefore / std::max(massAfter, 1e-300);

        std::printf("PRE3F0 GPU momentum residual: resInf=%.12e resL2=%.12e rhsInf=%.12e relInf=%.12e\n",
                    globalResInf,
                    globalResL2,
                    globalRhsInf,
                    globalResInf / std::max(globalRhsInf, 1e-300));

        std::printf("PRE3F0 momentum residual summary: massBeforeInf=%.12e massAfterInf=%.12e reduction=%.12e improvement=%.6e\n",
                    massBefore,
                    massAfter,
                    reduction,
                    improvement);
      }
    }

    double localL2 = 0.0;
    double localInf = 0.0;
    double localVol = 0.0;

    for(int c = 0; c < nLocal; ++c) {
      const double xc = std::abs(double(xhost[c]));
      localL2 += xc * xc * mesh.vol[c];
      localInf = std::max(localInf, xc);
      localVol += mesh.vol[c];
    }

    double globalL2 = 0.0, globalInf = 0.0, globalVol = 0.0;
    MPI_Allreduce(&localL2, &globalL2, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    MPI_Allreduce(&localInf, &globalInf, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
    MPI_Allreduce(&localVol, &globalVol, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

    globalL2 = std::sqrt(globalL2 / std::max(globalVol, 1e-300));

    double wall = t1 - t0;
    double maxWall = 0.0;
    MPI_Reduce(&wall, &maxWall, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    if(rank == 0) {
      std::printf("PRE3F0 axial momentum predictor W-solve RESULT: solveErr=%d its=%d finalRel=%.12e wL2=%.12e wInf=%.12e wall=%.6e s\n",
                  (int)solveErr, (int)its, double(rel), globalL2, globalInf, maxWall);
      if(std::isfinite(globalL2) && std::isfinite(globalInf)) {
        std::printf("PRE3F0 RESULT: PASS_RAN\n");
      } else {
        std::printf("PRE3F0 RESULT: FAIL_NAN_INF\n");
      }
    }

    HYPRE_CALL(HYPRE_BoomerAMGDestroy(prec));
    HYPRE_CALL(HYPRE_ParCSRPCGDestroy(solver));
    HYPRE_CALL(HYPRE_IJVectorDestroy(xij));
    HYPRE_CALL(HYPRE_IJVectorDestroy(bij));
    HYPRE_CALL(HYPRE_IJMatrixDestroy(Aij));
    HYPRE_Finalize();

    MPI_Finalize();
    return 0;
  }
  catch(const std::exception &e) {
    std::fprintf(stderr, "rank %d ERROR: %s\n", rank, e.what());
    MPI_Abort(MPI_COMM_WORLD, 1);
  }

  MPI_Finalize();
  return 1;
}
