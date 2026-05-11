#!/usr/bin/env python3
from pathlib import Path
import sys

p = Path("apps/pre3_pressure_correction_skeleton_gpu_mpi/src/main.cu")
s = p.read_text()

if "PRE3D2 pressure correction residual skeleton" in s:
    print("Already patched PRE3D2")
    raise SystemExit(0)

s = s.replace("PRE3D1", "PRE3D2")
s = s.replace("pressure/velocity bridge", "pressure correction skeleton")

# Insert residual kernels before mesh-copy helper.
anchor = "static void copy_mesh_arrays_to_device("
kernels = r'''
// -----------------------------------------------------------------------------
// PRE3D2 pressure correction residual skeleton.
// Applies the same geometric FV operator on GPU after solving:
//   residual = A*x - rhs
// This is the pressure-correction consistency check before full SIMPLE.
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

'''
if anchor not in s:
    raise SystemExit("Could not find copy_mesh_arrays_to_device anchor")
s = s.replace(anchor, kernels + "\n" + anchor, 1)

# Insert residual diagnostic after pCorr halo/correction bridge print.
old = r'''      if(rank == 0) {
        std::printf("PRE3D2 pCorr halo/correction bridge: nProcFacesLocal=%zu globalMaxHaloJump=%.12e globalMaxFaceCorr=%.12e globalSumAbsFaceCorr=%.12e\n",
                    pFace.size(), globalMaxHaloJump, globalMaxFaceCorr, globalSumAbsFaceCorr);
      }
    }

    double localL2 = 0.0;
'''

new = r'''      if(rank == 0) {
        std::printf("PRE3D2 pCorr halo/correction bridge: nProcFacesLocal=%zu globalMaxHaloJump=%.12e globalMaxFaceCorr=%.12e globalSumAbsFaceCorr=%.12e\n",
                    pFace.size(), globalMaxHaloJump, globalMaxFaceCorr, globalSumAbsFaceCorr);
      }

      // PRE3D2 pressure-correction residual: apply A*x-rhs on GPU.
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
        std::printf("PRE3D2 GPU pressure-correction residual: resInf=%.12e resL2=%.12e rhsInf=%.12e relInf=%.12e\n",
                    globalResInf,
                    globalResL2,
                    globalRhsInf,
                    globalResInf / std::max(globalRhsInf, 1e-300));
      }
    }

    double localL2 = 0.0;
'''

if old not in s:
    raise SystemExit("Could not find pCorr halo/correction bridge print block")
s = s.replace(old, new, 1)

# Final label.
s = s.replace("PRE3D2 pressure correction skeleton RESULT", "PRE3D2 pressure correction residual skeleton RESULT")

p.write_text(s)
print("OK: patched PRE3D2 pressure-correction residual skeleton")
