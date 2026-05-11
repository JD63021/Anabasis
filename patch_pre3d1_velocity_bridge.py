#!/usr/bin/env python3
from pathlib import Path
import sys

p = Path("apps/pre3_pressure_velocity_bridge_gpu_mpi/src/main.cu")
s = p.read_text()

if "PRE3D1 velocity bridge" in s:
    print("Already patched PRE3D1 velocity bridge")
    raise SystemExit(0)

s = s.replace("PRE3D0", "PRE3D1")
s = s.replace("pressure bridge", "pressure/velocity bridge")

# ---------------------------------------------------------------------
# Add vector-halo and velocity-correction kernels before mesh-copy helper.
# ---------------------------------------------------------------------
anchor = "static void copy_mesh_arrays_to_device("
kernels = r'''
// -----------------------------------------------------------------------------
// PRE3D1 velocity bridge kernels.
// These are diagnostic/skeleton kernels, not final SIMPLE discretization.
// They prove U/V/W halo exchange and pCorr-driven GPU velocity update plumbing.
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

'''
if anchor not in s:
    raise SystemExit("Could not find mesh-copy helper anchor")
s = s.replace(anchor, kernels + "\n" + anchor, 1)

# ---------------------------------------------------------------------
# Allocate/init dummy velocity after block constant is defined.
# ---------------------------------------------------------------------
old = r'''    const int block = 256;
    k_zero_values_rhs<<<(std::max(pat.nnz, nLocal) + block - 1)/block, block>>>(
'''

new = r'''    const int block = 256;

    double *d_u = nullptr;
    double *d_v = nullptr;
    double *d_w = nullptr;

    CUDA_CALL(cudaMalloc((void**)&d_u, sizeof(double) * nLocal));
    CUDA_CALL(cudaMalloc((void**)&d_v, sizeof(double) * nLocal));
    CUDA_CALL(cudaMalloc((void**)&d_w, sizeof(double) * nLocal));

    k_init_dummy_velocity<<<(nLocal + block - 1)/block, block>>>(
      nLocal,
      d_ccx,
      d_ccy,
      d_ccz,
      d_u,
      d_v,
      d_w);
    CUDA_CALL(cudaGetLastError());
    CUDA_CALL(cudaDeviceSynchronize());

    k_zero_values_rhs<<<(std::max(pat.nnz, nLocal) + block - 1)/block, block>>>(
'''

if old not in s:
    raise SystemExit("Could not find block constant / zero kernel anchor")
s = s.replace(old, new, 1)

# ---------------------------------------------------------------------
# Insert vector halo and velocity correction diagnostic after remote scalar halo copy.
# ---------------------------------------------------------------------
old = r'''      CUDA_CALL(cudaMemcpy(d_recvPhi, h_recvPhi.data(),
                           sizeof(double) * pFace.size(),
                           cudaMemcpyHostToDevice));

      k_proc_pcorr_bridge_diagnostic<<<((int)pFace.size() + block - 1)/block, block>>>(
'''

new = r'''      CUDA_CALL(cudaMemcpy(d_recvPhi, h_recvPhi.data(),
                           sizeof(double) * pFace.size(),
                           cudaMemcpyHostToDevice));

      // PRE3D1: vector U/V/W halo exchange, host staged first.
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
        std::printf("PRE3D1 vector halo / velocity correction bridge: globalMaxVecJump=%.12e globalMaxVelCorr=%.12e globalSumVelCorr=%.12e\n",
                    globalMaxVecJump, globalMaxVelCorr, globalSumVelCorr);
      }

      k_proc_pcorr_bridge_diagnostic<<<((int)pFace.size() + block - 1)/block, block>>>(
'''

if old not in s:
    raise SystemExit("Could not find scalar halo copy anchor")
s = s.replace(old, new, 1)

# Rename final pass labels where needed.
s = s.replace("PRE3D1 pressure solve RESULT", "PRE3D1 pressure/velocity bridge RESULT")

p.write_text(s)
print("OK: patched PRE3D1 vector halo and velocity correction bridge")
