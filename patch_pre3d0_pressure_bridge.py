#!/usr/bin/env python3
from pathlib import Path
import sys

p = Path("apps/pre3_pressure_bridge_gpu_mpi/src/main.cu")
s = p.read_text()

if "PRE3D0 pressure bridge" in s:
    print("Already patched pre3d0 pressure bridge")
    raise SystemExit(0)

# Rename print labels.
s = s.replace("PRE3C2 GPU geometric Poisson setup", "PRE3D0 pressure bridge setup")
s = s.replace("PRE3C2 GPU geometric Poisson RESULT", "PRE3D0 pressure bridge pressure-solve RESULT")
s = s.replace("PRE3C2 RESULT: PASS_RAN", "PRE3D0 pressure solve RESULT: PASS_RAN")
s = s.replace("PRE3C2 RESULT: FAIL_NAN_INF", "PRE3D0 pressure solve RESULT: FAIL_NAN_INF")

# Add kernels after k_processor_geom_poisson.
anchor = r'''__global__ static void k_processor_geom_poisson(
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
'''

insert = anchor + r'''

// -----------------------------------------------------------------------------
// PRE3D0 bridge kernels:
// Use the solved local pCorr/pressure field on GPU, exchange processor-patch
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

  // This is the pressure-correction-like face contribution across a processor
  // face. In SIMPLE this kind of data feeds flux/velocity correction.
  faceCorr[i] = D * (phiN - phiP);
}
'''

if anchor not in s:
    raise SystemExit("Could not find k_processor_geom_poisson anchor")
s = s.replace(anchor, insert, 1)

# Insert halo exchange after xhost copy and before error calculation.
old = r'''    HYPRE_CALL(HYPRE_IJVectorGetValues(xij, nLocal, d_rows, d_x));
    std::vector<HYPRE_Complex> xhost(nLocal);
    CUDA_CALL(cudaMemcpy(xhost.data(), d_x, sizeof(HYPRE_Complex) * nLocal, cudaMemcpyDeviceToHost));

    double localL2 = 0.0;
'''

new = r'''    HYPRE_CALL(HYPRE_IJVectorGetValues(xij, nLocal, d_rows, d_x));
    std::vector<HYPRE_Complex> xhost(nLocal);
    CUDA_CALL(cudaMemcpy(xhost.data(), d_x, sizeof(HYPRE_Complex) * nLocal, cudaMemcpyDeviceToHost));

    // -----------------------------------------------------------------------
    // PRE3D0: pCorr/pressure halo exchange bridge.
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
        throw std::runtime_error("PRE3D0 currently expects exactly one processor patch");
      }

      const int nbr = procPatches[0].neighbProcNo;
      int sendN = (int)pFace.size();
      int recvN = 0;

      MPI_Sendrecv(&sendN, 1, MPI_INT, nbr, 201,
                   &recvN, 1, MPI_INT, nbr, 201,
                   MPI_COMM_WORLD, MPI_STATUS_IGNORE);

      if(recvN != sendN) {
        throw std::runtime_error("PRE3D0 pCorr halo count mismatch");
      }

      MPI_Sendrecv(h_sendPhi.data(), sendN, MPI_DOUBLE, nbr, 202,
                   h_recvPhi.data(), recvN, MPI_DOUBLE, nbr, 202,
                   MPI_COMM_WORLD, MPI_STATUS_IGNORE);

      CUDA_CALL(cudaMemcpy(d_recvPhi, h_recvPhi.data(),
                           sizeof(double) * pFace.size(),
                           cudaMemcpyHostToDevice));

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
        std::printf("PRE3D0 pCorr halo/correction bridge: nProcFacesLocal=%zu globalMaxHaloJump=%.12e globalMaxFaceCorr=%.12e globalSumAbsFaceCorr=%.12e\n",
                    pFace.size(), globalMaxHaloJump, globalMaxFaceCorr, globalSumAbsFaceCorr);
      }
    }

    double localL2 = 0.0;
'''

if old not in s:
    raise SystemExit("Could not find xhost copy block")
s = s.replace(old, new, 1)

# Rename pass/fail final labels if any missed.
s = s.replace("PRE3C2", "PRE3D0")

p.write_text(s)
print("OK: patched PRE3D0 pressure bridge app")
