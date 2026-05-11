#!/usr/bin/env python3
from pathlib import Path
import sys

p = Path("apps/pre3_pipe_momentum_flux_audit_gpu_mpi/src/main.cu")
s = p.read_text()

if "PRE3F1 momentum flux audit" in s:
    print("Already patched PRE3F1")
    raise SystemExit(0)

s = s.replace("PRE3F0", "PRE3F1")
s = s.replace("axial momentum predictor", "momentum flux audit")

# Add kernels before mesh-copy helper.
anchor = "static void copy_mesh_arrays_to_device("

kernels = r'''
// -----------------------------------------------------------------------------
// PRE3F1 momentum flux audit.
// After solving W, copy W into the velocity field, exchange W across processor
// patches, and assemble div(phi(W)) on GPU to quantify mass imbalance before
// pressure correction.
// -----------------------------------------------------------------------------
__global__ static void k_set_velocity_from_solved_W(
    int nCells,
    const HYPRE_Complex *Wsol,
    double *u,
    double *v,
    double *w)
{
  int c = blockIdx.x * blockDim.x + threadIdx.x;
  if(c >= nCells) return;

  u[c] = 0.0;
  v[c] = 0.0;
  w[c] = (double)Wsol[c];
}

__global__ static void k_zero_scalar_double(
    int n,
    double *a)
{
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if(i >= n) return;
  a[i] = 0.0;
}

__global__ static void k_internal_flux_divergence_double(
    int nFaces,
    const int *owner,
    const int *neigh,
    const double *u,
    const double *v,
    const double *w,
    const double *Sfx,
    const double *Sfy,
    const double *Sfz,
    double *div)
{
  int f = blockIdx.x * blockDim.x + threadIdx.x;
  if(f >= nFaces) return;

  const int P = owner[f];
  const int N = neigh[f];

  const double uf = 0.5 * (u[P] + u[N]);
  const double vf = 0.5 * (v[P] + v[N]);
  const double wf = 0.5 * (w[P] + w[N]);

  const double phi = uf*Sfx[f] + vf*Sfy[f] + wf*Sfz[f];

  atomicAdd(&div[P], +phi);
  atomicAdd(&div[N], -phi);
}

__global__ static void k_processor_flux_divergence_double(
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
    double *div)
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

  atomicAdd(&div[P], +phi);
}

__global__ static void k_boundary_flux_divergence_from_bc_double(
    int nB,
    const int *bFace,
    const int *bOwner,
    const int *bType,
    double uMean,
    const double *Sfx,
    const double *Sfy,
    const double *Sfz,
    double *div)
{
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if(i >= nB) return;

  const int f = bFace[i];
  const int P = bOwner[i];
  const int typ = bType[i];

  double phi = 0.0;

  // wall: no penetration
  // inlet patch_2_0: prescribed +z axial velocity
  // outlet patch_1_0: use zero-gradient approximation from owner value later;
  // for this audit, prescribe +z uMean as a simple through-flow boundary.
  if(typ == 1 || typ == 2) {
    phi = uMean * Sfz[f];
  }

  atomicAdd(&div[P], +phi);
}

'''

if anchor not in s:
    raise SystemExit("Could not find copy_mesh_arrays_to_device anchor")
s = s.replace(anchor, kernels + "\n" + anchor, 1)

# Insert audit after xhost copy, before later diagnostics.
old = '''    HYPRE_CALL(HYPRE_IJVectorGetValues(xij, nLocal, d_rows, d_x));
    std::vector<HYPRE_Complex> xhost(nLocal);
    CUDA_CALL(cudaMemcpy(xhost.data(), d_x, sizeof(HYPRE_Complex) * nLocal, cudaMemcpyDeviceToHost));

    // -----------------------------------------------------------------------
'''

new = '''    HYPRE_CALL(HYPRE_IJVectorGetValues(xij, nLocal, d_rows, d_x));
    std::vector<HYPRE_Complex> xhost(nLocal);
    CUDA_CALL(cudaMemcpy(xhost.data(), d_x, sizeof(HYPRE_Complex) * nLocal, cudaMemcpyDeviceToHost));

    // -----------------------------------------------------------------------
    // PRE3F1: solved-W predictor flux audit.
    // Put solved W into the velocity field, exchange vector halo, and compute
    // div(phi(W)) on GPU.
    // -----------------------------------------------------------------------
    k_set_velocity_from_solved_W<<<(nLocal + block - 1)/block, block>>>(
      nLocal,
      d_x,
      d_u,
      d_v,
      d_w);
    CUDA_CALL(cudaGetLastError());
    CUDA_CALL(cudaDeviceSynchronize());

    double *d_fluxSendVecAudit = nullptr;
    double *d_fluxRecvVecAudit = nullptr;
    double *d_divAudit = nullptr;

    CUDA_CALL(cudaMalloc((void**)&d_fluxSendVecAudit, sizeof(double) * 3 * pFace.size()));
    CUDA_CALL(cudaMalloc((void**)&d_fluxRecvVecAudit, sizeof(double) * 3 * pFace.size()));
    CUDA_CALL(cudaMalloc((void**)&d_divAudit, sizeof(double) * nLocal));

    k_zero_scalar_double<<<(nLocal + block - 1)/block, block>>>(nLocal, d_divAudit);
    CUDA_CALL(cudaGetLastError());

    if(!pFace.empty()) {
      k_pack_proc_owner_vector<<<((int)pFace.size() + block - 1)/block, block>>>(
        (int)pFace.size(),
        d_pOwner,
        d_u,
        d_v,
        d_w,
        d_fluxSendVecAudit);
      CUDA_CALL(cudaGetLastError());
      CUDA_CALL(cudaDeviceSynchronize());

      std::vector<double> h_sendVecAudit(3 * pFace.size(), 0.0);
      std::vector<double> h_recvVecAudit(3 * pFace.size(), 0.0);

      CUDA_CALL(cudaMemcpy(h_sendVecAudit.data(), d_fluxSendVecAudit,
                           sizeof(double) * 3 * pFace.size(),
                           cudaMemcpyDeviceToHost));

      if(procPatches.size() != 1) {
        throw std::runtime_error("PRE3F1 currently expects exactly one processor patch");
      }

      const int nbrAudit = procPatches[0].neighbProcNo;
      int sendNAudit = (int)pFace.size();
      int recvNAudit = 0;

      MPI_Sendrecv(&sendNAudit, 1, MPI_INT, nbrAudit, 401,
                   &recvNAudit, 1, MPI_INT, nbrAudit, 401,
                   MPI_COMM_WORLD, MPI_STATUS_IGNORE);

      if(recvNAudit != sendNAudit) {
        throw std::runtime_error("PRE3F1 solved-W halo count mismatch");
      }

      MPI_Sendrecv(h_sendVecAudit.data(), 3 * sendNAudit, MPI_DOUBLE, nbrAudit, 402,
                   h_recvVecAudit.data(), 3 * recvNAudit, MPI_DOUBLE, nbrAudit, 402,
                   MPI_COMM_WORLD, MPI_STATUS_IGNORE);

      CUDA_CALL(cudaMemcpy(d_fluxRecvVecAudit, h_recvVecAudit.data(),
                           sizeof(double) * 3 * pFace.size(),
                           cudaMemcpyHostToDevice));
    }

    k_internal_flux_divergence_double<<<(mesh.nInternalFaces + block - 1)/block, block>>>(
      mesh.nInternalFaces,
      d_owner,
      d_neigh,
      d_u,
      d_v,
      d_w,
      d_Sfx,
      d_Sfy,
      d_Sfz,
      d_divAudit);
    CUDA_CALL(cudaGetLastError());

    k_processor_flux_divergence_double<<<((int)pFace.size() + block - 1)/block, block>>>(
      (int)pFace.size(),
      d_pFace,
      d_pOwner,
      d_u,
      d_v,
      d_w,
      d_fluxRecvVecAudit,
      d_Sfx,
      d_Sfy,
      d_Sfz,
      d_divAudit);
    CUDA_CALL(cudaGetLastError());

    k_boundary_flux_divergence_from_bc_double<<<((int)bFace.size() + block - 1)/block, block>>>(
      (int)bFace.size(),
      d_bFace,
      d_bOwner,
      d_bType,
      uMean,
      d_Sfx,
      d_Sfy,
      d_Sfz,
      d_divAudit);
    CUDA_CALL(cudaGetLastError());
    CUDA_CALL(cudaDeviceSynchronize());

    std::vector<double> h_divAudit(nLocal, 0.0);
    CUDA_CALL(cudaMemcpy(h_divAudit.data(), d_divAudit,
                         sizeof(double) * nLocal,
                         cudaMemcpyDeviceToHost));

    double localDivInf = 0.0;
    double localDivL2 = 0.0;
    double localWMin = 1e300;
    double localWMax = -1e300;
    double localWVolSum = 0.0;
    double localVolSum = 0.0;

    for(int c = 0; c < nLocal; ++c) {
      const double wc = (double)xhost[c];
      localDivInf = std::max(localDivInf, std::abs(h_divAudit[c]));
      localDivL2 += h_divAudit[c] * h_divAudit[c];

      localWMin = std::min(localWMin, wc);
      localWMax = std::max(localWMax, wc);
      localWVolSum += wc * mesh.vol[c];
      localVolSum += mesh.vol[c];
    }

    double globalDivInf = 0.0;
    double globalDivL2 = 0.0;
    double globalWMin = 0.0;
    double globalWMax = 0.0;
    double globalWVolSum = 0.0;
    double globalVolSum = 0.0;

    MPI_Allreduce(&localDivInf, &globalDivInf, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
    MPI_Allreduce(&localDivL2, &globalDivL2, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    MPI_Allreduce(&localWMin, &globalWMin, 1, MPI_DOUBLE, MPI_MIN, MPI_COMM_WORLD);
    MPI_Allreduce(&localWMax, &globalWMax, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
    MPI_Allreduce(&localWVolSum, &globalWVolSum, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    MPI_Allreduce(&localVolSum, &globalVolSum, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

    globalDivL2 = std::sqrt(globalDivL2 / (double)globalN);

    if(rank == 0) {
      std::printf("PRE3F1 solved-W flux audit: divInf=%.12e divL2=%.12e Wmin=%.12e Wmax=%.12e WvolAvg=%.12e\\n",
                  globalDivInf,
                  globalDivL2,
                  globalWMin,
                  globalWMax,
                  globalWVolSum / std::max(globalVolSum, 1e-300));
    }

    // -----------------------------------------------------------------------
'''

if old not in s:
    raise SystemExit("Could not find xhost-copy anchor")

s = s.replace(old, new, 1)

# Final labels.
s = s.replace("PRE3F1 axial momentum predictor W-solve RESULT", "PRE3F1 momentum flux audit W-solve RESULT")
s = s.replace("PRE3F1 RESULT: PASS_RAN", "PRE3F1 RESULT: PASS_RAN")

p.write_text(s)
print("OK: patched PRE3F1 momentum flux audit")
