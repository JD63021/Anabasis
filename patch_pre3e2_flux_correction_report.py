#!/usr/bin/env python3
from pathlib import Path
import re
import sys

p = Path("apps/pre3_pipe_flux_correction_gpu_mpi/src/main.cu")
s = p.read_text()

if "PRE3E2 explicit flux correction report" in s:
    print("Already patched PRE3E2 explicit flux correction report")
    raise SystemExit(0)

# Rename copied PRE3E1 labels.
s = s.replace("PRE3E1", "PRE3E2")
s = s.replace("pipe pressure correction report", "pipe flux correction report")
s = s.replace("mass correction summary", "mass/flux correction summary")

# ---------------------------------------------------------------------
# Add explicit flux-correction kernels before mesh-copy helper.
# ---------------------------------------------------------------------
anchor = "static void copy_mesh_arrays_to_device("

kernels = r'''
// -----------------------------------------------------------------------------
// PRE3E2 explicit flux correction report.
// Compute pressure-correction flux magnitudes on GPU for internal, boundary,
// and processor faces. This makes the SIMPLE pressure-correction action visible.
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
    const HYPRE_Complex *pCorr,
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
  const double pP = (double)pCorr[P];
  const double pN = (double)pCorr[N];

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
    const HYPRE_Complex *pCorr,
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
  const double pP = (double)pCorr[P];

  // PRE3E pressure-correction skeleton uses boundary pCorr=0 anchor.
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

'''

if anchor not in s:
    raise SystemExit("ERROR: could not find copy_mesh_arrays_to_device anchor")

s = s.replace(anchor, kernels + "\n" + anchor, 1)

# ---------------------------------------------------------------------
# Insert flux-correction diagnostic immediately before the existing
# pressure-correction residual section.
# ---------------------------------------------------------------------
marker_re = re.compile(
    r'(\s*//\s*PRE3[^\n]*pressure-correction residual:\s*apply A\*x-rhs on GPU\.)'
)

m = marker_re.search(s)
if not m:
    # fallback: search for the first residual malloc block
    fallback = "double *d_res = nullptr;"
    idx = s.find(fallback)
    if idx < 0:
        raise SystemExit("ERROR: could not find residual section marker")
    insert_pos = idx
    prefix = ""
else:
    insert_pos = m.start(1)
    prefix = ""

flux_block = r'''
      // PRE3E2 explicit pressure-correction face-flux report.
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
        std::printf("PRE3E2 explicit flux correction report: maxInternal=%.12e sumInternal=%.12e maxBoundary=%.12e sumBoundary=%.12e maxProcessor=%.12e sumProcessor=%.12e\n",
                    globalMaxIntCorr,
                    globalSumIntCorr,
                    globalMaxBndCorr,
                    globalSumBndCorr,
                    globalMaxProcCorr,
                    globalSumProcCorr);
      }

'''

s = s[:insert_pos] + flux_block + s[insert_pos:]

p.write_text(s)
print("OK: patched PRE3E2 explicit flux correction report")
