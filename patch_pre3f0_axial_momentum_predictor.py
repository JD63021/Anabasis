#!/usr/bin/env python3
from pathlib import Path
import sys

p = Path("apps/pre3_pipe_momentum_predictor_gpu_mpi/src/main.cu")
s = p.read_text()

if "PRE3F0 axial momentum predictor" in s:
    print("Already patched PRE3F0")
    raise SystemExit(0)

# Rename labels.
s = s.replace("PRE3E2", "PRE3F0")
s = s.replace("pipe flux correction report", "axial momentum predictor")
s = s.replace("pipe pressure correction report", "axial momentum predictor")
s = s.replace("pipe pressure correction", "axial momentum predictor")

# Add mu parser variable.
old = '''    double tol = 1e-7;
    double uMean = 1.0;
    int device = rank;
'''
new = '''    double tol = 1e-7;
    double uMean = 1.0;
    double mu = 0.05;
    int device = rank;
'''
if old not in s:
    raise SystemExit("Could not find tol/uMean/device block")
s = s.replace(old, new, 1)

old = '''      } else if(a == "-uMean") {
        need("-uMean");
        uMean = std::atof(argv[++i]);
      } else if(a == "-device") {
'''
new = '''      } else if(a == "-uMean") {
        need("-uMean");
        uMean = std::atof(argv[++i]);
      } else if(a == "-mu") {
        need("-mu");
        mu = std::atof(argv[++i]);
      } else if(a == "-device") {
'''
if old not in s:
    raise SystemExit("Could not find uMean/device parser block")
s = s.replace(old, new, 1)

# Replace pressure RHS construction with pure momentum RHS.
old = '''    // Pre-solve vector halo for predictor flux on processor faces.
    double *d_fluxSendVec = nullptr;
    double *d_fluxRecvVec = nullptr;
    CUDA_CALL(cudaMalloc((void**)&d_fluxSendVec, sizeof(double) * 3 * pFace.size()));
    CUDA_CALL(cudaMalloc((void**)&d_fluxRecvVec, sizeof(double) * 3 * pFace.size()));

    if(!pFace.empty()) {
      k_pack_proc_owner_vector<<<((int)pFace.size() + block - 1)/block, block>>>(
        (int)pFace.size(),
        d_pOwner,
        d_u,
        d_v,
        d_w,
        d_fluxSendVec);
      CUDA_CALL(cudaGetLastError());
      CUDA_CALL(cudaDeviceSynchronize());

      std::vector<double> h_fluxSendVec(3 * pFace.size(), 0.0);
      std::vector<double> h_fluxRecvVec(3 * pFace.size(), 0.0);

      CUDA_CALL(cudaMemcpy(h_fluxSendVec.data(), d_fluxSendVec,
                           sizeof(double) * 3 * pFace.size(),
                           cudaMemcpyDeviceToHost));

      if(procPatches.size() != 1) {
        throw std::runtime_error("PRE3F0 currently expects exactly one processor patch");
      }

      const int nbr = procPatches[0].neighbProcNo;
      int sendN = (int)pFace.size();
      int recvN = 0;

      MPI_Sendrecv(&sendN, 1, MPI_INT, nbr, 301,
                   &recvN, 1, MPI_INT, nbr, 301,
                   MPI_COMM_WORLD, MPI_STATUS_IGNORE);

      if(recvN != sendN) {
        throw std::runtime_error("PRE3F0 predictor velocity halo count mismatch");
      }

      MPI_Sendrecv(h_fluxSendVec.data(), 3 * sendN, MPI_DOUBLE, nbr, 302,
                   h_fluxRecvVec.data(), 3 * recvN, MPI_DOUBLE, nbr, 302,
                   MPI_COMM_WORLD, MPI_STATUS_IGNORE);

      CUDA_CALL(cudaMemcpy(d_fluxRecvVec, h_fluxRecvVec.data(),
                           sizeof(double) * 3 * pFace.size(),
                           cudaMemcpyHostToDevice));
    }

    k_zero_values_rhs<<<(std::max(pat.nnz, nLocal) + block - 1)/block, block>>>(
      pat.nnz, nLocal, d_values, d_rhs);
    CUDA_CALL(cudaGetLastError());

    // Assemble pressure-correction RHS = div(phiStar) from predictor flux.
    k_internal_flux_divergence<<<(mesh.nInternalFaces + block - 1)/block, block>>>(
      mesh.nInternalFaces,
      d_owner,
      d_neigh,
      d_u,
      d_v,
      d_w,
      d_Sfx,
      d_Sfy,
      d_Sfz,
      d_rhs);
    CUDA_CALL(cudaGetLastError());

    k_processor_flux_divergence<<<((int)pFace.size() + block - 1)/block, block>>>(
      (int)pFace.size(),
      d_pFace,
      d_pOwner,
      d_u,
      d_v,
      d_w,
      d_fluxRecvVec,
      d_Sfx,
      d_Sfy,
      d_Sfz,
      d_rhs);
    CUDA_CALL(cudaGetLastError());

    k_boundary_flux_divergence<<<((int)bFace.size() + block - 1)/block, block>>>(
      (int)bFace.size(),
      d_bFace,
      d_bOwner,
      d_bType,
      uMean,
      d_Sfx,
      d_Sfy,
      d_Sfz,
      d_rhs);
    CUDA_CALL(cudaGetLastError());
    CUDA_CALL(cudaDeviceSynchronize());

    k_internal_geom_poisson<<<(mesh.nInternalFaces + block - 1)/block, block>>>(
'''
new = '''    k_zero_values_rhs<<<(std::max(pat.nnz, nLocal) + block - 1)/block, block>>>(
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
'''
if old not in s:
    raise SystemExit("Could not find pre-solve flux/RHS block")
s = s.replace(old, new, 1)

# Add a new boundary momentum kernel before copy_mesh_arrays_to_device.
anchor = "static void copy_mesh_arrays_to_device("
kernel = r'''
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

'''
if anchor not in s:
    raise SystemExit("Could not find copy_mesh_arrays_to_device anchor")
s = s.replace(anchor, kernel + "\n" + anchor, 1)

# Replace old boundary_dirichlet_geom_poisson call with axial momentum bc call.
old = '''    k_boundary_dirichlet_geom_poisson<<<((int)bFace.size() + block - 1)/block, block>>>(
      (int)bFace.size(),
      d_bFace, d_bOwner, d_bDiag,
      d_ccx, d_ccy, d_ccz,
      d_xfx, d_xfy, d_xfz,
      d_Sfx, d_Sfy, d_Sfz,
      d_values, d_rhs);
    CUDA_CALL(cudaGetLastError());
'''
new = '''    k_boundary_axial_momentum_bc<<<((int)bFace.size() + block - 1)/block, block>>>(
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
'''
if old not in s:
    raise SystemExit("Could not find boundary_dirichlet_geom_poisson call")
s = s.replace(old, new, 1)

# Remove/neutralize correction bridge sections? Keep them for now but relabel as post-solve diagnostics.
s = s.replace("pCorr", "W")
s = s.replace("pressure-correction", "momentum-solve")
s = s.replace("pressure correction", "momentum predictor")
s = s.replace("pipe flux correction report", "axial momentum predictor")

# Final solution metrics should be W field magnitude.
s = s.replace("xL2", "wL2")
s = s.replace("xInf", "wInf")
s = s.replace("solve RESULT", "W-solve RESULT")

# Some residual labels remain; make them momentum residual.
s = s.replace("GPU momentum-solve residual", "GPU momentum residual")
s = s.replace("mass/flux correction summary", "momentum residual summary")

p.write_text(s)
print("OK: patched PRE3F0 axial momentum predictor")
