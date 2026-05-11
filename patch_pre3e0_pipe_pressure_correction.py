#!/usr/bin/env python3
from pathlib import Path
import sys

p = Path("apps/pre3_pipe_pressure_correction_gpu_mpi/src/main.cu")
s = p.read_text()

if "PRE3E0 pipe pressure correction" in s:
    print("Already patched PRE3E0")
    raise SystemExit(0)

# Rename labels.
s = s.replace("PRE3D2", "PRE3E0")
s = s.replace("pressure correction skeleton", "pipe pressure correction")
s = s.replace("pressure correction residual skeleton", "pipe pressure correction")

# --------------------------------------------------------------------
# 1. Add uMean parser variable.
# --------------------------------------------------------------------
old = '''    std::string caseRoot = "/tmp/case";
    int maxit = 500;
    double tol = 1e-7;
    int device = rank;
'''

new = '''    std::string caseRoot = "/tmp/case";
    int maxit = 500;
    double tol = 1e-7;
    double uMean = 1.0;
    int device = rank;
'''

if old not in s:
    raise SystemExit("Could not find parser variable block")
s = s.replace(old, new, 1)

old = '''      } else if(a == "-tol") {
        need("-tol");
        tol = std::atof(argv[++i]);
      } else if(a == "-device") {
'''

new = '''      } else if(a == "-tol") {
        need("-tol");
        tol = std::atof(argv[++i]);
      } else if(a == "-uMean") {
        need("-uMean");
        uMean = std::atof(argv[++i]);
      } else if(a == "-device") {
'''

if old not in s:
    raise SystemExit("Could not find parser tol/device block")
s = s.replace(old, new, 1)

# --------------------------------------------------------------------
# 2. Boundary pCorr should be zero-value anchor for this skeleton.
#    Remove MMS boundary RHS addition from the pressure matrix assembly.
# --------------------------------------------------------------------
old = '''  const double phiB = d_phi_exact_xyz(xfx[f], xfy[f], xfz[f]);

  atomicAdd(&vals[bDiag[i]], (HYPRE_Complex)D);
  atomicAdd(&rhs[P], (HYPRE_Complex)(D * phiB));
'''

new = '''  // PRE3E0 pressure-correction skeleton:
  // use pCorr = 0 as a simple physical-boundary anchor.
  // The predictor mass imbalance enters only through div(phiStar).
  atomicAdd(&vals[bDiag[i]], (HYPRE_Complex)D);
'''

if old not in s:
    raise SystemExit("Could not find boundary MMS RHS block")
s = s.replace(old, new, 1)

# --------------------------------------------------------------------
# 3. Add flux/divergence kernels before mesh-copy helper.
# --------------------------------------------------------------------
anchor = "static void copy_mesh_arrays_to_device("

kernels = r'''
// -----------------------------------------------------------------------------
// PRE3E0 pipe pressure-correction kernels.
// This is not full SIMPLE yet. It creates a predictor flux from a dummy axial
// velocity field, assembles div(phiStar) as pressure-correction RHS, solves pCorr,
// and checks whether A*pCorr removes that algebraic mass imbalance.
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

'''

if anchor not in s:
    raise SystemExit("Could not find copy_mesh_arrays_to_device anchor")
s = s.replace(anchor, kernels + "\n" + anchor, 1)

# --------------------------------------------------------------------
# 4. Add bType vector to physical boundary face list.
# --------------------------------------------------------------------
old = '''    std::vector<int> bFace, bOwner, bDiag;
    std::vector<int> pFace, pOwner, pDiag, pOff;
'''

new = '''    std::vector<int> bFace, bOwner, bDiag, bType;
    std::vector<int> pFace, pOwner, pDiag, pOff;
'''

if old not in s:
    raise SystemExit("Could not find bFace declaration")
s = s.replace(old, new, 1)

old = '''      } else {
        bFace.push_back(f);
        bOwner.push_back(P);
        bDiag.push_back(pos[P][local_row(P)]);
      }
'''

new = '''      } else {
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
'''

if old not in s:
    raise SystemExit("Could not find physical boundary face push block")
s = s.replace(old, new, 1)

old = '''    int *d_bFace = copy_vec_to_device(bFace);
    int *d_bOwner = copy_vec_to_device(bOwner);
    int *d_bDiag = copy_vec_to_device(bDiag);
'''

new = '''    int *d_bFace = copy_vec_to_device(bFace);
    int *d_bOwner = copy_vec_to_device(bOwner);
    int *d_bDiag = copy_vec_to_device(bDiag);
    int *d_bType = copy_vec_to_device(bType);
'''

if old not in s:
    raise SystemExit("Could not find device bFace block")
s = s.replace(old, new, 1)

# --------------------------------------------------------------------
# 5. Replace dummy velocity init with pipe predictor init.
# --------------------------------------------------------------------
old = '''    k_init_dummy_velocity<<<(nLocal + block - 1)/block, block>>>(
      nLocal,
      d_ccx,
      d_ccy,
      d_ccz,
      d_u,
      d_v,
      d_w);
'''

new = '''    k_init_pipe_predictor_velocity<<<(nLocal + block - 1)/block, block>>>(
      nLocal,
      d_ccx,
      d_ccy,
      d_ccz,
      uMean,
      d_u,
      d_v,
      d_w);
'''

if old not in s:
    raise SystemExit("Could not find dummy velocity init")
s = s.replace(old, new, 1)

# --------------------------------------------------------------------
# 6. Insert pre-solve vector halo exchange before zeroing/assembling RHS.
# --------------------------------------------------------------------
old = '''    k_zero_values_rhs<<<(std::max(pat.nnz, nLocal) + block - 1)/block, block>>>(
      pat.nnz, nLocal, d_values, d_rhs);
    CUDA_CALL(cudaGetLastError());

    k_source_rhs<<<(nLocal + block - 1)/block, block>>>(
      nLocal, d_ccx, d_ccy, d_ccz, d_vol, d_rhs);
    CUDA_CALL(cudaGetLastError());

    k_internal_geom_poisson<<<(mesh.nInternalFaces + block - 1)/block, block>>>(
'''

new = '''    // Pre-solve vector halo for predictor flux on processor faces.
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
        throw std::runtime_error("PRE3E0 currently expects exactly one processor patch");
      }

      const int nbr = procPatches[0].neighbProcNo;
      int sendN = (int)pFace.size();
      int recvN = 0;

      MPI_Sendrecv(&sendN, 1, MPI_INT, nbr, 301,
                   &recvN, 1, MPI_INT, nbr, 301,
                   MPI_COMM_WORLD, MPI_STATUS_IGNORE);

      if(recvN != sendN) {
        throw std::runtime_error("PRE3E0 predictor velocity halo count mismatch");
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

if old not in s:
    raise SystemExit("Could not find RHS source/zero block")
s = s.replace(old, new, 1)

# --------------------------------------------------------------------
# 7. Replace final exact-error calculation with pressure-correction magnitude.
# --------------------------------------------------------------------
old = '''    double localL2 = 0.0;
    double localInf = 0.0;
    double localVol = 0.0;

    for(int c = 0; c < nLocal; ++c) {
      const double ue = phi_exact(mesh.cc[c]);
      const double e = std::abs(double(xhost[c]) - ue);
      localL2 += e * e * mesh.vol[c];
      localInf = std::max(localInf, e);
      localVol += mesh.vol[c];
    }

    double globalL2 = 0.0, globalInf = 0.0, globalVol = 0.0;
    MPI_Allreduce(&localL2, &globalL2, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    MPI_Allreduce(&localInf, &globalInf, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
    MPI_Allreduce(&localVol, &globalVol, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

    globalL2 = std::sqrt(globalL2 / std::max(globalVol, 1e-300));
'''

new = '''    double localL2 = 0.0;
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
'''

if old not in s:
    raise SystemExit("Could not find exact-error block")
s = s.replace(old, new, 1)

# Relabel final print L2/Linf as xL2/xInf.
s = s.replace(" L2=%.12e Linf=%.12e ", " xL2=%.12e xInf=%.12e ")
s = s.replace("PRE3E0 pipe pressure correction pressure-solve RESULT",
              "PRE3E0 pipe pressure correction solve RESULT")
s = s.replace("PRE3E0 pipe pressure correction RESULT: PASS_RAN",
              "PRE3E0 RESULT: PASS_RAN")
s = s.replace("PRE3E0 pipe pressure correction RESULT: FAIL_NAN_INF",
              "PRE3E0 RESULT: FAIL_NAN_INF")

p.write_text(s)
print("OK: patched PRE3E0 pipe pressure-correction skeleton")
