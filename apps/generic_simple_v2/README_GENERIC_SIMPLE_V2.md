# generic_simple_v2 handoff notes

This app is a copy of `apps/generic_simple_v1` with the pressure-velocity coupling ideas from the recent monolithic SIMPLE debugging branch folded in.

The Poisson and scalar transport libraries were not functionally changed. The new logic is isolated in `apps/generic_simple_v2/src/main.cu` and the top-level `CMakeLists.txt` only adds a new app target.

## Important changes from v1

- Added OpenFOAM-like stabilized face delta coefficient:

  `delta = 1 / max(nHat.d, 0.05*|d|)`

  This is used consistently in the SIMPLE pressure matrix, pressure equation flux correction, explicit pressure non-orthogonal flux, and old explicit Rhie-Chow term.

- Added pressure equation modes:

  `-p-mode pcorr` keeps v1-style pressure correction.

  `-p-mode absolute` reconstructs HbyA by adding `rAU*grad(pOld)` back to the momentum predictor, solves the pressure equation for absolute pressure, relaxes `p = pOld + pRelax*(pAbs-pOld)`, and corrects `U = HbyA - rAU*grad(p)`.

- Added Rhie-Chow switch:

  `-rc-mode old` keeps the explicit pressure-consistency term in predictor flux.

  `-rc-mode oflike` removes the explicit term from predictor flux; pressure enters through the pressure equation flux and velocity correction.

- Added rAU mode:

  `-rAU-mode relaxed` uses `V/aP_relaxed`, matching OpenFOAM's usual `rAU()` after equation relaxation.

  `-rAU-mode raw` uses `V/aP_raw`; internally this rescales the relaxed matrix diagonal by `uRelax` before extracting rAU.

- Added pressure and momentum non-orthogonal scaling:

  `-p-nonorth-scale 0` gives uncorrected pressure laplacian/flux.

  `-mom-nonorth-scale 0` gives uncorrected momentum diffusion correction.

- Added pressure coefficient scale:

  `-p-coeff-scale 1` is normal. Other values are diagnostic only.

- Added robust geometry mode:

  `-geom-method robust` uses triangulated/area-weighted face geometry and robust cell centroid/volume reconstruction.

  `-geom-method legacy` keeps old geometry behavior.

- Added LSQ weight option:

  `-lsq-weight-power 1` or `2` changes compact LSQ weighting.

  `-lsq-stencil extended` is accepted for command compatibility but this generic app currently still uses compact LSQ coefficients. Full extended second-ring LSQ was not ported in this pass.

## Recommended robust starting command

Use this first when comparing to the recent monolithic branch:

```bash
mpirun -n 1 "$EXE" \
  -polyMeshDir /tmp/meshCase/constant/polyMesh \
  -out-prefix "$RUNROOT/case" \
  -device 0 \
  -rho 1 \
  -mu 1e5 \
  -uMean 1.0e-3 \
  -wall-patch patch_0_0 \
  -inlet-patch patch_2_0 \
  -outlet-patch patch_1_0 \
  -geom-method robust \
  -lsq-stencil compact \
  -lsq-weight-power 1 \
  -mom-nonorth-scale 0 \
  -p-nonorth-scale 0 \
  -p-mode absolute \
  -p-coeff-scale 1 \
  -rc-mode old \
  -rAU-mode raw \
  -u-relax 0.7 \
  -p-relax 0.3 \
  -tolMass 1e-3 \
  -tolVel 1e-3 \
  -vel-maxit 2000 \
  -vel-tol 0 \
  -vel-reltol 1e-10 \
  -p-use-amg 1 \
  -p-maxit 4000 \
  -p-tol 1e-10 \
  -p-reltol 0 \
  -p-amg-coarsen-type 8 \
  -p-amg-interp-type 6 \
  -p-amg-agg-levels 1 \
  -p-amg-keep-transpose 1 \
  -p-amg-rebuild-every 1 \
  -nVelNonOrthCorr 0 \
  -nNonOrthCorr 0 \
  -nPressureCorr 0 \
  -nsteps 500 \
  -print-every 5 \
  -write-vtu 1 \
  -write-every 0 \
  2>&1 | tee "$RUNROOT/case.log"
```

Also test `-rc-mode oflike` and `-rAU-mode relaxed`; in the monolithic branch these sometimes behaved similarly on refined meshes but could differ on borderline coarse/bad meshes.

## Build status

This package was patched in an environment without CUDA/nvcc, so it has not been compiled here. The changes are intentionally localized and the compile commands below are explicit so build errors can be reported cleanly if any remain.
