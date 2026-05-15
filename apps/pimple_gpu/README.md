# apps/pimple_gpu

Transient incompressible PIMPLE solver app for OpenFOAM `polyMesh` input.

Validated v1.1d-final path:

```text
timeScheme BDF2
transientMomentum 1
nOuterCorrectors 2
nCorrectors 2
pSolveMode ofAbsolute
rcMode oflike
pGradScheme gauss
pDeltaMinCos 0.1
```

Build:

```bash
./build_pimple_gpu_bdf2_hypre31_double.sh
```

Run reference case:

```bash
mpirun -n 1 ./pimple_gpu_bdf2_dp \
  -case-config cases/pimple/cylinder_3d3z_sine_re100_bdf2_800k.case
```
