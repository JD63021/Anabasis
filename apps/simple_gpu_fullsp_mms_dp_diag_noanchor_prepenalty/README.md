# Anabasis `simple_gpu`

`simple_gpu` is the current CUDA/HYPRE finite-volume SIMPLE solver app in Anabasis v1.1.

It reads an OpenFOAM `polyMesh`, runtime boundary conditions from a flat `.case` file, and solves steady incompressible flow with a GPU-assembled/GPU-solved segregated SIMPLE loop. It also exposes Poisson and passive-scalar module options for the modular PDE-solver direction of the codebase.

Recommended robust pressure-velocity mode:

```text
pMode absolute
pSolveMode ofAbsolute
rcMode oflike
pGradScheme gauss
pDeltaMode of
```

Do not use `pSolveMode ofAbsolute` with `rcMode old`.
