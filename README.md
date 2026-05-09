# Anabasis

Anabasis is a journey towards a modular GPU finite-volume codebase for CFD and transport problems.

The current public focus is a CUDA/HYPRE steady incompressible SIMPLE solver that reads OpenFOAM `polyMesh` meshes and runtime boundary conditions from a flat `.case` file. The code is being organized toward a modular PDE-solver layout with separate flow, Poisson, and scalar-transport options.

## Current solver

The main application is:

```text
apps/simple_gpu
```

Direct-build executable:

```text
./simple_gpu
```

The solver currently supports:

- Steady segregated SIMPLE incompressible flow.
- OpenFOAM-inspired absolute-pressure/HbyA mode:

  ```text
  pMode absolute
  pSolveMode ofAbsolute
  rcMode oflike
  pGradScheme gauss
  ```

- CUDA assembly and GPU linear solves through HYPRE/PETSc linkage.
- Runtime velocity and pressure boundary conditions from the `.case` file.
- Momentum convection choice:

  ```text
  momentumConvectionScheme central
  momentumConvectionScheme upwind
  ```

- Poisson module gradient choice:

  ```text
  poissonGradientScheme gauss
  poissonGradientScheme lsq
  ```

- Passive scalar transport module options, including:

  ```text
  scalarConvectionScheme upwind
  scalarConvectionScheme central
  ```

- Optional cylinder / patch force postprocessing, enabled only by:

  ```text
  forceEnable 1
  ```

## Repository layout

```text
apps/simple_gpu/       Main SIMPLE flow solver app
libpoisson/            Mesh, BC, gradient, HYPRE, and Poisson/elliptic utilities
libscalar/             Passive scalar transport library
cases/reference.case   Verbose documented reference case
cases/cylinder.case    Cylinder benchmark case
docs/INSTALL.md        Build notes for RTX 3060 and A100
docs/RUN_CYLINDER.md   Full cylinder run command
build_simple_gpu.sh    Direct NVCC build script
```

## Build on RTX 3060

```bash
cd ~/anabasis_v1_1

export PETSC_DIR=$HOME/src/petsc
export PETSC_ARCH=arch-linux-cuda-opt
export LD_LIBRARY_PATH="$PETSC_DIR/$PETSC_ARCH/lib:${LD_LIBRARY_PATH:-}"

SM_ARCH=sm_86 ./build_simple_gpu.sh
```

## Build on A100

```bash
cd ~/anabasis_v1_1

export PETSC_DIR=$HOME/src/petsc
export PETSC_ARCH=arch-linux-cuda-opt
export LD_LIBRARY_PATH="$PETSC_DIR/$PETSC_ARCH/lib:${LD_LIBRARY_PATH:-}"

SM_ARCH=sm_80 ./build_simple_gpu.sh
```

## Run the cylinder case

The mesh should be available at:

```text
/tmp/meshCase/constant/polyMesh
```

Run:

```bash
cd ~/anabasis_v1_1

mkdir -p runs/cylinder

mpirun -n 1 ./simple_gpu \
  -case-config cases/cylinder.case \
  -out-prefix runs/cylinder/case \
  2>&1 | tee runs/cylinder/run.log
```

Check the force output:

```bash
grep -E "Ubar, D, H|CD_vector|CL_y_vector|CL_z_vector|Wrote VTU" runs/cylinder/run.log
```

## Case files

The case format is intentionally flat key/value text. Sections such as `[mesh/output]`, `[physics/fluid]`, `[Poisson module options]`, and `[Scalar transport module options]` are comments only. They are there to make future multiphysics additions easier without changing the parser.

Use `cases/reference.case` as the documented template and `cases/cylinder.case` as the cylinder benchmark run file.

## Notes

For the OpenFOAM-like absolute-pressure path, keep:

```text
pSolveMode ofAbsolute
rcMode oflike
pGradScheme gauss
```

Avoid:

```text
pSolveMode ofAbsolute
rcMode old
```

That combination was unstable in development because the old explicit Rhie-Chow pressure term conflicts with the OpenFOAM-style absolute pressure flux path.
