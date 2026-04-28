# Anabasis v0

**Anabasis v0** is a CUDA/HYPRE finite-volume solver checkpoint built around a fast SIMPLE-style incompressible-flow loop on OpenFOAM `polyMesh` input. The name marks this as the beginning of a broader journey toward a modular, configurable finite-volume CFD codebase.

This repository currently focuses on a steady laminar incompressible SIMPLE solver with:

- OpenFOAM `constant/polyMesh` reader
- cell-centered finite-volume unknowns
- unstructured polyhedral/tetrahedral mesh support through OpenFOAM-style owner/neighbour/connectivity arrays
- GPU linear-system assembly for the SIMPLE loop
- HYPRE IJ/ParCSR GPU solves
- BoomerAMG pressure preconditioning
- BiCGSTAB velocity solves with diagonal scaling
- GPU pressure correction machinery
- modular boundary-condition wrappers
- runtime case configuration files
- direct `rho` and `mu` input
- normal-velocity inlet support

The currently recommended executable is:

```bash
simple
```

The main source file is:

```text
apps/simple/src/main.cu
```

---

## Current status

This is a validated working checkpoint, not a finished general CFD package.

The validated path is:

```text
OpenFOAM polyMesh
    -> Anabasis mesh reader
    -> modular BC/case config setup
    -> GPU SIMPLE loop
    -> HYPRE GPU ParCSR solves
    -> VTU output
```

The old development tree had multiple experimental executables. In this cleaned checkpoint, the important path has been renamed to:

```text
apps/simple
```

and builds to:

```text
build/apps/simple/simple
```

---

## Solver overview

The solver implements a steady SIMPLE-like segregated incompressible-flow algorithm.

At a high level, each SIMPLE outer iteration does:

1. Apply boundary conditions from the runtime case/BC configuration.
2. Assemble and solve the three scalar momentum equations for `u`, `v`, and `w`.
3. Extract `rAU = V / aP` from the relaxed momentum matrix diagonal.
4. Build Rhie-Chow-style predicted face fluxes.
5. Assemble/update the pressure equation using `rAU`.
6. Solve pressure correction with HYPRE PCG + BoomerAMG.
7. Apply non-orthogonal pressure correction loops.
8. Correct face fluxes.
9. Update pressure.
10. Correct velocity using the pressure-correction gradient.
11. Check continuity and relative field changes.

The code is currently intended for **single MPI rank** GPU runs:

```bash
mpirun -n 1 ./build/apps/simple/simple -case-config cases/<case>.case
```

The HYPRE interface uses MPI, but this checkpoint is not a domain-decomposed multi-rank solver.

---

## Numerical method

### Spatial layout

The current solver is cell-centered:

- pressure stored at cells
- velocity components stored at cells
- face fluxes computed on faces
- owner/neighbour face connectivity from OpenFOAM `polyMesh`

### Momentum equations

The velocity components are solved as scalar finite-volume equations. The velocity matrix assembly is GPU-based. The momentum solve uses HYPRE ParCSR BiCGSTAB with diagonal scaling.

Important controls include:

```text
velMaxit
velTol
velRelTol
uRelax
nVelNonOrthCorr
```

### Pressure correction

The pressure equation uses HYPRE ParCSR PCG with BoomerAMG. The pressure matrix is assembled on the GPU and can reuse/rebuild the AMG setup depending on the selected settings.

Important controls include:

```text
pUseAmg
pMaxit
pTol
pRelTol
pAmgSetupScope
pAmgMaxit
pAmgCoarsenType
pAmgInterpType
pAmgAggLevels
pAmgRelaxType
pAmgPmax
pAmgKeepTranspose
pAmgRebuildEvery
nNonOrthCorr
nPressureCorr
pRelax
```

### Non-orthogonal corrections

The solver supports pressure non-orthogonal correction loops through:

```text
nNonOrthCorr
```

For the validated pipe cases, `nNonOrthCorr 5` was used.

---

## GPU/HYPRE design

The solver uses a CUDA + HYPRE architecture:

- CUDA kernels assemble matrix coefficients and RHS vectors.
- HYPRE IJ/ParCSR is used as the sparse linear algebra backend.
- Velocity components use BiCGSTAB.
- Pressure uses PCG + BoomerAMG.
- Pressure RHS, face flux correction, continuity residual, pressure-correction gradient, and velocity correction are on the GPU in the fast path.
- HYPRE must be built with real CUDA support.

A correct HYPRE build should show these in `HYPRE_config.h`:

```c
#define HYPRE_USING_CUDA 1
#define HYPRE_USING_CUSPARSE 1
#define HYPRE_USING_GPU 1
```

A CPU-only HYPRE build is not sufficient.

---

## Repository layout

Typical cleaned checkpoint layout:

```text
anabasis_v0/
├── apps/
│   └── simple/
│       ├── CMakeLists.txt
│       └── src/
│           ├── main.cu
│           ├── bc_runtime_config.cu
│           ├── bc_runtime_config.h
│           ├── bc_specs.cu
│           ├── bc_specs.h
│           ├── patch_geometry.cu
│           ├── patch_geometry.h
│           ├── pressure_lib_adapter.cu
│           ├── pressure_lib_adapter.h
│           ├── velocity_bc_eval.cu
│           ├── velocity_bc_eval.h
│           ├── velocity_lib_adapter.cu
│           └── velocity_lib_adapter.h
├── bc_configs/
├── cases/
├── libpoisson/
├── libscalar/
├── poisson/
├── scripts/
├── CMakeLists.txt
└── ANABASIS_V0_CHECKPOINT.md
```

### `apps/simple/src/main.cu`

Main SIMPLE solver driver. It contains:

- argument parsing
- case-config expansion
- OpenFOAM `polyMesh` loading
- mesh/device mesh setup
- case and BC setup
- GPU SIMPLE loop
- HYPRE solver setup
- VTU output

### `bc_specs.*`

Defines boundary-condition specification types for velocity and pressure.

Supported velocity concepts include:

- no-slip wall
- zero-gradient velocity
- fixed uniform vector
- fixed normal speed
- fixed flow rate

Supported pressure concepts include:

- zero-gradient
- fixed value
- open/outlet-like pressure handling

### `bc_runtime_config.*`

Loads boundary-condition lines from a config file.

Typical syntax:

```text
velocity patch_0_0 wall_noslip
velocity patch_1_0 fixed_normal_speed -1.0 average_patch_normal
velocity patch_2_0 zero_gradient

pressure patch_0_0 zero_gradient
pressure patch_1_0 zero_gradient
pressure patch_2_0 fixed_value 0.0
```

### `patch_geometry.*`

Builds patch geometry summaries:

- patch face list
- total patch area
- patch centroid
- average patch normal
- planarity metric

This is useful for diagnosing inlet/outlet orientation and normal-speed sign conventions.

### `libpoisson` and `libscalar`

Reusable finite-volume library components used during earlier validation work and still kept in the tree. They include Poisson/scalar transport routines and MMS validation targets.

---

## Boundary conditions

Boundary conditions can be provided inline inside a `.case` file or through a separate BC config file.

### Important normal-speed sign convention

`fixed_normal_speed` is signed along the **outward mesh normal**.

So:

- positive speed means flow along the outward normal
- negative speed means flow into the domain if the outward normal points out of the inlet

Example:

If an inlet patch has outward normal approximately `+z`, then inflow into the domain is in `-z`, so use:

```text
velocity patch_2_0 fixed_normal_speed -1.0 average_patch_normal
```

If an inlet patch has outward normal approximately `-z`, then inflow into the domain is in `+z`, so also use:

```text
velocity patch_1_0 fixed_normal_speed -1.0 average_patch_normal
```

because the speed is measured along the outward normal.

### Example: reverse pipe case

In this case, `patch_2_0` is the normal-speed inlet and `patch_1_0` is the pressure outlet:

```text
velocity patch_0_0 wall_noslip
velocity patch_2_0 fixed_normal_speed -1.0 average_patch_normal
velocity patch_1_0 zero_gradient

pressure patch_0_0 zero_gradient
pressure patch_2_0 zero_gradient
pressure patch_1_0 fixed_value 0.0
```

---

## Case configuration files

The preferred run style is:

```bash
mpirun -n 1 ./build/apps/simple/simple -case-config cases/<case>.case
```

A case file can include:

- mesh path
- output prefix
- GPU device
- fluid properties
- patch names
- SIMPLE controls
- velocity solver controls
- pressure/HYPRE/AMG controls
- profiling controls
- boundary conditions

Example case:

```text
# Mesh / output
polyMeshDir /tmp/meshCase/constant/polyMesh
outPrefix runs/case_pipe_reverse_normal_inlet_mu
device 0

# Fluid properties
rho 1.0
mu 0.05

# Patch names used for summaries and initial guess
wallPatch patch_0_0
inletPatch patch_2_0
outletPatch patch_1_0

# SIMPLE controls
nsteps 500
printEvery 10
writeVtu 1
writeEvery 0
monitor 1

nVelNonOrthCorr 0
nNonOrthCorr 5
nPressureCorr 0

uRelax 0.7
pRelax 0.3

corrTol 1e-10
tolMass 1e-3
tolVel 1e-3

# Velocity solver
velMaxit 100
velTol 1e-10
velRelTol 0

# Pressure solver / AMG
pUseAmg 1
pAmgSetupScope outer
pMaxit 4000
pTol 1e-10
pRelTol 0
pAmgMaxit 1
pAmgCoarsenType 8
pAmgInterpType 6
pAmgAggLevels 1
pAmgRelaxType 7
pAmgPmax 4
pAmgKeepTranspose 1
pAmgRebuildEvery 25

# Profiling
profileSteps 0

# Boundary conditions
velocity patch_0_0 wall_noslip
velocity patch_2_0 fixed_normal_speed -1.0 average_patch_normal
velocity patch_1_0 zero_gradient

pressure patch_0_0 zero_gradient
pressure patch_2_0 zero_gradient
pressure patch_1_0 fixed_value 0.0
```

The case-config parser generates a temporary `.generated.bc` file from inline `velocity` and `pressure` lines, then passes it internally through the existing BC config loader.

Command-line arguments after `-case-config` can override case file values. For example:

```bash
mpirun -n 1 ./build/apps/simple/simple \
  -case-config cases/pipe_reverse_normal_inlet_mu.case \
  -write-vtu 0 \
  -out-prefix runs/no_vtu_override
```

---

## Build requirements

### Core requirements

- Linux
- CMake
- CUDA toolkit
- MPI compiler wrappers (`mpicc`, `mpicxx`)
- HYPRE built with CUDA support
- OpenMPI
- C++ compiler
- NVIDIA GPU

### Tested environments

Local development was done on an RTX 3060-class machine with CUDA 12.x and HYPRE/PETSc-based GPU builds.

Cloud testing was done on an A100-SXM4-80GB MIG 7g.80gb device using:

- CUDA toolkit 12.6
- NVIDIA driver 570.133.20
- HYPRE v3.0.0 built with CUDA support
- `CMAKE_CUDA_ARCHITECTURES=80`

---

## Building on local RTX 3060

For RTX 3060, use CUDA architecture 86:

```bash
cd ~/anabasis_v0

rm -rf build

export CUDACXX=/usr/local/cuda-12.2/bin/nvcc
export PATH=/usr/local/cuda-12.2/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda-12.2/lib64:$LD_LIBRARY_PATH

export PETSC_DIR=$HOME/src/petsc
export PETSC_ARCH=arch-linux-cuda-opt
export HYPRE_ROOT=$PETSC_DIR/$PETSC_ARCH
export HYPRE_DIR=$PETSC_DIR/$PETSC_ARCH
export LD_LIBRARY_PATH=$PETSC_DIR/$PETSC_ARCH/lib:$LD_LIBRARY_PATH

cmake -S . -B build \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CUDA_COMPILER=/usr/local/cuda-12.2/bin/nvcc \
  -DCMAKE_CUDA_ARCHITECTURES=86 \
  -DPETSC_DIR=$PETSC_DIR \
  -DPETSC_ARCH=$PETSC_ARCH \
  -DHYPRE_ROOT=$HYPRE_ROOT \
  -DHYPRE_DIR=$HYPRE_DIR

cmake --build build -j"$(nproc)"
```

Run:

```bash
mpirun -n 1 ./build/apps/simple/simple \
  -case-config cases/pipe_reverse_normal_inlet_mu.case
```

---

## Building on A100

For A100, use CUDA architecture 80:

```bash
cd /root/anabasis_v0

unset PETSC_DIR
unset PETSC_ARCH

export CUDA_HOME=/usr/local/cuda-12.6
export HYPRE_ROOT=/opt/hypre-cuda-real
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$HYPRE_ROOT/lib:$CUDA_HOME/lib64:$LD_LIBRARY_PATH

rm -rf build_a100

cmake -S . -B build_a100 \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CUDA_COMPILER=$CUDA_HOME/bin/nvcc \
  -DCMAKE_CUDA_ARCHITECTURES=80 \
  -DHYPRE_ROOT=/opt/hypre-cuda-real \
  -DHYPRE_INCLUDE_DIR=/opt/hypre-cuda-real/include \
  -DHYPRE_LIBRARY=/opt/hypre-cuda-real/lib/libHYPRE.so

cmake --build build_a100 -j"$(nproc)"
```

Run:

```bash
mpirun --allow-run-as-root -n 1 ./build_a100/apps/simple/simple \
  -case-config cases/pipe_reverse_normal_inlet_mu.case
```

---

## Important CUDA architecture note

Do not hardcode the architecture in `apps/simple/CMakeLists.txt`.

Use the configure-time value:

```bash
-DCMAKE_CUDA_ARCHITECTURES=86   # RTX 3060
-DCMAKE_CUDA_ARCHITECTURES=80   # A100
```

If an A100 run fails with a CUDA kernel error like:

```text
named symbol not found
```

check that the executable was not accidentally built for `sm_86`.

---

## Mesh input

The solver reads OpenFOAM `polyMesh` directories:

```text
constant/polyMesh/
├── boundary
├── faces
├── neighbour
├── owner
└── points
```

The case file expects:

```text
polyMeshDir /tmp/meshCase/constant/polyMesh
```

A common cloud workflow is to package only the polyMesh:

```bash
cd /tmp/meshCase/constant

tar -I 'zstd -19 -T0' -cf meshCase_polyMesh.tar.zst polyMesh
```

On the cloud node:

```bash
rm -rf /tmp/meshCase
mkdir -p /tmp/meshCase/constant

tar -I unzstd -xf /root/meshCase_polyMesh.tar.zst -C /tmp/meshCase/constant
```

---

## Running cases

Preferred:

```bash
mpirun -n 1 ./build/apps/simple/simple \
  -case-config cases/pipe_reverse_normal_inlet_mu.case
```

or on a root cloud node:

```bash
mpirun --allow-run-as-root -n 1 ./build_a100/apps/simple/simple \
  -case-config cases/pipe_reverse_normal_inlet_mu.case
```

A convenience script may be available:

```bash
./scripts/run_case.sh cases/pipe_reverse_normal_inlet_mu.case
```

---

## Output

The solver can write `.vtu` files for visualization in ParaView.

Case controls:

```text
writeVtu 1
writeEvery 0
outPrefix runs/case_pipe_reverse_normal_inlet_mu
```

`writeEvery 0` with `writeVtu 1` writes the final VTU.

Typical output fields include:

- pressure `p`
- velocity magnitude
- cell volume
- divergence/continuity residual
- vector velocity `U`

Disable VTU output for timing:

```bash
mpirun -n 1 ./build/apps/simple/simple \
  -case-config cases/pipe_reverse_normal_inlet_mu.case \
  -write-vtu 0
```

---

## Validated pipe case behavior

For the coarse pipe mesh used during validation, the solver reached convergence in about 23 SIMPLE iterations.

Representative results:

```text
Iterations    ≈ 23
max|w|        ≈ 2.01
max|p|        ≈ 3.2e2
massRes       ≈ 1e-12 to 1e-13
```

The exact `max|p|` changes slightly depending on mesh, inlet choice, and whether the case uses the older parabolic fallback or a normal-speed inlet.

---

## Performance notes

The fast path was introduced because the older modular glue path spent significant time in repeated CPU-side gradients and CPU pressure-correction loops.

The current `simple` path is based on the faster GPU path and uses:

- GPU LSQ gradients in the SIMPLE loop
- GPU pressure non-orthogonal flux/divergence
- GPU pressure RHS construction
- GPU face-flux correction
- GPU velocity correction
- direct HYPRE matrix-value writes/reuse
- reusable U/V/W momentum matrix pattern
- reusable pressure AMG setup according to rebuild controls

For small meshes, wall-clock timing is dominated by setup, MPI launch, HYPRE setup, and I/O. Large meshes are better tests of GPU performance.

---

## Known limitations

This is an early checkpoint. Important current limitations include:

1. **Single-rank operation**
   - Current validation is with `mpirun -n 1`.
   - The code is not yet a domain-decomposed parallel CFD solver.

2. **Laminar SIMPLE focus**
   - Current main target is steady laminar incompressible flow.
   - RANS, turbulence models, transient schemes, and multiphase models are not yet included.

3. **Boundary-condition expressions**
   - Normal speed, fixed vectors, walls, fixed pressure, and zero-gradient cases are supported.
   - General analytic `x,y,z,t` expression parsing from config files is not yet a full scripting system.

4. **OpenFOAM mesh reading**
   - The code expects an OpenFOAM-style `polyMesh`.
   - Only the mesh features exercised in the current tests are validated.

5. **Pressure reference**
   - The current validated pipe cases use a fixed pressure outlet, so no separate pressure anchor is needed.
   - Closed-domain pressure-nullspace handling is not the main validated path in this checkpoint.

6. **GPU architecture**
   - Build with the correct `CMAKE_CUDA_ARCHITECTURES` for your GPU.
   - `86` for RTX 3060.
   - `80` for A100.

7. **HYPRE build**
   - HYPRE must be CUDA-enabled.
   - Linking against a CPU HYPRE will fail or give invalid GPU behavior.

---

## Troubleshooting

### CUDA kernel error: `named symbol not found`

Likely wrong CUDA architecture. Rebuild with the architecture for the GPU:

```bash
# A100
-DCMAKE_CUDA_ARCHITECTURES=80

# RTX 3060
-DCMAKE_CUDA_ARCHITECTURES=86
```

Then delete the build folder and rebuild.

### HYPRE pointer or CUDA memory errors

Check HYPRE config:

```bash
grep -E "HYPRE_USING_CUDA|HYPRE_USING_GPU|HYPRE_USING_CUSPARSE" \
  /opt/hypre-cuda-real/include/HYPRE_config.h
```

Expected:

```c
#define HYPRE_USING_CUDA 1
#define HYPRE_USING_CUSPARSE 1
#define HYPRE_USING_GPU 1
```

Check linkage:

```bash
ldd ./build/apps/simple/simple | egrep -i "hypre|cuda|cusparse|cublas|curand|cusolver"
```

### Flow goes the wrong way

Check patch normal summaries printed at startup.

`fixed_normal_speed` is signed along outward normal.

For inflow, the normal speed is usually negative:

```text
velocity inlet_patch fixed_normal_speed -1.0 average_patch_normal
```

### Case file changes do not seem to apply

Look for:

```text
BC config source : <path>.generated.bc
```

If it says:

```text
BC config source : <hardcoded-fallback>
```

then the case file did not provide inline BC lines or `bcConfig`.

---

## Development direction

Anabasis v0 is a foundation for a more general finite-volume CFD codebase.

Near-term development directions:

- make case-config the standard interface
- remove old pipe-specific fallback assumptions
- improve BC expression/function support
- add more robust inlet/outlet BC variants
- add additional discretization scheme controls
- expand validation cases
- modularize solver kernels and HYPRE backend further
- add RANS turbulence-model support
- improve output/diagnostic tooling
- eventually explore multi-rank/domain-decomposed operation

---

## Suggested first test

After building, run:

```bash
mpirun -n 1 ./build/apps/simple/simple \
  -case-config cases/pipe_reverse_normal_inlet_mu.case \
  -write-vtu 1
```

On cloud/root:

```bash
mpirun --allow-run-as-root -n 1 ./build_a100/apps/simple/simple \
  -case-config cases/pipe_reverse_normal_inlet_mu.case \
  -write-vtu 1
```

Open the resulting `.vtu` in ParaView and verify:

- velocity direction matches the chosen inlet/outlet patches
- pressure is zero at the outlet patch
- continuity residual converges
- velocity magnitude is physically reasonable

---
