# Anabasis v0

Anabasis v0 is an early CUDA/HYPRE finite-volume CFD solver checkpoint focused on a GPU-accelerated SIMPLE-style incompressible-flow loop using OpenFOAM polyMesh input.

The current recommended app is:

    apps/generic_simple_v1

It reads an OpenFOAM constant/polyMesh, applies runtime boundary conditions from a .case file, solves the steady incompressible SIMPLE loop on the GPU, writes VTU output, and can integrate raw Cartesian forces over any named wall patch.

---

## Current validated example

The repository includes a small cylinder/channel polyhedral mesh:

    examples/cylinder/constant/polyMesh

and a matching case file:

    cases/cylinder.case

This example uses:

- patch_1_0 as inlet
- patch_2_0 as outlet
- patch_0_0 and patch_5_0 as outer no-slip walls
- patch_3_0 as the cylinder wall and force-integration patch

The case validates the current runtime BC system and generic raw Cartesian force output.

---

## Build

Example local build for RTX 3060 / CUDA architecture 86:

    cd ~/anabasis_v0

    cmake -S . -B build \
      -DCMAKE_BUILD_TYPE=Release \
      -DCMAKE_CUDA_ARCHITECTURES=86

    cmake --build build --target generic_simple_v1 -j"$(nproc)"

The GPU path requires a CUDA-enabled HYPRE build. A CPU-only HYPRE build is not sufficient for the intended GPU solver path.

If you use a PETSc-installed HYPRE, set your PETSc/HYPRE environment before configuring, for example:

    export PETSC_DIR=$HOME/src/petsc
    export PETSC_ARCH=arch-linux-cuda-opt
    export HYPRE_ROOT=$PETSC_DIR/$PETSC_ARCH
    export HYPRE_DIR=$PETSC_DIR/$PETSC_ARCH
    export LD_LIBRARY_PATH=$PETSC_DIR/$PETSC_ARCH/lib:$LD_LIBRARY_PATH

For A100, use CUDA architecture 80 instead of 86.

---

## Run the cylinder example

    cd ~/anabasis_v0

    mpirun -n 1 ./build/apps/generic_simple_v1/generic_simple_v1 \
      -case-config cases/cylinder.case

The solver prints:

- mesh statistics
- all detected boundary patches
- runtime BC patch coverage
- SIMPLE iteration history
- final residual summary
- raw Cartesian force output
- VTU output path

The final VTU file can be opened in ParaView.

---

## Runtime boundary conditions

Every boundary patch in the OpenFOAM polyMesh/boundary file must have exactly one velocity BC and one pressure BC.

Example:

    velocity patch_0_0 wall_noslip
    velocity patch_5_0 wall_noslip
    velocity patch_3_0 wall_noslip
    velocity patch_1_0 parabolic_box_inlet 0.41 0.45 y z average_patch_normal
    velocity patch_2_0 zero_gradient

    pressure patch_0_0 zero_gradient
    pressure patch_5_0 zero_gradient
    pressure patch_3_0 zero_gradient
    pressure patch_1_0 zero_gradient
    pressure patch_2_0 fixed_value 0.0

Supported velocity BCs currently include:

    velocity <patch> wall_noslip
    velocity <patch> zero_gradient
    velocity <patch> fixed_uniform_vector <ux> <uy> <uz>
    velocity <patch> fixed_normal_speed <value> [average_patch_normal|local_face_normal]
    velocity <patch> fixed_flow_rate <value> [average_patch_normal|local_face_normal]
    velocity <patch> parabolic_box_inlet <H> <Umax> <coord1> <coord2> [average_patch_normal|local_face_normal]

Supported pressure BCs currently include:

    pressure <patch> zero_gradient
    pressure <patch> fixed_value <value>
    pressure <patch> open

If any patch is missing a velocity or pressure BC, generic_simple_v1 aborts and prints a patch coverage table.

---

## Raw Cartesian force integration

The generic force postprocessor integrates pressure and viscous traction over a named patch.

Example:

    forceEnable 1
    forcePatch patch_3_0
    forceNormalSign -1

The output is dimensional Cartesian force, not drag/lift coefficient:

    pressureForce     = [Fx, Fy, Fz]
    viscousForce      = [Fx, Fy, Fz]
    totalForce        = [Fx, Fy, Fz]

    pressureForce_x   = ...
    pressureForce_y   = ...
    pressureForce_z   = ...

    viscousForce_x    = ...
    viscousForce_y    = ...
    viscousForce_z    = ...

    totalForce_x      = ...
    totalForce_y      = ...
    totalForce_z      = ...

This is intentionally general and does not require a cylinder diameter, reference velocity, reference area, drag direction, or lift direction.

---

## Numerical method summary

The current generic_simple_v1 app implements a steady SIMPLE-like segregated incompressible finite-volume algorithm.

Each outer iteration does approximately:

1. Apply runtime boundary conditions.
2. Assemble and solve scalar momentum equations for u, v, and w.
3. Extract rAU = V/aP.
4. Build Rhie-Chow-style predicted face fluxes.
5. Assemble/update the pressure equation.
6. Solve pressure correction using HYPRE PCG + BoomerAMG.
7. Apply pressure non-orthogonal correction loops.
8. Correct face fluxes.
9. Update pressure.
10. Correct velocity using pressure-correction gradients.
11. Check continuity and field-change convergence.

The current validated path is steady laminar incompressible flow.

---

## Output

With:

    writeVtu 1
    writeEvery 0

the solver writes a final VTU file.

The example cylinder case writes:

    runs/cylinder_final.vtu

Open this file in ParaView.

---

## Current limitations

- Single MPI rank only.
- No domain-decomposed multi-rank CFD yet.
- Current focus is steady laminar incompressible SIMPLE.
- Runtime BC support is useful but not yet a full OpenFOAM-style dictionary system.
- HYPRE must be CUDA-enabled for the GPU path.
- The codebase is still a research/development checkpoint.

---

## Development direction

Near-term goals:

- keep generic_simple_v1 as the clean main SIMPLE app
- remove older cylinder-specific print blocks from the generic app
- improve force-output file handling
- add more boundary-condition variants
- add stronger validation cases
- modularize solver kernels further
- later add RANS/turbulence support
