# Anabasis v1.1b: hypre 3.1 internal SpGEMM default

v1.1b makes standalone hypre 3.1 the default build target and forces hypre's internal device SpGEMM at runtime using:

```cpp
HYPRE_Initialize();
HYPRE_DeviceInitialize();
HYPRE_SetSpGemmUseVendor(0);
```

This is the robustness path validated earlier for large A100 meshes where cuSPARSE/vendor SpGEMM can fail during BoomerAMG setup.

## Precision variants

- `simple_gpu_dp`: double-precision HYPRE build, expected `sizeof(HYPRE_Complex)==8`.
- `simple_gpu_sp`: single-precision HYPRE build, expected `sizeof(HYPRE_Complex)==4`.

The CFD fields remain stored as double in Anabasis. The single-HYPRE executable converts double field guesses/results to/from HYPRE's single-precision solver vectors. Matrix values and RHS use `HYPRE_Complex`, so in the single build the linear solves and AMG hierarchy use single precision.

## Build scripts

```bash
./build_simple_gpu_hypre31_double.sh
./build_simple_gpu_hypre31_single.sh
```

`./build_simple_gpu.sh` defaults to the double hypre 3.1 build.
