# Anabasis v1.1b A100 / hypre 3.1 internal SpGEMM build

Large A100 runs can fail in hypre BoomerAMG setup when hypre uses the vendor/cuSPARSE SpGEMM path. The v1.1b cloud build supports forcing hypre's internal device SpGEMM:

```cpp
HYPRE_Initialize();
HYPRE_DeviceInitialize();
HYPRE_SetSpGemmUseVendor(0);
```

Use hypre 3.1.0 static CUDA build and compile Anabasis with `ANABASIS_USE_HYPRE_INTERNAL_SPGEMM=1`.

Direct build:

```bash
export CUDA_HOME=/usr/local/cuda-12.6
export HYPRE_ROOT=/opt/hypre-3.1.0-cuda-real
export HYPRE_LIBRARY=$HYPRE_ROOT/lib/libHYPRE.a
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}
SM_ARCH=sm_80 ./build_simple_gpu_a100_hypre31_internalspgemm.sh
```

CMake build:

```bash
export CUDA_HOME=/usr/local/cuda-12.6
export HYPRE_ROOT=/opt/hypre-3.1.0-cuda-real
export HYPRE_LIBRARY=$HYPRE_ROOT/lib/libHYPRE.a
./build_simple_gpu_a100_hypre31_internalspgemm_cmake.sh
```

Expected startup line:

```text
HYPRE SpGEMM backend switch: forced internal SpGEMM via HYPRE_SetSpGemmUseVendor(0)
```
