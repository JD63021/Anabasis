# Build notes

## Dependencies

The direct build script expects:

- CUDA 12.x with `nvcc`
- OpenMPI / `mpicxx`
- PETSc/HYPRE build with GPU support, usually under `$HOME/src/petsc/$PETSC_ARCH`

## RTX 3060 build

```bash
cd ~/anabasis_v1_1

export PETSC_DIR=$HOME/src/petsc
export PETSC_ARCH=arch-linux-cuda-opt
export LD_LIBRARY_PATH="$PETSC_DIR/$PETSC_ARCH/lib:${LD_LIBRARY_PATH:-}"

SM_ARCH=sm_86 ./build_simple_gpu.sh
```

## A100 build

```bash
cd ~/anabasis_v1_1

export PETSC_DIR=$HOME/src/petsc
export PETSC_ARCH=arch-linux-cuda-opt
export LD_LIBRARY_PATH="$PETSC_DIR/$PETSC_ARCH/lib:${LD_LIBRARY_PATH:-}"

SM_ARCH=sm_80 ./build_simple_gpu.sh
```

The executable is:

```text
./simple_gpu
```
