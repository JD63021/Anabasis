#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

NVCC=${NVCC:-/usr/local/cuda-12.2/bin/nvcc}
PETSC_DIR=${PETSC_DIR:-$HOME/src/petsc}
PETSC_ARCH=${PETSC_ARCH:-arch-linux-cuda-opt}
SM_ARCH=${SM_ARCH:-sm_86}

SRC_APP=(
  apps/simple_gpu/src/main.cu
  apps/simple_gpu/src/bc_specs.cu
  apps/simple_gpu/src/patch_geometry.cu
  apps/simple_gpu/src/velocity_bc_eval.cu
  apps/simple_gpu/src/bc_runtime_config.cu
)
SRC_LIBPOISSON=(
  libpoisson/src/poisson_library.cu
  libpoisson/src/scalar_elliptic.cu
  libpoisson/src/gradient.cu
  libpoisson/src/mesh.cu
  libpoisson/src/hypre_backend.cu
  libpoisson/src/bc.cu
)
SRC_LIBSCALAR=(
  libscalar/src/scalar_transport_library.cu
)

EXE=simple_gpu
LOG=build_simple_gpu.log

MPI_COMPILE="$(mpicxx --showme:compile 2>/dev/null || true)"
MPI_LINK="$(mpicxx --showme:link 2>/dev/null || true)"

rm -f "$EXE"

"$NVCC" -std=c++17 -O3 \
  -arch="$SM_ARCH" \
  -ccbin "$(command -v mpicxx)" \
  -Iapps/simple_gpu/src \
  -Ilibpoisson/include \
  -Ilibscalar/include \
  -I"$PETSC_DIR/$PETSC_ARCH/include" \
  -I"$PETSC_DIR/include" \
  $MPI_COMPILE \
  "${SRC_APP[@]}" \
  "${SRC_LIBPOISSON[@]}" \
  "${SRC_LIBSCALAR[@]}" \
  -L"$PETSC_DIR/$PETSC_ARCH/lib" \
  -Xlinker -rpath -Xlinker "$PETSC_DIR/$PETSC_ARCH/lib" \
  -lHYPRE \
  $MPI_LINK \
  -lcudart -lcusparse -lcublas -lcurand -lcusolver \
  -o "$EXE" \
  2>&1 | tee "$LOG"

if [[ -x "$EXE" ]]; then
  echo "Built successfully: ./$(basename "$EXE")"
  ls -lh "$EXE"
else
  echo "BUILD FAILED. See $LOG"
  exit 1
fi
