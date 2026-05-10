#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"

if [[ -z "${CUDA_HOME:-}" ]]; then
  if command -v nvcc >/dev/null 2>&1; then
    CUDA_HOME="$(cd "$(dirname "$(command -v nvcc)")/.." && pwd)"
  else
    CUDA_HOME=/usr/local/cuda
  fi
fi
NVCC=${NVCC:-$CUDA_HOME/bin/nvcc}
HYPRE_ROOT=${HYPRE_ROOT:-/opt/hypre-3.1.0-cuda-single}
HYPRE_LIBRARY=${HYPRE_LIBRARY:-$HYPRE_ROOT/lib/libHYPRE.a}
SM_ARCH=${SM_ARCH:-sm_86}
EXE=${EXE:-simple_gpu_sp}
LOG=${LOG:-build_simple_gpu_hypre31_single.log}

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

MPI_COMPILE="$(mpicxx --showme:compile 2>/dev/null || true)"
MPI_LINK="$(mpicxx --showme:link 2>/dev/null || true)"

if [[ ! -x "$NVCC" ]]; then echo "ERROR: nvcc not found at $NVCC"; exit 1; fi
if [[ ! -f "$HYPRE_LIBRARY" ]]; then echo "ERROR: HYPRE_LIBRARY not found: $HYPRE_LIBRARY"; exit 1; fi
if [[ ! -d "$HYPRE_ROOT/include" ]]; then echo "ERROR: HYPRE include dir not found: $HYPRE_ROOT/include"; exit 1; fi

rm -f "$EXE"

"$NVCC" -std=c++17 -O3 \
  -DANABASIS_EXPECT_HYPRE_COMPLEX_BYTES=4 \
  -arch="$SM_ARCH" \
  -ccbin "$(command -v mpicxx)" \
  -Iapps/simple_gpu/src \
  -Ilibpoisson/include \
  -Ilibscalar/include \
  -I"$HYPRE_ROOT/include" \
  $MPI_COMPILE \
  "${SRC_APP[@]}" \
  "${SRC_LIBPOISSON[@]}" \
  "${SRC_LIBSCALAR[@]}" \
  "$HYPRE_LIBRARY" \
  $MPI_LINK \
  -L"$CUDA_HOME/lib64" \
  -lcudart -lcusparse -lcublas -lcublasLt -lcurand -lcusolver -ldl \
  -o "$EXE" \
  2>&1 | tee "$LOG"

if [[ -x "$EXE" ]]; then
  echo "Built single-HYPRE executable: ./$EXE"
  ls -lh "$EXE"
else
  echo "BUILD FAILED. See $LOG"
  exit 1
fi
