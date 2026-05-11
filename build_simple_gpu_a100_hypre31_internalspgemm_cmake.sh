#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

CUDA_HOME=${CUDA_HOME:-/usr/local/cuda-12.6}
HYPRE_ROOT=${HYPRE_ROOT:-/opt/hypre-3.1.0-cuda-real}
HYPRE_LIBRARY=${HYPRE_LIBRARY:-$HYPRE_ROOT/lib/libHYPRE.a}
BUILD_DIR=${BUILD_DIR:-build_a100_hypre31_static_sm80_internalspgemm}

rm -rf "$BUILD_DIR"

cmake -S . -B "$BUILD_DIR" \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CUDA_COMPILER="$CUDA_HOME/bin/nvcc" \
  -DCMAKE_CUDA_ARCHITECTURES=80 \
  -DHYPRE_ROOT="$HYPRE_ROOT" \
  -DHYPRE_INCLUDE_DIR="$HYPRE_ROOT/include" \
  -DHYPRE_LIBRARY="$HYPRE_LIBRARY" \
  -DANABASIS_USE_HYPRE_INTERNAL_SPGEMM=ON \
  -DANABASIS_HYPRE_STATIC_CUDA_DEPS=ON

cmake --build "$BUILD_DIR" --target simple_gpu -j"$(nproc)"

cp -f "$BUILD_DIR/apps/simple_gpu/simple_gpu" ./simple_gpu
ls -lh ./simple_gpu
