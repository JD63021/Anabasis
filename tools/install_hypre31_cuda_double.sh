#!/usr/bin/env bash
set -euo pipefail

CUDA_HOME=${CUDA_HOME:-/usr/local/cuda}
CUDA_ARCH=${CUDA_ARCH:-86}
PREFIX=${PREFIX:-/opt/hypre-3.1.0-cuda-real}
SRCROOT=${SRCROOT:-$HOME/src/hypre-3.1.0-cuda-real-src}
BUILD=${BUILD:-$SRCROOT/build-sm${CUDA_ARCH}-double}

export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"

rm -rf "$SRCROOT"
git clone --depth 1 --branch v3.1.0 https://github.com/hypre-space/hypre.git "$SRCROOT"

cmake -S "$SRCROOT/src" -B "$BUILD" -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_INSTALL_PREFIX="$PREFIX" \
  -DBUILD_SHARED_LIBS=OFF \
  -DHYPRE_ENABLE_MPI=ON \
  -DHYPRE_ENABLE_CUDA=ON \
  -DHYPRE_ENABLE_CUSPARSE=ON \
  -DHYPRE_ENABLE_CUBLAS=ON \
  -DHYPRE_ENABLE_CURAND=ON \
  -DHYPRE_ENABLE_CUSOLVER=ON \
  -DHYPRE_ENABLE_UMPIRE=OFF \
  -DCMAKE_CUDA_ARCHITECTURES="$CUDA_ARCH" \
  -DCMAKE_C_COMPILER=mpicc \
  -DCMAKE_CXX_COMPILER=mpicxx \
  -DCMAKE_CUDA_COMPILER="$CUDA_HOME/bin/nvcc"

cmake --build "$BUILD" -j"$(nproc)"
cmake --install "$BUILD"

cat >/tmp/check_hypre_double.cpp <<'CPP'
#include "HYPRE.h"
static_assert(sizeof(HYPRE_Real) == 8, "Expected double-precision HYPRE_Real");
int main(){ return 0; }
CPP
mpicxx -I"$PREFIX/include" -c /tmp/check_hypre_double.cpp -o /tmp/check_hypre_double.o

echo "Installed double-precision hypre 3.1 CUDA to: $PREFIX"
ls -lh "$PREFIX/lib/libHYPRE.a"
