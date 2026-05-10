#!/usr/bin/env bash
set -euo pipefail

CUDA_HOME=${CUDA_HOME:-/usr/local/cuda}
CUDA_ARCH=${CUDA_ARCH:-86}
PREFIX=${PREFIX:-/opt/hypre-3.1.0-cuda-single}
SRCROOT=${SRCROOT:-$HOME/src/hypre-3.1.0-cuda-single-src}
BUILD=${BUILD:-$SRCROOT/build-sm${CUDA_ARCH}-single}

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
  -DHYPRE_ENABLE_SINGLE=ON \
  -DCMAKE_CUDA_ARCHITECTURES="$CUDA_ARCH" \
  -DCMAKE_C_COMPILER=mpicc \
  -DCMAKE_CXX_COMPILER=mpicxx \
  -DCMAKE_CUDA_COMPILER="$CUDA_HOME/bin/nvcc"

cmake --build "$BUILD" -j"$(nproc)"
cmake --install "$BUILD"

cat >/tmp/check_hypre_single.cpp <<'CPP'
#include "HYPRE.h"
static_assert(sizeof(HYPRE_Real) == 4, "Expected single-precision HYPRE_Real. Check whether HYPRE_ENABLE_SINGLE was accepted by hypre 3.1 CMake.");
int main(){ return 0; }
CPP
mpicxx -I"$PREFIX/include" -c /tmp/check_hypre_single.cpp -o /tmp/check_hypre_single.o

echo "Installed single-precision hypre 3.1 CUDA to: $PREFIX"
ls -lh "$PREFIX/lib/libHYPRE.a"
