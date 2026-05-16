#!/usr/bin/env bash
set -euo pipefail

# Build Anabasis v1.1e.7 coupled transient app with BDF1/BDF2 time scheme, sine inlet, and force CSV logging.
# Expected local defaults from Jaydeep's workstation:
#   CUDA_HOME=/usr/local/cuda-12.2
#   HYPRE_ROOT=/opt/hypre-3.1.0-cuda-real
#   HYPRE_LIBRARY=/opt/hypre-3.1.0-cuda-real/lib/libHYPRE.a
#   SM_ARCH=sm_86

CUDA_HOME=${CUDA_HOME:-/usr/local/cuda-12.2}
HYPRE_ROOT=${HYPRE_ROOT:-/opt/hypre-3.1.0-cuda-real}
HYPRE_LIBRARY=${HYPRE_LIBRARY:-${HYPRE_ROOT}/lib/libHYPRE.a}
SM_ARCH=${SM_ARCH:-sm_86}
ARCH_NUM=${SM_ARCH#sm_}
BUILD_DIR=${BUILD_DIR:-build_v1_1e_7_coupled_transient_nonorth_dp}

export CUDACXX="${CUDA_HOME}/bin/nvcc"

cmake -S . -B "${BUILD_DIR}" \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CUDA_COMPILER="${CUDACXX}" \
  -DCMAKE_CUDA_ARCHITECTURES="${ARCH_NUM}" \
  -DHYPRE_ROOT="${HYPRE_ROOT}" \
  -DHYPRE_LIBRARY="${HYPRE_LIBRARY}" \
  -DANABASIS_HYPRE_STATIC_CUDA_DEPS=ON \
  -DANABASIS_USE_HYPRE_INTERNAL_SPGEMM=ON

cmake --build "${BUILD_DIR}" --target coupled_transient_nonorth_gpu -j "$(nproc)"

cp "${BUILD_DIR}/apps/coupled_transient_nonorth_gpu/coupled_transient_nonorth_gpu" ./coupled_transient_nonorth_gpu_dp
ln -sf coupled_transient_nonorth_gpu_dp ./coupled_transient_nonorth_gpu

echo "Built double-precision v1.1e.7 executable: ./coupled_transient_nonorth_gpu_dp"
echo "Symlinked ./coupled_transient_nonorth_gpu -> coupled_transient_nonorth_gpu_dp"
