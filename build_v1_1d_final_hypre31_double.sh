#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"

echo "=== Building steady SIMPLE double executable ==="
EXE=simple_gpu_dp LOG=build_simple_gpu_hypre31_double.log ./build_simple_gpu_hypre31_double.sh

echo

echo "=== Building transient PIMPLE/BDF2 double executable ==="
EXE=pimple_gpu_bdf2_dp LOG=build_pimple_gpu_bdf2_hypre31_double.log ./build_pimple_gpu_bdf2_hypre31_double.sh

echo

echo "Built both final double-precision executables:"
ls -lh simple_gpu_dp simple_gpu pimple_gpu_bdf2_dp pimple_gpu
