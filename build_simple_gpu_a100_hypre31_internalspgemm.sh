#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
SM_ARCH=${SM_ARCH:-sm_80} EXE=${EXE:-simple_gpu_dp} ./build_simple_gpu_hypre31_double.sh "$@"
