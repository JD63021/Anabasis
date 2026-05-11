#!/usr/bin/env bash
set -euo pipefail
# v1.1b default build: standalone HYPRE 3.1 double precision with internal SpGEMM forced in source.
cd "$(dirname "$0")"
exec ./build_simple_gpu_hypre31_double.sh "$@"
