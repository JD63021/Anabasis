#!/usr/bin/env bash
set -euo pipefail

CASE_FILE="${1:-cases/pipe_reverse_normal_inlet_mu.case}"

mpirun -n 1 ./build/apps/simple/simple \
  -case-config "$CASE_FILE"
