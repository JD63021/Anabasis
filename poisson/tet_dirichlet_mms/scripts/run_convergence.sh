#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -lt 2 ]; then
  echo "Usage: $0 <executable> <meshDir1> [meshDir2 ...]"
  exit 1
fi

EXE="$1"
shift

printf "%12s  %16s  %16s  %16s\n" "mesh" "L1" "L2" "Linf"
for MESH in "$@"; do
  OUT=$(mpirun -n 1 "$EXE" -polyMeshDir "$MESH" -monitor 0)
  L1=$(echo "$OUT"   | awk '/L1 error/   {print $4}')
  L2=$(echo "$OUT"   | awk '/L2 error/   {print $4}')
  LI=$(echo "$OUT"   | awk '/Linf error/ {print $4}')
  printf "%12s  %16s  %16s  %16s\n" "$(basename "$MESH")" "$L1" "$L2" "$LI"
done
