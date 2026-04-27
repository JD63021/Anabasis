#!/usr/bin/env bash
set -euo pipefail

EXE="./build/apps/simple/simple"
CACHE="./build/CMakeCache.txt"

echo "============================================================"
echo "HOST / GPU"
echo "============================================================"
hostname || true
nvidia-smi || true
nvcc --version || true

echo
echo "============================================================"
echo "CMake HYPRE variables"
echo "============================================================"
if [ -f "$CACHE" ]; then
  grep -E "HYPRE|CMAKE_CUDA_ARCHITECTURES|CMAKE_CUDA_COMPILER|CUDAToolkit" "$CACHE" || true
else
  echo "No $CACHE found"
fi

echo
echo "============================================================"
echo "Executable ldd"
echo "============================================================"
if [ -x "$EXE" ]; then
  ldd "$EXE" | egrep -i "hypre|cuda|cudart|mpi|stdc" || true
else
  echo "No executable found at $EXE"
fi

echo
echo "============================================================"
echo "Resolved HYPRE library symbols"
echo "============================================================"
if [ -x "$EXE" ]; then
  HYPRE_SO=$(ldd "$EXE" | awk '/[Hh][Yy][Pp][Rr][Ee]/ {print $3; exit}')
  echo "HYPRE_SO=$HYPRE_SO"
  if [ -n "${HYPRE_SO:-}" ] && [ -f "$HYPRE_SO" ]; then
    nm -D "$HYPRE_SO" | egrep "HYPRE_IJMatrixInitialize_v2|HYPRE_IJVectorInitialize_v2|HYPRE_IJMatrixMigrate|HYPRE_IJVectorMigrate|hypre_CSRMatrixMigrate|hypre_ParCSRMatrixMigrate" || true
    echo
    echo "strings GPU/CUDA hints:"
    strings "$HYPRE_SO" | egrep -i "cuda|gpu|cusparse|cublas|thrust|device|unified" | head -80 || true
  fi
fi

echo
echo "============================================================"
echo "HYPRE header config hints"
echo "============================================================"
if [ -f "$CACHE" ]; then
  INC=$(grep '^HYPRE_INCLUDE_DIR:' "$CACHE" | sed 's/^[^=]*=//')
  echo "HYPRE_INCLUDE_DIR=$INC"
  if [ -d "$INC" ]; then
    grep -R "HYPRE_USING_CUDA\|HYPRE_USING_GPU\|HYPRE_USING_CUSPARSE\|HYPRE_MEMORY_DEVICE\|HYPRE_USING_UNIFIED_MEMORY" "$INC" 2>/dev/null | head -120 || true
  fi
fi
