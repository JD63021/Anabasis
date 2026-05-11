#!/usr/bin/env bash
set -euo pipefail

OUTER_RANK="${OMPI_COMM_WORLD_RANK:-0}"
OUTER_LOCAL_RANK="${OMPI_COMM_WORLD_LOCAL_RANK:-0}"
HOST="$(hostname -s)"

APPDIR="$HOME/anabasis_v1_1b_sp"
EXE="$APPDIR/simple_gpu_sp"
MESH="/tmp/meshCase/constant/polyMesh"
CASECFG="$APPDIR/cases/test.case"

RUNROOT="$APPDIR/runs/mpi2_replicated_${HOST}_outerrank${OUTER_RANK}"
mkdir -p "$RUNROOT"

echo "============================================================"
echo "Replicated Anabasis singleton run"
echo "Outer MPI rank   : $OUTER_RANK"
echo "Outer local rank : $OUTER_LOCAL_RANK"
echo "Host             : $HOST"
echo "Executable       : $EXE"
echo "Mesh             : $MESH"
echo "Runroot          : $RUNROOT"
echo "============================================================"

# One GPU per node for now.
export CUDA_VISIBLE_DEVICES=0

# Auto-detect CUDA toolkit path on each node.
if [ -d /usr/local/cuda-12.8 ]; then
  export CUDA_HOME=/usr/local/cuda-12.8
elif [ -d /usr/local/cuda-12.6 ]; then
  export CUDA_HOME=/usr/local/cuda-12.6
elif [ -d /usr/local/cuda-12.2 ]; then
  export CUDA_HOME=/usr/local/cuda-12.2
elif [ -d /usr/local/cuda ]; then
  export CUDA_HOME=/usr/local/cuda
else
  echo "ERROR: Could not find CUDA toolkit under /usr/local/cuda*"
  exit 2
fi

export HYPRE_ROOT=/opt/hypre-3.1.0-cuda-single
export HYPRE_LIBRARY=$HYPRE_ROOT/lib/libHYPRE.a
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}

echo "CUDA_HOME        : $CUDA_HOME"
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
echo "GPU visible:"
nvidia-smi --query-gpu=index,name,memory.used,memory.total --format=csv,noheader || true

# Important:
# The current Anabasis code is single-rank. If we directly run it inside
# the outer mpirun environment, HYPRE/MPI may see MPI_COMM_WORLD size=2.
# So we scrub MPI launcher variables before starting the solver.
for v in $(env | awk -F= '/^(OMPI_|PMIX_|PMI_|OPAL_)/ {print $1}'); do
  unset "$v"
done

cd "$APPDIR"

"$EXE" \
  -case-config "$CASECFG" \
  -polyMeshDir "$MESH" \
  -out-prefix "$RUNROOT/case" \
  -rho 1 \
  -mu 0.05 \
  -uMean 1.0 \
  -pipeD 0.1 \
  -pipeL 2.5 \
  -vel-solver bicgstab \
  -vel-maxit 2 \
  -vel-tol 1e-7 \
  -vel-reltol 0 \
  -p-tol 1e-7 \
  -tolMass 1e-2 \
  -tolVel 1e-2 \
  -nVelNonOrthCorr 0 \
  -nNonOrthCorr 0 \
  -nPressureCorr 0 \
  -p-amg-rebuild-every 500 \
  -write-vtu 0 \
  -write-every 0 \
  -nsteps 50 \
  -print-every 10 \
  2>&1 | tee "$RUNROOT/run.log"
