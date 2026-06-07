#!/usr/bin/env bash
set -euo pipefail

cd /home/jd/anabasis_v0

echo "=== Building/checking simple_gpu_fullsp_mms SP executable ==="

cmake -S /home/jd/anabasis_v0 \
  -B /home/jd/anabasis_v0/build_simple_gpu_fullsp_mms_sp_noanchor_prepenalty \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CUDA_COMPILER=/usr/local/cuda-12.2/bin/nvcc \
  -DCMAKE_CUDA_ARCHITECTURES=86 \
  -DHYPRE_ROOT=/opt/hypre-3.1.0-cuda-single \
  -DHYPRE_INCLUDE_DIR=/opt/hypre-3.1.0-cuda-single/include \
  -DHYPRE_LIBRARY=/opt/hypre-3.1.0-cuda-single/lib/libHYPRE.a \
  -DANABASIS_HYPRE_STATIC_CUDA_DEPS=ON

cmake --build /home/jd/anabasis_v0/build_simple_gpu_fullsp_mms_sp_noanchor_prepenalty \
  --target simple_gpu_fullsp_mms \
  -j1

EXE="/home/jd/anabasis_v0/build_simple_gpu_fullsp_mms_sp_noanchor_prepenalty/apps/simple_gpu_fullsp_mms/simple_gpu_fullsp_mms"
CASE_FILE="/home/jd/anabasis_v0/cases/regression_checks/industrial1.case"
RUN_DIR="/home/jd/anabasis_v0/runs/regression_checks/industrial1"
LOG_FILE="/home/jd/anabasis_v0/runs/regression_checks/industrial1/run.log"

if [ ! -x "${EXE}" ]; then
  echo "ERROR: executable not found: ${EXE}"
  exit 1
fi

if [ ! -d /home/jd/Desktop/meshes/industrial1/constant/polyMesh ]; then
  echo "ERROR: missing mesh: /home/jd/Desktop/meshes/industrial1/constant/polyMesh"
  exit 1
fi

echo
echo "=== Boundary file preview ==="
sed -n '1,220p' /home/jd/Desktop/meshes/industrial1/constant/polyMesh/boundary || true

echo
echo "=== Case file ==="
cat "${CASE_FILE}"

rm -rf "${RUN_DIR}"
mkdir -p "${RUN_DIR}"

echo
echo "=== Running industrial1 ==="
echo "Case: ${CASE_FILE}"
echo "Log : ${LOG_FILE}"
echo

set +e
ANABASIS_SIMPLE_FACE_SKEW_CORR=1 \
ANABASIS_SIMPLE_SUBTRI_CONTINUITY=1 \
ANABASIS_SIMPLE_SUBTRI_FLUX_RECON=1 \
mpirun -n 1 \
  stdbuf -oL -eL \
  "${EXE}" \
  -case-config "${CASE_FILE}" \
  2>&1 | tee "${LOG_FILE}"
RUN_STATUS=${PIPESTATUS[0]}
set -e

echo
echo "=== Solver process status ==="
echo "mpirun exit code: ${RUN_STATUS}"

echo
echo "=== Error/warning/write summary from log ==="
grep -Ei "WARNING|FATAL|ERROR|terminate called|what\(\):|Signal:|Aborted|Segmentation fault|mpirun noticed|Primary job|CUDA|HYPRE|sticky|diverged|nan|inf|vtu|vtk|pvd|write|wrote|writing|saved|output" "${LOG_FILE}" | tail -120 || true

if [ "${RUN_STATUS}" -ne 0 ]; then
  echo
  echo "ERROR: solver failed. Full log:"
  echo "${LOG_FILE}"
  exit "${RUN_STATUS}"
fi

echo
echo "=== Final summary lines ==="
grep -E "Iterations|massRes|CD|CL|C_drag|C_lift|last pcg it|FATAL|ERROR|nan|inf" "${LOG_FILE}" | tail -40 || true

echo
echo "Full log:"
echo "${LOG_FILE}"
