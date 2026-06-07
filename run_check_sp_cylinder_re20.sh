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
CASE_FILE="/home/jd/anabasis_v0/cases/regression_checks/sp_cylinder_re20.case"
RUN_DIR="/home/jd/anabasis_v0/runs/regression_checks/sp_cylinder_re20"
LOG_FILE="/home/jd/anabasis_v0/runs/regression_checks/sp_cylinder_re20/run.log"

if [ ! -x "${EXE}" ]; then
  echo "ERROR: executable not found: ${EXE}"
  exit 1
fi

if [ ! -d /home/jd/Desktop/meshes/cylinder/9mesh/constant/polyMesh ]; then
  echo "ERROR: missing mesh: /home/jd/Desktop/meshes/cylinder/9mesh/constant/polyMesh"
  exit 1
fi

rm -rf "${RUN_DIR}"
mkdir -p "${RUN_DIR}"

echo
echo "=== Running SP Re=20 cylinder physical benchmark ==="
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

python3 - <<'PY'
from pathlib import Path
import re
import sys
import math

log = Path("/home/jd/anabasis_v0/runs/regression_checks/sp_cylinder_re20/run.log")
text = log.read_text(errors="replace")

def last_float(pattern):
    m = re.findall(pattern, text, flags=re.MULTILINE)
    return float(m[-1]) if m else float("nan")

def last_word(pattern):
    m = re.findall(pattern, text, flags=re.MULTILINE)
    return m[-1].strip().lower() if m else "unknown"

def fmt(x):
    return f"{x:.12e}" if math.isfinite(x) else "nan"

p_ref = last_word(r"Pressure reference required:\s*(yes|no)")
anchor = last_word(r"Pressure anchor enabled\s*:\s*(yes|no)")
cd = last_float(r"CD_vector\s*=*\s*([0-9.eE+-]+)")
cl_y = last_float(r"CL_y_vector\s*=*\s*([0-9.eE+-]+)")
cl_z = last_float(r"CL_z_vector\s*=*\s*([0-9.eE+-]+)")
mass = last_float(r"massRes\s*=\s*([0-9.eE+-]+)")

bad = bool(re.search(r"FATAL|ERROR|nan|inf", text, re.IGNORECASE))

print()
print("=== SP cylinder Re=20 summary ===")
print("log      =", log)
print("pRefReq  =", p_ref)
print("anchor   =", anchor)
print("massRes  =", fmt(mass))
print("CD       =", fmt(cd))
print("CL_y     =", fmt(cl_y))
print("CL_z     =", fmt(cl_z))

ok = (
    not bad
    and p_ref == "no"
    and anchor == "no"
    and math.isfinite(cd)
    and 5.5 <= cd <= 6.7
    and math.isfinite(cl_z)
    and abs(cl_z) <= 5.0e-2
)

if ok:
    print("RESULT   = PASS")
    sys.exit(0)

print("RESULT   = FAIL")
sys.exit(2)
PY
