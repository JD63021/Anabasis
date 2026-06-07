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
CASE_FILE="/home/jd/anabasis_v0/cases/simple_fullsp_mms_reference/blockmesh_32cube_sp_noanchor_prepenalty.case"
RUN_DIR="/home/jd/anabasis_v0/runs/simple_fullsp_mms_reference/blockmesh_32cube_sp_noanchor_prepenalty"
LOG_FILE="/home/jd/anabasis_v0/runs/simple_fullsp_mms_reference/blockmesh_32cube_sp_noanchor_prepenalty/run.log"

if [ ! -x "${EXE}" ]; then
  echo "ERROR: executable not found: ${EXE}"
  exit 1
fi

if [ ! -d /home/jd/Desktop/meshes/unitcube/blockmesh/32cube/constant/polyMesh ]; then
  echo "ERROR: missing mesh: /home/jd/Desktop/meshes/unitcube/blockmesh/32cube/constant/polyMesh"
  exit 1
fi

rm -rf "${RUN_DIR}"
mkdir -p "${RUN_DIR}"

echo
echo "=== Running SP MMS blockMesh 32cube smoke test ==="
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

log = Path("/home/jd/anabasis_v0/runs/simple_fullsp_mms_reference/blockmesh_32cube_sp_noanchor_prepenalty/run.log")
text = log.read_text(errors="replace")

def last_float_any(patterns):
    for pattern in patterns:
        vals = re.findall(pattern, text, flags=re.MULTILINE)
        if vals:
            return float(vals[-1])
    return float("nan")

def last_word(pattern):
    m = re.findall(pattern, text, flags=re.MULTILINE)
    return m[-1].strip().lower() if m else "unknown"

def fmt(x):
    return f"{x:.12e}" if math.isfinite(x) else "nan"

anchor = last_word(r"Pressure anchor enabled\s*:\s*(yes|no)")
uerr = last_float_any([
    r"SIMPLE_MMS\s+U\s+rel\s+L2\s*=\s*([0-9.eE+-]+)",
    r"SIMPLE_MMS\s+U\s+abs\s+L2\s*=\s*([0-9.eE+-]+)",
    r"simpleMmsURelL2\s*[:= ]+\s*([0-9.eE+-]+)",
])
perr = last_float_any([
    r"SIMPLE_MMS\s+p\s+rel\s+L2\s*=\s*([0-9.eE+-]+)",
    r"SIMPLE_MMS\s+p\s+abs\s+L2\s*=\s*([0-9.eE+-]+)",
    r"simpleMmsPRelL2MeanFree\s*[:= ]+\s*([0-9.eE+-]+)",
    r"simpleMmsPRelL2\s*[:= ]+\s*([0-9.eE+-]+)",
])
mass = last_float_any([r"massRes\s*=\s*([0-9.eE+-]+)"])

bad = bool(re.search(r"FATAL|ERROR|nan|inf", text, re.IGNORECASE))

print()
print("=== SP MMS blockMesh 32cube summary ===")
print("log      =", log)
print("anchor   =", anchor)
print("massRes  =", fmt(mass))
print("UrelL2   =", fmt(uerr))
print("PrelL2   =", fmt(perr))

ok = (
    not bad
    and anchor != "yes"
    and math.isfinite(uerr)
    and math.isfinite(perr)
    and uerr < 1.0e-2
    and perr < 2.0e-2
)

if ok:
    print("RESULT   = PASS")
    sys.exit(0)

print("RESULT   = FAIL")
sys.exit(2)
PY
