#!/usr/bin/env python3
from pathlib import Path
import sys

p = Path("apps/pre3_pipe_pressure_correction_report_gpu_mpi/src/main.cu")
s = p.read_text()

if "PRE3E1 mass correction summary" in s:
    print("Already patched PRE3E1")
    raise SystemExit(0)

s = s.replace("PRE3E0", "PRE3E1")
s = s.replace("pipe pressure correction", "pipe pressure correction report")

old = '''      if(rank == 0) {
        std::printf("PRE3E1 GPU pressure-correction residual: resInf=%.12e resL2=%.12e rhsInf=%.12e relInf=%.12e\\n",
                    globalResInf,
                    globalResL2,
                    globalRhsInf,
                    globalResInf / std::max(globalRhsInf, 1e-300));
      }
'''

new = '''      if(rank == 0) {
        const double massBefore = globalRhsInf;
        const double massAfter  = globalResInf;
        const double reduction  = massAfter / std::max(massBefore, 1e-300);
        const double improvement = massBefore / std::max(massAfter, 1e-300);

        std::printf("PRE3E1 GPU pressure-correction residual: resInf=%.12e resL2=%.12e rhsInf=%.12e relInf=%.12e\\n",
                    globalResInf,
                    globalResL2,
                    globalRhsInf,
                    globalResInf / std::max(globalRhsInf, 1e-300));

        std::printf("PRE3E1 mass correction summary: massBeforeInf=%.12e massAfterInf=%.12e reduction=%.12e improvement=%.6e\\n",
                    massBefore,
                    massAfter,
                    reduction,
                    improvement);
      }
'''

if old not in s:
    raise SystemExit("Could not find PRE3E1 pressure residual print block")

s = s.replace(old, new, 1)

s = s.replace("PRE3E1 pipe pressure correction report solve RESULT",
              "PRE3E1 pipe pressure correction report solve RESULT")
s = s.replace("PRE3E1 RESULT: PASS_RAN",
              "PRE3E1 RESULT: PASS_RAN")

p.write_text(s)
print("OK: patched PRE3E1 mass correction summary")
