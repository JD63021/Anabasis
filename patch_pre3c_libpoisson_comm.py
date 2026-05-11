#!/usr/bin/env python3
from pathlib import Path
import sys

h = Path("libpoisson/include/hypre_backend.h")
cu = Path("libpoisson/src/hypre_backend.cu")

hs = h.read_text()
cs = cu.read_text()

if "libpoisson_set_hypre_comm" in hs:
    print("Already patched libpoisson communicator")
    raise SystemExit(0)

# Header declarations.
hs = hs.replace(
'''struct HypreSolveInfo {
  int iterations = 0;
  double finalRelResNorm = 0.0;
};
''',
'''struct HypreSolveInfo {
  int iterations = 0;
  double finalRelResNorm = 0.0;
};

// pre3: allow libpoisson HYPRE solves to use MPI_COMM_SELF for
// per-rank local decomposed-mesh tests, or MPI_COMM_WORLD for normal use.
void libpoisson_set_hypre_comm(MPI_Comm comm);
MPI_Comm libpoisson_get_hypre_comm();
''',
1)

# Source global helpers after include.
cs = cs.replace(
'''#include "hypre_backend.h"
''',
'''#include "hypre_backend.h"

namespace {
MPI_Comm g_libpoisson_hypre_comm = MPI_COMM_WORLD;
}

void libpoisson_set_hypre_comm(MPI_Comm comm) {
  g_libpoisson_hypre_comm = comm;
}

MPI_Comm libpoisson_get_hypre_comm() {
  return g_libpoisson_hypre_comm;
}
''',
1)

# Replace MPI_COMM_WORLD inside hypre backend.
cs = cs.replace("MPI_COMM_WORLD", "libpoisson_get_hypre_comm()")

# But undo accidental replacement inside global initializer if any.
cs = cs.replace("MPI_Comm g_libpoisson_hypre_comm = libpoisson_get_hypre_comm();",
                "MPI_Comm g_libpoisson_hypre_comm = MPI_COMM_WORLD;")

h.write_text(hs)
cu.write_text(cs)

print("OK: patched libpoisson selectable HYPRE communicator")
