#pragma once

#include "poisson_library.h"

#include <mpi.h>

struct ProcPatchDecomp {
  std::string name;
  int nFaces = 0;
  int startFace = 0;
  int myProcNo = -1;
  int neighbProcNo = -1;
};

struct DecompMesh {
  Mesh mesh;
  MPI_Comm comm = MPI_COMM_WORLD;

  int rank = 0;
  int size = 1;
  int nLocal = 0;

  std::vector<int> counts;
  std::vector<HYPRE_BigInt> offsets;

  HYPRE_BigInt ilower = 0;
  HYPRE_BigInt iupper = -1;
  HYPRE_BigInt globalN = 0;

  std::vector<ProcPatchDecomp> procPatches;
  std::vector<char> isProcFace;

  std::vector<HYPRE_BigInt> remoteRowForFace;
  std::vector<std::array<double,3>> remoteCCForFace;
};

struct DistCSRPattern {
  int nRows = 0;
  int nnz = 0;

  std::vector<HYPRE_Int> ncols;
  std::vector<int> rowOffsets;
  std::vector<int> diagPos;

  std::vector<HYPRE_BigInt> rows;
  std::vector<HYPRE_BigInt> cols;

  std::vector<int> facePP, facePN, faceNP, faceNN;

  std::vector<int> procFace;
  std::vector<int> procOwner;
  std::vector<int> procDiag;
  std::vector<int> procOff;
};

struct DistEllipticOptions {
  int nNonOrthCorr = 2;
  std::string gradScheme = "lsq";
  std::string laplacianScheme = "nonorth";

  bool useReferenceCell = false;
  HYPRE_BigInt referenceGlobalCell = 0;
  double referenceValue = 0.0;

  HypreOptions hypre;
};

struct DistEllipticResult {
  std::vector<double> phi;
  HypreSolveInfo lastSolveInfo;

  int nOuter = 0;
  long long globalNnz = 0;

  double l2 = 0.0;
  double linf = 0.0;
};

DecompMesh read_decomposed_openfoam_case(
    const std::string& caseRoot,
    MPI_Comm comm = MPI_COMM_WORLD);

DistCSRPattern build_decomp_scalar_pattern(const DecompMesh& dm);

DistEllipticResult solve_scalar_elliptic_decomp(
    const DecompMesh& dm,
    const std::vector<double>& gammaFace,
    const std::vector<double>& cellSource,
    const ScalarBCSet& bcSet,
    const DistEllipticOptions& opts);

inline DistEllipticResult solve_poisson_decomp(
    const DecompMesh& dm,
    const std::vector<double>& cellSource,
    const ScalarBCSet& bcSet,
    const DistEllipticOptions& opts) {
  std::vector<double> gammaFace(dm.mesh.nFaces, 1.0);
  return solve_scalar_elliptic_decomp(dm, gammaFace, cellSource, bcSet, opts);
}

std::vector<double> exchange_proc_face_scalar_owner_values(
    const DecompMesh& dm,
    const std::vector<double>& phi);

std::vector<std::array<double,3>> exchange_proc_face_vector_owner_values(
    const DecompMesh& dm,
    const std::vector<std::array<double,3>>& vec);
