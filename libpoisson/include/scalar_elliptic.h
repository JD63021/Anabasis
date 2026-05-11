#pragma once

#include "gradient.h"

struct CSRPattern {
  int nRows = 0;
  int nnz = 0;
  std::vector<HYPRE_Int> ncols;
  std::vector<int> rowOffsets;
  std::vector<int> diagPos;
  std::vector<HYPRE_BigInt> rows;
  std::vector<HYPRE_BigInt> cols;
  std::vector<int> facePP, facePN, faceNP, faceNN;
};

CSRPattern build_scalar_pattern(const Mesh& mesh);

void assemble_scalar_elliptic_system(
    const Mesh& mesh,
    const CSRPattern& pat,
    const std::vector<double>& gammaFace,
    const std::vector<double>& cellSource,
    const BoundaryFaceData& bcFaceData,
    const std::vector<std::array<double,3>>& grad,
    std::vector<HYPRE_Complex>& values,
    std::vector<HYPRE_Complex>& rhs,
    bool includeNonOrth,
    bool useReferenceCell,
    int refCell,
    double refValue);
