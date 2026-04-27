#pragma once

#include <array>
#include <string>
#include <vector>

struct PressureLibMeshData {
  std::vector<std::array<double,3>> P;
  std::vector<std::vector<int>> faces;
  std::vector<int> owner, neigh, bPatch;
  std::vector<std::string> patchNames;
  int nFaces = 0;
  int nInternalFaces = 0;
  int nCells = 0;
  std::vector<std::array<double,3>> cc, xf, nf, Sf;
  std::vector<double> vol, Af;
  std::vector<std::vector<int>> cellNbrs, cellBFace, cellFaces, cellOrient;
  double maxNonOrthDeg = 0.0;
};

struct PressureLibOptions {
  int maxIter = 4000;
  double relTol = 0.0;
  double absTol = 1e-10;
  int monitor = 0;

  int amgMaxIter = 1;
  int amgRelaxType = 7;
  int amgCoarsenType = 8;
  int amgInterpType = 6;
  int amgAggLevels = 1;
  int amgAggInterpType = 4;
  int amgNumSweeps = 1;
  int amgRAP2 = 0;
  int amgPmax = 4;
  int amgKeepTranspose = 1;
  double amgTruncFactor = 0.0;
  double amgStrongThreshold = -1.0;

  bool useReferenceCell = false;
  int referenceCell = 0;
  double referenceValue = 0.0;
};

struct PressureLibAssembly {
  std::vector<double> values;   // matrix coefficients, same CSR order every call
  std::vector<double> rhs;      // assembled rhs
  double wallTime = 0.0;        // assembly time only
};

struct PressureLibAssemblyHandle {
  void* impl = nullptr;
  int nRows = 0;
  int nnz = 0;
};

void init_pressure_lib_assembly_handle(
    PressureLibAssemblyHandle& handle,
    const PressureLibMeshData& meshData);

PressureLibAssembly assemble_pressure_with_libpoisson(
    PressureLibAssemblyHandle& handle,
    const std::vector<double>& rAUCell,
    const std::vector<std::string>& bcPatchType,
    const std::vector<double>& bcFaceValue,
    const std::vector<double>& rhs,
    const PressureLibOptions& opt);

void destroy_pressure_lib_assembly_handle(PressureLibAssemblyHandle& handle);
