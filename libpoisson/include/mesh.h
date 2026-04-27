#pragma once

#include "common.h"

struct PatchInfo {
  std::string name;
  int nFaces = 0;
  int startFace = 0;
  std::string type;
};

struct Mesh {
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

Mesh read_openfoam_polymesh(const std::string& polyMeshDir);
