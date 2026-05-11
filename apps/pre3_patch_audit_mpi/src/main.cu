#include <mpi.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

#include "mesh.h"

static double norm3(const std::array<double,3> &a)
{
  return std::sqrt(a[0]*a[0] + a[1]*a[1] + a[2]*a[2]);
}

int main(int argc, char **argv)
{
  MPI_Init(&argc, &argv);

  int rank = 0, size = 1;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  std::string caseRoot = "/tmp/pre3_meshCase_test";

  for(int i = 1; i < argc; ++i) {
    std::string a = argv[i];
    if(a == "-case-root" && i + 1 < argc) {
      caseRoot = argv[++i];
    }
  }

  try {
    const std::string polyMeshDir =
      caseRoot + "/processor" + std::to_string(rank) + "/constant/polyMesh";

    Mesh mesh = read_openfoam_polymesh(polyMeshDir);

    MPI_Barrier(MPI_COMM_WORLD);

    for(int r = 0; r < size; ++r) {
      MPI_Barrier(MPI_COMM_WORLD);

      if(r == rank) {
        std::printf("================================================================================\n");
        std::printf("PRE3 patch audit rank %d/%d mesh=%s\n",
                    rank, size, polyMeshDir.c_str());
        std::printf("cells=%d faces=%d internalFaces=%d nPatches=%zu maxNonOrth=%.3f\n",
                    mesh.nCells, mesh.nFaces, mesh.nInternalFaces,
                    mesh.patchNames.size(), mesh.maxNonOrthDeg);

        for(size_t p = 0; p < mesh.patchNames.size(); ++p) {
          const int start = mesh.patchStartFace[p];
          const int nFaces = mesh.patchNFaces[p];

          double area = 0.0;
          std::array<double,3> meanX = {0.0, 0.0, 0.0};
          std::array<double,3> meanS = {0.0, 0.0, 0.0};
          std::array<double,3> minX = {1e300, 1e300, 1e300};
          std::array<double,3> maxX = {-1e300, -1e300, -1e300};

          for(int i = 0; i < nFaces; ++i) {
            const int f = start + i;
            const double a = norm3(mesh.Sf[f]);
            area += a;

            for(int d = 0; d < 3; ++d) {
              meanX[d] += mesh.xf[f][d];
              meanS[d] += mesh.Sf[f][d];
              minX[d] = std::min(minX[d], mesh.xf[f][d]);
              maxX[d] = std::max(maxX[d], mesh.xf[f][d]);
            }
          }

          if(nFaces > 0) {
            for(int d = 0; d < 3; ++d) {
              meanX[d] /= (double)nFaces;
              meanS[d] /= (double)nFaces;
            }
          }

          std::printf(
            "patch[%zu] name=%s start=%d nFaces=%d area=%.12e meanX=(%.6e %.6e %.6e) minX=(%.6e %.6e %.6e) maxX=(%.6e %.6e %.6e) meanSf=(%.6e %.6e %.6e)\n",
            p,
            mesh.patchNames[p].c_str(),
            start,
            nFaces,
            area,
            meanX[0], meanX[1], meanX[2],
            minX[0], minX[1], minX[2],
            maxX[0], maxX[1], maxX[2],
            meanS[0], meanS[1], meanS[2]);
        }

        std::fflush(stdout);
      }
    }

    MPI_Barrier(MPI_COMM_WORLD);

    if(rank == 0) {
      std::printf("PRE3 PATCH AUDIT RESULT: PASS_RAN\n");
    }

    MPI_Finalize();
    return 0;
  }
  catch(const std::exception &e) {
    std::fprintf(stderr, "rank %d ERROR: %s\n", rank, e.what());
    MPI_Abort(MPI_COMM_WORLD, 1);
  }

  MPI_Finalize();
  return 1;
}
