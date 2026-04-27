#include <mpi.h>
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <vector>

extern "C" {
#include "HYPRE.h"
#include "HYPRE_IJ_mv.h"
#include "HYPRE_parcsr_ls.h"
#include "_hypre_parcsr_mv.h"
}

#define CUDA_CALL(stmt) do { \
  cudaError_t err = (stmt); \
  if (err != cudaSuccess) { \
    std::fprintf(stderr, "CUDA ERROR %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
    MPI_Abort(MPI_COMM_WORLD, 1); \
  } \
} while(0)

#define HYPRE_CALL(stmt) do { \
  HYPRE_Int ierr = (stmt); \
  if (ierr) { \
    std::fprintf(stderr, "HYPRE ERROR %s:%d code=%d\n", __FILE__, __LINE__, (int)ierr); \
    MPI_Abort(MPI_COMM_WORLD, (int)ierr); \
  } \
} while(0)

static const char* memtype_name(int t) {
#if CUDART_VERSION >= 10000
  if (t == (int)cudaMemoryTypeUnregistered) return "Unregistered";
#endif
  if (t == (int)cudaMemoryTypeHost) return "Host";
  if (t == (int)cudaMemoryTypeDevice) return "Device";
  if (t == (int)cudaMemoryTypeManaged) return "Managed";
  return "Unknown";
}

static void print_ptr(const char* name, const void* p) {
  cudaPointerAttributes attr{};
  cudaError_t err = cudaPointerGetAttributes(&attr, p);
  if (err != cudaSuccess) {
    std::printf("%-28s %p : cudaPointerGetAttributes failed: %s\n",
                name, p, cudaGetErrorString(err));
    cudaGetLastError();
    return;
  }

#if CUDART_VERSION >= 10000
  int mt = (int)attr.type;
#else
  int mt = (int)attr.memoryType;
#endif

  std::printf("%-28s %p : type=%s(%d), device=%d\n",
              name, p, memtype_name(mt), mt, attr.device);
}

int main(int argc, char** argv) {
  MPI_Init(&argc, &argv);
  CUDA_CALL(cudaSetDevice(0));

  HYPRE_CALL(HYPRE_Initialize());
  HYPRE_CALL(HYPRE_SetMemoryLocation(HYPRE_MEMORY_DEVICE));
  HYPRE_CALL(HYPRE_SetExecutionPolicy(HYPRE_EXEC_DEVICE));

  HYPRE_IJMatrix Aij = nullptr;
  HYPRE_ParCSRMatrix Apar = nullptr;

  const int n = 3;
  HYPRE_BigInt ilower = 0, iupper = n - 1;

  HYPRE_Int ncols[3] = {2, 3, 2};
  HYPRE_BigInt rows[3] = {0, 1, 2};
  HYPRE_BigInt cols[7] = {0,1, 0,1,2, 1,2};
  HYPRE_Complex vals[7] = {2,-1, -1,2,-1, -1,2};

  HYPRE_CALL(HYPRE_IJMatrixCreate(MPI_COMM_WORLD, ilower, iupper, ilower, iupper, &Aij));
  HYPRE_CALL(HYPRE_IJMatrixSetObjectType(Aij, HYPRE_PARCSR));
  HYPRE_CALL(HYPRE_IJMatrixSetRowSizes(Aij, ncols));

  std::printf("Initializing matrix with HYPRE_MEMORY_HOST, then migrating to DEVICE\n");
  HYPRE_CALL(HYPRE_IJMatrixInitialize_v2(Aij, HYPRE_MEMORY_HOST));
  HYPRE_CALL(HYPRE_IJMatrixSetValues(Aij, n, ncols, rows, cols, vals));
  HYPRE_CALL(HYPRE_IJMatrixAssemble(Aij));
  HYPRE_CALL(HYPRE_IJMatrixGetObject(Aij, (void**)&Apar));

  hypre_CSRMatrix* diag0 = hypre_ParCSRMatrixDiag(Apar);
  print_ptr("diag data before migrate", hypre_CSRMatrixData(diag0));
  print_ptr("diag I before migrate", hypre_CSRMatrixI(diag0));
  print_ptr("diag J before migrate", hypre_CSRMatrixJ(diag0));

  HYPRE_CALL(HYPRE_IJMatrixMigrate(Aij, HYPRE_MEMORY_DEVICE));
  HYPRE_CALL(HYPRE_IJMatrixGetObject(Aij, (void**)&Apar));

  hypre_CSRMatrix* diag1 = hypre_ParCSRMatrixDiag(Apar);
  print_ptr("diag data after migrate", hypre_CSRMatrixData(diag1));
  print_ptr("diag I after migrate", hypre_CSRMatrixI(diag1));
  print_ptr("diag J after migrate", hypre_CSRMatrixJ(diag1));

  HYPRE_CALL(HYPRE_IJMatrixDestroy(Aij));
  HYPRE_CALL(HYPRE_Finalize());
  MPI_Finalize();
  return 0;
}
