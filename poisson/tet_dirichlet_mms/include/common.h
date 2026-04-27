#pragma once

#include <mpi.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <array>
#include <cctype>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <map>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

extern "C" {
#include "HYPRE.h"
#include "HYPRE_IJ_mv.h"
#include "HYPRE_parcsr_ls.h"
#include "HYPRE_krylov.h"
}

#define CUDA_CALL(stmt) do { \
  cudaError_t _err = (stmt); \
  if (_err != cudaSuccess) { \
    std::fprintf(stderr, "CUDA ERROR at %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(_err)); \
    MPI_Abort(MPI_COMM_WORLD, -1); \
  } \
} while (0)

#define HYPRE_CALL(stmt) do { \
  HYPRE_Int _ierr = (stmt); \
  if (_ierr) { \
    int _rank = 0; MPI_Comm_rank(MPI_COMM_WORLD, &_rank); \
    std::fprintf(stderr, "[%d] HYPRE ERROR %s:%d code=%d\n", _rank, __FILE__, __LINE__, (int)_ierr); \
    MPI_Abort(MPI_COMM_WORLD, (int)_ierr); \
  } \
} while (0)

constexpr double kPi = 3.141592653589793238462643383279502884;

inline std::array<double,3> add3(const std::array<double,3>& a,const std::array<double,3>& b){ return {a[0]+b[0], a[1]+b[1], a[2]+b[2]}; }
inline std::array<double,3> sub3(const std::array<double,3>& a,const std::array<double,3>& b){ return {a[0]-b[0], a[1]-b[1], a[2]-b[2]}; }
inline std::array<double,3> mul3(double s,const std::array<double,3>& a){ return {s*a[0], s*a[1], s*a[2]}; }
inline double dot3(const std::array<double,3>& a,const std::array<double,3>& b){ return a[0]*b[0] + a[1]*b[1] + a[2]*b[2]; }
inline std::array<double,3> cross3(const std::array<double,3>& a,const std::array<double,3>& b){ return {a[1]*b[2]-a[2]*b[1], a[2]*b[0]-a[0]*b[2], a[0]*b[1]-a[1]*b[0]}; }
inline double norm3(const std::array<double,3>& a){ return std::sqrt(dot3(a,a)); }

inline void print_device_info(int device){
  cudaDeviceProp prop{};
  CUDA_CALL(cudaGetDeviceProperties(&prop, device));
  std::printf("Running on GPU \"%s\" (cc %d.%d), total memory %.2f GiB\n",
              prop.name, prop.major, prop.minor,
              static_cast<double>(prop.totalGlobalMem)/(1024.0*1024.0*1024.0));
}
