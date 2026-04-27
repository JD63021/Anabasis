cmake_minimum_required(VERSION 3.21)

find_package(MPI REQUIRED)
find_package(CUDAToolkit REQUIRED)

set(HYPRE_ROOT "" CACHE PATH "Root of HYPRE install that contains include/ and lib/")

find_path(HYPRE_INCLUDE_DIR
  NAMES HYPRE.h
  HINTS
    ${HYPRE_ROOT}/include
    $ENV{HYPRE_ROOT}/include
    $ENV{HYPRE_DIR}/include
    $ENV{PETSC_DIR}/$ENV{PETSC_ARCH}/include
    $ENV{PETSC_DIR}/include
)

find_library(HYPRE_LIBRARY
  NAMES HYPRE hypre
  HINTS
    ${HYPRE_ROOT}/lib
    ${HYPRE_ROOT}/lib64
    $ENV{HYPRE_ROOT}/lib
    $ENV{HYPRE_ROOT}/lib64
    $ENV{HYPRE_DIR}/lib
    $ENV{HYPRE_DIR}/lib64
    $ENV{PETSC_DIR}/$ENV{PETSC_ARCH}/lib
)

if(NOT HYPRE_INCLUDE_DIR OR NOT HYPRE_LIBRARY)
  message(FATAL_ERROR "Could not find HYPRE. Set HYPRE_ROOT or HYPRE_DIR, or ensure PETSC_DIR/PETSC_ARCH point to a PETSc build that includes hypre headers/libs.")
endif()

add_library(libpoisson STATIC
  src/mesh.cu
  src/bc.cu
  src/gradient.cu
  src/scalar_elliptic.cu
  src/hypre_backend.cu
  src/poisson_library.cu
)

target_include_directories(libpoisson PUBLIC
  ${CMAKE_CURRENT_SOURCE_DIR}/include
  ${HYPRE_INCLUDE_DIR}
)

target_link_libraries(libpoisson PUBLIC
  MPI::MPI_CXX
  CUDA::cudart
  ${HYPRE_LIBRARY}
)

set_target_properties(libpoisson PROPERTIES
  CUDA_SEPARABLE_COMPILATION ON
  CUDA_ARCHITECTURES native
)
