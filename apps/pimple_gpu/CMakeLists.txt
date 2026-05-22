option(ANABASIS_USE_HYPRE_INTERNAL_SPGEMM "Force hypre internal device SpGEMM instead of vendor/cuSPARSE SpGEMM" OFF)

find_package(CUDAToolkit REQUIRED)

add_executable(pimple_gpu
  src/main.cu
  src/bc_specs.cu
  src/patch_geometry.cu
  src/velocity_bc_eval.cu
  src/bc_runtime_config.cu
)

target_include_directories(pimple_gpu
  PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    ${MPI_CXX_INCLUDE_DIRS}
    ${CMAKE_SOURCE_DIR}/libscalar/include
    ${CMAKE_SOURCE_DIR}/libpoisson/include
)

target_link_libraries(pimple_gpu
  PRIVATE
    libpoisson
    libscalar
    ${MPI_CXX_LIBRARIES}
    CUDA::cudart
    CUDA::cusparse
    CUDA::cublas
    CUDA::cusolver
    CUDA::curand
)

if(MPI_CXX_COMPILE_OPTIONS)
  target_compile_options(pimple_gpu PRIVATE ${MPI_CXX_COMPILE_OPTIONS})
endif()

if(MPI_CXX_LINK_FLAGS)
  separate_arguments(_MPI_CXX_LINK_FLAGS_PIMPLE NATIVE_COMMAND "${MPI_CXX_LINK_FLAGS}")
  target_link_options(pimple_gpu PRIVATE ${_MPI_CXX_LINK_FLAGS_PIMPLE})
endif()

if(ANABASIS_USE_HYPRE_INTERNAL_SPGEMM)
  target_compile_definitions(pimple_gpu PRIVATE ANABASIS_USE_HYPRE_INTERNAL_SPGEMM=1)
endif()

set_target_properties(pimple_gpu PROPERTIES
  CUDA_SEPARABLE_COMPILATION ON
  CUDA_ARCHITECTURES ${CMAKE_CUDA_ARCHITECTURES}
)


# Static GPU HYPRE contains references to CUDA vendor libraries.
# Repeat libHYPRE near the tail, followed by CUDA vendor libs, so GNU ld
# resolves cusolver/curand/cusparse/cublas symbols in one-pass archive linking.
set(_ANABASIS_HYPRE_HINTS)
if(DEFINED HYPRE_ROOT)
  list(APPEND _ANABASIS_HYPRE_HINTS "${HYPRE_ROOT}/lib")
endif()
if(DEFINED ENV{HYPRE_ROOT})
  list(APPEND _ANABASIS_HYPRE_HINTS "$ENV{HYPRE_ROOT}/lib")
endif()

find_library(ANABASIS_PIMPLE_HYPRE_LIBRARY
  NAMES HYPRE
  PATHS ${_ANABASIS_HYPRE_HINTS}
  NO_DEFAULT_PATH
)

if(NOT ANABASIS_PIMPLE_HYPRE_LIBRARY)
  message(FATAL_ERROR "pimple_gpu could not find libHYPRE for static CUDA tail link. Set HYPRE_ROOT.")
endif()

find_library(ANABASIS_CUDART_LIBRARY   NAMES cudart   PATHS "${CUDAToolkit_LIBRARY_DIR}" NO_DEFAULT_PATH)
find_library(ANABASIS_CUSPARSE_LIBRARY NAMES cusparse PATHS "${CUDAToolkit_LIBRARY_DIR}" NO_DEFAULT_PATH)
find_library(ANABASIS_CUBLAS_LIBRARY   NAMES cublas   PATHS "${CUDAToolkit_LIBRARY_DIR}" NO_DEFAULT_PATH)
find_library(ANABASIS_CUSOLVER_LIBRARY NAMES cusolver PATHS "${CUDAToolkit_LIBRARY_DIR}" NO_DEFAULT_PATH)
find_library(ANABASIS_CURAND_LIBRARY   NAMES curand   PATHS "${CUDAToolkit_LIBRARY_DIR}" NO_DEFAULT_PATH)

foreach(_lib
    ANABASIS_CUDART_LIBRARY
    ANABASIS_CUSPARSE_LIBRARY
    ANABASIS_CUBLAS_LIBRARY
    ANABASIS_CUSOLVER_LIBRARY
    ANABASIS_CURAND_LIBRARY)
  if(NOT ${_lib})
    message(FATAL_ERROR "pimple_gpu could not find CUDA library ${_lib}; CUDAToolkit_LIBRARY_DIR=${CUDAToolkit_LIBRARY_DIR}")
  endif()
endforeach()

target_link_libraries(pimple_gpu
  PRIVATE
    ${ANABASIS_PIMPLE_HYPRE_LIBRARY}
    ${ANABASIS_CUSPARSE_LIBRARY}
    ${ANABASIS_CUBLAS_LIBRARY}
    ${ANABASIS_CUSOLVER_LIBRARY}
    ${ANABASIS_CURAND_LIBRARY}
    ${ANABASIS_CUDART_LIBRARY}
)

