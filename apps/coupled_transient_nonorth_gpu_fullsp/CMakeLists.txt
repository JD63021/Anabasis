option(ANABASIS_USE_HYPRE_INTERNAL_SPGEMM "Force hypre internal device SpGEMM instead of vendor/cuSPARSE SpGEMM" OFF)

find_package(CUDAToolkit REQUIRED)

add_executable(coupled_transient_nonorth_gpu_fullsp
  src/main.cu
  src/bc_specs.cu
  src/patch_geometry.cu
  src/velocity_bc_eval.cu
  src/bc_runtime_config.cu
)

target_include_directories(coupled_transient_nonorth_gpu_fullsp
  PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    ${MPI_CXX_INCLUDE_DIRS}
    ${CMAKE_SOURCE_DIR}/libscalar/include
    ${CMAKE_SOURCE_DIR}/libpoisson/include
)

target_link_libraries(coupled_transient_nonorth_gpu_fullsp
  PRIVATE
    libpoisson
    libscalar
    ${MPI_CXX_LIBRARIES}
)

# Static CUDA HYPRE dependency closure.
# Required when HYPRE_LIBRARY points to libHYPRE.a.
# Without these, link fails with unresolved cusparse/cublas/curand/cusolver symbols.
if(ANABASIS_HYPRE_STATIC_CUDA_DEPS)
  target_link_libraries(coupled_transient_nonorth_gpu_fullsp
    PRIVATE
      CUDA::cusparse
      CUDA::cublas
      CUDA::cublasLt
      CUDA::curand
      CUDA::cusolver
      dl
  )
endif()

if(MPI_CXX_COMPILE_OPTIONS)
  target_compile_options(coupled_transient_nonorth_gpu_fullsp PRIVATE ${MPI_CXX_COMPILE_OPTIONS})
endif()

if(MPI_CXX_LINK_FLAGS)
  separate_arguments(_MPI_CXX_LINK_FLAGS_SIMPLE NATIVE_COMMAND "${MPI_CXX_LINK_FLAGS}")
  target_link_options(coupled_transient_nonorth_gpu_fullsp PRIVATE ${_MPI_CXX_LINK_FLAGS_SIMPLE})
endif()

target_compile_definitions(coupled_transient_nonorth_gpu_fullsp PRIVATE ANABASIS_FULLSP_APP=1)

if(ANABASIS_USE_HYPRE_INTERNAL_SPGEMM)
  target_compile_definitions(coupled_transient_nonorth_gpu_fullsp PRIVATE ANABASIS_USE_HYPRE_INTERNAL_SPGEMM=1)
endif()

set_target_properties(coupled_transient_nonorth_gpu_fullsp PROPERTIES
  CUDA_SEPARABLE_COMPILATION ON
  CUDA_ARCHITECTURES ${CMAKE_CUDA_ARCHITECTURES}
)

# CUDA-enabled static HYPRE depends on CUDA vendor libraries.
# Keep these explicit so standalone builds do not rely on PETSc's link line.
target_link_libraries(coupled_transient_nonorth_gpu_fullsp PRIVATE
  /usr/local/cuda-12.2/targets/x86_64-linux/lib/libcusparse.so
  /usr/local/cuda-12.2/targets/x86_64-linux/lib/libcublas.so
  /usr/local/cuda-12.2/targets/x86_64-linux/lib/libcusolver.so
  /usr/local/cuda-12.2/targets/x86_64-linux/lib/libcurand.so
  /usr/local/cuda-12.2/targets/x86_64-linux/lib/libcudart.so
)


# Final static-HYPRE closure.
# This must appear after all other target_link_libraries calls:
# libHYPRE.a first, then CUDA vendor dependencies, because static archive
# resolution is left-to-right.
target_link_libraries(coupled_transient_nonorth_gpu_fullsp PRIVATE
  /opt/hypre-3.1.0-cuda-single/lib/libHYPRE.a
  /usr/local/cuda-12.2/targets/x86_64-linux/lib/libcusparse.so
  /usr/local/cuda-12.2/targets/x86_64-linux/lib/libcublas.so
  /usr/local/cuda-12.2/targets/x86_64-linux/lib/libcublasLt.so
  /usr/local/cuda-12.2/targets/x86_64-linux/lib/libcusolver.so
  /usr/local/cuda-12.2/targets/x86_64-linux/lib/libcurand.so
  /usr/local/cuda-12.2/targets/x86_64-linux/lib/libcudart.so
)

