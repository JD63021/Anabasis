option(ANABASIS_USE_HYPRE_INTERNAL_SPGEMM "Force hypre internal device SpGEMM instead of vendor/cuSPARSE SpGEMM" OFF)

find_package(CUDAToolkit REQUIRED)

add_executable(coupled_transient_nonorth_gpu
  src/main.cu
  src/bc_specs.cu
  src/patch_geometry.cu
  src/velocity_bc_eval.cu
  src/bc_runtime_config.cu
)

target_include_directories(coupled_transient_nonorth_gpu
  PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    ${MPI_CXX_INCLUDE_DIRS}
    ${CMAKE_SOURCE_DIR}/libscalar/include
    ${CMAKE_SOURCE_DIR}/libpoisson/include
)

target_link_libraries(coupled_transient_nonorth_gpu
  PRIVATE
    libpoisson
    libscalar
    ${MPI_CXX_LIBRARIES}
)

# Static CUDA HYPRE dependency closure.
# Required when HYPRE_LIBRARY points to libHYPRE.a.
# Without these, link fails with unresolved cusparse/cublas/curand/cusolver symbols.
if(ANABASIS_HYPRE_STATIC_CUDA_DEPS)
  target_link_libraries(coupled_transient_nonorth_gpu
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
  target_compile_options(coupled_transient_nonorth_gpu PRIVATE ${MPI_CXX_COMPILE_OPTIONS})
endif()

if(MPI_CXX_LINK_FLAGS)
  separate_arguments(_MPI_CXX_LINK_FLAGS_SIMPLE NATIVE_COMMAND "${MPI_CXX_LINK_FLAGS}")
  target_link_options(coupled_transient_nonorth_gpu PRIVATE ${_MPI_CXX_LINK_FLAGS_SIMPLE})
endif()

if(ANABASIS_USE_HYPRE_INTERNAL_SPGEMM)
  target_compile_definitions(coupled_transient_nonorth_gpu PRIVATE ANABASIS_USE_HYPRE_INTERNAL_SPGEMM=1)
endif()

set_target_properties(coupled_transient_nonorth_gpu PROPERTIES
  CUDA_SEPARABLE_COMPILATION ON
  CUDA_ARCHITECTURES ${CMAKE_CUDA_ARCHITECTURES}
)
