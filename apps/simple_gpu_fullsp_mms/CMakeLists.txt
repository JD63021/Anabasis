find_package(CUDAToolkit REQUIRED)

option(ANABASIS_USE_HYPRE_INTERNAL_SPGEMM "Force hypre internal device SpGEMM instead of vendor/cuSPARSE SpGEMM" OFF)

add_executable(simple_gpu_fullsp_mms
  src/main.cu
  src/bc_specs.cu
  src/patch_geometry.cu
  src/velocity_bc_eval.cu
  src/bc_runtime_config.cu
)

target_include_directories(simple_gpu_fullsp_mms
  PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    ${MPI_CXX_INCLUDE_DIRS}
    ${CMAKE_SOURCE_DIR}/libscalar/include
    ${CMAKE_SOURCE_DIR}/libpoisson/include
)

target_link_libraries(simple_gpu_fullsp_mms
  PRIVATE
    libpoisson
    libscalar
    ${MPI_CXX_LIBRARIES}
    CUDA::cudart
    CUDA::cusparse
    CUDA::cublas
    CUDA::cublasLt
    CUDA::curand
    CUDA::cusolver
    dl
)

if(MPI_CXX_COMPILE_OPTIONS)
  target_compile_options(simple_gpu_fullsp_mms PRIVATE ${MPI_CXX_COMPILE_OPTIONS})
endif()

if(MPI_CXX_LINK_FLAGS)
  separate_arguments(_MPI_CXX_LINK_FLAGS_SIMPLE NATIVE_COMMAND "${MPI_CXX_LINK_FLAGS}")
  target_link_options(simple_gpu_fullsp_mms PRIVATE ${_MPI_CXX_LINK_FLAGS_SIMPLE})
endif()

if(ANABASIS_USE_HYPRE_INTERNAL_SPGEMM)
  target_compile_definitions(simple_gpu_fullsp_mms PRIVATE ANABASIS_USE_HYPRE_INTERNAL_SPGEMM=1)
endif()

target_compile_definitions(simple_gpu_fullsp_mms PRIVATE ANABASIS_FULL_SP_APP=1)

set_target_properties(simple_gpu_fullsp_mms PROPERTIES
  CUDA_SEPARABLE_COMPILATION ON
  CUDA_ARCHITECTURES ${CMAKE_CUDA_ARCHITECTURES}
)
