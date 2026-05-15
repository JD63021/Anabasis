option(ANABASIS_USE_HYPRE_INTERNAL_SPGEMM "Force hypre internal device SpGEMM instead of vendor/cuSPARSE SpGEMM" OFF)

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
