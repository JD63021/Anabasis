find_package(CUDAToolkit REQUIRED)

add_executable(dg2_dg1_pmass_simple_diag
  src/main.cu
)

target_include_directories(dg2_dg1_pmass_simple_diag
  PRIVATE
    ${MPI_CXX_INCLUDE_DIRS}
    ${CMAKE_SOURCE_DIR}/libpoisson/include
)

target_link_libraries(dg2_dg1_pmass_simple_diag
  PRIVATE
    libpoisson
    ${MPI_CXX_LIBRARIES}
    CUDA::cudart
)

if(MPI_CXX_COMPILE_OPTIONS)
  target_compile_options(dg2_dg1_pmass_simple_diag PRIVATE ${MPI_CXX_COMPILE_OPTIONS})
endif()

if(MPI_CXX_LINK_FLAGS)
  separate_arguments(_MPI_CXX_LINK_FLAGS_DG2_CG1_PMASS NATIVE_COMMAND "${MPI_CXX_LINK_FLAGS}")
  target_link_options(dg2_dg1_pmass_simple_diag PRIVATE ${_MPI_CXX_LINK_FLAGS_DG2_CG1_PMASS})
endif()

set_target_properties(dg2_dg1_pmass_simple_diag PROPERTIES
  CUDA_SEPARABLE_COMPILATION ON
  CUDA_ARCHITECTURES ${CMAKE_CUDA_ARCHITECTURES}
)
