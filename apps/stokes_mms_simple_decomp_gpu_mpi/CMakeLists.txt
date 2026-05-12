add_executable(stokes_mms_simple_decomp_gpu_mpi
  src/main.cu
)

target_include_directories(stokes_mms_simple_decomp_gpu_mpi PRIVATE
  ${CMAKE_SOURCE_DIR}/libpoisson/include
  ${CMAKE_SOURCE_DIR}/libpoisson_decomp/include
  ${CMAKE_SOURCE_DIR}/libscalar_decomp/include
)

target_link_libraries(stokes_mms_simple_decomp_gpu_mpi PRIVATE
  libscalar_decomp
  libpoisson_decomp
  libpoisson
  MPI::MPI_CXX
)

set_target_properties(stokes_mms_simple_decomp_gpu_mpi PROPERTIES
  CUDA_SEPARABLE_COMPILATION ON
)
