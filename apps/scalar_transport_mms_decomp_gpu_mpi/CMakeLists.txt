add_executable(scalar_transport_mms_decomp_gpu_mpi
  src/main.cu
  ${CMAKE_SOURCE_DIR}/apps/simple_gpu/src/bc_specs.cu
  ${CMAKE_SOURCE_DIR}/apps/simple_gpu/src/bc_runtime_config.cu
)

target_include_directories(scalar_transport_mms_decomp_gpu_mpi PRIVATE
  ${CMAKE_SOURCE_DIR}/libpoisson/include
  ${CMAKE_SOURCE_DIR}/libpoisson_decomp/include
  ${CMAKE_SOURCE_DIR}/libscalar_decomp/include
  ${CMAKE_SOURCE_DIR}/apps/simple_gpu/src
)

target_link_libraries(scalar_transport_mms_decomp_gpu_mpi PRIVATE
  libscalar_decomp
  libpoisson_decomp
  libpoisson
  MPI::MPI_CXX
)

set_target_properties(scalar_transport_mms_decomp_gpu_mpi PROPERTIES
  CUDA_SEPARABLE_COMPILATION ON
)
