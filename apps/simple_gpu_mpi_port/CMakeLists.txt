add_executable(simple_gpu_mpi_port
  src/main.cu
  ${CMAKE_SOURCE_DIR}/apps/simple_gpu/src/bc_specs.cu
  ${CMAKE_SOURCE_DIR}/apps/simple_gpu/src/bc_runtime_config.cu
  ${CMAKE_SOURCE_DIR}/apps/simple_gpu/src/patch_geometry.cu
  ${CMAKE_SOURCE_DIR}/apps/simple_gpu/src/velocity_bc_eval.cu
)

target_include_directories(simple_gpu_mpi_port PRIVATE
  ${CMAKE_SOURCE_DIR}/libpoisson/include
  ${CMAKE_SOURCE_DIR}/libpoisson_decomp/include
  ${CMAKE_SOURCE_DIR}/libscalar_decomp/include
  ${CMAKE_SOURCE_DIR}/apps/simple_gpu/src
)

target_link_libraries(simple_gpu_mpi_port PRIVATE
  libscalar_decomp
  libpoisson_decomp
  libpoisson
  MPI::MPI_CXX
)

set_target_properties(simple_gpu_mpi_port PROPERTIES
  CUDA_SEPARABLE_COMPILATION ON
)
