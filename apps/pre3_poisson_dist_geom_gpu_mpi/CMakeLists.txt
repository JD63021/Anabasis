add_executable(pre3_poisson_dist_geom_gpu_mpi
  src/main.cu
)

target_link_libraries(pre3_poisson_dist_geom_gpu_mpi PRIVATE
  libpoisson
)

target_include_directories(pre3_poisson_dist_geom_gpu_mpi PRIVATE
  ${CMAKE_SOURCE_DIR}/libpoisson/include
)

set_target_properties(pre3_poisson_dist_geom_gpu_mpi PROPERTIES
  CUDA_SEPARABLE_COMPILATION ON
  CUDA_ARCHITECTURES native
)
