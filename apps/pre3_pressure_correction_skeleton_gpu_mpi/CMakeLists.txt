add_executable(pre3_pressure_correction_skeleton_gpu_mpi
  src/main.cu
)

target_link_libraries(pre3_pressure_correction_skeleton_gpu_mpi PRIVATE
  libpoisson
)

target_include_directories(pre3_pressure_correction_skeleton_gpu_mpi PRIVATE
  ${CMAKE_SOURCE_DIR}/libpoisson/include
)

set_target_properties(pre3_pressure_correction_skeleton_gpu_mpi PROPERTIES
  CUDA_SEPARABLE_COMPILATION ON
  CUDA_ARCHITECTURES native
)
