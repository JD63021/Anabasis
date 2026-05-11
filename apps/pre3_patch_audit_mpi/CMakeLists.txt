add_executable(pre3_patch_audit_mpi
  src/main.cu
)

target_link_libraries(pre3_patch_audit_mpi PRIVATE
  libpoisson
)

target_include_directories(pre3_patch_audit_mpi PRIVATE
  ${CMAKE_SOURCE_DIR}/libpoisson/include
)

set_target_properties(pre3_patch_audit_mpi PROPERTIES
  CUDA_SEPARABLE_COMPILATION ON
  CUDA_ARCHITECTURES native
)
