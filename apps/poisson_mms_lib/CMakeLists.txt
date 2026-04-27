cmake_minimum_required(VERSION 3.21)

add_executable(poisson_mms_lib
  main.cu
)

target_link_libraries(poisson_mms_lib PRIVATE libpoisson)

target_include_directories(poisson_mms_lib PRIVATE
  ${CMAKE_SOURCE_DIR}/libpoisson/include
)

set_target_properties(poisson_mms_lib PROPERTIES
  CUDA_SEPARABLE_COMPILATION ON
  CUDA_ARCHITECTURES native
)
