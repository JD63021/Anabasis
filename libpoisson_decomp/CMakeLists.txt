add_library(libpoisson_decomp STATIC
  src/poisson_decomp_library.cu
)

target_include_directories(libpoisson_decomp PUBLIC
  ${CMAKE_CURRENT_SOURCE_DIR}/include
  ${CMAKE_SOURCE_DIR}/libpoisson/include
)

target_link_libraries(libpoisson_decomp PUBLIC
  libpoisson
  MPI::MPI_CXX
)

set_target_properties(libpoisson_decomp PROPERTIES
  CUDA_SEPARABLE_COMPILATION ON
)
