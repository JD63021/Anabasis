add_library(libscalar_decomp STATIC
  src/scalar_decomp_library.cu
)

target_include_directories(libscalar_decomp PUBLIC
  ${CMAKE_CURRENT_SOURCE_DIR}/include
  ${CMAKE_SOURCE_DIR}/libpoisson/include
  ${CMAKE_SOURCE_DIR}/libpoisson_decomp/include
)

target_link_libraries(libscalar_decomp PUBLIC
  libpoisson_decomp
  libpoisson
  MPI::MPI_CXX
)

set_target_properties(libscalar_decomp PROPERTIES
  CUDA_SEPARABLE_COMPILATION ON
)
