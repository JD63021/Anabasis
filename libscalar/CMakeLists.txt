add_library(libscalar STATIC
  src/scalar_transport_library.cu
)

target_include_directories(libscalar
  PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/libpoisson/include
    ${MPI_CXX_INCLUDE_DIRS}
)

target_link_libraries(libscalar
  PUBLIC
    libpoisson
    ${MPI_CXX_LIBRARIES}
)

if(MPI_CXX_COMPILE_OPTIONS)
  target_compile_options(libscalar PUBLIC ${MPI_CXX_COMPILE_OPTIONS})
endif()

if(MPI_CXX_LINK_FLAGS)
  separate_arguments(_MPI_CXX_LINK_FLAGS NATIVE_COMMAND "${MPI_CXX_LINK_FLAGS}")
  target_link_options(libscalar PUBLIC ${_MPI_CXX_LINK_FLAGS})
endif()

set_target_properties(libscalar PROPERTIES
  CUDA_SEPARABLE_COMPILATION ON
  CUDA_ARCHITECTURES 86
)
