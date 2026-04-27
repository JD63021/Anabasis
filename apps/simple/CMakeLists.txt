add_executable(simple
  src/main.cu
  src/bc_specs.cu
  src/patch_geometry.cu
  src/velocity_bc_eval.cu
  src/bc_runtime_config.cu
)

target_include_directories(simple
  PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    ${MPI_CXX_INCLUDE_DIRS}
    ${CMAKE_SOURCE_DIR}/libscalar/include
    ${CMAKE_SOURCE_DIR}/libpoisson/include
)

target_link_libraries(simple
  PRIVATE
    libpoisson
    libscalar
    ${MPI_CXX_LIBRARIES}
)

if(MPI_CXX_COMPILE_OPTIONS)
  target_compile_options(simple PRIVATE ${MPI_CXX_COMPILE_OPTIONS})
endif()

if(MPI_CXX_LINK_FLAGS)
  separate_arguments(_MPI_CXX_LINK_FLAGS_SIMPLE NATIVE_COMMAND "${MPI_CXX_LINK_FLAGS}")
  target_link_options(simple PRIVATE ${_MPI_CXX_LINK_FLAGS_SIMPLE})
endif()

set_target_properties(simple PROPERTIES
  CUDA_SEPARABLE_COMPILATION ON
  CUDA_ARCHITECTURES 86
)
