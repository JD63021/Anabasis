add_executable(scalar_mms_lib
  main.cu
)

target_include_directories(scalar_mms_lib
  PRIVATE
    ${MPI_CXX_INCLUDE_DIRS}
)

target_link_libraries(scalar_mms_lib
  PRIVATE
    libscalar
    ${MPI_CXX_LIBRARIES}
)

if(MPI_CXX_COMPILE_OPTIONS)
  target_compile_options(scalar_mms_lib PRIVATE ${MPI_CXX_COMPILE_OPTIONS})
endif()

if(MPI_CXX_LINK_FLAGS)
  separate_arguments(_MPI_CXX_LINK_FLAGS NATIVE_COMMAND "${MPI_CXX_LINK_FLAGS}")
  target_link_options(scalar_mms_lib PRIVATE ${_MPI_CXX_LINK_FLAGS})
endif()

set_target_properties(scalar_mms_lib PROPERTIES
  CUDA_SEPARABLE_COMPILATION ON
  CUDA_ARCHITECTURES 86
)
