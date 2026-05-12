cmake_minimum_required(VERSION 3.21)
project(anabasis LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

find_package(MPI REQUIRED COMPONENTS CXX)

add_subdirectory(libpoisson)
add_subdirectory(libpoisson_decomp)
add_subdirectory(libscalar_decomp)
add_subdirectory(libscalar)
add_subdirectory(apps/simple_gpu)

add_subdirectory(apps/pre3_poisson_local_mpi)

add_subdirectory(apps/pre3_poisson_dist_graph_mpi)

add_subdirectory(apps/pre3_poisson_dist_geom_gpu_mpi)

add_subdirectory(apps/pre3_pressure_bridge_gpu_mpi)

add_subdirectory(apps/pre3_pressure_velocity_bridge_gpu_mpi)

add_subdirectory(apps/pre3_pressure_correction_skeleton_gpu_mpi)

add_subdirectory(apps/pre3_patch_audit_mpi)

add_subdirectory(apps/pre3_pipe_pressure_correction_gpu_mpi)

add_subdirectory(apps/pre3_pipe_pressure_correction_report_gpu_mpi)

add_subdirectory(apps/pre3_pipe_flux_correction_gpu_mpi)

add_subdirectory(apps/pre3_pipe_momentum_predictor_gpu_mpi)

add_subdirectory(apps/pre3_pipe_momentum_flux_audit_gpu_mpi)

add_subdirectory(apps/poisson_mms_decomp_gpu_mpi)

add_subdirectory(apps/scalar_transport_mms_decomp_gpu_mpi)

add_subdirectory(apps/stokes_mms_simple_decomp_gpu_mpi)

add_subdirectory(apps/simple_gpu_mpi_port)
