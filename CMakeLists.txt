cmake_minimum_required(VERSION 3.21)
project(anabasis LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

find_package(MPI REQUIRED COMPONENTS CXX)

add_subdirectory(libpoisson)
add_subdirectory(libscalar)
add_subdirectory(apps/simple_gpu)
add_subdirectory(apps/pimple_gpu)
add_subdirectory(apps/coupled_transient_nonorth_gpu)
add_subdirectory(apps/simple_gpu_subtri)

# Single-precision full SIMPLE GPU app
add_subdirectory(apps/simple_gpu_fullsp)
