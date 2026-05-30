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

# Old steady SIMPLE MMS reference app restored from backup
add_subdirectory(apps/simple_gpu_subtri_mms_ref)

# Single-precision full SIMPLE GPU app
add_subdirectory(apps/simple_gpu_fullsp)

# Isolated full-SP MMS verification app
add_subdirectory(apps/simple_gpu_fullsp_mms)

# Double-precision copy of the memory-optimized full-SP MMS app
add_subdirectory(apps/simple_gpu_fullsp_mms_dp)

# Diagnostic DP MMS app
add_subdirectory(apps/simple_gpu_fullsp_mms_dp_diag)

# Diagnostic MMS app: no point anchor, distributed pressure penalty before AMG setup
add_subdirectory(apps/simple_gpu_fullsp_mms_dp_diag_noanchor_prepenalty)

# Diagnostic MMS app: pressure anchor row-only, keep ref column in neighbor rows
add_subdirectory(apps/simple_gpu_fullsp_mms_dp_diag_anchor_rowonly)

# Diagnostic MMS app: anchored pressure solve plus post-solve/post-update volume mean projection
add_subdirectory(apps/simple_gpu_fullsp_mms_dp_diag_anchor_meanproj)

# Diagnostic MMS app with center reference anchored to exact MMS pressure
add_subdirectory(apps/simple_gpu_fullsp_mms_dp_diag_centerref_exactanchor)

# Diagnostic MMS app with pressure anchor disabled
add_subdirectory(apps/simple_gpu_fullsp_mms_dp_diag_noanchor)

# Diagnostic MMS no-anchor app with distributed pressure penalty
add_subdirectory(apps/simple_gpu_fullsp_mms_dp_diag_noanchor_penalty)

# Diagnostic MMS app with center pressure reference
add_subdirectory(apps/simple_gpu_fullsp_mms_dp_diag_centerref)
