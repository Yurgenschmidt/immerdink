# Benchmark harness (M4). A standalone CMake project, so the top-level build stays untouched:
#   cmake -S bench -B build/bench -G Ninja [-DGEMMUL8_DIR=...] [-DIMMERDINK_LIBRARY=...]
# bench/run_sweep.sh does this (with the immerdink conda env) and runs the sweep.
#
# libimmerdink: by default the repository's own CMake project is added as a subdirectory
# (tests off), so the bench always links the library built from this checkout. Set
# IMMERDINK_LIBRARY=/path/to/libimmerdink.so to link a prebuilt one instead.
cmake_minimum_required(VERSION 3.28)
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES 89)
endif()
project(immerdink_bench LANGUAGES C CXX CUDA)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

set(IMMERDINK_LIBRARY "" CACHE FILEPATH "prebuilt libimmerdink.so (empty: build from ..)")
set(GEMMUL8_DIR "${CMAKE_CURRENT_SOURCE_DIR}/_deps/GEMMul8"
    CACHE PATH "GEMMul8 checkout built by tools/get_gemmul8.sh")

find_package(CUDAToolkit REQUIRED)
find_library(CUTENSOR_LIB cutensor HINTS "${CUDAToolkit_LIBRARY_DIR}" "${CUDAToolkit_LIBRARY_ROOT}/lib"
             "$ENV{CONDA_PREFIX}/lib" REQUIRED)
find_path(CUTENSOR_INCLUDE cutensor.h HINTS "${CUDAToolkit_INCLUDE_DIRS}"
          "${CUDAToolkit_LIBRARY_ROOT}/include" "$ENV{CONDA_PREFIX}/include" REQUIRED)

if(IMMERDINK_LIBRARY)
  add_library(immerdink SHARED IMPORTED)
  set_target_properties(immerdink PROPERTIES IMPORTED_LOCATION "${IMMERDINK_LIBRARY}"
    INTERFACE_INCLUDE_DIRECTORIES "${CMAKE_CURRENT_SOURCE_DIR}/../include")
else()
  set(IMMERDINK_BUILD_TESTS OFF CACHE BOOL "" FORCE)
  set(IMMERDINK_PROFILE ON CACHE BOOL "" FORCE)  # per-phase timing (src/profile_hooks.h)
  add_subdirectory("${CMAKE_CURRENT_SOURCE_DIR}/.." immerdink EXCLUDE_FROM_ALL)
endif()

add_executable(imd_bench imd_bench.cu)
if(NOT IMMERDINK_LIBRARY)
  target_compile_definitions(imd_bench PRIVATE IMD_BENCH_PROFILE)
  target_include_directories(imd_bench PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/../src")
endif()
set_target_properties(imd_bench PROPERTIES CUDA_STANDARD 20 CUDA_STANDARD_REQUIRED ON)
target_include_directories(imd_bench PRIVATE "${CUTENSOR_INCLUDE}")
target_link_libraries(imd_bench PRIVATE immerdink CUDA::cudart CUDA::cublas CUDA::cublasLt
                      "${CUTENSOR_LIB}")

if(EXISTS "${GEMMUL8_DIR}/lib/libgemmul8.a")
  message(STATUS "GEMMul8 baseline: ${GEMMUL8_DIR}")
  find_library(NVML_LIB nvidia-ml HINTS /usr/lib/x86_64-linux-gnu "${CUDAToolkit_LIBRARY_DIR}/stubs" REQUIRED)
  target_compile_definitions(imd_bench PRIVATE IMD_BENCH_GEMMUL8)
  target_include_directories(imd_bench PRIVATE "${GEMMUL8_DIR}/include")
  target_link_libraries(imd_bench PRIVATE "${GEMMUL8_DIR}/lib/libgemmul8.a" CUDA::cuda_driver
                        "${NVML_LIB}" ${CMAKE_DL_LIBS})
else()
  message(STATUS "GEMMul8 not built (tools/get_gemmul8.sh): the gemmul8 method is disabled")
endif()
