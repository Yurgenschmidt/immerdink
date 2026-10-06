# Benchmark harness (M4). A standalone CMake project, so the top-level build stays untouched:
#   cmake -S bench -B build/bench -G Ninja [-DIMMERDINK_LIBRARY=...]
# bench/run_sweep.sh does this (with the immerdink conda env) and runs the sweep.
#
# libimmerdink: by default the repository's own CMake project is added as a subdirectory
# (tests off), so the bench always links the library built from this checkout. Set
# IMMERDINK_LIBRARY=/path/to/libimmerdink.so to link a prebuilt one instead.
# immerdink only: no other library is needed (comparison baselines live in a separate
# evaluation workspace).
cmake_minimum_required(VERSION 3.28)
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES 89)
endif()
project(immerdink_bench LANGUAGES C CXX CUDA)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

set(IMMERDINK_LIBRARY "" CACHE FILEPATH "prebuilt libimmerdink.so (empty: build from ..)")

find_package(CUDAToolkit REQUIRED)

if(IMMERDINK_LIBRARY)
  # libimmerdink needs libimmerdink_host (DESIGN §6a: imd_workspace_size, imd_status_string),
  # built next to it.
  get_filename_component(_imd_dir "${IMMERDINK_LIBRARY}" DIRECTORY)
  add_library(immerdink_host SHARED IMPORTED)
  set_target_properties(immerdink_host PROPERTIES
    IMPORTED_LOCATION "${_imd_dir}/libimmerdink_host.so"
    INTERFACE_INCLUDE_DIRECTORIES "${CMAKE_CURRENT_SOURCE_DIR}/../include")
  add_library(immerdink SHARED IMPORTED)
  set_target_properties(immerdink PROPERTIES IMPORTED_LOCATION "${IMMERDINK_LIBRARY}"
    INTERFACE_INCLUDE_DIRECTORIES "${CMAKE_CURRENT_SOURCE_DIR}/../include"
    INTERFACE_LINK_LIBRARIES immerdink_host)
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
find_package(Threads REQUIRED)
# v1.10: method sustained samples NVML (dlopen) in a host thread.
target_link_libraries(imd_bench PRIVATE immerdink CUDA::cudart Threads::Threads ${CMAKE_DL_LIBS})
