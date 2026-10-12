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
# Optional baselines: methods cutensor / cutensor-emu (native and INT8-emulated FP64 cuTENSOR contractions).
# Found via CUTENSOR_ROOT / CUTENSOR_DIR / CUTENSOR_HOME (cmake variable or environment), the CUDA
# toolkit, or CONDA_PREFIX; without it the methods are compiled out.
set(_ct_hints "${CUTENSOR_ROOT}" "${CUTENSOR_DIR}" "$ENV{CUTENSOR_ROOT}" "$ENV{CUTENSOR_DIR}"
              "$ENV{CUTENSOR_HOME}" "$ENV{CUTENSOR_PATH}" "${CUDAToolkit_ROOT}" "$ENV{CONDA_PREFIX}")
find_path(CUTENSOR_INCLUDE cutensor.h HINTS ${_ct_hints} ${CUDAToolkit_INCLUDE_DIRS}
          PATH_SUFFIXES include include/libcutensor libcutensor/include)
find_library(CUTENSOR_LIB cutensor HINTS ${_ct_hints} ${CUDAToolkit_LIBRARY_DIR}
             PATH_SUFFIXES lib lib64 lib/x86_64-linux-gnu libcutensor/lib libcutensor/lib64)
if(CUTENSOR_INCLUDE AND CUTENSOR_LIB)
  message(STATUS "cuTENSOR: ${CUTENSOR_LIB} (${CUTENSOR_INCLUDE}): methods cutensor, cutensor-emu enabled")
  target_compile_definitions(imd_bench PRIVATE IMD_BENCH_CUTENSOR)
  target_include_directories(imd_bench PRIVATE "${CUTENSOR_INCLUDE}")
  target_link_libraries(imd_bench PRIVATE "${CUTENSOR_LIB}")
else()
  message(STATUS "cuTENSOR not found (set CUTENSOR_ROOT): methods cutensor, cutensor-emu are compiled out")
endif()
# v1.10: method sustained samples NVML (dlopen) in a host thread.
target_link_libraries(imd_bench PRIVATE immerdink CUDA::cudart Threads::Threads ${CMAKE_DL_LIBS})
