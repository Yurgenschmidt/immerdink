# Benchmark harness (M4). A standalone CMake project, so the top-level build stays untouched:
#   cmake -S bench -B build/bench -G Ninja [-DIMMERDINK_LIBRARY=...]
# bench/run_sweep.sh does this (with the immerdink conda env) and runs the sweep.
#
# libimmerdink: by default the repository's own CMake project is added as a subdirectory
# (tests off), so the bench always links the library built from this checkout. Set
# IMMERDINK_LIBRARY=/path/to/libimmerdink.so to link a prebuilt one instead.
# immerdink alone needs no other library. The comparison baselines are optional:
#   cuBLAS (always linked): methods dgemm (FP64 GEMM) and emu (FP64 fixed-point emulation, compiled in
#     only if the cuBLAS headers have it: a CUDA 13 cuBLAS (checked by compiling its API); otherwise "needs CUDA 13 cuBLAS");
#   cuTENSOR (CUTENSOR_ROOT): methods cutensor, cutensor-emu and the permutations of perm+X;
#   GEMMul8 (GEMMUL8_DIR, bench/build_gemmul8.sh): method gemmul8.
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
# cuBLAS: methods dgemm and emu (the emulation API exists from CUDA 13; compile-tested here).
include(CheckCXXSourceCompiles)
set(_save_inc "${CMAKE_REQUIRED_INCLUDES}")
set(CMAKE_REQUIRED_INCLUDES ${CUDAToolkit_INCLUDE_DIRS})
set(CMAKE_TRY_COMPILE_TARGET_TYPE STATIC_LIBRARY)  # headers only: the check must not need to link cuBLAS
check_cxx_source_compiles("
#include <cublas_v2.h>
int main() {
  cublasHandle_t h = nullptr;
  cudaEmulationMantissaControl mc{};
  (void)mc;
  cublasSetEmulationStrategy(h, CUBLAS_EMULATION_STRATEGY_EAGER);
  cublasSetFixedPointEmulationMantissaControl(h, CUDA_EMULATION_MANTISSA_CONTROL_FIXED);
  cublasSetFixedPointEmulationMaxMantissaBitCount(h, 55);
  cublasGetFixedPointEmulationMantissaBitOffset(h, nullptr);
  cublasComputeType_t t = CUBLAS_COMPUTE_64F_EMULATED_FIXEDPOINT;
  (void)t;
  return 0;
}" IMD_CUBLAS_HAS_EMULATION)
set(CMAKE_REQUIRED_INCLUDES "${_save_inc}")
unset(CMAKE_TRY_COMPILE_TARGET_TYPE)
if(IMD_CUBLAS_HAS_EMULATION)
  message(STATUS "cuBLAS FP64 emulation API found: method emu enabled")
  target_compile_definitions(imd_bench PRIVATE IMD_BENCH_CUBLAS_EMU)
else()
  message(STATUS "cuBLAS has no FP64 emulation API (needs CUDA 13 cuBLAS): method emu records an error")
endif()

# Optional GEMMul8 baseline (method gemmul8): GEMMUL8_DIR holds include/gemmul8.hpp and lib/libgemmul8.a
# (bench/build_gemmul8.sh builds the pinned version and prints it). Name / commit / API generation
# come from the BUILD_INFO that script writes there, else from bench/GEMMUL8_PIN.
set(GEMMUL8_DIR "" CACHE PATH "GEMMul8 install (include/, lib/libgemmul8.a); empty: method gemmul8 is compiled out")
if(NOT GEMMUL8_DIR AND DEFINED ENV{GEMMUL8_DIR})
  set(GEMMUL8_DIR "$ENV{GEMMUL8_DIR}")
endif()
if(GEMMUL8_DIR)
  if(NOT EXISTS "${GEMMUL8_DIR}/lib/libgemmul8.a" OR NOT EXISTS "${GEMMUL8_DIR}/include/gemmul8.hpp")
    message(FATAL_ERROR "GEMMUL8_DIR=${GEMMUL8_DIR}: no lib/libgemmul8.a or include/gemmul8.hpp (bench/build_gemmul8.sh)")
  endif()
  set(_g8_name "")
  set(_g8_commit "")
  set(_g8_api "")
  if(EXISTS "${GEMMUL8_DIR}/BUILD_INFO")
    file(STRINGS "${GEMMUL8_DIR}/BUILD_INFO" _g8_info)
    foreach(_l ${_g8_info})
      if(_l MATCHES "^name=(.*)$")
        set(_g8_name "${CMAKE_MATCH_1}")
      elseif(_l MATCHES "^commit=(.*)$")
        set(_g8_commit "${CMAKE_MATCH_1}")
      elseif(_l MATCHES "^api=(.*)$")
        set(_g8_api "${CMAKE_MATCH_1}")
      endif()
    endforeach()
  endif()
  if(NOT _g8_name OR NOT _g8_commit OR NOT _g8_api)
    file(STRINGS "${CMAKE_CURRENT_SOURCE_DIR}/GEMMUL8_PIN" _g8_pin REGEX "^[^#]")
    list(GET _g8_pin 0 _g8_line)
    string(REGEX MATCHALL "[^ ]+" _g8_f "${_g8_line}")
    list(GET _g8_f 0 _g8_name)
    list(GET _g8_f 1 _g8_commit)
    list(GET _g8_f 2 _g8_api)
  endif()
  message(STATUS "GEMMul8 ${_g8_name} (${_g8_commit}, API ${_g8_api}): ${GEMMUL8_DIR}: method gemmul8 enabled")
  target_compile_definitions(imd_bench PRIVATE IMD_BENCH_GEMMUL8 G8_API=${_g8_api}
                             G8_NAME="${_g8_name}" G8_COMMIT="${_g8_commit}")
  target_include_directories(imd_bench PRIVATE "${GEMMUL8_DIR}/include")
  target_link_libraries(imd_bench PRIVATE "${GEMMUL8_DIR}/lib/libgemmul8.a" CUDA::cublasLt)
else()
  message(STATUS "GEMMUL8_DIR not set: method gemmul8 is compiled out")
endif()
# cuBLAS goes AFTER the static GEMMul8 library, which needs it (a linker with --as-needed drops a shared
# library listed before the archive that uses it).
target_link_libraries(imd_bench PRIVATE CUDA::cublas)
# v1.10: method sustained samples NVML (dlopen) in a host thread.
target_link_libraries(imd_bench PRIVATE immerdink CUDA::cudart Threads::Threads ${CMAKE_DL_LIBS})

# INT8 ceilings (issue 16): imd_ceiling measures mma.sync, cuBLASLt and stock CUTLASS (sm80, and on a Hopper build the CUTLASS
# Hopper GEMMs with wgmma). Not built by default (bench/build.sh builds imd_bench only; WITH_CEILING=1 adds it).
# Needs CUTLASS (the submodule or IMMERDINK_CUTLASS_DIR, as the library) and cuBLASLt. A build whose architectures include 90
# compiles this one target for sm_90a (wgmma exists only there) and defines IMD_CEIL_HOPPER.
find_path(IMD_CUTLASS_INCLUDE cutlass/cutlass.h HINTS "${IMMERDINK_CUTLASS_DIR}" "$ENV{IMMERDINK_CUTLASS_DIR}"
          "${CMAKE_CURRENT_SOURCE_DIR}/../third_party/cutlass" PATH_SUFFIXES include NO_DEFAULT_PATH)
if(IMD_CUTLASS_INCLUDE)
  add_executable(imd_ceiling ceiling/int8_ceiling.cu)
  set_target_properties(imd_ceiling PROPERTIES CUDA_STANDARD 20 CUDA_STANDARD_REQUIRED ON)
  target_include_directories(imd_ceiling SYSTEM PRIVATE "${IMD_CUTLASS_INCLUDE}")
  target_compile_options(imd_ceiling PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr -diag-suppress=20012>)
  target_link_libraries(imd_ceiling PRIVATE CUDA::cublasLt CUDA::cudart)
  set(_imd_has90 FALSE)
  foreach(_a IN LISTS CMAKE_CUDA_ARCHITECTURES)
    if(_a MATCHES "^90")    # 90, 90a, 90-real, 90a-real
      set(_imd_has90 TRUE)
    endif()
  endforeach()
  if(_imd_has90)
    set_target_properties(imd_ceiling PROPERTIES CUDA_ARCHITECTURES "90a-real")
    target_compile_definitions(imd_ceiling PRIVATE IMD_CEIL_HOPPER)
    message(STATUS "imd_ceiling: Hopper (sm_90a) variants enabled")
  endif()
else()
  message(STATUS "imd_ceiling: no CUTLASS include dir found (third_party/cutlass or IMMERDINK_CUTLASS_DIR): not built")
endif()
