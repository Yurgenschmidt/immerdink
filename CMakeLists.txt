cmake_minimum_required(VERSION 3.28)

if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES 89)  # RTX 4090 (sm_89) first: DESIGN §4
endif()

# IMMERDINK_HOST_ONLY: build only libimmerdink_host (DESIGN §6a: lowering, legality and the
# memory model as a CUDA-free C ABI) and its CPU tests. Needs no CUDA toolkit, no nvcc and no
# CUTLASS; tests/check_host_cuda_free.sh builds the repository this way with nvcc off PATH.
option(IMMERDINK_HOST_ONLY "Build only the CUDA-free libimmerdink_host" OFF)

project(immerdink VERSION 0.0.0 LANGUAGES C CXX)
if(NOT IMMERDINK_HOST_ONLY)
  enable_language(CUDA)
endif()

set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CUDA_STANDARD 17)  # CUTLASS 2.x translation units (DESIGN §4, PLAN M0)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_EXTENSIONS OFF)
set(CMAKE_CUDA_RUNTIME_LIBRARY Shared)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

option(IMMERDINK_WERROR "Treat warnings in immerdink's own code as errors" OFF)
option(IMMERDINK_BUILD_TESTS "Build the tests (and the test-only hooks of src/test_hooks.cpp)" ON)
option(IMMERDINK_PROFILE "Benchmark-only per-phase event timing hooks (src/profile.hpp)" OFF)
set(IMMERDINK_CUTLASS_DIR "${PROJECT_SOURCE_DIR}/third_party/cutlass"
    CACHE PATH "CUTLASS v4.8.0 source tree (default: the git submodule)")

if(NOT IMMERDINK_HOST_ONLY)
  find_package(CUDAToolkit REQUIRED)
endif()
find_package(Python3 COMPONENTS Interpreter)

# --- Warnings ---------------------------------------------------------------------------
add_library(imd_warnings INTERFACE)
target_compile_options(imd_warnings INTERFACE
  $<$<COMPILE_LANGUAGE:C,CXX>:-Wall -Wextra -Wpedantic -Wshadow>
  $<$<AND:$<COMPILE_LANGUAGE:C,CXX>,$<BOOL:${IMMERDINK_WERROR}>>:-Werror>
  $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-Wall,-Wextra>
  $<$<AND:$<COMPILE_LANGUAGE:CUDA>,$<BOOL:${IMMERDINK_WERROR}>>:-Werror=all-warnings>)

# --- Device constant tables (spec/params.py --emit-header) -------------------------------
# The committed include/immerdink/detail/constants.hpp lets the project build without
# Python. With Python the build regenerates the header into the build tree, which then
# shadows the committed copy, and the test `constants_header_current` fails if the two differ.
set(IMD_CONSTANTS_HEADER include/immerdink/detail/constants.hpp)
add_library(imd_constants INTERFACE)
if(Python3_FOUND)
  set(IMD_GENERATED_INCLUDE "${PROJECT_BINARY_DIR}/generated/include")
  add_custom_command(
    OUTPUT "${IMD_GENERATED_INCLUDE}/immerdink/detail/constants.hpp"
    COMMAND Python3::Interpreter "${PROJECT_SOURCE_DIR}/spec/params.py"
            --emit-header "${IMD_GENERATED_INCLUDE}/immerdink/detail/constants.hpp"
    DEPENDS "${PROJECT_SOURCE_DIR}/spec/params.py"
    COMMENT "Generating the device constant tables from spec/params.py"
    VERBATIM)
  add_custom_target(imd_constants_gen
    DEPENDS "${IMD_GENERATED_INCLUDE}/immerdink/detail/constants.hpp")
  target_include_directories(imd_constants BEFORE INTERFACE "${IMD_GENERATED_INCLUDE}")
  add_dependencies(imd_constants imd_constants_gen)
else()
  message(STATUS "Python3 not found: using the committed ${IMD_CONSTANTS_HEADER}")
endif()
target_include_directories(imd_constants INTERFACE "${PROJECT_SOURCE_DIR}/include")

# --- libimmerdink_host (CUDA-free; DESIGN §6a) ------------------------------------------
# The single implementation of argument checks, lowering, legality (§5, §3.2 RN acceptance)
# and the memory model (§4a, plan bytes), behind the C ABI of
# include/immerdink/immerdink_host.h (imd_workspace_size, imd_workspace_layout,
# imd_plan_bytes, imd_lowering, ...). Plain C++17: no CUDA headers, no CUDA libraries.
# libimmerdink links it and sizes its plans with it. Its internal C++ entry points
# (IMD_HOST_INTERNAL, src/host/export.hpp) are exported for libimmerdink and the tests only.
# v1.8 (§6c): the performance model (src/host/perf.cpp) and its shipped calibration.
add_library(immerdink_host SHARED
  src/host/index_map.cpp src/host/workspace.cpp src/host/analysis.cpp src/host/k4_config.cpp
  src/host/pack.cpp src/host/host_api.cpp src/host/perf.cpp src/host/accuracy.cpp)
target_include_directories(immerdink_host PUBLIC
  $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include> $<INSTALL_INTERFACE:include>)
# DESIGN §6c.2: the shipped calibration file is embedded as a raw string literal (the file is
# the one source; editing it re-runs the configure step).
set(IMD_PERF_CALIB_DEFAULT "${PROJECT_SOURCE_DIR}/src/host/calib/rtx4090_sm89.calib")
set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS "${IMD_PERF_CALIB_DEFAULT}")
file(READ "${IMD_PERF_CALIB_DEFAULT}" IMD_PERF_CALIB_TEXT)
file(CONFIGURE OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/generated_perf/perf_calib_default.inc"
     CONTENT "// Generated by CMake from src/host/calib/rtx4090_sm89.calib (DESIGN §6c.2).\nconstexpr char kPerfCalibDefault[] = R\"IMDCALIB(@IMD_PERF_CALIB_TEXT@)IMDCALIB\";\n"
     @ONLY)
target_include_directories(immerdink_host PRIVATE "${CMAKE_CURRENT_BINARY_DIR}/generated_perf")
target_link_libraries(immerdink_host PRIVATE imd_constants imd_warnings)
set_target_properties(immerdink_host PROPERTIES CXX_STANDARD 17 CXX_STANDARD_REQUIRED ON
                                                CXX_EXTENSIONS OFF
                                                CXX_VISIBILITY_PRESET hidden
                                                VISIBILITY_INLINES_HIDDEN ON)
# Internal headers (src/, src/host/) for libimmerdink and the unit tests.
add_library(imd_host_internal INTERFACE)
target_include_directories(imd_host_internal INTERFACE
  "${PROJECT_SOURCE_DIR}/src" "${PROJECT_SOURCE_DIR}/src/host")
target_link_libraries(imd_host_internal INTERFACE immerdink_host imd_constants)
# DESIGN §6e: the C floor of the planner queries (ns per host-library call in a C loop),
# driven by bench/planner_overhead.py. CPU only; built in host-only builds too.
add_executable(imd_planner_floor bench/planner_floor.c)
target_link_libraries(imd_planner_floor PRIVATE immerdink_host imd_warnings)

if(NOT IMMERDINK_HOST_ONLY)
# --- CUTLASS (header-only, CUDA translation units only) ----------------------------------
if(NOT EXISTS "${IMMERDINK_CUTLASS_DIR}/include/cutlass/cutlass.h")
  message(FATAL_ERROR
    "CUTLASS not found at ${IMMERDINK_CUTLASS_DIR}. Run\n"
    "  git submodule update --init --depth 1 third_party/cutlass\n"
    "or configure with -DIMMERDINK_CUTLASS_DIR=/path/to/cutlass (v4.8.0).")
endif()
add_library(imd_cutlass INTERFACE)
target_include_directories(imd_cutlass SYSTEM INTERFACE "${IMMERDINK_CUTLASS_DIR}/include")
target_compile_options(imd_cutlass INTERFACE
  $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr -diag-suppress=20012>)

# --- Library -----------------------------------------------------------------------------
# The library: C ABI (api.cpp), K0-K5 + DIRECT kernels (contract.cu), K4 CUTLASS GEMM
# (gemm_k4.cu, one K4Gemm instantiation per configuration in src/k4/).
# Test-only hooks (src/test_hooks.cpp: launch-limit override, lowering dump) are built only
# with IMMERDINK_BUILD_TESTS=ON, and the benchmark phase marks (src/profile.cpp) only with
# IMMERDINK_PROFILE=ON; a production build exports neither.
add_library(immerdink SHARED src/api.cpp src/contract.cu src/gemm_k4.cu
  src/k4/k4_cfg0.cu src/k4/k4_cfg1.cu src/k4/k4_cfg2.cu src/k4/k4_cfg3.cu)
if(IMMERDINK_BUILD_TESTS)
  target_sources(immerdink PRIVATE src/test_hooks.cpp)
endif()
if(IMMERDINK_PROFILE)
  target_sources(immerdink PRIVATE src/profile.cpp)
  target_compile_definitions(immerdink PRIVATE IMMERDINK_PROFILE=1)
endif()
target_include_directories(immerdink PUBLIC
  $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include> $<INSTALL_INTERFACE:include>)
# PUBLIC immerdink_host: the CUDA-free half of the ABI (imd_workspace_size, imd_status_string,
# the host queries) lives there; libimmerdink has it as a DT_NEEDED dependency.
target_link_libraries(immerdink PUBLIC immerdink_host CUDA::cudart
                                PRIVATE imd_host_internal imd_cutlass imd_warnings)
set_target_properties(immerdink PROPERTIES CXX_VISIBILITY_PRESET hidden
                                           CUDA_VISIBILITY_PRESET hidden
                                           VISIBILITY_INLINES_HIDDEN ON)

endif()  # NOT IMMERDINK_HOST_ONLY

# --- Tests -------------------------------------------------------------------------------
if(IMMERDINK_BUILD_TESTS)
  enable_testing()
  add_subdirectory(tests)
endif()
