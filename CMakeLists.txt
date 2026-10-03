cmake_minimum_required(VERSION 3.28)

if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES 89)  # RTX 4090 (sm_89) first: DESIGN §4
endif()

project(immerdink VERSION 0.0.0 LANGUAGES C CXX CUDA)

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
option(IMMERDINK_BUILD_TESTS "Build the tests" ON)
set(IMMERDINK_CUTLASS_DIR "${PROJECT_SOURCE_DIR}/third_party/cutlass"
    CACHE PATH "CUTLASS v4.8.0 source tree (default: the git submodule)")

find_package(CUDAToolkit REQUIRED)
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
# imd_core: internal C++ (IndexMap lowering, workspace layout), also linked by unit tests.
add_library(imd_core OBJECT src/index_map.cpp src/workspace.cpp)
target_include_directories(imd_core PUBLIC "${PROJECT_SOURCE_DIR}/src")
target_link_libraries(imd_core PUBLIC imd_constants CUDA::cudart PRIVATE imd_warnings)
set_target_properties(imd_core PROPERTIES CXX_VISIBILITY_PRESET hidden
                                          VISIBILITY_INLINES_HIDDEN ON)

add_library(immerdink SHARED src/api.cpp)
target_include_directories(immerdink PUBLIC
  $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include> $<INSTALL_INTERFACE:include>)
target_link_libraries(immerdink PUBLIC CUDA::cudart PRIVATE imd_core imd_warnings)
set_target_properties(immerdink PROPERTIES CXX_VISIBILITY_PRESET hidden
                                           VISIBILITY_INLINES_HIDDEN ON)

# --- Tests -------------------------------------------------------------------------------
if(IMMERDINK_BUILD_TESTS)
  enable_testing()
  add_subdirectory(tests)
endif()
