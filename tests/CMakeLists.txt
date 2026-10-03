# Correctness tests (pass/fail; run ungated, see tools/run_quiet.sh). No timing here.

# C ABI from C (also proves the public header is valid C11).
add_executable(test_c_api test_c_api.c)
target_link_libraries(test_c_api PRIVATE immerdink imd_warnings)
add_test(NAME c_api COMMAND test_c_api)

# Generated constant tables re-checked from their definitions in C++.
add_executable(test_constants test_constants.cpp)
target_link_libraries(test_constants PRIVATE imd_constants imd_warnings)
add_test(NAME constants_tables COMMAND test_constants)

# CUDA smoke test: __constant__ tables on sm_89 + exhaustive epilogue mod reduction.
add_executable(test_device_constants test_device_constants.cu)
target_include_directories(test_device_constants PRIVATE "${PROJECT_SOURCE_DIR}/src")
target_link_libraries(test_device_constants PRIVATE imd_constants imd_warnings)
add_test(NAME device_constants COMMAND test_device_constants)

# CUTLASS de-risking for M1: int8 batched TN GEMM, and the mod-p -> u8 EVT epilogue.
add_executable(test_cutlass_int8_gemm test_cutlass_int8_gemm.cu)
target_link_libraries(test_cutlass_int8_gemm PRIVATE imd_constants imd_cutlass)
add_test(NAME cutlass_int8_gemm COMMAND test_cutlass_int8_gemm)

add_executable(test_cutlass_mod_epilogue test_cutlass_mod_epilogue.cu)
target_include_directories(test_cutlass_mod_epilogue PRIVATE "${PROJECT_SOURCE_DIR}/src")
target_link_libraries(test_cutlass_mod_epilogue PRIVATE imd_constants imd_cutlass)
add_test(NAME cutlass_mod_epilogue COMMAND test_cutlass_mod_epilogue)

# PLAN M1: exhaustive CPU check of the §3.5 / §3.6 modular reductions (the kernels' code).
find_package(Threads REQUIRED)
add_executable(test_mod_exhaustive test_mod_exhaustive.cpp)
target_include_directories(test_mod_exhaustive PRIVATE "${PROJECT_SOURCE_DIR}/src")
target_link_libraries(test_mod_exhaustive PRIVATE imd_constants imd_warnings Threads::Threads)
add_test(NAME mod_exhaustive_cpu COMMAND test_mod_exhaustive)

if(Python3_FOUND)
  # imd_contract on the GPU vs spec/oracle.py, bit for bit (tests/test_contract.py).
  foreach(suite api stages e2e tensor bigk replay)
    add_test(NAME contract_${suite}
             COMMAND Python3::Interpreter "${CMAKE_CURRENT_SOURCE_DIR}/test_contract.py"
                     --lib "$<TARGET_FILE:immerdink>" --suite ${suite} -v)
    set_tests_properties(contract_${suite} PROPERTIES RESOURCE_LOCK gpu TIMEOUT 900)
  endforeach()

  # imd_workspace_size and the C++ lowering vs spec/params.py on generated descriptors.
  set(vectors "${CMAKE_CURRENT_BINARY_DIR}/workspace_vectors.txt")
  add_custom_command(
    OUTPUT "${vectors}"
    COMMAND Python3::Interpreter "${CMAKE_CURRENT_SOURCE_DIR}/gen_workspace_vectors.py" "${vectors}"
    DEPENDS gen_workspace_vectors.py "${PROJECT_SOURCE_DIR}/spec/params.py"
    COMMENT "Generating workspace test vectors from spec/params.py"
    VERBATIM)
  add_custom_target(workspace_vectors ALL DEPENDS "${vectors}")
  add_executable(test_workspace test_workspace.cpp)
  target_link_libraries(test_workspace PRIVATE immerdink imd_core imd_warnings)
  add_dependencies(test_workspace workspace_vectors)
  add_test(NAME workspace_vs_params COMMAND test_workspace "${vectors}")

  add_test(NAME python_harness
           COMMAND Python3::Interpreter "${CMAKE_CURRENT_SOURCE_DIR}/harness.py"
                   --lib "$<TARGET_FILE:immerdink>")
  add_test(NAME params_check COMMAND Python3::Interpreter "${PROJECT_SOURCE_DIR}/spec/params.py" --check)
  add_test(NAME constants_header_current
           COMMAND "${CMAKE_COMMAND}" -E compare_files
                   "${IMD_GENERATED_INCLUDE}/immerdink/detail/constants.hpp"
                   "${PROJECT_SOURCE_DIR}/${IMD_CONSTANTS_HEADER}")
endif()

# GPU tests share one device; run them one at a time.
set_tests_properties(device_constants cutlass_int8_gemm cutlass_mod_epilogue
                     PROPERTIES RESOURCE_LOCK gpu)
