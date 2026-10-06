# Correctness tests (pass/fail; run ungated, see tools/run_quiet.sh). No timing here.

# --- libimmerdink_host (CUDA-free; DESIGN §6a): also built with IMMERDINK_HOST_ONLY -------
# The host query ABI from C, linked against libimmerdink_host alone (also proves that
# immerdink_host.h is valid C11 without any CUDA header).
add_executable(test_host_c test_host_c.c)
target_link_libraries(test_host_c PRIVATE immerdink_host imd_warnings)
add_test(NAME host_c_api COMMAND test_host_c)
# Host API 10 (DESIGN §6c.3): the configure-time string check of a calibration file passes on
# the shipped file and fails on a notes value one byte over its field.
add_test(NAME calib_strings_shipped
         COMMAND "${CMAKE_COMMAND}"
                 "-DIMD_CALIB_FILE=${PROJECT_SOURCE_DIR}/src/host/calib/rtx4090_sm89.calib"
                 -P "${PROJECT_SOURCE_DIR}/src/host/calib/check_strings.cmake")
add_test(NAME calib_strings_too_long
         COMMAND "${CMAKE_COMMAND}"
                 "-DIMD_CALIB_FILE=${CMAKE_CURRENT_SOURCE_DIR}/data/perf_calib_notes_too_long.calib"
                 -P "${PROJECT_SOURCE_DIR}/src/host/calib/check_strings.cmake")
# (CMake wraps the message: blanks or line breaks between the words; no ';' in a regex here,
# a property value is a list.)
set_tests_properties(calib_strings_too_long PROPERTIES PASS_REGULAR_EXPRESSION
  "too_long[.]calib:9:[ \n]+'notes'[ \n]+is[ \n]+128[ \n]+bytes.[ \n]+the[ \n]+calibration[ \n]+field[ \n]+holds[ \n]+127")

if(Python3_FOUND)
  # The host library and its C++ lowering vs spec/params.py on generated descriptors: the
  # workspace vectors (legality, layouts, plan device bytes), every vec_mode and g = 1..N.
  set(vectors "${CMAKE_CURRENT_BINARY_DIR}/workspace_vectors.txt")
  add_custom_command(
    OUTPUT "${vectors}"
    COMMAND Python3::Interpreter "${CMAKE_CURRENT_SOURCE_DIR}/gen_workspace_vectors.py" "${vectors}"
    DEPENDS gen_workspace_vectors.py "${PROJECT_SOURCE_DIR}/spec/params.py"
    COMMENT "Generating workspace test vectors from spec/params.py"
    VERBATIM)
  add_custom_target(workspace_vectors ALL DEPENDS "${vectors}")
  add_executable(test_workspace test_workspace.cpp)
  target_link_libraries(test_workspace PRIVATE imd_host_internal imd_warnings)
  add_dependencies(test_workspace workspace_vectors)
  add_test(NAME workspace_vs_params COMMAND test_workspace "${vectors}")

  # DESIGN v1.13 §4b stage 2: the K-alignment resolver and its test hook, A passed explicitly,
  # the pack format word and analyze_packed's PACK_MISMATCH reason order (FORMAT via the hook,
  # an unknown format word, the k_hat drift check). CPU only.
  add_executable(test_k_align test_k_align.cpp)
  target_link_libraries(test_k_align PRIVATE imd_host_internal imd_warnings)
  add_test(NAME host_k_align COMMAND test_k_align)

  # The Python wrapper (python/immerdink/host.py) over libimmerdink_host vs the C ABI and
  # vs spec/params.py and spec/oracle.py (lowering summary, sizes, g sweep). CPU only.
  add_test(NAME host_python
           COMMAND Python3::Interpreter "${CMAKE_CURRENT_SOURCE_DIR}/test_host.py"
                   --host-lib "$<TARGET_FILE:immerdink_host>" --suite host -v)
  # DESIGN v1.8 §6c: the performance model (calibration, features, rank). CPU only.
  add_test(NAME host_perf
           COMMAND Python3::Interpreter "${CMAKE_CURRENT_SOURCE_DIR}/test_host.py"
                   --host-lib "$<TARGET_FILE:immerdink_host>" --suite perf -v)
  # DESIGN v1.11 §6e: the fast planner queries (python/immerdink/planner.py) == host.py, bit
  # for bit, on random queries; Planner / Memo bounds and threads. CPU only.
  add_test(NAME host_planner
           COMMAND Python3::Interpreter "${CMAKE_CURRENT_SOURCE_DIR}/test_planner.py"
                   --host-lib "$<TARGET_FILE:immerdink_host>" -v)
  # DESIGN v1.12 §6f: accuracy-aware planning (imd_err_*, predict / rank err fields) vs
  # spec/params.py error_estimate (exact rational enclosure). CPU only.
  add_test(NAME host_accuracy
           COMMAND Python3::Interpreter "${CMAKE_CURRENT_SOURCE_DIR}/test_accuracy.py"
                   --host-lib "$<TARGET_FILE:immerdink_host>" -v)
  add_test(NAME params_check COMMAND Python3::Interpreter "${PROJECT_SOURCE_DIR}/spec/params.py" --check)
  add_test(NAME constants_header_current
           COMMAND "${CMAKE_COMMAND}" -E compare_files
                   "${IMD_GENERATED_INCLUDE}/immerdink/detail/constants.hpp"
                   "${PROJECT_SOURCE_DIR}/${IMD_CONSTANTS_HEADER}")
endif()

# libimmerdink_host has no CUDA dependency or symbol (readelf / ldd / nm); in the full build
# also: the repository configures, builds and passes its host tests with IMMERDINK_HOST_ONLY
# and no nvcc on PATH (a nested build in tests/host_only).
if(IMMERDINK_HOST_ONLY)
  add_test(NAME host_cuda_free
           COMMAND bash "${CMAKE_CURRENT_SOURCE_DIR}/check_host_cuda_free.sh"
                   "$<TARGET_FILE:immerdink_host>")
  return()
endif()
add_test(NAME host_cuda_free
         COMMAND bash "${CMAKE_CURRENT_SOURCE_DIR}/check_host_cuda_free.sh"
                 "$<TARGET_FILE:immerdink_host>" --standalone "${PROJECT_SOURCE_DIR}"
                 "${CMAKE_CURRENT_BINARY_DIR}/host_only" "${CMAKE_CXX_COMPILER}"
                 "${Python3_EXECUTABLE}" "${CMAKE_COMMAND}")
set_tests_properties(host_cuda_free PROPERTIES TIMEOUT 600)

# --- libimmerdink (CUDA) -------------------------------------------------------------------
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

# DESIGN v1.2 §6: concurrent host threads (shared and own plans, both vec modes, own streams
# and workspaces), bit-identical to a sequential run.
add_executable(test_threads test_threads.cpp)
target_link_libraries(test_threads PRIVATE immerdink CUDA::cudart Threads::Threads imd_warnings)
add_test(NAME threads COMMAND test_threads)
set_tests_properties(threads PROPERTIES RESOURCE_LOCK gpu TIMEOUT 300)

# DESIGN v1.2 §6: imd_contract captured into a CUDA graph (global capture mode) and replayed,
# bit-exact, in each vec_mode.
add_executable(test_graph test_graph.cpp)
target_link_libraries(test_graph PRIVATE immerdink CUDA::cudart imd_warnings)
add_test(NAME graph_capture COMMAND test_graph)
set_tests_properties(graph_capture PROPERTIES RESOURCE_LOCK gpu TIMEOUT 300)

# LD_PRELOAD counter of CUDA runtime allocations: imd_contract must allocate nothing (M3).
add_library(imd_alloc_shim SHARED alloc_shim.c)
target_include_directories(imd_alloc_shim PRIVATE ${CUDAToolkit_INCLUDE_DIRS})
target_link_libraries(imd_alloc_shim PRIVATE ${CMAKE_DL_LIBS})
target_compile_definitions(imd_alloc_shim PRIVATE
  IMD_CUDART_SONAME="libcudart.so.${CUDAToolkit_VERSION_MAJOR}")

if(Python3_FOUND)
  # imd_contract on the GPU vs spec/oracle.py, bit for bit (tests/test_contract.py), with
  # the allocation shim preloaded; tensor front end and robustness (tests/test_frontend.py).
  # contract_highrank (operand ranks 6-12) has hard size guards; run it alone with
  # `tools/test.sh -R contract_highrank`, or skip it with `-E contract_highrank`.
  set(shim "$<TARGET_FILE:imd_alloc_shim>")
  foreach(suite api stages e2e tensor bigk replay every_n k_align pad_mutation)
    add_test(NAME contract_${suite}
             COMMAND Python3::Interpreter "${CMAKE_CURRENT_SOURCE_DIR}/test_contract.py"
                     --lib "$<TARGET_FILE:immerdink>" --shim "${shim}" --suite ${suite} -v)
    set_tests_properties(contract_${suite} PROPERTIES RESOURCE_LOCK gpu TIMEOUT 900
                         ENVIRONMENT "LD_PRELOAD=${shim}")
  endforeach()
  foreach(suite lowering legality random fixtures robust highrank)
    add_test(NAME contract_${suite}
             COMMAND Python3::Interpreter "${CMAKE_CURRENT_SOURCE_DIR}/test_frontend.py"
                     --lib "$<TARGET_FILE:immerdink>" --shim "${shim}" --suite ${suite} -v)
    set_tests_properties(contract_${suite} PROPERTIES RESOURCE_LOCK gpu TIMEOUT 900
                         ENVIRONMENT "LD_PRELOAD=${shim}")
  endforeach()
  set_tests_properties(contract_highrank PROPERTIES TIMEOUT 300)
  # DESIGN v1.13 §4b.8 (hygiene): the initcheck mutation test for one-sided padding. Needs
  # compute-sanitizer: IMMERDINK_COMPUTE_SANITIZER (a path; tools/test.sh passes the variable
  # of the same name), else found next to nvcc; without it the test is not registered.
  set(IMMERDINK_COMPUTE_SANITIZER "" CACHE FILEPATH "compute-sanitizer (contract_initcheck_pad)")
  set(_imd_san "${IMMERDINK_COMPUTE_SANITIZER}")
  if(NOT _imd_san)
    find_program(IMD_COMPUTE_SANITIZER_FOUND compute-sanitizer
                 HINTS "${CUDAToolkit_BIN_DIR}" "${CUDAToolkit_BIN_DIR}/../compute-sanitizer")
    if(IMD_COMPUTE_SANITIZER_FOUND)
      set(_imd_san "${IMD_COMPUTE_SANITIZER_FOUND}")
    endif()
  endif()
  if(_imd_san)
    add_test(NAME contract_initcheck_pad
             COMMAND bash "${CMAKE_CURRENT_SOURCE_DIR}/initcheck_pad_mutation.sh" "${_imd_san}"
                     "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/test_contract.py"
                     "$<TARGET_FILE:immerdink>")
    set_tests_properties(contract_initcheck_pad PROPERTIES RESOURCE_LOCK gpu TIMEOUT 900)
  else()
    message(STATUS "compute-sanitizer not found: contract_initcheck_pad not registered "
                   "(set IMMERDINK_COMPUTE_SANITIZER)")
  endif()
  # DESIGN v1.7 §6b packed operands: imd_pack (every byte) and imd_contract_packed vs
  # spec/oracle.py, bit for bit, with the allocation shim; reuse; statuses.
  foreach(suite packed reuse api k_align)
    add_test(NAME packed_${suite}
             COMMAND Python3::Interpreter "${CMAKE_CURRENT_SOURCE_DIR}/test_packed.py"
                     --lib "$<TARGET_FILE:immerdink>" --shim "${shim}" --suite ${suite} -v)
    set_tests_properties(packed_${suite} PROPERTIES RESOURCE_LOCK gpu TIMEOUT 1800
                         ENVIRONMENT "LD_PRELOAD=${shim}")
  endforeach()

  # DESIGN v1.9 §6d: the Python device API (python/immerdink: Plan, Pack, contract) on the
  # GPU, bit for bit vs spec/oracle.py, with the allocation shim (pytest; CAI / DLPack
  # producers built with ctypes, no CuPy or PyTorch needed).
  add_test(NAME python_device
           COMMAND Python3::Interpreter -m pytest -q -p no:cacheprovider
                   "${CMAKE_CURRENT_SOURCE_DIR}/test_python_device.py")
  set_tests_properties(python_device PROPERTIES RESOURCE_LOCK gpu TIMEOUT 900
                       ENVIRONMENT "IMMERDINK_LIB=$<TARGET_FILE:immerdink>;IMMERDINK_SHIM=${shim};LD_PRELOAD=${shim}")

  # libimmerdink's real plans vs libimmerdink_host's queries (DESIGN §6a): the plan's GEMM
  # view, workspace layout, device bytes, K4 parameter sets and launches, and host payload
  # (src/test_hooks.cpp imd_test_plan_sizes) equal imd_lowering / imd_plan_bytes /
  # imd_workspace_layout on random descriptors, every vec_mode and g = 1..N.
  add_test(NAME host_vs_plan
           COMMAND Python3::Interpreter "${CMAKE_CURRENT_SOURCE_DIR}/test_host.py"
                   --host-lib "$<TARGET_FILE:immerdink_host>" --lib "$<TARGET_FILE:immerdink>"
                   --suite plan -v)
  set_tests_properties(host_vs_plan PROPERTIES RESOURCE_LOCK gpu TIMEOUT 600)

  add_test(NAME python_harness
           COMMAND Python3::Interpreter "${CMAKE_CURRENT_SOURCE_DIR}/harness.py"
                   --lib "$<TARGET_FILE:immerdink>")
  # DESIGN §3.8 bound on CC-shaped fixtures with stressing data (docs/ACCURACY.md), small
  # extents; exit 1 on any element whose error exceeds the bound.
  add_test(NAME accuracy_bound_quick
           COMMAND Python3::Interpreter "${PROJECT_SOURCE_DIR}/bench/accuracy/contractions.py"
                   --lib "$<TARGET_FILE:immerdink>" --quick --rows 16)
  set_tests_properties(accuracy_bound_quick PROPERTIES RESOURCE_LOCK gpu TIMEOUT 300)
  # DESIGN v1.12 §6f: measured errors <= libimmerdink_host's rigorous estimates (every metric)
  # on random small cases; reports the typical estimate's coverage. Exit 1 on a violation.
  add_test(NAME accuracy_err_api_quick
           COMMAND Python3::Interpreter "${PROJECT_SOURCE_DIR}/bench/accuracy/err_api.py"
                   --lib "$<TARGET_FILE:immerdink>" --host-lib "$<TARGET_FILE:immerdink_host>"
                   --quick)
  set_tests_properties(accuracy_err_api_quick PROPERTIES RESOURCE_LOCK gpu TIMEOUT 300)
endif()

# GPU tests share one device; run them one at a time.
set_tests_properties(device_constants cutlass_int8_gemm cutlass_mod_epilogue
                     PROPERTIES RESOURCE_LOCK gpu)
