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
# Issue 22 (docs/NOTES-22.md "Library integration (design)"): the Hopper K4 backend's host rules (device-class table, legality, the R2 tile /
# scheduler table, the profitability floor), CUDA-free: runs on every machine, with the backend built in or not.
add_executable(test_k4_wgmma_select test_k4_wgmma_select.cpp)
target_link_libraries(test_k4_wgmma_select PRIVATE imd_host_internal imd_warnings)
add_test(NAME k4_wgmma_select COMMAND test_k4_wgmma_select)
# tools/test.sh's detection of a caller's ctest selection option (-R/-E/-L/...): any such
# option must put the whole run under the GPU lock (CPU-only; no build, no GPU).
add_test(NAME tools_test_sh_selftest
         COMMAND bash "${PROJECT_SOURCE_DIR}/tools/test.sh" --self-test)
# The memory-budget wrapper must not make tools/test.sh relaunch itself forever when there is no
# systemd user manager (systemctl stubbed to fail). CPU-only.
add_test(NAME tools_memcap_loop_check
         COMMAND bash "${CMAKE_CURRENT_SOURCE_DIR}/memcap_loop_check.sh" "${PROJECT_SOURCE_DIR}")

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

  # Issue 21: K1's L2 budgets as fractions of the card's L2 (the 4090's values exactly). CPU only.
  add_executable(test_l2_knobs test_l2_knobs.cpp)
  target_link_libraries(test_l2_knobs PRIVATE imd_host_internal imd_warnings)
  add_test(NAME host_l2_knobs COMMAND test_l2_knobs)

  # Issue 21 / R2: the per-device-class sequence defaults (sm_90 = E1 + F1 + limb CRT = 13, sm_89 = 1, sm_120 = 0) and the forcing rules. CPU only, CUDA-free.
  add_executable(test_seq_table test_seq_table.cpp)
  target_link_libraries(test_seq_table PRIVATE imd_host_internal imd_warnings)
  if(NOT IMMERDINK_SEQ_TWINS)
    target_compile_definitions(test_seq_table PRIVATE IMD_NO_SEQ_TWINS=1)
  endif()
  add_test(NAME host_seq_table COMMAND test_seq_table)

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
  # DESIGN v1.19 §6g (NOTES-C1 §4.1): the footprints precision phases rely on, on the library
  # == spec/params.py: F1 / F2 (published from API 13 on: workspace and pack bytes
  # nondecreasing in N and g), C-ws / C-pack, and the internal facts I1-I3. CPU only.
  add_test(NAME host_footprint_monotone
           COMMAND Python3::Interpreter "${CMAKE_CURRENT_SOURCE_DIR}/test_footprint.py"
                   --host-lib "$<TARGET_FILE:immerdink_host>" -v)
  # DESIGN v1.19 §6g.10-§6g.11 (NOTES-C1 §4.2, stage 1): python/immerdink/precision.py, the
  # PrecisionPlan phase rule == a brute-force composition from host.py's queries (per-role
  # pack cost, share groups, BUDGET / MISSED / DIRECT, monotone floor), its footprints, and
  # rho_from_output == spec/params.py rho_recipe bit for bit. CPU only (needs numpy).
  add_test(NAME host_precision
           COMMAND Python3::Interpreter "${CMAKE_CURRENT_SOURCE_DIR}/test_precision.py"
                   --host-lib "$<TARGET_FILE:immerdink_host>" -v)
  # NOTES-C1 §4.3 (stage 2): the C2 study's exact engine (bench/accuracy/kw_engine.py: sigma,
  # K-weights, emulation, exact references) and its several-partner / batched paths
  # (c2_gate.py run_multi / run_batched) == spec/oracle.py, so a later change to the oracle
  # cannot silently invalidate the M-a / M-d1 engine. CPU only (numpy), about 10 s.
  add_test(NAME c2_engine_selftest
           COMMAND Python3::Interpreter "${PROJECT_SOURCE_DIR}/bench/accuracy/c2_gate.py" selftest)
  # NOTES-C1 §4.3 M-c (§5.9): bench/sweep_disjoint.py, the check that a hold-out shares no
  # GEMM view with the fit sweeps. Its matching rule (another spec, A and B swapped, a tensor
  # equal to a square, near sizes, the batch), its exit codes (--against fails, --report only
  # lists), and the committed sweeps: M-c against sweep_calib.txt and sweep_calib_n.txt, and
  # the Q24 cases (sweep_calib_n.txt) against M-c and the product hold-out
  # (sweep_perf_validate.txt). CPU only, standard library, under a second.
  add_test(NAME sweep_disjoint_selftest
           COMMAND Python3::Interpreter "${PROJECT_SOURCE_DIR}/bench/sweep_disjoint.py" --self-test)
  # NOTES-C1 §4.3 M-d2 (§5.10): bench/accuracy/md2_time.py's pre-registered rules on
  # synthetic records: the decision per system and data set (rules (a)-(d), STOPPED on any
  # bit-exact failure in the file), the bars, bands and protocol constants pinned as
  # literals, and the provenance checks (calibration, registered_check, the --md2-cleared
  # citation, driver, parameters, configuration, one library per file). Registering it does
  # not change the driver's sha256. CPU only, standard library, under a second.
  add_test(NAME md2_time_selftest
           COMMAND Python3::Interpreter "${PROJECT_SOURCE_DIR}/bench/accuracy/md2_time.py" selftest)
  # Campaign kit (issue 15; bench/campaign, docs/CAMPAIGN.md): the statistics (paired interval, tie rule, regression
  # guard), the manifest lint (gate G1), the resumable runner (kill -9, resume, time limits, copy-back) and the analyzer
  # (a damaged folder gives INVALID; the issue-11 folders reproduce their tables when they are on this machine).
  # CPU only, standard library, about 25 s in all.
  foreach(t stats manifest runner analyze archgate summarize_tests)
    add_test(NAME campaign_${t}_selftest
             COMMAND Python3::Interpreter -I "${PROJECT_SOURCE_DIR}/bench/campaign/test_${t}.py")
  endforeach()
  add_test(NAME params_check COMMAND Python3::Interpreter "${PROJECT_SOURCE_DIR}/spec/params.py" --check)
  # Issue 21 (docs/NOTES-21.md): the analysis of the bandwidth microbenchmark (names the limiter of the pack / CRT phases). CPU only.
  add_test(NAME bench_membw_selftest
           COMMAND Python3::Interpreter -I "${PROJECT_SOURCE_DIR}/bench/membw/test_analyze.py")
  # math.fma for Python before 3.13 (the PACE env), used by the oracle tests. CPU only.
  add_test(NAME spec_fma_compat
           COMMAND Python3::Interpreter -I "${PROJECT_SOURCE_DIR}/spec/test_fma_compat.py")
  # Issue 21: the CRT with the sums in binary64 limbs against the reference for every N (CPU only; the GPU kernel: crt_f64_gpu).
  add_test(NAME spec_crt_f64
           COMMAND Python3::Interpreter -I "${PROJECT_SOURCE_DIR}/spec/test_crt_f64.py")
  # Issue 23 (known divergence): the decision logic of contract_api's per-W pack-occupancy check, on mocked tables (the Hopper pattern is
  # accepted as known, everything else decides as before). CPU only.
  add_test(NAME contract_occupancy_known
           COMMAND Python3::Interpreter -I "${CMAKE_CURRENT_SOURCE_DIR}/test_occupancy_known.py")
  # Issue 20 (docs/NOTES-20.md): the K4 tile-table fit and its adoption rule (reproduces the shipped 4090 constants
  # from the A3 data where they are on this machine). CPU only.
  add_test(NAME bench_k4_fit_selftest
           COMMAND Python3::Interpreter -I "${PROJECT_SOURCE_DIR}/bench/test_k4_fit.py")
  # Issue 18 (docs/NOTES-18.md): the device-class rules (libimmerdink_host against spec/devices.py) and the
  # coverage of every kernel launch site by the device-status check, with the K4 limits. CPU only.
  add_test(NAME host_devices
           COMMAND Python3::Interpreter "${CMAKE_CURRENT_SOURCE_DIR}/test_devices.py"
                   --host-lib "$<TARGET_FILE:immerdink_host>")
  add_test(NAME device_kernel_sites
           COMMAND Python3::Interpreter "${CMAKE_CURRENT_SOURCE_DIR}/check_kernel_sites.py" "${PROJECT_SOURCE_DIR}")
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

# DESIGN v1.14 §3.4.1: the pack emitter (|a'| as W words) vs an exact host reference, every
# N, every eps, E = 16 / 8 / 4, W = 2 / 3, RN and TRUNC, device asserts on.
add_executable(test_pack_emitter test_pack_emitter.cu)
target_include_directories(test_pack_emitter PRIVATE "${PROJECT_SOURCE_DIR}/src")
target_link_libraries(test_pack_emitter PRIVATE imd_constants imd_warnings)
add_test(NAME pack_emitter COMMAND test_pack_emitter)
# Issue 21: sequence F1 (the CRT finish through a hardware u64 -> f64 conversion) against crt_finish + round_scaled, every N, with mutations.
add_executable(test_finish_f1 test_finish_f1.cu)
target_include_directories(test_finish_f1 PRIVATE "${PROJECT_SOURCE_DIR}/src")
target_link_libraries(test_finish_f1 PRIVATE imd_constants imd_warnings)
add_test(NAME finish_f1 COMMAND test_finish_f1)

# CUTLASS de-risking for M1: int8 batched TN GEMM, and the mod-p -> u8 EVT epilogue.
add_executable(test_cutlass_int8_gemm test_cutlass_int8_gemm.cu)
target_link_libraries(test_cutlass_int8_gemm PRIVATE imd_constants imd_cutlass)
add_test(NAME cutlass_int8_gemm COMMAND test_cutlass_int8_gemm)

add_executable(test_cutlass_mod_epilogue test_cutlass_mod_epilogue.cu)
target_include_directories(test_cutlass_mod_epilogue PRIVATE "${PROJECT_SOURCE_DIR}/src")
target_link_libraries(test_cutlass_mod_epilogue PRIVATE imd_constants imd_cutlass)
add_test(NAME cutlass_mod_epilogue COMMAND test_cutlass_mod_epilogue)

# Issue 17 (docs/NOTES-17.md): the Hopper K4 epilogue candidate on accumulators placed in wgmma order, against
# K4's own epilogue and a host reference. Runs on any sm_80+ GPU.
add_executable(test_hopper_epilogue test_hopper_epilogue.cu)
target_include_directories(test_hopper_epilogue PRIVATE "${PROJECT_SOURCE_DIR}/src")
target_link_libraries(test_hopper_epilogue PRIVATE imd_constants imd_cutlass)
add_test(NAME hopper_epilogue COMMAND test_hopper_epilogue)
set_tests_properties(hopper_epilogue PROPERTIES RESOURCE_LOCK gpu TIMEOUT 300)

# The register order the Hopper fast epilogue assumes, against CuTe's partition_C of the real TiledMma of every variant. Host code only
# (no GPU), but built for sm_90a because the variants are.
if(CMAKE_CUDA_COMPILER_VERSION VERSION_GREATER_EQUAL 12.0)   # sm_90a needs nvcc 12+; the target adds a heavy CUTLASS compile
  add_executable(test_hopper_partition test_hopper_partition.cu)
  target_include_directories(test_hopper_partition PRIVATE "${PROJECT_SOURCE_DIR}/src")
  target_link_libraries(test_hopper_partition PRIVATE imd_constants imd_cutlass)
  set_target_properties(test_hopper_partition PROPERTIES CUDA_ARCHITECTURES "90a-real")
  target_compile_options(test_hopper_partition PRIVATE --expt-relaxed-constexpr)
  add_test(NAME hopper_partition COMMAND test_hopper_partition)
endif()

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
  foreach(suite api stages e2e tensor bigk replay every_n k_align pad_mutation eps)
    add_test(NAME contract_${suite}
             COMMAND Python3::Interpreter "${CMAKE_CURRENT_SOURCE_DIR}/test_contract.py"
                     --lib "$<TARGET_FILE:immerdink>" --shim "${shim}" --suite ${suite} -v)
    set_tests_properties(contract_${suite} PROPERTIES RESOURCE_LOCK gpu TIMEOUT 900
                         ENVIRONMENT "LD_PRELOAD=${shim}")
  endforeach()
  # Issue 21 (docs/NOTES-21.md "Pack emitter and finish"): the same oracle tests with the other instruction sequences forced
  # for every plan (IMD_TEST_SEQ, tests/harness.py: the library must report that set back, or the run fails): 0 = the baseline
  # sequences (the production path on every class without a default set, and the reference now that the sm_89 / sm_90
  # default is E1), 1 = E1 emitter, 7 = E1 + Q1 quantize + F1 finish, 1031 = 7 + O1 (issue 21, task 17: the offset-binary emitter input, every N
  # through every_n: W = 2 takes O1, W = 3 falls back to E1), 3079 = 1031 + fq (task 18: O1's quotient in binary32). The suites without a suffix run the class default.
  foreach(seq 0 1 7 1031 3079)
    if((seq EQUAL 7 OR seq EQUAL 1031 OR seq EQUAL 3079) AND NOT IMMERDINK_SEQ_TWINS)
      continue()
    endif()
    foreach(suite e2e tensor every_n eps bigk)
      if(NOT (seq EQUAL 7 OR seq EQUAL 1031 OR seq EQUAL 3079) AND (suite STREQUAL "tensor" OR suite STREQUAL "every_n" OR suite STREQUAL "bigk"))
        continue()
      endif()
      add_test(NAME contract_${suite}_seq${seq}
               COMMAND Python3::Interpreter "${CMAKE_CURRENT_SOURCE_DIR}/test_contract.py"
                       --lib "$<TARGET_FILE:immerdink>" --shim "${shim}" --suite ${suite} -v)
      set_tests_properties(contract_${suite}_seq${seq} PROPERTIES RESOURCE_LOCK gpu TIMEOUT 900
                           ENVIRONMENT "LD_PRELOAD=${shim};IMD_TEST_SEQ=${seq}")
    endforeach()
    add_test(NAME output_hashes_seq${seq}
             COMMAND Python3::Interpreter "${CMAKE_CURRENT_SOURCE_DIR}/test_output_hashes.py"
                     --lib "$<TARGET_FILE:immerdink>" --tags small mid)
    set_tests_properties(output_hashes_seq${seq} PROPERTIES RESOURCE_LOCK gpu TIMEOUT 900
                         ENVIRONMENT "IMD_TEST_SEQ=${seq}")
  endforeach()
  # Issue 23 (docs/NOTES-23.md): the KL fixed-part cuts (bits 16 klwarp, 32 klload, 64 klshape) forced for every plan on top of E1 (17, 33, 65, 113) and on top of
  # the baseline emitter (112): the same oracle suites and the committed hashes. They are kernel twins like Q1 / F1 (not built with IMMERDINK_SEQ_TWINS=OFF).
  if(IMMERDINK_SEQ_TWINS)
    # Issue 21 (docs/NOTES-21-klf64.md): klf64 (256, needs klwarp) on E1 + klwarp (273), + klload (305) and on the baseline emitter (272).
    # Issue 21 kllong (512: klload's direct loads for rows of nc > 2 chunks): alone on E1 (513), with klwarp (529), with klwarp + klf64 (785), and the sm_90 twin 439 + 512 (951).
    # Issue 21 O1 (1024, task 17): on E1 alone (1025, the sm_89 shape), on the sm_90 twin (1463 = 439 + 1024) and with kllong too (1975 = 951 + 1024).
    # Issue 21 fq (2048, task 18, needs O1): the same three shapes, 3073, 3511, 4023.
    # Issue 21 task 20 klcs (4096) on the sm_90 long-row set 4023: 8119; klph (8192) on top: 16311.
    foreach(seq 17 33 65 113 112 273 305 272 513 529 785 951 1025 1463 1975 3073 3511 4023 8119 16311)
      foreach(suite e2e tensor eps)
        add_test(NAME contract_${suite}_kl${seq}
                 COMMAND Python3::Interpreter "${CMAKE_CURRENT_SOURCE_DIR}/test_contract.py"
                         --lib "$<TARGET_FILE:immerdink>" --shim "${shim}" --suite ${suite} -v)
        set_tests_properties(contract_${suite}_kl${seq} PROPERTIES RESOURCE_LOCK gpu TIMEOUT 900
                             ENVIRONMENT "LD_PRELOAD=${shim};IMD_TEST_SEQ=${seq}")
      endforeach()
      add_test(NAME output_hashes_kl${seq}
               COMMAND Python3::Interpreter "${CMAKE_CURRENT_SOURCE_DIR}/test_output_hashes.py"
                       --lib "$<TARGET_FILE:immerdink>" --tags small mid)
      set_tests_properties(output_hashes_kl${seq} PROPERTIES RESOURCE_LOCK gpu TIMEOUT 900
                           ENVIRONMENT "IMD_TEST_SEQ=${seq}")
    endforeach()
    # Issue 21: the threshold rows (suite klf64: square sums on and next to the sigma thresholds, the exact fallback) under the klf64 sets and the default set.
    foreach(seq -1 273 305 272 437 439 785 951 1463 3511)
      add_test(NAME contract_klf64_seq${seq}
               COMMAND Python3::Interpreter "${CMAKE_CURRENT_SOURCE_DIR}/test_contract.py"
                       --lib "$<TARGET_FILE:immerdink>" --shim "${shim}" --suite klf64 -v)
      set_tests_properties(contract_klf64_seq${seq} PROPERTIES RESOURCE_LOCK gpu TIMEOUT 1800
                           ENVIRONMENT "LD_PRELOAD=${shim};IMD_TEST_SEQ=${seq}")
    endforeach()
    foreach(seq 17 113 273 529 785 1025 1975 3073 4023 8119 16311)
      add_test(NAME contract_bigk_kl${seq}
               COMMAND Python3::Interpreter "${CMAKE_CURRENT_SOURCE_DIR}/test_contract.py"
                       --lib "$<TARGET_FILE:immerdink>" --shim "${shim}" --suite bigk -v)
      set_tests_properties(contract_bigk_kl${seq} PROPERTIES RESOURCE_LOCK gpu TIMEOUT 900
                           ENVIRONMENT "LD_PRELOAD=${shim};IMD_TEST_SEQ=${seq}")
      add_test(NAME output_hashes_large_kl${seq}
               COMMAND Python3::Interpreter "${CMAKE_CURRENT_SOURCE_DIR}/test_output_hashes.py"
                       --lib "$<TARGET_FILE:immerdink>" --tags large)
      set_tests_properties(output_hashes_large_kl${seq} PROPERTIES RESOURCE_LOCK gpu TIMEOUT 900
                           ENVIRONMENT "IMD_TEST_SEQ=${seq}")
    endforeach()
  endif()
  set(_seq_twin_arg "")
  if(NOT IMMERDINK_SEQ_TWINS)
    set(_seq_twin_arg --no-seq-twins)
  endif()
  # Issue 23: every combination of the KL bits over four emitter sets, bit-identical, with the KL launch counter showing which kernel each operand ran.
  add_test(NAME kl_bits
           COMMAND Python3::Interpreter "${CMAKE_CURRENT_SOURCE_DIR}/test_kl_bits.py" --lib "$<TARGET_FILE:immerdink>"
                   ${_seq_twin_arg})
  set_tests_properties(kl_bits PROPERTIES RESOURCE_LOCK gpu TIMEOUT 1800)
  # Issue 21 klstrided: the sm_90 class's KL rule (forced with imd_test_set_fused(6) on any device) is bit-identical to KL for both operands.
  add_test(NAME kl_strided
           COMMAND Python3::Interpreter "${CMAKE_CURRENT_SOURCE_DIR}/test_kl_strided.py" --lib "$<TARGET_FILE:immerdink>")
  set_tests_properties(kl_strided PROPERTIES RESOURCE_LOCK gpu TIMEOUT 600)
  add_test(NAME seq_select
           COMMAND Python3::Interpreter "${CMAKE_CURRENT_SOURCE_DIR}/test_seq_select.py" --lib "$<TARGET_FILE:immerdink>"
                   ${_seq_twin_arg})
  set_tests_properties(seq_select PROPERTIES RESOURCE_LOCK gpu TIMEOUT 300)
  # Issue 22 (docs/NOTES-22.md "Library integration (design)"): the Hopper wgmma K4 backend. Registered on every build and device and
  # passing on all of them (the campaign counts a skipped test as not passed): off sm_90 (or in a build without IMMERDINK_K4_WGMMA) the
  # backend is unavailable, a forced wgmma plan reports and runs mma.sync, and the launch counters must stay 0; on a Hopper the forced
  # runs must be served by the wgmma kernel and equal the mma.sync K4 bit for bit. IMD_TEST_K4 (tests/harness.py) forces the backend of
  # every plan of the process and the library must report it back: wgmma = every legal launch, the profitability floor lifted.
  add_test(NAME k4_wgmma_exact
           COMMAND Python3::Interpreter "${CMAKE_CURRENT_SOURCE_DIR}/test_k4_wgmma.py" --lib "$<TARGET_FILE:immerdink>")
  set_tests_properties(k4_wgmma_exact PROPERTIES RESOURCE_LOCK gpu TIMEOUT 900)
  foreach(suite e2e every_n bigk)
    add_test(NAME contract_${suite}_k4wgmma
             COMMAND Python3::Interpreter "${CMAKE_CURRENT_SOURCE_DIR}/test_contract.py"
                     --lib "$<TARGET_FILE:immerdink>" --shim "${shim}" --suite ${suite} -v)
    set_tests_properties(contract_${suite}_k4wgmma PROPERTIES RESOURCE_LOCK gpu TIMEOUT 900
                         ENVIRONMENT "LD_PRELOAD=${shim};IMD_TEST_K4=wgmma")
  endforeach()
  add_test(NAME output_hashes_k4wgmma
           COMMAND Python3::Interpreter "${CMAKE_CURRENT_SOURCE_DIR}/test_output_hashes.py"
                   --lib "$<TARGET_FILE:immerdink>" --tags small mid)
  set_tests_properties(output_hashes_k4wgmma PROPERTIES RESOURCE_LOCK gpu TIMEOUT 900 ENVIRONMENT "IMD_TEST_K4=wgmma")
  add_test(NAME output_hashes_large_k4wgmma
           COMMAND Python3::Interpreter "${CMAKE_CURRENT_SOURCE_DIR}/test_output_hashes.py"
                   --lib "$<TARGET_FILE:immerdink>" --tags large)
  set_tests_properties(output_hashes_large_k4wgmma PROPERTIES RESOURCE_LOCK gpu TIMEOUT 900 ENVIRONMENT "IMD_TEST_K4=wgmma")
  # the fallback on a Hopper: every plan forced to the mma.sync K4 reproduces the committed hashes
  add_test(NAME output_hashes_k4mma
           COMMAND Python3::Interpreter "${CMAKE_CURRENT_SOURCE_DIR}/test_output_hashes.py"
                   --lib "$<TARGET_FILE:immerdink>" --tags small mid)
  set_tests_properties(output_hashes_k4mma PROPERTIES RESOURCE_LOCK gpu TIMEOUT 900 ENVIRONMENT "IMD_TEST_K4=mma")
  # imd_bench --seq / --crt (method imd): needs the profiling build of the bench (bench/build.sh), so only when
  # IMMERDINK_BENCH_BIN names it (environment or cache variable); bench/test_seq_crt.py documents the checks.
  if(NOT IMMERDINK_BENCH_BIN AND DEFINED ENV{IMMERDINK_BENCH_BIN})
    set(IMMERDINK_BENCH_BIN "$ENV{IMMERDINK_BENCH_BIN}")
  endif()
  if(IMMERDINK_BENCH_BIN)
    add_test(NAME bench_seq_crt
             COMMAND Python3::Interpreter -I "${PROJECT_SOURCE_DIR}/bench/test_seq_crt.py" --bench "${IMMERDINK_BENCH_BIN}"
                     ${_seq_twin_arg})
    set_tests_properties(bench_seq_crt PROPERTIES RESOURCE_LOCK gpu TIMEOUT 600)
    # imd_bench accuracy fields (acc_*, --accuracy; docs/NOTES-BENCH-ACCURACY.md).
    add_test(NAME bench_accuracy_fields
             COMMAND Python3::Interpreter -I "${PROJECT_SOURCE_DIR}/bench/test_accuracy_fields.py" --bench "${IMMERDINK_BENCH_BIN}")
    set_tests_properties(bench_accuracy_fields PROPERTIES RESOURCE_LOCK gpu TIMEOUT 600)
  endif()
  if(IMMERDINK_SEQ_TWINS)
  # Issue 23 (docs/NOTES-23-k5.md): the k5fin kernels (bit 128: set 133 = E1 + F1 + 128 with the limb CRT forced, IMD_TEST_K5_CRT=1) against the oracle and the committed hashes.
  # (On an RTX 4090 the limb CRT is slow but exact; tests/test_crt_f64_gpu.py compares them with the shipped kernels and checks that they ran.)
  foreach(suite e2e eps tensor every_n bigk)
    add_test(NAME contract_${suite}_k5fin
             COMMAND Python3::Interpreter "${CMAKE_CURRENT_SOURCE_DIR}/test_contract.py"
                     --lib "$<TARGET_FILE:immerdink>" --shim "${shim}" --suite ${suite} -v)
    set_tests_properties(contract_${suite}_k5fin PROPERTIES RESOURCE_LOCK gpu TIMEOUT 1800
                         ENVIRONMENT "LD_PRELOAD=${shim};IMD_TEST_SEQ=133;IMD_TEST_K5_CRT=1")
  endforeach()
  # graph capture and replay of the k5fin kernels (test_graph reads the same two variables and requires at least one k5fin launch)
  add_test(NAME graph_capture_k5fin COMMAND test_graph)
  set_tests_properties(graph_capture_k5fin PROPERTIES RESOURCE_LOCK gpu TIMEOUT 600
                       ENVIRONMENT "IMD_TEST_SEQ=133;IMD_TEST_K5_CRT=1;IMD_TEST_REQUIRE_K5FIN=1")
  add_test(NAME output_hashes_k5fin
           COMMAND Python3::Interpreter "${CMAKE_CURRENT_SOURCE_DIR}/test_output_hashes.py"
                   --lib "$<TARGET_FILE:immerdink>" --tags small mid)
  set_tests_properties(output_hashes_k5fin PROPERTIES RESOURCE_LOCK gpu TIMEOUT 1800 ENVIRONMENT "IMD_TEST_SEQ=133;IMD_TEST_K5_CRT=1")
  add_test(NAME output_hashes_large_k5fin
           COMMAND Python3::Interpreter "${CMAKE_CURRENT_SOURCE_DIR}/test_output_hashes.py"
                   --lib "$<TARGET_FILE:immerdink>" --tags large)
  set_tests_properties(output_hashes_large_k5fin PROPERTIES RESOURCE_LOCK gpu TIMEOUT 1800 ENVIRONMENT "IMD_TEST_SEQ=133;IMD_TEST_K5_CRT=1")
  # R4 integration (docs/NOTES-R4-INTEGRATION.md): the duel sides' sets forced TOGETHER (E1 1, F1 4, klwarp 16, klload 32, klshape 64, k5fin 128), each with the K5 CRT of the side
  # (crt=f64; the sets without 128 also crt=int): the oracle suites e2e / eps and the 19 committed hashes, one test per (set, crt); and the in-process check of all of them
  # at once with the KL and k5fin launch counters (test_forced_sets.py).
  foreach(seq 5 21 37 69 53 101 85 117 133 149 165 197 245 437 439)
    foreach(crt f64 int)
      if(crt STREQUAL "int" AND seq GREATER 127)
        continue()
      endif()
      add_test(NAME r4_set_${seq}_${crt}
               COMMAND Python3::Interpreter "${CMAKE_CURRENT_SOURCE_DIR}/run_forced_set.py"
                       --lib "$<TARGET_FILE:immerdink>" --shim "${shim}" --seq ${seq} --crt ${crt})
      set_tests_properties(r4_set_${seq}_${crt} PROPERTIES RESOURCE_LOCK gpu TIMEOUT 3600)
    endforeach()
  endforeach()
  add_test(NAME forced_sets
           COMMAND Python3::Interpreter "${CMAKE_CURRENT_SOURCE_DIR}/test_forced_sets.py" --lib "$<TARGET_FILE:immerdink>")
  set_tests_properties(forced_sets PROPERTIES RESOURCE_LOCK gpu TIMEOUT 3600)
  add_test(NAME output_hashes_large_seq7
           COMMAND Python3::Interpreter "${CMAKE_CURRENT_SOURCE_DIR}/test_output_hashes.py"
                   --lib "$<TARGET_FILE:immerdink>" --tags large)
  set_tests_properties(output_hashes_large_seq7 PROPERTIES RESOURCE_LOCK gpu TIMEOUT 900
                       ENVIRONMENT "IMD_TEST_SEQ=7")
  endif()
  # Committed output hashes (tests/data/output_hashes.json): the library's C on this GPU against
  # hashes from the CPU oracle (small shapes) and from the RTX 4090 (large ones). The same file
  # is checked on every shipped GPU. output_hashes_oracle recomputes the CPU hashes.
  add_test(NAME output_hashes
           COMMAND Python3::Interpreter "${CMAKE_CURRENT_SOURCE_DIR}/test_output_hashes.py"
                   --lib "$<TARGET_FILE:immerdink>" --tags small mid)
  add_test(NAME output_hashes_oracle
           COMMAND Python3::Interpreter "${CMAKE_CURRENT_SOURCE_DIR}/test_output_hashes.py"
                   --lib "$<TARGET_FILE:immerdink>" --tags small mid --oracle)
  add_test(NAME output_hashes_large
           COMMAND Python3::Interpreter "${CMAKE_CURRENT_SOURCE_DIR}/test_output_hashes.py"
                   --lib "$<TARGET_FILE:immerdink>" --tags large)
  # Issue 21: K5's CRT with the sums in binary64 limbs equals the integer CRT bit for bit (and reproduces the committed hashes).
  add_test(NAME crt_f64_gpu
           COMMAND Python3::Interpreter "${CMAKE_CURRENT_SOURCE_DIR}/test_crt_f64_gpu.py"
                   --lib "$<TARGET_FILE:immerdink>" ${_seq_twin_arg})
  set_tests_properties(crt_f64_gpu PROPERTIES RESOURCE_LOCK gpu TIMEOUT 900)
  # Issue 21: the real device functions on the edge vectors (before round_scaled), with the mutation check.
  add_test(NAME crt_f64_unit
           COMMAND Python3::Interpreter "${CMAKE_CURRENT_SOURCE_DIR}/test_crt_f64_unit.py"
                   --lib "$<TARGET_FILE:immerdink>")
  set_tests_properties(crt_f64_unit PROPERTIES RESOURCE_LOCK gpu TIMEOUT 900)
  set_tests_properties(output_hashes output_hashes_oracle output_hashes_large PROPERTIES
                       RESOURCE_LOCK gpu TIMEOUT 900)
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

  # DESIGN v1.19 §6g (NOTES-C1 §4.1, stage 0): precision phases on the GPU vs spec/oracle.py,
  # one fresh process per suite (graph_phases captures phases whose N never ran in its
  # process): a buffer re-packed in place at another N (PackedBuffer refuses the stale plan),
  # one workspace for plans at every N, CUDA graphs per phase with cold captures, and F3's
  # write extent of imd_pack (sentinels, every pack launch shape) with packed contracts on
  # exactly sized packs on every producer path.
  foreach(pair packed_repack_n:repack_n workspace_shared_n:workspace_shared_n
               graph_phases:graph_phases pack_write_extent:write_extent)
    string(REPLACE ":" ";" pair "${pair}")
    list(GET pair 0 name)
    list(GET pair 1 suite)
    add_test(NAME ${name}
             COMMAND Python3::Interpreter "${CMAKE_CURRENT_SOURCE_DIR}/test_phases.py"
                     --lib "$<TARGET_FILE:immerdink>" --shim "${shim}" --suite ${suite} -v)
    set_tests_properties(${name} PROPERTIES RESOURCE_LOCK gpu TIMEOUT 900
                         ENVIRONMENT "LD_PRELOAD=${shim}")
  endforeach()
  # F3 under compute-sanitizer memcheck: every allocation exactly sized, clean; a pack
  # allocation 16 bytes short must be reported (tests/memcheck_pack_extent.sh).
  if(_imd_san)
    add_test(NAME pack_write_extent_memcheck
             COMMAND bash "${CMAKE_CURRENT_SOURCE_DIR}/memcheck_pack_extent.sh" "${_imd_san}"
                     "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/test_phases.py"
                     "$<TARGET_FILE:immerdink>")
    set_tests_properties(pack_write_extent_memcheck PROPERTIES RESOURCE_LOCK gpu TIMEOUT 1800)
  endif()

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

  # Issue 17 (docs/NOTES-17.md): the layouts the Hopper K4 design depends on, read from CUTLASS's CuTe tables
  # (compiles tests/wgmma_layout_dump.cpp with the host compiler; no GPU).
  add_test(NAME wgmma_layout
           COMMAND Python3::Interpreter "${CMAKE_CURRENT_SOURCE_DIR}/test_wgmma_layout.py"
                   --cutlass "${IMMERDINK_CUTLASS_DIR}" --cxx "${CMAKE_CXX_COMPILER}"
                   --include "${CUDAToolkit_INCLUDE_DIRS}")
  # The status of this GPU (imd_device_info, imd_plan_info), with lazy and with eager module loading.
  add_test(NAME device_status
           COMMAND Python3::Interpreter "${CMAKE_CURRENT_SOURCE_DIR}/test_device_status.py"
                   --lib "$<TARGET_FILE:immerdink>")
  add_test(NAME device_status_eager
           COMMAND Python3::Interpreter "${CMAKE_CURRENT_SOURCE_DIR}/test_device_status.py"
                   --lib "$<TARGET_FILE:immerdink>")
  set_tests_properties(device_status device_status_eager PROPERTIES RESOURCE_LOCK gpu TIMEOUT 300)
  set_tests_properties(device_status_eager PROPERTIES ENVIRONMENT "CUDA_MODULE_LOADING=EAGER")
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
set_tests_properties(device_constants pack_emitter finish_f1 cutlass_int8_gemm cutlass_mod_epilogue
                     PROPERTIES RESOURCE_LOCK gpu)
