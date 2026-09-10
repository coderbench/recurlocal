cmake_minimum_required(VERSION 3.24)

# The public header is the single source of truth for the version, so a consumer's
# compile-time TENSORTRANSIT_VERSION_* and the installed package version cannot drift.
file(READ "${CMAKE_CURRENT_SOURCE_DIR}/include/tensortransit/version.h" _tt_version_header)
foreach(_part MAJOR MINOR PATCH)
    string(REGEX MATCH "TENSORTRANSIT_VERSION_${_part}[ \t]+([0-9]+)" _ "${_tt_version_header}")
    set(_tt_${_part} "${CMAKE_MATCH_1}")
endforeach()

project(TensorTransit
        VERSION ${_tt_MAJOR}.${_tt_MINOR}.${_tt_PATCH}
        DESCRIPTION "Engine-independent cross-kernel tensor-locality planner for AI inference"
        LANGUAGES CXX)

option(TENSORTRANSIT_BUILD_CUDA "Build the CUDA executor, benchmark and adapter" OFF)
option(TENSORTRANSIT_BUILD_TESTS "Build tests" ON)
option(TENSORTRANSIT_BUILD_CLI "Build the tensortransit command-line tool" ON)
option(TENSORTRANSIT_INSTALL "Generate install and package rules" ${PROJECT_IS_TOP_LEVEL})

# Accept every earlier spelling so an existing script keeps working. RECURLLOCAL_* was a
# pre-0.2 typo; RECURLOCAL_* is the 0.1 name. Both are deprecated, neither is removed.
foreach(_opt BUILD_CUDA BUILD_TESTS)
    foreach(_old RECURLLOCAL RECURLOCAL)
        if(DEFINED ${_old}_${_opt})
            message(DEPRECATION "${_old}_${_opt} is deprecated; use TENSORTRANSIT_${_opt}")
            set(TENSORTRANSIT_${_opt} "${${_old}_${_opt}}" CACHE BOOL "" FORCE)
        endif()
    endforeach()
endforeach()
if(DEFINED RECURLOCAL_INSTALL)
    set(TENSORTRANSIT_INSTALL "${RECURLOCAL_INSTALL}" CACHE BOOL "" FORCE)
endif()

# This is a performance project; an unspecified build type would otherwise give
# every documented build command an unoptimised -O0 host build.
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
    set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS Debug Release RelWithDebInfo MinSizeRel)
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

include(GNUInstallDirs)

# The engine-independent core plus the planner family. No CUDA, no model, no runtime: every
# policy decision in this library is host arithmetic and is tested without a GPU.
add_library(tensortransit STATIC
    src/tensor_registry.cpp
    src/transit_graph.cpp
    src/plan.cpp
    src/device.cpp
    src/ceiling.cpp
    src/planner_registry.cpp
    src/executor.cpp
    src/runtime.cpp
    src/trace.cpp
    src/recurrent_planner.cpp
    planners/common.cpp
    planners/baseline/baseline_planner.cpp
    planners/recurrent_v0/recurrent_v0_planner.cpp
    planners/greedy/greedy_planner.cpp
    planners/budgeted/budgeted_planner.cpp
    planners/concurrency/concurrency_planner.cpp)
add_library(TensorTransit::tensortransit ALIAS tensortransit)
# The 0.1 target name, so a consumer's target_link_libraries keeps resolving.
add_library(RecurLocal::recurlocal ALIAS tensortransit)
target_include_directories(tensortransit PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>)
# planners/common.h is implementation-private and is not installed; only the build tree
# needs to reach it.
target_include_directories(tensortransit PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_compile_features(tensortransit PUBLIC cxx_std_17)
set_target_properties(tensortransit PROPERTIES VERSION ${PROJECT_VERSION} EXPORT_NAME tensortransit)
if(NOT MSVC)
    target_compile_options(tensortransit PRIVATE -Wall -Wextra)
endif()

add_executable(tensortransit_info tools/info.cpp)
target_link_libraries(tensortransit_info PRIVATE tensortransit)
# The 0.1 tool name. Same binary, so a script that calls either gets the same output.
set_target_properties(tensortransit_info PROPERTIES OUTPUT_NAME tensortransit_info)
add_custom_command(TARGET tensortransit_info POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy $<TARGET_FILE:tensortransit_info>
            $<TARGET_FILE_DIR:tensortransit_info>/recur_local_info)

if(TENSORTRANSIT_BUILD_CLI)
    # inspect / plan / compare / report. Deliberately NOT required by the runtime library:
    # a runtime embeds the core and never needs the tool (spec section 47).
    add_executable(tensortransit_cli tools/tensortransit_cli.cpp)
    target_link_libraries(tensortransit_cli PRIVATE tensortransit)
    set_target_properties(tensortransit_cli PROPERTIES OUTPUT_NAME tensortransit)
    if(NOT MSVC)
        target_compile_options(tensortransit_cli PRIVATE -Wall -Wextra)
    endif()
endif()

if(TENSORTRANSIT_BUILD_CUDA)
    enable_language(CUDA)
    find_package(CUDAToolkit REQUIRED)
    add_library(tensortransit_cuda STATIC
        executors/cuda/cache_control.cu
        executors/cuda/prefetch.cu
        executors/cuda/transit_executor.cu)
    add_library(TensorTransit::tensortransit_cuda ALIAS tensortransit_cuda)
    add_library(RecurLocal::recurlocal_cuda ALIAS tensortransit_cuda)
    target_include_directories(tensortransit_cuda PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>)
    target_link_libraries(tensortransit_cuda PUBLIC tensortransit CUDA::cudart)
    # RECURLOCAL_WITH_CUDA and the RECURLLOCAL_ typo are the deprecated spellings, kept so a
    # vendored copy of an old header guard still compiles.
    target_compile_definitions(tensortransit_cuda PUBLIC
        TENSORTRANSIT_WITH_CUDA=1 RECURLOCAL_WITH_CUDA=1 RECURLLOCAL_WITH_CUDA=1)
    # Whole-program compilation, deliberately. Relocatable device code would oblige every
    # consumer to run a CUDA device-link step over this library, and a runtime that links it
    # as one more static archive gets "undefined reference to __cudaRegisterLinkedBinary_*"
    # at link time instead. TensorTransit is meant to be embedded, so its kernels resolve at
    # compile time and it links like any other archive.
    set_target_properties(tensortransit_cuda PROPERTIES CUDA_STANDARD 17 CUDA_STANDARD_REQUIRED ON
                          CUDA_SEPARABLE_COMPILATION OFF EXPORT_NAME tensortransit_cuda)

    target_link_libraries(tensortransit_info PRIVATE tensortransit_cuda)

    # The SparkInfer adapter ships as its own target, so a runtime links the integration it
    # wants and nothing else. It compiles as C++ (no kernels of its own).
    add_library(tensortransit_sparkinfer STATIC
        adapters/sparkinfer/tensortransit_sparkinfer.cpp
        adapters/sparkinfer/transit_backend.cpp)
    add_library(TensorTransit::tensortransit_sparkinfer ALIAS tensortransit_sparkinfer)
    add_library(RecurLocal::recurlocal_sparkinfer ALIAS tensortransit_sparkinfer)
    target_link_libraries(tensortransit_sparkinfer PUBLIC tensortransit_cuda)
    set_target_properties(tensortransit_sparkinfer PROPERTIES EXPORT_NAME tensortransit_sparkinfer)
    if(NOT MSVC)
        target_compile_options(tensortransit_sparkinfer PRIVATE -Wall -Wextra)
    endif()

    # Device-side contract tests. They skip cleanly with exit 0 when no GPU is present, so
    # this stays registered even on a device-less CUDA build box - the alternative is a
    # suite that only exists on the one machine that can run it.
    if(TENSORTRANSIT_BUILD_TESTS)
        add_executable(test_cuda_controller tests/test_cuda_controller.cu)
        target_link_libraries(test_cuda_controller PRIVATE tensortransit_cuda CUDA::cudart)
        set_target_properties(test_cuda_controller PROPERTIES CUDA_STANDARD 17 CUDA_STANDARD_REQUIRED ON)
        add_test(NAME cuda_controller COMMAND test_cuda_controller)

        add_executable(test_cuda_executor tests/test_cuda_executor.cu)
        target_link_libraries(test_cuda_executor PRIVATE tensortransit_cuda CUDA::cudart)
        set_target_properties(test_cuda_executor PROPERTIES CUDA_STANDARD 17 CUDA_STANDARD_REQUIRED ON)
        add_test(NAME cuda_executor COMMAND test_cuda_executor)
    endif()

    add_executable(tensortransit_bench workloads/recurrent/synthetic/cuda_bench.cu)
    target_link_libraries(tensortransit_bench PRIVATE tensortransit_cuda CUDA::cudart)
    set_target_properties(tensortransit_bench PROPERTIES CUDA_STANDARD 17 CUDA_STANDARD_REQUIRED ON)
    # The 0.1 binary name, so eval/run_eval.py and every recorded command still work.
    add_custom_command(TARGET tensortransit_bench POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy $<TARGET_FILE:tensortransit_bench>
                $<TARGET_FILE_DIR:tensortransit_bench>/recur_local_cuda_bench)

    # Standalone: does an access-policy window set on a node of a graph that is STILL BEING
    # CAPTURED survive into the instantiated graph, and does it take effect on replay? It
    # links nothing of TensorTransit's - it reproduces the pattern in isolation, which is the
    # point - but it is built here so it cannot rot while the mechanism it justifies ships.
    # Not registered as a test: it needs a real device and it times replays.
    add_executable(capture_attr_probe profiling/capture_attr_probe.cu)
    target_link_libraries(capture_attr_probe PRIVATE CUDA::cudart)
    set_target_properties(capture_attr_probe PROPERTIES CUDA_STANDARD 17 CUDA_STANDARD_REQUIRED ON)
endif()

if(TENSORTRANSIT_BUILD_TESTS)
    enable_testing()

    # The recurrent locality planner: the 0.1 suite, unchanged.
    add_executable(test_planner tests/test_planner.cpp)
    target_link_libraries(test_planner PRIVATE tensortransit)
    add_test(NAME planner COMMAND test_planner)

    # The core: registry lifetime, graph reuse analysis, plan structure and serialization.
    add_executable(test_core tests/test_core.cpp)
    target_link_libraries(test_core PRIVATE tensortransit)
    add_test(NAME core COMMAND test_core)

    # The planner family, including the five comparison arms and the equivalence between
    # RecurrentV0Planner and the 0.1 policy it is supposed to reproduce.
    add_executable(test_transit_planner tests/test_transit_planner.cpp)
    target_link_libraries(test_transit_planner PRIVATE tensortransit)
    add_test(NAME transit_planner COMMAND test_transit_planner)

    # Golden traces: CPU CI validates planner behaviour against committed plan digests, so a
    # policy change that nobody intended fails a build rather than a benchmark.
    # Spec section 80's planner+executor CPU budget, gated rather than described.
    add_executable(test_overhead_budget tests/test_overhead_budget.cpp)
    target_link_libraries(test_overhead_budget PRIVATE tensortransit)
    add_test(NAME overhead_budget COMMAND test_overhead_budget)

    add_executable(test_golden tests/test_golden.cpp)
    target_link_libraries(test_golden PRIVATE tensortransit)
    target_compile_definitions(test_golden PRIVATE
        TENSORTRANSIT_GOLDEN_DIR="${CMAKE_CURRENT_SOURCE_DIR}/tests/golden")
    add_test(NAME golden COMMAND test_golden)

    # The deprecated recurlocal/* headers and namespace alias still compile and still name
    # the same entities. A compatibility promise nothing exercises is a compatibility hope.
    add_executable(test_compat tests/test_compat.cpp)
    target_link_libraries(test_compat PRIVATE tensortransit)
    add_test(NAME compat COMMAND test_compat)

    # The decision bands are policy: drift there silently redefines every past verdict,
    # so they are covered by the same `ctest` run as the planner.
    find_package(Python3 COMPONENTS Interpreter)
    if(Python3_Interpreter_FOUND)
        add_test(NAME decision COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/eval/test_decide.py)
        add_test(NAME schemas COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/eval/test_schemas.py)
        # The Transit Frontier Ledger's golden tests. Registered here rather than run by hand
        # because the scoring math is the ruler: a change to normalization, dominance,
        # hypervolume, aggregation or the receipt's status derivation silently redefines every
        # past receipt, and this suite is what makes that a build failure.
        add_test(NAME frontier COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/eval/test_frontier.py)
        # The COMMAND LINE, end to end, on a synthetic raw file built from the frozen
        # generation. The golden tests above cover the scoring library; this covers the wiring
        # a trusted evaluator and a contributor actually run -- argument parsing, the
        # generation lookup, where the ledger is written, whether a non-zero exit means what it
        # should. A break there would stay invisible until somebody spent GPU time to find it.
        add_test(NAME frontier_cli COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/eval/test_cli_pipeline.py)
    else()
        message(STATUS "Python3 not found - skipping the eval decision and schema tests")
    endif()
endif()

if(TENSORTRANSIT_INSTALL)
    include(CMakePackageConfigHelpers)
    set(_tt_targets tensortransit)
    if(TENSORTRANSIT_BUILD_CUDA)
        list(APPEND _tt_targets tensortransit_cuda tensortransit_sparkinfer)
    endif()
    install(TARGETS ${_tt_targets} EXPORT TensorTransitTargets
            ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
            LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
            RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})
    install(DIRECTORY include/ DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
    install(EXPORT TensorTransitTargets NAMESPACE TensorTransit::
            DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/TensorTransit)
    configure_package_config_file(cmake/TensorTransitConfig.cmake.in
        ${CMAKE_CURRENT_BINARY_DIR}/TensorTransitConfig.cmake
        INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/TensorTransit)
    write_basic_package_version_file(
        ${CMAKE_CURRENT_BINARY_DIR}/TensorTransitConfigVersion.cmake
        COMPATIBILITY SameMinorVersion)
    install(FILES ${CMAKE_CURRENT_BINARY_DIR}/TensorTransitConfig.cmake
                  ${CMAKE_CURRENT_BINARY_DIR}/TensorTransitConfigVersion.cmake
            DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/TensorTransit)

    # The 0.1 package, so `find_package(RecurLocal)` and the RecurLocal:: targets keep
    # working through the deprecation window. SameMajorVersion here and SameMinorVersion
    # above is deliberate and is the only place the two differ: a consumer pinned to
    # `RecurLocal 0.1` must keep resolving until the shim is removed, while a consumer
    # asking for TensorTransit by version gets the honest unstable-ABI answer.
    # docs/STABILITY.md section 8.
    configure_package_config_file(cmake/RecurLocalConfig.cmake.in
        ${CMAKE_CURRENT_BINARY_DIR}/RecurLocalConfig.cmake
        INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/RecurLocal)
    write_basic_package_version_file(
        ${CMAKE_CURRENT_BINARY_DIR}/RecurLocalConfigVersion.cmake
        COMPATIBILITY SameMajorVersion)
    install(FILES ${CMAKE_CURRENT_BINARY_DIR}/RecurLocalConfig.cmake
                  ${CMAKE_CURRENT_BINARY_DIR}/RecurLocalConfigVersion.cmake
            DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/RecurLocal)
endif()
