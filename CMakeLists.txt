cmake_minimum_required(VERSION 3.24)

# The public header is the single source of truth for the version, so a consumer's
# compile-time RECURLOCAL_VERSION_* and the installed package version cannot drift.
file(READ "${CMAKE_CURRENT_SOURCE_DIR}/include/recurlocal/version.h" _recurlocal_version_header)
foreach(_part MAJOR MINOR PATCH)
    string(REGEX MATCH "RECURLOCAL_VERSION_${_part}[ \t]+([0-9]+)" _ "${_recurlocal_version_header}")
    set(_recurlocal_${_part} "${CMAKE_MATCH_1}")
endforeach()

project(RecurLocal
        VERSION ${_recurlocal_MAJOR}.${_recurlocal_MINOR}.${_recurlocal_PATCH}
        DESCRIPTION "Software-directed CUDA locality for mutable recurrent LLM state"
        LANGUAGES CXX)

option(RECURLOCAL_BUILD_CUDA "Build CUDA locality controller and benchmark" OFF)
option(RECURLOCAL_BUILD_TESTS "Build tests" ON)
option(RECURLOCAL_INSTALL "Generate install and package rules" ${PROJECT_IS_TOP_LEVEL})

# Accept the pre-0.2 misspelling so existing scripts keep working.
foreach(_opt BUILD_CUDA BUILD_TESTS)
    if(DEFINED RECURLLOCAL_${_opt})
        message(DEPRECATION "RECURLLOCAL_${_opt} is deprecated; use RECURLOCAL_${_opt}")
        set(RECURLOCAL_${_opt} "${RECURLLOCAL_${_opt}}" CACHE BOOL "" FORCE)
    endif()
endforeach()

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

add_library(recurlocal STATIC src/planner.cpp)
add_library(RecurLocal::recurlocal ALIAS recurlocal)
target_include_directories(recurlocal PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>)
target_compile_features(recurlocal PUBLIC cxx_std_17)
set_target_properties(recurlocal PROPERTIES VERSION ${PROJECT_VERSION} EXPORT_NAME recurlocal)
if(NOT MSVC)
    target_compile_options(recurlocal PRIVATE -Wall -Wextra)
endif()

add_executable(recur_local_info tools/info.cpp)
target_link_libraries(recur_local_info PRIVATE recurlocal)

if(RECURLOCAL_BUILD_CUDA)
    enable_language(CUDA)
    find_package(CUDAToolkit REQUIRED)
    add_library(recurlocal_cuda STATIC src/cuda/cache_control.cu src/cuda/prefetch.cu)
    add_library(RecurLocal::recurlocal_cuda ALIAS recurlocal_cuda)
    target_include_directories(recurlocal_cuda PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>)
    target_link_libraries(recurlocal_cuda PUBLIC recurlocal CUDA::cudart)
    # RECURLLOCAL_WITH_CUDA is the deprecated spelling, kept so vendored copies of the
    # old header guard still compile.
    target_compile_definitions(recurlocal_cuda PUBLIC RECURLOCAL_WITH_CUDA=1 RECURLLOCAL_WITH_CUDA=1)
    # Whole-program compilation, deliberately. Relocatable device code would oblige every
    # consumer to run a CUDA device-link step over this library, and a runtime that links it
    # as one more static archive gets "undefined reference to __cudaRegisterLinkedBinary_*"
    # at link time instead. RecurLocal is meant to be embedded, so its kernels resolve at
    # compile time and it links like any other archive. Nothing here calls across
    # translation units on the device, so nothing is lost.
    set_target_properties(recurlocal_cuda PROPERTIES CUDA_STANDARD 17 CUDA_STANDARD_REQUIRED ON
                          CUDA_SEPARABLE_COMPILATION OFF EXPORT_NAME recurlocal_cuda)

    # The info tool reports the real device when a CUDA build is available.
    target_link_libraries(recur_local_info PRIVATE recurlocal_cuda)

    # The SparkInfer adapter ships as its own target, so a runtime links the integration
    # it wants and nothing else, and so the hook can be built and installed without the
    # benchmark. It compiles as C++ (no kernels of its own) against recurlocal_cuda.
    add_library(recurlocal_sparkinfer STATIC integrations/sparkinfer/recurlocal_sparkinfer.cpp)
    add_library(RecurLocal::recurlocal_sparkinfer ALIAS recurlocal_sparkinfer)
    target_link_libraries(recurlocal_sparkinfer PUBLIC recurlocal_cuda)
    set_target_properties(recurlocal_sparkinfer PROPERTIES EXPORT_NAME recurlocal_sparkinfer)
    if(NOT MSVC)
        target_compile_options(recurlocal_sparkinfer PRIVATE -Wall -Wextra)
    endif()

    # Device-side contract tests. They skip cleanly with exit 0 when no GPU is present, so
    # this stays registered even on a device-less CUDA build box - the alternative is a
    # suite that only exists on the one machine that can run it.
    if(RECURLOCAL_BUILD_TESTS)
        add_executable(test_cuda_controller tests/test_cuda_controller.cu)
        target_link_libraries(test_cuda_controller PRIVATE recurlocal_cuda CUDA::cudart)
        set_target_properties(test_cuda_controller PROPERTIES CUDA_STANDARD 17 CUDA_STANDARD_REQUIRED ON)
        add_test(NAME cuda_controller COMMAND test_cuda_controller)
    endif()

    add_executable(recur_local_cuda_bench bench/cuda_bench.cu)
    target_link_libraries(recur_local_cuda_bench PRIVATE recurlocal_cuda CUDA::cudart)
    set_target_properties(recur_local_cuda_bench PROPERTIES CUDA_STANDARD 17 CUDA_STANDARD_REQUIRED ON)
endif()

if(RECURLOCAL_BUILD_TESTS)
    enable_testing()
    add_executable(test_planner tests/test_planner.cpp)
    target_link_libraries(test_planner PRIVATE recurlocal)
    add_test(NAME planner COMMAND test_planner)

    # The decision bands are policy: drift there silently redefines every past verdict,
    # so they are covered by the same `ctest` run as the planner.
    find_package(Python3 COMPONENTS Interpreter)
    if(Python3_Interpreter_FOUND)
        add_test(NAME decision COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/eval/test_decide.py)
    else()
        message(STATUS "Python3 not found - skipping the eval decision tests")
    endif()
endif()

if(RECURLOCAL_INSTALL)
    include(CMakePackageConfigHelpers)
    set(_recurlocal_targets recurlocal)
    if(RECURLOCAL_BUILD_CUDA)
        list(APPEND _recurlocal_targets recurlocal_cuda recurlocal_sparkinfer)
    endif()
    install(TARGETS ${_recurlocal_targets} EXPORT RecurLocalTargets
            ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
            LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
            RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})
    install(DIRECTORY include/ DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
    install(EXPORT RecurLocalTargets NAMESPACE RecurLocal::
            DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/RecurLocal)
    configure_package_config_file(cmake/RecurLocalConfig.cmake.in
        ${CMAKE_CURRENT_BINARY_DIR}/RecurLocalConfig.cmake
        INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/RecurLocal)
    write_basic_package_version_file(
        ${CMAKE_CURRENT_BINARY_DIR}/RecurLocalConfigVersion.cmake
        COMPATIBILITY SameMinorVersion)
    install(FILES ${CMAKE_CURRENT_BINARY_DIR}/RecurLocalConfig.cmake
                  ${CMAKE_CURRENT_BINARY_DIR}/RecurLocalConfigVersion.cmake
            DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/RecurLocal)
endif()
