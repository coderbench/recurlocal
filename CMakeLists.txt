cmake_minimum_required(VERSION 3.24)
project(RecurLocal VERSION 0.1.0 LANGUAGES CXX)

option(RECURLLOCAL_BUILD_CUDA "Build CUDA locality controller and benchmark" OFF)
option(RECURLLOCAL_BUILD_TESTS "Build tests" ON)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

add_library(recurlocal STATIC src/planner.cpp)
target_include_directories(recurlocal PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_compile_features(recurlocal PUBLIC cxx_std_17)

add_executable(recur_local_info tools/info.cpp)
target_link_libraries(recur_local_info PRIVATE recurlocal)

if(RECURLLOCAL_BUILD_CUDA)
    enable_language(CUDA)
    find_package(CUDAToolkit REQUIRED)
    add_library(recurlocal_cuda STATIC src/cuda/cache_control.cu src/cuda/prefetch.cu)
    target_include_directories(recurlocal_cuda PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
    target_link_libraries(recurlocal_cuda PUBLIC recurlocal CUDA::cudart)
    target_compile_definitions(recurlocal_cuda PUBLIC RECURLLOCAL_WITH_CUDA=1)
    set_target_properties(recurlocal_cuda PROPERTIES CUDA_STANDARD 17 CUDA_STANDARD_REQUIRED ON CUDA_SEPARABLE_COMPILATION ON)

    add_executable(recur_local_cuda_bench bench/cuda_bench.cu)
    target_link_libraries(recur_local_cuda_bench PRIVATE recurlocal_cuda CUDA::cudart)
    set_target_properties(recur_local_cuda_bench PROPERTIES CUDA_STANDARD 17 CUDA_STANDARD_REQUIRED ON)
endif()

if(RECURLLOCAL_BUILD_TESTS)
    enable_testing()
    add_executable(test_planner tests/test_planner.cpp)
    target_link_libraries(test_planner PRIVATE recurlocal)
    add_test(NAME planner COMMAND test_planner)
endif()
