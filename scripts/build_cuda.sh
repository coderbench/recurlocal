#!/usr/bin/env bash
set -euo pipefail
ARCH="${CMAKE_CUDA_ARCHITECTURES:-120}"
cmake -S . -B build -DTENSORTRANSIT_BUILD_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES="${ARCH}"
cmake --build build -j
ctest --test-dir build --output-on-failure
