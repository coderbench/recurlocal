#!/usr/bin/env bash
set -euo pipefail
cmake -S . -B build -DRECURLLOCAL_BUILD_CUDA=OFF
cmake --build build -j
ctest --test-dir build --output-on-failure
