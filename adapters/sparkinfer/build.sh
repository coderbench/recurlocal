#!/usr/bin/env bash
# Build a SparkInfer that carries the TensorTransit locality hook.
#
#   adapters/sparkinfer/build.sh [workdir]
#
# Clones the pinned commit, installs TensorTransit, applies the hook patch, builds. The result
# is ONE binary that runs every configuration: with TENSORTRANSIT unset it is the control, and
# with TENSORTRANSIT=<mode> it is the candidate. That is deliberate - a same-box A/B between
# two separately linked binaries cannot separate the policy from the link.
set -euo pipefail

here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
tt_root="$(cd "$here/../.." && pwd)"
work="${1:-${TENSORTRANSIT_WORKDIR:-$PWD/si-work}}"
arch="${CMAKE_CUDA_ARCHITECTURES:-120}"
jobs="${JOBS:-$(nproc)}"

read_pin() { python3 -c "import json,sys; print(json.load(open('$here/pin.json'))$1)"; }
repo="$(read_pin "['repository']")"
commit="$(read_pin "['commit']")"

mkdir -p "$work"
src="$work/sparkinfer"
prefix="$work/tensortransit-install"

echo ">> TensorTransit -> $prefix"
# From scratch every time. TensorTransit is a handful of translation units, so a full rebuild costs
# seconds, and reusing the tree across a change to a CUDA target property (separable
# compilation, architecture) leaves object files CMake will not recompile and the link fails
# with an undefined __cudaRegisterLinkedBinary_*. SparkInfer's tree is NOT wiped: that build
# is minutes, and nothing in it depends on TensorTransit's compile settings.
rm -rf "$work/tensortransit-build" "$prefix"
cmake -S "$tt_root" -B "$work/tensortransit-build" \
      -DCMAKE_BUILD_TYPE=Release -DTENSORTRANSIT_BUILD_CUDA=ON -DTENSORTRANSIT_BUILD_TESTS=OFF -DTENSORTRANSIT_BUILD_CLI=OFF \
      -DCMAKE_CUDA_ARCHITECTURES="$arch" -DCMAKE_INSTALL_PREFIX="$prefix" >/dev/null
cmake --build "$work/tensortransit-build" -j "$jobs" >/dev/null
cmake --install "$work/tensortransit-build" >/dev/null

echo ">> SparkInfer @ $commit"
if [ ! -d "$src/.git" ]; then
    git clone --filter=blob:none --no-checkout "$repo" "$src"
fi
git -C "$src" fetch --depth 1 origin "$commit" -q
# A hard reset, not a checkout: the patch must apply to the pinned tree and nothing else,
# so a previous run's applied hook is discarded rather than merged into.
git -C "$src" checkout --detach -q "$commit"
git -C "$src" reset --hard -q "$commit"
git -C "$src" clean -fdq -e build

echo ">> hook patch"
# From pin.json, not hardcoded: the pin file is the single source of truth for which patch
# belongs to which commit, and a second copy of the name here is a second thing to forget.
patch_name="$(read_pin "['patch']")"
git -C "$src" apply --verbose "$here/$patch_name"

echo ">> build (sm_$arch)"
cmake -S "$src" -B "$src/build" -DCMAKE_BUILD_TYPE=Release \
      -DCMAKE_CUDA_ARCHITECTURES="$arch" -DBUILD_TESTS=OFF -DBUILD_EXAMPLES=ON \
      -DCMAKE_PREFIX_PATH="$prefix" >/dev/null
# Three binaries: batch-1 decode, the token-exact greedy replay gate, and concurrent
# continuous batching. All from one build, so no arm of the matrix is measured against
# a differently compiled runtime.
cmake --build "$src/build" -j "$jobs" --target qwen3_gguf_bench qwen3_gguf_generate qwen3_gguf_cb_bench

echo
echo "binaries: $src/build/runtime/{qwen3_gguf_bench,qwen3_gguf_generate,qwen3_gguf_cb_bench}"
echo "control: TENSORTRANSIT unset"
echo "cand.  : TENSORTRANSIT=combined TENSORTRANSIT_PREFETCH_DISTANCE=1 ..."
