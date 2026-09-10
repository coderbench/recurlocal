#include "tensortransit/cuda_recurrent.h"
#if defined(RECURLOCAL_WITH_CUDA) || defined(RECURLLOCAL_WITH_CUDA)
#include <cuda_runtime.h>
#include <algorithm>
#include <cstdint>

// Pre-touch strategies.
//
// Every strategy must be read-only over the caller's state and must leave a small
// observable result so the compiler cannot delete the loads. They differ only in how the
// bytes are moved. That is deliberately the whole contract: a new strategy is a new kernel
// plus an enumerator, and it can be measured against every existing one without touching
// them. See docs/OPTIMIZATION-SURFACES.md.

namespace {

__global__ void pre_touch_scalar(const float* __restrict__ ptr, std::size_t count,
                                 float* __restrict__ scratch, std::size_t scratch_count) {
    const std::size_t tid=blockIdx.x*blockDim.x+threadIdx.x;
    const std::size_t stride=(std::size_t)blockDim.x*gridDim.x;
    float acc=0.0f;
    for (std::size_t i=tid;i<count;i+=stride) acc += ptr[i];
    if (threadIdx.x==0 && blockIdx.x<scratch_count) scratch[blockIdx.x]=acc;
}

// 128-bit loads: a quarter of the memory instructions for the same bytes, so the walk
// finishes sooner and competes with the running kernel for less time.
__global__ void pre_touch_vec4(const float4* __restrict__ vec, std::size_t vec_count,
                               const float* __restrict__ tail, std::size_t tail_count,
                               float* __restrict__ scratch, std::size_t scratch_count) {
    const std::size_t tid=blockIdx.x*blockDim.x+threadIdx.x;
    const std::size_t stride=(std::size_t)blockDim.x*gridDim.x;
    float acc=0.0f;
    for (std::size_t i=tid;i<vec_count;i+=stride) { const float4 v=vec[i]; acc+=v.x+v.y+v.z+v.w; }
    for (std::size_t i=tid;i<tail_count;i+=stride) acc+=tail[i];
    if (threadIdx.x==0 && blockIdx.x<scratch_count) scratch[blockIdx.x]=acc;
}

// __ldcg caches at the global level and does not keep the line in L1. That is what a
// pre-touch actually wants: warm the cache the next layer will read from without evicting
// the L1 working set of the kernel currently running.
__global__ void pre_touch_vec4_ldcg(const float4* __restrict__ vec, std::size_t vec_count,
                                    const float* __restrict__ tail, std::size_t tail_count,
                                    float* __restrict__ scratch, std::size_t scratch_count) {
    const std::size_t tid=blockIdx.x*blockDim.x+threadIdx.x;
    const std::size_t stride=(std::size_t)blockDim.x*gridDim.x;
    float acc=0.0f;
    for (std::size_t i=tid;i<vec_count;i+=stride) { const float4 v=__ldcg(&vec[i]); acc+=v.x+v.y+v.z+v.w; }
    for (std::size_t i=tid;i<tail_count;i+=stride) acc+=__ldcg(&tail[i]);
    if (threadIdx.x==0 && blockIdx.x<scratch_count) scratch[blockIdx.x]=acc;
}

// prefetch.global.L2 moves a line into L2 without loading it into a register or occupying
// L1. The overview assumed CUDA exposes no such primitive for arbitrary allocations; PTX
// does, and it is the closest thing to "put this range in L2 now" that the ISA offers.
// Nothing is written back, so the observable scratch result is a constant.
__global__ void pre_touch_ptx_l2(const char* __restrict__ ptr, std::size_t bytes,
                                 float* __restrict__ scratch, std::size_t scratch_count) {
    constexpr std::size_t kLine = 128;
    const std::size_t lines = (bytes + kLine - 1) / kLine;
    const std::size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    const std::size_t stride = (std::size_t)blockDim.x * gridDim.x;
    for (std::size_t i = tid; i < lines; i += stride) {
        const char* line = ptr + i * kLine;
        asm volatile("prefetch.global.L2 [%0];" :: "l"(line) : "memory");
    }
    if (threadIdx.x == 0 && blockIdx.x < scratch_count) scratch[blockIdx.x] = (float)lines;
}

// One warp per contiguous tile. Each warp's 32 lanes issue a fully coalesced 128-byte
// request per step, instead of every warp striding across the whole buffer.
__global__ void pre_touch_warp_tile(const float4* __restrict__ vec, std::size_t vec_count,
                                    float* __restrict__ scratch, std::size_t scratch_count) {
    const unsigned lane = threadIdx.x & 31u;
    const unsigned warp = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    const unsigned warps = (gridDim.x * blockDim.x) >> 5;
    const std::size_t per_warp = (vec_count + warps - 1) / warps;
    const std::size_t begin = (std::size_t)warp * per_warp;
    const std::size_t end = min(begin + per_warp, vec_count);
    float acc = 0.0f;
    for (std::size_t i = begin + lane; i < end; i += 32) { const float4 v = vec[i]; acc += v.x + v.y + v.z + v.w; }
    if (threadIdx.x == 0 && blockIdx.x < scratch_count) scratch[blockIdx.x] = acc;
}

// Row-major pre-touch: one launch for every concurrently decoding sequence.
//
// At concurrency a runtime does not hold one recurrent-state allocation, it holds one per
// sequence and gathers their base pointers into a device-side array (SparkInfer's
// packed_lin_state). Walking them with one launch per row would put `rows` extra kernel
// nodes per layer into a captured decode graph - 3072 of them at 32 sequences and 48
// recurrent layers - which costs more than any locality it could buy. This reads the base
// pointer from device memory instead, so the launch count does not depend on concurrency.
__global__ void pre_touch_rows_vec4(const void* const* __restrict__ bases, int rows,
                                    std::size_t byte_offset, std::size_t vec_per_row,
                                    float* __restrict__ scratch, std::size_t scratch_count) {
    const unsigned row = blockIdx.y;
    if (row >= (unsigned)rows) return;
    const char* base = static_cast<const char*>(bases[row]);
    if (!base) return;
    const float4* vec = reinterpret_cast<const float4*>(base + byte_offset);
    const std::size_t tid = (std::size_t)blockIdx.x * blockDim.x + threadIdx.x;
    const std::size_t stride = (std::size_t)blockDim.x * gridDim.x;
    float acc = 0.0f;
    for (std::size_t i = tid; i < vec_per_row; i += stride) { const float4 v = vec[i]; acc += v.x + v.y + v.z + v.w; }
    if (threadIdx.x == 0 && blockIdx.x < scratch_count) scratch[blockIdx.x] = acc;
}

// Byte-exact counterpart for a slice that is not a whole number of float4 - the bf16
// convolution window at an odd stride, for instance.
__global__ void pre_touch_rows_ptx_l2(const void* const* __restrict__ bases, int rows,
                                      std::size_t byte_offset, std::size_t bytes,
                                      float* __restrict__ scratch, std::size_t scratch_count) {
    const unsigned row = blockIdx.y;
    if (row >= (unsigned)rows) return;
    const char* base = static_cast<const char*>(bases[row]);
    if (!base) return;
    const char* ptr = base + byte_offset;
    constexpr std::size_t kLine = 128;
    const std::size_t lines = (bytes + kLine - 1) / kLine;
    const std::size_t tid = (std::size_t)blockIdx.x * blockDim.x + threadIdx.x;
    const std::size_t stride = (std::size_t)blockDim.x * gridDim.x;
    for (std::size_t i = tid; i < lines; i += stride)
        asm volatile("prefetch.global.L2 [%0];" :: "l"(ptr + i * kLine) : "memory");
    if (threadIdx.x == 0 && blockIdx.x < scratch_count) scratch[blockIdx.x] = (float)lines;
}

constexpr int kThreads = 256;

unsigned grid_for(std::size_t work, std::size_t scratch_count) {
    const auto wanted = static_cast<unsigned>((work + kThreads - 1) / kThreads);
    return std::max(1u, std::min(wanted, static_cast<unsigned>(scratch_count)));
}

} // namespace

// Launch status is reported with cudaPeekAtLastError(), never cudaGetLastError().
//
// The difference matters because this library lives inside someone else's process.
// cudaGetLastError() CLEARS the per-thread error slot, so a pre-touch would silently consume
// an error the host runtime had not yet checked - its own next error check would then report
// success for whatever really failed. Peek reports without consuming. It can still surface a
// stale error as though it were ours, which is why the fork/join bookkeeping no longer
// depends on this call succeeding.
namespace tensortransit {

cudaError_t pre_touch_async(PreTouchStrategy strategy, const float* ptr, std::size_t count,
                            float* scratch, std::size_t scratch_count, cudaStream_t stream) noexcept {
    if (!ptr || !count || !scratch || !scratch_count || !stream) return cudaErrorInvalidValue;

    // Vector strategies need 16-byte alignment. cudaMalloc gives it, but a runtime handing
    // us a sub-allocation may not, so fall back rather than fault.
    const bool aligned = (reinterpret_cast<std::uintptr_t>(ptr) % sizeof(float4)) == 0;
    if (!aligned && strategy != PreTouchStrategy::Scalar) strategy = PreTouchStrategy::Scalar;

    if (strategy == PreTouchStrategy::Scalar) {
        pre_touch_scalar<<<grid_for(count, scratch_count), kThreads, 0, stream>>>(
            ptr, count, scratch, scratch_count);
        return cudaPeekAtLastError();
    }

    if (strategy == PreTouchStrategy::PtxL2) {
        const std::size_t bytes = count * sizeof(float);
        const std::size_t lines = (bytes + 127) / 128;
        pre_touch_ptx_l2<<<grid_for(lines, scratch_count), kThreads, 0, stream>>>(
            reinterpret_cast<const char*>(ptr), bytes, scratch, scratch_count);
        return cudaPeekAtLastError();
    }

    // Warming only the leading part of the state: cheaper, and enough if the recurrent
    // kernel reads the buffer in order.
    if (strategy == PreTouchStrategy::Partial) count = std::max<std::size_t>(count / 2, 1);

    const std::size_t vec_count = count / 4;
    const std::size_t tail_count = count - vec_count * 4;
    const auto* vec = reinterpret_cast<const float4*>(ptr);
    const float* tail = ptr + vec_count * 4;
    const unsigned blocks = grid_for(vec_count ? vec_count : 1, scratch_count);

    if (strategy == PreTouchStrategy::WarpTile) {
        pre_touch_warp_tile<<<blocks, kThreads, 0, stream>>>(vec, vec_count, scratch, scratch_count);
        return cudaPeekAtLastError();
    }
    if (strategy == PreTouchStrategy::Vec4 || strategy == PreTouchStrategy::Partial) {
        pre_touch_vec4<<<blocks, kThreads, 0, stream>>>(
            vec, vec_count, tail, tail_count, scratch, scratch_count);
    } else {
        pre_touch_vec4_ldcg<<<blocks, kThreads, 0, stream>>>(
            vec, vec_count, tail, tail_count, scratch, scratch_count);
    }
    return cudaPeekAtLastError();
}

cudaError_t pre_touch_async(const float* ptr, std::size_t count, float* scratch,
                            std::size_t scratch_count, cudaStream_t stream) noexcept {
    return pre_touch_async(PreTouchStrategy::Scalar, ptr, count, scratch, scratch_count, stream);
}

cudaError_t pre_touch_rows_async(PreTouchStrategy strategy, const void* const* device_bases,
                                 int rows, std::size_t byte_offset, std::size_t bytes,
                                 float* scratch, std::size_t scratch_count,
                                 cudaStream_t stream) noexcept {
    if (!device_bases || rows <= 0 || !bytes || !scratch || !scratch_count || !stream)
        return cudaErrorInvalidValue;
    // The base pointers live in device memory, so alignment of each row cannot be checked
    // from the host. cudaMalloc guarantees 256 bytes; a slice offset that is not a multiple
    // of 16 is the only way to lose it, and that is checkable.
    const bool vec_shaped = (bytes % sizeof(float4)) == 0 && (byte_offset % sizeof(float4)) == 0;
    dim3 grid(1, (unsigned)rows);
    if (strategy == PreTouchStrategy::PtxL2 || !vec_shaped) {
        const std::size_t lines = (bytes + 127) / 128;
        grid.x = grid_for(lines, scratch_count);
        pre_touch_rows_ptx_l2<<<grid, kThreads, 0, stream>>>(
            device_bases, rows, byte_offset, bytes, scratch, scratch_count);
        return cudaPeekAtLastError();
    }
    std::size_t effective = bytes;
    if (strategy == PreTouchStrategy::Partial)
        effective = std::max<std::size_t>((bytes / 2) & ~(sizeof(float4) - 1), sizeof(float4));
    const std::size_t vec_per_row = effective / sizeof(float4);
    grid.x = grid_for(vec_per_row, scratch_count);
    pre_touch_rows_vec4<<<grid, kThreads, 0, stream>>>(
        device_bases, rows, byte_offset, vec_per_row, scratch, scratch_count);
    return cudaPeekAtLastError();
}

// Byte-oriented entry point. A hybrid model's second recurrent state is bf16 (Qwen3.8-27B's
// convolution window), and a walk expressed in floats cannot touch it. The strategies read
// and discard, so the element type only matters for how many bytes the tail is: whatever
// does not divide into floats is walked line-wise by the PTX strategy, which is byte-exact.
cudaError_t pre_touch_bytes_async(PreTouchStrategy strategy, const void* ptr, std::size_t bytes,
                                  float* scratch, std::size_t scratch_count,
                                  cudaStream_t stream) noexcept {
    if (!ptr || !bytes || !scratch || !scratch_count || !stream) return cudaErrorInvalidValue;
    // PtxL2 is already byte-addressed; everything else walks floats, so a buffer that is
    // not a whole number of floats (or is not float-aligned) goes through it instead of
    // being silently truncated or read past its end.
    const bool float_shaped = (bytes % sizeof(float)) == 0 &&
                              (reinterpret_cast<std::uintptr_t>(ptr) % alignof(float)) == 0;
    if (strategy == PreTouchStrategy::PtxL2 || !float_shaped) {
        const std::size_t lines = (bytes + 127) / 128;
        pre_touch_ptx_l2<<<grid_for(lines, scratch_count), kThreads, 0, stream>>>(
            static_cast<const char*>(ptr), bytes, scratch, scratch_count);
        return cudaPeekAtLastError();
    }
    return pre_touch_async(strategy, static_cast<const float*>(ptr), bytes / sizeof(float),
                           scratch, scratch_count, stream);
}

} // namespace tensortransit
#endif
