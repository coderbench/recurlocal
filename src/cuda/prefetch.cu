#include "recurlocal/cuda_api.h"
#ifdef RECURLLOCAL_WITH_CUDA
#include <cuda_runtime.h>
#include <algorithm>

namespace {
__global__ void pre_touch_kernel(const float* __restrict__ ptr, std::size_t count,
                                 float* __restrict__ scratch, std::size_t scratch_count) {
    const std::size_t tid=blockIdx.x*blockDim.x+threadIdx.x;
    const std::size_t stride=blockDim.x*gridDim.x;
    float acc=0.0f;
    for (std::size_t i=tid;i<count;i+=stride) acc += ptr[i];
    if (threadIdx.x==0 && blockIdx.x<scratch_count) scratch[blockIdx.x]=acc;
}
}

namespace recurlocal {
cudaError_t pre_touch_async(const float* ptr, std::size_t count, float* scratch,
                            std::size_t scratch_count, cudaStream_t stream) {
    if (!ptr || !count || !scratch || !scratch_count || !stream) return cudaErrorInvalidValue;
    constexpr int threads=256;
    const auto wanted=static_cast<unsigned>((count+threads-1)/threads);
    const unsigned blocks=std::max(1u,std::min(wanted,static_cast<unsigned>(scratch_count)));
    pre_touch_kernel<<<blocks,threads,0,stream>>>(ptr,count,scratch,scratch_count);
    return cudaGetLastError();
}
}
#endif
