#include "recurlocal/cuda_api.h"
#ifdef RECURLLOCAL_WITH_CUDA
#include <algorithm>
#include <stdexcept>

namespace recurlocal {

DeviceCaps query_device_caps(int device) {
    cudaDeviceProp prop{};
    auto err = cudaGetDeviceProperties(&prop, device);
    if (err != cudaSuccess) throw std::runtime_error(cudaGetErrorString(err));
    return DeviceCaps{
        static_cast<std::size_t>(prop.l2CacheSize),
        static_cast<std::size_t>(prop.persistingL2CacheMaxSize),
        static_cast<std::size_t>(prop.accessPolicyMaxWindowSize)};
}

cudaError_t configure_persisting_l2(int device, std::size_t requested_bytes, std::size_t* actual_bytes) {
    cudaDeviceProp prop{};
    auto err = cudaGetDeviceProperties(&prop, device); if (err != cudaSuccess) return err;
    const auto desired = std::min(requested_bytes, static_cast<std::size_t>(prop.persistingL2CacheMaxSize));
    err = cudaSetDevice(device); if (err != cudaSuccess) return err;
    err = cudaDeviceSetLimit(cudaLimitPersistingL2CacheSize, desired); if (err != cudaSuccess) return err;
    std::size_t actual=0; err = cudaDeviceGetLimit(&actual, cudaLimitPersistingL2CacheSize);
    if (err==cudaSuccess && actual_bytes) *actual_bytes=actual;
    return err;
}

cudaError_t set_access_policy_window(cudaStream_t stream, void* ptr, std::size_t bytes, double hit_ratio) {
    if (!stream || !ptr || !bytes) return cudaErrorInvalidValue;
    cudaStreamAttrValue a{};
    a.accessPolicyWindow.base_ptr=ptr;
    a.accessPolicyWindow.num_bytes=bytes;
    a.accessPolicyWindow.hitRatio=static_cast<float>(std::clamp(hit_ratio,0.0,1.0));
    a.accessPolicyWindow.hitProp=cudaAccessPropertyPersisting;
    a.accessPolicyWindow.missProp=cudaAccessPropertyStreaming;
    return cudaStreamSetAttribute(stream,cudaStreamAttributeAccessPolicyWindow,&a);
}

cudaError_t clear_access_policy_window(cudaStream_t stream) {
    if (!stream) return cudaErrorInvalidValue;
    cudaStreamAttrValue a{};
    a.accessPolicyWindow.base_ptr=nullptr;
    a.accessPolicyWindow.num_bytes=0;
    a.accessPolicyWindow.hitRatio=0.0f;
    a.accessPolicyWindow.hitProp=cudaAccessPropertyNormal;
    a.accessPolicyWindow.missProp=cudaAccessPropertyNormal;
    return cudaStreamSetAttribute(stream,cudaStreamAttributeAccessPolicyWindow,&a);
}

CudaLocalityController::CudaLocalityController(int device, PlannerConfig config)
    : device_(device), planner_(query_device_caps(device), config) {
    auto err=configure_persisting_l2(device_,planner_.recommended_l2_set_aside(),&l2_set_aside_bytes_);
    if (err!=cudaSuccess) { l2_set_aside_bytes_=0; cudaGetLastError(); }
    scratch_count_=8192;
    cudaMalloc(&scratch_,scratch_count_*sizeof(float));
}

CudaLocalityController::~CudaLocalityController() {
    if (compute_stream_) clear_access_policy_window(compute_stream_);
    if (scratch_) cudaFree(scratch_);
}

cudaError_t CudaLocalityController::bind_streams(cudaStream_t compute_stream, cudaStream_t prefetch_stream) {
    if (!compute_stream || !prefetch_stream) return cudaErrorInvalidValue;
    compute_stream_=compute_stream; prefetch_stream_=prefetch_stream; return cudaSuccess;
}

cudaError_t CudaLocalityController::before_layer(void* current_state, std::size_t current_state_bytes,
                                                  const float* next_state, std::size_t next_state_count,
                                                  bool has_next_recurrent_layer,
                                                  std::size_t concurrently_hot_bytes) {
    if (!compute_stream_ || !prefetch_stream_) return cudaErrorInvalidResourceHandle;
    const auto plan=planner_.plan_for_layer(current_state_bytes,has_next_recurrent_layer,concurrently_hot_bytes);
    if (plan.use_persisting_window && l2_set_aside_bytes_>0) {
        auto e=set_access_policy_window(compute_stream_,current_state,plan.hot_window_bytes,plan.hit_ratio);
        if (e!=cudaSuccess) return e;
    } else {
        clear_access_policy_window(compute_stream_);
    }
    if (plan.prefetch_next && next_state && next_state_count) {
        auto e=pre_touch_async(next_state,next_state_count,scratch_,scratch_count_,prefetch_stream_);
        if (e!=cudaSuccess) return e;
    }
    return cudaSuccess;
}

cudaError_t CudaLocalityController::after_layer() { return cudaSuccess; }
cudaError_t CudaLocalityController::reset() {
    if (compute_stream_) clear_access_policy_window(compute_stream_);
    return cudaCtxResetPersistingL2Cache();
}

} // namespace recurlocal
#endif
