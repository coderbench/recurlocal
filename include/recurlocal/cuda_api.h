#pragma once
#include <cstddef>
#include "recurlocal/planner.h"

#ifdef RECURLLOCAL_WITH_CUDA
#include <cuda_runtime_api.h>
namespace recurlocal {

DeviceCaps query_device_caps(int device);
cudaError_t configure_persisting_l2(int device, std::size_t requested_bytes, std::size_t* actual_bytes = nullptr);
cudaError_t set_access_policy_window(cudaStream_t stream, void* ptr, std::size_t bytes, double hit_ratio);
cudaError_t clear_access_policy_window(cudaStream_t stream);
cudaError_t pre_touch_async(const float* ptr, std::size_t count, float* scratch, std::size_t scratch_count, cudaStream_t stream);

class CudaLocalityController {
public:
    CudaLocalityController(int device, PlannerConfig config);
    ~CudaLocalityController();
    CudaLocalityController(const CudaLocalityController&) = delete;
    CudaLocalityController& operator=(const CudaLocalityController&) = delete;

    cudaError_t bind_streams(cudaStream_t compute_stream, cudaStream_t prefetch_stream);
    cudaError_t before_layer(void* current_state, std::size_t current_state_bytes,
                             const float* next_state, std::size_t next_state_count,
                             bool has_next_recurrent_layer,
                             std::size_t concurrently_hot_bytes = 0);
    cudaError_t after_layer();
    cudaError_t reset();

    const LocalityPlanner& planner() const noexcept { return planner_; }
    std::size_t l2_set_aside_bytes() const noexcept { return l2_set_aside_bytes_; }

private:
    int device_;
    LocalityPlanner planner_;
    cudaStream_t compute_stream_ = nullptr;
    cudaStream_t prefetch_stream_ = nullptr;
    float* scratch_ = nullptr;
    std::size_t scratch_count_ = 0;
    std::size_t l2_set_aside_bytes_ = 0;
};

} // namespace recurlocal
#endif
