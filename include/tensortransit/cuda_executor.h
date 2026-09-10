#pragma once
#include "tensortransit/executor.h"
#include "tensortransit/tensor.h"

#if defined(TENSORTRANSIT_WITH_CUDA) || defined(RECURLOCAL_WITH_CUDA) || \
    defined(RECURLLOCAL_WITH_CUDA)
#include <cuda_runtime_api.h>

#include "tensortransit/device.h"

namespace tensortransit {

// Fills a DeviceProfile from the device. Non-throwing: a runtime built with -fno-exceptions,
// or driven from a Python binding, cannot take a throw at an integration boundary.
cudaError_t query_device_profile(int device, DeviceProfile* out) noexcept;

// Applies a TransitPlan on CUDA.
//
// It decides NOTHING. Every choice -- which region, which hit ratio, when to clear, which
// stream -- came from the planner and is in the plan, where it is serializable and testable
// on a machine with no GPU. This class only knows how to turn each action into the CUDA call
// that performs it, and how to refuse the ones that would be unsafe.
//
// Two things it refuses, both because they have actually gone wrong here:
//
//   * A tensor whose registry entry no longer matches the plan's pointer. The runtime
//     recycled the memory under a compiled plan, and placing a window on whatever is at that
//     address now is worse than doing nothing. Counted in `stale_tensor_refs`.
//   * A tensor with device == -1, which is how read_trace() marks a synthetic address. A
//     plan built offline from a trace is for inspection, never for execution, and the only
//     safe place to enforce that is at the point where an address would reach the driver.
//
// Thread-safety: one instance is not synchronised. Hold one per compute stream, which is
// also the only scope on which the shared locality budget it manages means anything.
class CudaTransitExecutor final : public ITransitExecutor {
public:
    CudaTransitExecutor() noexcept = default;
    ~CudaTransitExecutor() override;
    CudaTransitExecutor(const CudaTransitExecutor&) = delete;
    CudaTransitExecutor& operator=(const CudaTransitExecutor&) = delete;

    // Reserves the persisting-L2 set-aside and records the previous value so it can be
    // given back. Never throws; failure is reported by status().
    //
    // The set-aside is DEVICE-WIDE and context-lifetime: without restoring it, merely
    // constructing an executor carves a permanent hole out of L2 for every other kernel in
    // the process, including after the executor is gone.
    cudaError_t initialize(int device, std::size_t set_aside_bytes) noexcept;
    cudaError_t status() const noexcept { return status_; }

    // Both streams are BORROWED: this class never destroys them and must never touch one
    // after the owner has destroyed it. Call release() before destroying the streams.
    // The two must differ -- a prefetch issued on the compute stream is not a prefetch, it
    // is extra work on the critical path.
    cudaError_t bind_streams(cudaStream_t compute, cudaStream_t prefetch) noexcept;

    // Borrowed, and required: without it the executor cannot tell a live tensor from a
    // recycled one and would place windows on stale addresses.
    void set_registry(const TensorRegistry* registry) noexcept { registry_ = registry; }

    // Scratch for the pre-touch kernels' (discarded) results. Borrowed device memory.
    void set_scratch(float* scratch, std::size_t count) noexcept;

    void set_plan(const TransitPlan* plan) noexcept override;
    const TransitPlan* plan() const noexcept override { return plan_; }
    void begin_step() noexcept override;
    void before_kernel(KernelId kernel) noexcept override;
    void after_kernel(KernelId kernel) noexcept override;
    void end_step() noexcept override;
    const ExecutorStats& stats() const noexcept override { return stats_; }
    void reset_stats() noexcept override { stats_ = ExecutorStats{}; }
    const char* name() const noexcept override { return "cuda"; }

    // Attach the window the last before_kernel() deferred to the graph kernel node(s) the
    // runtime has just recorded. Call immediately after launching the kernel, while capture
    // is still active.
    //
    // Production decode is captured into a CUDA graph, and a stream access-policy window is
    // host-side state the graph never records -- so under capture the window vanishes from
    // every replay unless it is set on the NODE. This is the only path by which a persisting
    // policy reaches a captured decode loop at all. No-op when nothing was deferred, so it
    // is safe to call unconditionally.
    cudaError_t attach_window_to_captured_node() noexcept;

    bool graph_capture_active() const noexcept;
    std::size_t set_aside_bytes() const noexcept { return set_aside_bytes_; }
    // The largest set-aside this executor ever held. Survives release(), which is what a
    // telemetry dump at process exit reads -- set_aside_bytes() there is 0, because shutdown
    // has already given the partition back, and reporting that makes every run look as
    // though it reserved nothing.
    std::size_t set_aside_peak_bytes() const noexcept { return set_aside_peak_; }

    // Drops any window, waits for outstanding prefetch work, returns the device-wide
    // set-aside to what it was before initialize() claimed it, and DROPS the borrowed stream
    // handles -- after which this object touches neither stream, so the caller is free to
    // destroy them. Safe to call more than once; the destructor calls it.
    cudaError_t release() noexcept;

private:
    void apply(const TransitAction* const* actions, int count) noexcept;
    // nullptr when the action must not be executed; increments the right counter and says
    // why through `reason`.
    const TensorDesc* resolve(const TransitAction& action) noexcept;

    const TransitPlan* plan_ = nullptr;
    const TensorRegistry* registry_ = nullptr;
    int device_ = -1;
    cudaError_t status_ = cudaErrorNotPermitted;  // "not initialised" until initialize() runs
    cudaStream_t compute_ = nullptr;
    cudaStream_t prefetch_ = nullptr;
    float* scratch_ = nullptr;
    std::size_t scratch_count_ = 0;
    std::size_t set_aside_bytes_ = 0;
    std::size_t set_aside_peak_ = 0;
    std::size_t previous_set_aside_ = 0;
    bool set_aside_owned_ = false;
    bool window_active_ = false;
    // Set under capture by an action that could not install its window on the stream, and
    // consumed by attach_window_to_captured_node().
    bool window_pending_node_attach_ = false;
    cudaAccessPolicyWindow pending_window_{};
    // Latched after the first capture this executor invalidated. One is a measurement;
    // continuing to do it is vandalism.
    bool node_attach_disabled_ = false;
    // Events for the fork/join pairs a Prefetch action needs. Indexed by event_id.
    static constexpr int kMaxEvents = 64;
    cudaEvent_t events_[kMaxEvents] = {};
    bool fork_outstanding_ = false;
    ExecutorStats stats_{};
};

}  // namespace tensortransit
#endif
