#include <chrono>

#include "tensortransit/cuda_executor.h"
#include "tensortransit/cuda_recurrent.h"

namespace tensortransit {

namespace {
std::uint64_t now_ns() noexcept {
    return static_cast<std::uint64_t>(
        std::chrono::duration_cast<std::chrono::nanoseconds>(
            std::chrono::steady_clock::now().time_since_epoch())
            .count());
}
}  // namespace

cudaError_t query_device_profile(int device, DeviceProfile* out) noexcept {
    if (!out) return cudaErrorInvalidValue;
    cudaDeviceProp prop{};
    const cudaError_t err = cudaGetDeviceProperties(&prop, device);
    if (err != cudaSuccess) return err;
    DeviceProfile profile{};
    profile.device = device;
    profile.l2_bytes = static_cast<std::size_t>(prop.l2CacheSize);
    profile.persisting_l2_max_bytes = static_cast<std::size_t>(prop.persistingL2CacheMaxSize);
    profile.access_policy_max_window_bytes =
        static_cast<std::size_t>(prop.accessPolicyMaxWindowSize);
    profile.sm_count = prop.multiProcessorCount;
    profile.major = prop.major;
    profile.minor = prop.minor;
    profile.global_memory_bytes = prop.totalGlobalMem;
    // Memory bus width is in bits and the clock in kHz; DDR moves two words per cycle.
    // CUDA 13 removed `memoryClockRate` from cudaDeviceProp, so it comes from the attribute
    // API instead -- and if that fails the field stays 0, which every consumer reads as
    // "unknown" and degrades on. A fabricated bandwidth would silently rescale every
    // bandwidth-derived estimate in the cost model.
    int memory_clock_khz = 0;
    if (cudaDeviceGetAttribute(&memory_clock_khz, cudaDevAttrMemoryClockRate, device) ==
            cudaSuccess &&
        memory_clock_khz > 0) {
        profile.peak_bandwidth_bytes_per_s =
            static_cast<std::uint64_t>(memory_clock_khz) * 1000ull *
            (static_cast<std::uint64_t>(prop.memoryBusWidth) / 8ull) * 2ull;
    } else {
        cudaGetLastError();
    }
    *out = profile;
    return cudaSuccess;
}

CudaTransitExecutor::~CudaTransitExecutor() { release(); }

cudaError_t CudaTransitExecutor::initialize(int device, std::size_t set_aside_bytes) noexcept {
    release();
    device_ = device;
    const cudaError_t select = cudaSetDevice(device);
    if (select != cudaSuccess) return status_ = select;

    // Remember what the device's limit was BEFORE we touched it, so release() can put it
    // back. The limit is device-wide and lives as long as the context.
    std::size_t previous = 0;
    if (cudaDeviceGetLimit(&previous, cudaLimitPersistingL2CacheSize) == cudaSuccess)
        previous_set_aside_ = previous;

    if (set_aside_bytes) {
        std::size_t granted = 0;
        const cudaError_t err = configure_persisting_l2(device, set_aside_bytes, &granted);
        if (err != cudaSuccess) {
            // An optional optimization layer must not take a model server down with it.
            // Persisting L2 unsupported means no persistence, not no inference.
            set_aside_bytes_ = 0;
        } else {
            set_aside_bytes_ = granted;
            set_aside_owned_ = true;
            if (granted > set_aside_peak_) set_aside_peak_ = granted;
        }
    }
    return status_ = cudaSuccess;
}

cudaError_t CudaTransitExecutor::bind_streams(cudaStream_t compute, cudaStream_t prefetch) noexcept {
    if (compute == prefetch) return cudaErrorInvalidValue;
    compute_ = compute;
    prefetch_ = prefetch;
    return cudaSuccess;
}

void CudaTransitExecutor::set_scratch(float* scratch, std::size_t count) noexcept {
    scratch_ = scratch;
    scratch_count_ = count;
}

void CudaTransitExecutor::set_plan(const TransitPlan* plan) noexcept { plan_ = plan; }

bool CudaTransitExecutor::graph_capture_active() const noexcept {
    if (!compute_) return false;
    cudaStreamCaptureStatus status = cudaStreamCaptureStatusNone;
    if (cudaStreamIsCapturing(compute_, &status) != cudaSuccess) return false;
    return status != cudaStreamCaptureStatusNone;
}

void CudaTransitExecutor::begin_step() noexcept { ++stats_.steps; }

const TensorDesc* CudaTransitExecutor::resolve(const TransitAction& action) noexcept {
    if (action.tensor == kInvalidTensorId) return nullptr;
    if (!registry_) {
        // Without a registry there is no way to tell a live tensor from a recycled one.
        // Refusing is the only safe answer, and it is counted rather than silent.
        ++stats_.stale_tensor_refs;
        return nullptr;
    }
    const TensorDesc* desc = registry_->find(action.tensor);
    if (!desc || desc->ptr != action.ptr) {
        // The runtime recycled the memory under a compiled plan. Placing a window on
        // whatever is at that address now is worse than doing nothing.
        ++stats_.stale_tensor_refs;
        return nullptr;
    }
    // A plan built offline from a trace carries synthetic addresses. read_trace() marks them
    // with device -1 precisely so that this check exists in one place, at the boundary where
    // a fabricated address would otherwise reach the driver.
    if (desc->device < 0) {
        ++stats_.stale_tensor_refs;
        return nullptr;
    }
    return desc;
}

void CudaTransitExecutor::apply(const TransitAction* const* actions, int count) noexcept {
    const auto start = now_ns();
    for (int i = 0; i < count; ++i) {
        const TransitAction& action = *actions[i];
        switch (action.kind) {
            case TransitActionKind::Normal:
                break;

            case TransitActionKind::Persist:
            case TransitActionKind::RotateWindow: {
                const TensorDesc* desc = resolve(action);
                if (!desc) { ++stats_.actions_skipped; break; }
                void* ptr = const_cast<void*>(action.ptr);
                if (graph_capture_active()) {
                    // A stream attribute is host-side state the graph never records, so
                    // setting one here would be a policy absent from every replay -- which
                    // is a null candidate wearing a policy's telemetry. Defer instead.
                    LayerPlan shim{};
                    shim.hot_window_bytes = action.bytes;
                    shim.hit_ratio = action.hit_ratio;
                    pending_window_ = make_access_policy_window(shim, ptr);
                    window_pending_node_attach_ = true;
                    ++stats_.persist_deferred;
                    ++stats_.actions_applied;
                    break;
                }
                const cudaError_t err =
                    set_access_policy_window(compute_, ptr, action.bytes, action.hit_ratio);
                if (err != cudaSuccess) { ++stats_.actions_failed; break; }
                window_active_ = true;
                ++stats_.persist_applied;
                stats_.persist_bytes += action.bytes;
                ++stats_.actions_applied;
                break;
            }

            case TransitActionKind::Stream: {
                const TensorDesc* desc = resolve(action);
                if (!desc) { ++stats_.actions_skipped; break; }
                if (graph_capture_active()) { ++stats_.actions_skipped; break; }
                const cudaError_t err = set_streaming_policy_window(
                    compute_, const_cast<void*>(action.ptr), action.bytes);
                if (err != cudaSuccess) { ++stats_.actions_failed; break; }
                window_active_ = true;
                ++stats_.stream_applied;
                ++stats_.actions_applied;
                break;
            }

            case TransitActionKind::Prefetch: {
                const TensorDesc* desc = resolve(action);
                if (!desc || !prefetch_ || !scratch_) {
                    ++stats_.actions_skipped;
                    ++stats_.prefetch_skipped;
                    break;
                }
                const cudaError_t err = pre_touch_bytes_async(
                    PreTouchStrategy::Vec4, action.ptr, action.bytes, scratch_, scratch_count_,
                    prefetch_);
                if (err != cudaSuccess) { ++stats_.actions_failed; break; }
                ++stats_.prefetch_applied;
                stats_.prefetch_bytes += action.bytes;
                ++stats_.actions_applied;
                break;
            }

            case TransitActionKind::ClearPolicy: {
                if (graph_capture_active()) { ++stats_.actions_skipped; break; }
                if (!window_active_) { ++stats_.actions_skipped; break; }
                const cudaError_t err = clear_access_policy_window(compute_);
                if (err != cudaSuccess) { ++stats_.actions_failed; break; }
                window_active_ = false;
                ++stats_.clear_applied;
                ++stats_.actions_applied;
                break;
            }

            case TransitActionKind::RecordEvent: {
                if (!compute_ || !prefetch_) { ++stats_.actions_skipped; break; }
                const int slot = static_cast<int>(action.event_id % kMaxEvents);
                if (!events_[slot] &&
                    cudaEventCreateWithFlags(&events_[slot], cudaEventDisableTiming) != cudaSuccess) {
                    ++stats_.actions_failed;
                    break;
                }
                if (cudaEventRecord(events_[slot], compute_) != cudaSuccess ||
                    cudaStreamWaitEvent(prefetch_, events_[slot], 0) != cudaSuccess) {
                    ++stats_.actions_failed;
                    break;
                }
                fork_outstanding_ = true;
                ++stats_.events_recorded;
                ++stats_.actions_applied;
                break;
            }

            case TransitActionKind::WaitEvent: {
                if (!compute_ || !prefetch_) { ++stats_.actions_skipped; break; }
                const int slot = static_cast<int>(action.event_id % kMaxEvents);
                if (!events_[slot]) { ++stats_.actions_skipped; break; }
                if (cudaEventRecord(events_[slot], prefetch_) != cudaSuccess ||
                    cudaStreamWaitEvent(compute_, events_[slot], 0) != cudaSuccess) {
                    ++stats_.actions_failed;
                    break;
                }
                fork_outstanding_ = false;
                ++stats_.events_waited;
                ++stats_.actions_applied;
                break;
            }
        }
    }
    stats_.host_ns += now_ns() - start;
}

void CudaTransitExecutor::before_kernel(KernelId kernel) noexcept {
    ++stats_.kernels;
    if (!plan_) return;
    int count = 0;
    const TransitAction* const* actions = plan_->before(kernel, &count);
    if (actions) apply(actions, count);
}

void CudaTransitExecutor::after_kernel(KernelId kernel) noexcept {
    if (!plan_) return;
    int count = 0;
    const TransitAction* const* actions = plan_->after(kernel, &count);
    if (actions) apply(actions, count);
}

cudaError_t CudaTransitExecutor::attach_window_to_captured_node() noexcept {
    if (!window_pending_node_attach_ || node_attach_disabled_) return cudaSuccess;
    window_pending_node_attach_ = false;
    if (!compute_) return cudaSuccess;

    cudaStreamCaptureStatus status = cudaStreamCaptureStatusNone;
    cudaGraph_t graph = nullptr;
    const cudaGraphNode_t* dependencies = nullptr;
    std::size_t dependency_count = 0;
    unsigned long long capture_id = 0;
    // CUDA 13 dropped the _v2 spelling and redefined the plain name to the edge-data form,
    // which takes one more parameter. Both toolkits build from this one source.
#if CUDART_VERSION >= 13000
    const cudaGraphEdgeData* edges = nullptr;
    const cudaError_t query = cudaStreamGetCaptureInfo(compute_, &status, &capture_id, &graph,
                                                       &dependencies, &edges, &dependency_count);
#else
    const cudaError_t query = cudaStreamGetCaptureInfo_v2(compute_, &status, &capture_id,
                                                          &graph, &dependencies,
                                                          &dependency_count);
#endif
    if (query != cudaSuccess || status != cudaStreamCaptureStatusActive || !dependencies) {
        ++stats_.actions_skipped;
        return query;
    }

    cudaKernelNodeAttrValue value{};
    value.accessPolicyWindow = pending_window_;
    std::uint64_t marked = 0;
    for (std::size_t i = 0; i < dependency_count; ++i) {
        cudaGraphNodeType type = cudaGraphNodeTypeEmpty;
        if (cudaGraphNodeGetType(dependencies[i], &type) != cudaSuccess) continue;
        if (type != cudaGraphNodeTypeKernel) continue;
        if (cudaGraphKernelNodeSetAttribute(dependencies[i],
                                            cudaKernelNodeAttributeAccessPolicyWindow,
                                            &value) != cudaSuccess) {
            ++stats_.actions_failed;
            continue;
        }
        ++marked;
    }
    if (!marked) return cudaSuccess;

    stats_.persist_nodes_attached += marked;
    ++stats_.persist_attached_to_node;
    stats_.persist_bytes += pending_window_.num_bytes;

    // If the attach invalidated the capture, whatever the throughput number for this run
    // was, it is measuring the runtime's fallback path. Latch the mechanism off: one
    // invalidation is a measurement, continuing to cause them is vandalism.
    cudaStreamCaptureStatus after = cudaStreamCaptureStatusNone;
    if (cudaStreamIsCapturing(compute_, &after) == cudaSuccess &&
        after == cudaStreamCaptureStatusInvalidated) {
        node_attach_disabled_ = true;
        ++stats_.capture_invalidations;
        ++stats_.actions_failed;
    }
    return cudaSuccess;
}

void CudaTransitExecutor::end_step() noexcept {
    // An unjoined fork does not merely leak a stream: under capture it ends the capture
    // INVALID and takes the runtime's whole decode path with it. Close it unconditionally.
    if (fork_outstanding_ && compute_ && prefetch_) {
        for (int i = 0; i < kMaxEvents; ++i) {
            if (!events_[i]) continue;
            if (cudaEventRecord(events_[i], prefetch_) == cudaSuccess)
                cudaStreamWaitEvent(compute_, events_[i], 0);
            break;
        }
        fork_outstanding_ = false;
    }
}

cudaError_t CudaTransitExecutor::release() noexcept {
    cudaError_t first = cudaSuccess;
    const bool capturing = graph_capture_active();
    if (capturing) ++stats_.released_during_capture;

    // Never spray failing CUDA calls into a caller's context mid-capture. Being torn down
    // during a capture is a caller bug (an exception unwinding, usually), but it is one this
    // library must not make worse.
    if (window_active_ && compute_ && !capturing) {
        const cudaError_t err = clear_access_policy_window(compute_);
        if (err != cudaSuccess && first == cudaSuccess) first = err;
    }
    window_active_ = false;
    window_pending_node_attach_ = false;

    for (int i = 0; i < kMaxEvents; ++i) {
        if (!events_[i]) continue;
        if (!capturing) cudaEventDestroy(events_[i]);
        events_[i] = nullptr;
    }
    fork_outstanding_ = false;

    if (set_aside_owned_ && device_ >= 0 && !capturing) {
        // Give the L2 partition back. Without this, merely constructing an executor carves a
        // permanent hole out of L2 for every other kernel in the process, forever.
        const cudaError_t err = configure_persisting_l2(device_, previous_set_aside_, nullptr);
        if (err != cudaSuccess && first == cudaSuccess) first = err;
    }
    set_aside_owned_ = false;
    set_aside_bytes_ = 0;
    plan_ = nullptr;
    status_ = cudaErrorNotPermitted;

    // DROP the borrowed handles. They are the caller's streams and the caller is entitled to
    // destroy them the moment release() returns -- so keeping them means the destructor's own
    // release() probes a destroyed stream, and cudaStreamIsCapturing segfaults inside
    // libcuda. Not a leak: a crash, in the teardown path of an optional optimization layer,
    // in somebody else's model server. Found by tests/test_cuda_executor.cu doing exactly
    // what this header tells a caller to do.
    compute_ = nullptr;
    prefetch_ = nullptr;
    registry_ = nullptr;
    scratch_ = nullptr;
    scratch_count_ = 0;
    return first;
}

}  // namespace tensortransit
