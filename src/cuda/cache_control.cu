#include "recurlocal/cuda_api.h"
#if defined(RECURLOCAL_WITH_CUDA) || defined(RECURLLOCAL_WITH_CUDA)
#include <algorithm>
#include <stdexcept>

namespace recurlocal {

cudaError_t query_device_caps(int device, DeviceCaps* out) noexcept {
    if (!out) return cudaErrorInvalidValue;
    cudaDeviceProp prop{};
    auto err = cudaGetDeviceProperties(&prop, device);
    if (err != cudaSuccess) return err;
    out->l2_bytes = static_cast<std::size_t>(prop.l2CacheSize);
    out->persisting_l2_max_bytes = static_cast<std::size_t>(prop.persistingL2CacheMaxSize);
    out->access_policy_max_window_bytes = static_cast<std::size_t>(prop.accessPolicyMaxWindowSize);
    return cudaSuccess;
}

DeviceCaps query_device_caps(int device) {
    DeviceCaps caps{};
    auto err = query_device_caps(device, &caps);
    if (err != cudaSuccess) throw std::runtime_error(cudaGetErrorString(err));
    return caps;
}

cudaError_t configure_persisting_l2(int device, std::size_t requested_bytes, std::size_t* actual_bytes) noexcept {
    cudaDeviceProp prop{};
    auto err = cudaGetDeviceProperties(&prop, device); if (err != cudaSuccess) return err;
    const auto desired = std::min(requested_bytes, static_cast<std::size_t>(prop.persistingL2CacheMaxSize));

    // The limit is per-device, so setting it means selecting that device first - but the
    // current device belongs to the CALLER, not to us. A runtime that drives two GPUs and
    // happens to construct a controller for device 1 would find its thread silently
    // repointed, and the next unqualified launch would land on the wrong GPU. Save and
    // restore, including on every early return.
    int previous = device;
    err = cudaGetDevice(&previous); if (err != cudaSuccess) return err;
    const bool switched = previous != device;
    if (switched) { err = cudaSetDevice(device); if (err != cudaSuccess) return err; }

    err = cudaDeviceSetLimit(cudaLimitPersistingL2CacheSize, desired);
    if (err == cudaSuccess) {
        std::size_t actual = 0;
        err = cudaDeviceGetLimit(&actual, cudaLimitPersistingL2CacheSize);
        if (err == cudaSuccess && actual_bytes) *actual_bytes = actual;
    }
    if (switched) {
        const auto restore = cudaSetDevice(previous);
        if (err == cudaSuccess) err = restore;
    }
    return err;
}

cudaAccessPolicyWindow make_access_policy_window(const LayerPlan& plan, void* ptr) noexcept {
    cudaAccessPolicyWindow w{};
    w.base_ptr=ptr;
    w.num_bytes=plan.hot_window_bytes;
    w.hitRatio=static_cast<float>(std::clamp(plan.hit_ratio,0.0,1.0));
    w.hitProp=cudaAccessPropertyPersisting;
    w.missProp=cudaAccessPropertyStreaming;
    return w;
}

cudaError_t set_access_policy_window(cudaStream_t stream, void* ptr, std::size_t bytes, double hit_ratio) noexcept {
    if (!stream || !ptr || !bytes) return cudaErrorInvalidValue;
    cudaStreamAttrValue a{};
    a.accessPolicyWindow.base_ptr=ptr;
    a.accessPolicyWindow.num_bytes=bytes;
    a.accessPolicyWindow.hitRatio=static_cast<float>(std::clamp(hit_ratio,0.0,1.0));
    a.accessPolicyWindow.hitProp=cudaAccessPropertyPersisting;
    a.accessPolicyWindow.missProp=cudaAccessPropertyStreaming;
    return cudaStreamSetAttribute(stream,cudaStreamAttributeAccessPolicyWindow,&a);
}

cudaError_t clear_access_policy_window(cudaStream_t stream) noexcept {
    if (!stream) return cudaErrorInvalidValue;
    cudaStreamAttrValue a{};
    a.accessPolicyWindow.base_ptr=nullptr;
    a.accessPolicyWindow.num_bytes=0;
    a.accessPolicyWindow.hitRatio=0.0f;
    a.accessPolicyWindow.hitProp=cudaAccessPropertyNormal;
    a.accessPolicyWindow.missProp=cudaAccessPropertyNormal;
    return cudaStreamSetAttribute(stream,cudaStreamAttributeAccessPolicyWindow,&a);
}

cudaError_t set_streaming_policy_window(cudaStream_t stream, void* ptr, std::size_t bytes) noexcept {
    if (!stream || !ptr || !bytes) return cudaErrorInvalidValue;
    cudaStreamAttrValue a{};
    a.accessPolicyWindow.base_ptr=ptr;
    a.accessPolicyWindow.num_bytes=bytes;
    a.accessPolicyWindow.hitRatio=1.0f;
    a.accessPolicyWindow.hitProp=cudaAccessPropertyStreaming;
    a.accessPolicyWindow.missProp=cudaAccessPropertyStreaming;
    return cudaStreamSetAttribute(stream,cudaStreamAttributeAccessPolicyWindow,&a);
}

CudaLocalityController::CudaLocalityController(int device, PlannerConfig config) noexcept {
    initialize(device, config);
}

cudaError_t CudaLocalityController::initialize(int device, PlannerConfig config) noexcept {
    release();
    if (validate(config)) return status_ = cudaErrorInvalidValue;

    DeviceCaps caps{};
    auto err = query_device_caps(device, &caps);
    if (err != cudaSuccess) return status_ = err;

    device_ = device;
    planner_ = LocalityPlanner(caps, config);

    // A device that cannot reserve a set-aside is not an error; the planner simply will
    // not ask for a persisting window on it. Record what the limit was first, so release()
    // can give the device back exactly what it had rather than guessing at zero.
    previous_l2_set_aside_ = 0;
    l2_set_aside_owned_ = false;
    const auto wanted = planner_.recommended_l2_set_aside();
    if (wanted) {
        if (cudaDeviceGetLimit(&previous_l2_set_aside_, cudaLimitPersistingL2CacheSize) != cudaSuccess) {
            previous_l2_set_aside_ = 0; cudaGetLastError();
        }
        if (configure_persisting_l2(device_, wanted, &l2_set_aside_bytes_) == cudaSuccess) {
            l2_set_aside_owned_ = true;
            l2_set_aside_requested_ = wanted;
            if (l2_set_aside_bytes_ > l2_set_aside_peak_) l2_set_aside_peak_ = l2_set_aside_bytes_;
            // Tell the planner what we GOT. Without this it keeps budgeting against the
            // request, and every hot-set decision inherits the driver's rounding error.
            planner_.set_granted_l2_set_aside(l2_set_aside_bytes_);
        } else {
            l2_set_aside_bytes_ = 0; cudaGetLastError();
        }
    } else {
        l2_set_aside_bytes_ = 0;
        l2_set_aside_requested_ = 0;
    }

    scratch_count_ = 8192;
    if (cudaMalloc(&scratch_, scratch_count_ * sizeof(float)) != cudaSuccess) {
        // Pre-touch degrades to a no-op rather than leaving a sticky error on the
        // context for an unrelated cudaGetLastError() to surface later.
        scratch_ = nullptr; scratch_count_ = 0; cudaGetLastError();
    }
    // Timing is disabled: these only order the two streams.
    if (cudaEventCreateWithFlags(&fork_event_, cudaEventDisableTiming) != cudaSuccess) { fork_event_ = nullptr; cudaGetLastError(); }
    if (cudaEventCreateWithFlags(&join_event_, cudaEventDisableTiming) != cudaSuccess) { join_event_ = nullptr; cudaGetLastError(); }
    return status_ = cudaSuccess;
}

cudaError_t CudaLocalityController::release() noexcept {
    // Two states in which none of the cleanup below is legal, and in which attempting it
    // makes things worse rather than better:
    //
    //  1. The compute stream is still capturing. cudaMalloc/cudaFree/cudaDeviceSetLimit are
    //     all forbidden during capture, so every call would fail AND leave an error on the
    //     caller's context for their next unrelated cudaGetLastError() to trip over.
    //     Confirmed with compute-sanitizer: destroying a controller mid-capture produced 14
    //     cudaErrorStreamCaptureUnsupported errors before this guard existed.
    //  2. The CUDA runtime is already unloading, which is where a static or long-lived
    //     member's destructor runs.
    //
    // In both cases we drop our handles without touching the driver. That leaks one scratch
    // buffer and two events in case 1, which is the lesser evil: the caller is already in a
    // broken state and a poisoned context would hide the real cause.
    bool capturing = false;
    if (compute_stream_) {
        cudaStreamCaptureStatus st = cudaStreamCaptureStatusNone;
        if (cudaStreamIsCapturing(compute_stream_, &st) == cudaSuccess)
            capturing = st != cudaStreamCaptureStatusNone;
        else cudaGetLastError();
    }
    if (capturing) ++stats_.released_during_capture;
    const bool runtime_alive = !capturing && cudaFree(nullptr) != cudaErrorCudartUnloading;
    cudaGetLastError();

    if (runtime_alive) {
        if (compute_stream_ && window_active_) clear_access_policy_window(compute_stream_);
        if (scratch_) cudaFree(scratch_);
        if (fork_event_) cudaEventDestroy(fork_event_);
        if (join_event_) cudaEventDestroy(join_event_);
        // Give the device-wide persisting-L2 set-aside back. Nothing else does: reset()
        // calls cudaCtxResetPersistingL2Cache(), which evicts persisting LINES but leaves
        // the reservation standing, so without this a process that ever built a controller
        // ran the rest of its life with a smaller L2.
        if (l2_set_aside_owned_) {
            configure_persisting_l2(device_, previous_l2_set_aside_, nullptr);
            cudaGetLastError();
        }
    }
    window_active_ = false;
    scratch_ = nullptr; scratch_count_ = 0;
    fork_event_ = nullptr; join_event_ = nullptr;
    compute_stream_ = prefetch_stream_ = nullptr;
    prefetch_fork_outstanding_ = false;
    window_pending_node_attach_ = false;
    node_attach_disabled_ = false;
    l2_set_aside_bytes_ = 0;
    previous_l2_set_aside_ = 0;
    l2_set_aside_owned_ = false;
    status_ = cudaErrorNotPermitted;
    return cudaSuccess;
}

CudaLocalityController::~CudaLocalityController() { release(); }

cudaError_t CudaLocalityController::bind_streams(cudaStream_t compute_stream, cudaStream_t prefetch_stream) noexcept {
    if (status_ != cudaSuccess) return status_;
    if (!compute_stream || !prefetch_stream) return cudaErrorInvalidValue;
    // Same stream would serialise the pre-touch into the critical path and quietly turn
    // "prefetch" into "the same work, earlier".
    if (compute_stream == prefetch_stream) return cudaErrorInvalidValue;
    compute_stream_=compute_stream; prefetch_stream_=prefetch_stream;
    // Recorded, not enforced: a prefetch stream at higher priority than compute will
    // steal SMs from the kernel it is trying to help.
    cudaStreamGetPriority(compute_stream_, &stats_.compute_stream_priority);
    cudaStreamGetPriority(prefetch_stream_, &stats_.prefetch_stream_priority);
    return cudaSuccess;
}

bool CudaLocalityController::graph_capture_active() const noexcept {
    if (!compute_stream_) return false;
    cudaStreamCaptureStatus capture = cudaStreamCaptureStatusNone;
    if (cudaStreamIsCapturing(compute_stream_, &capture) != cudaSuccess) { cudaGetLastError(); return false; }
    // Only Active is a capture we can still contribute to. Invalidated means the capture is
    // already doomed and will fail at cudaStreamEndCapture; treating it as active kept us
    // deferring windows and forking streams into a graph that can never be instantiated.
    return capture == cudaStreamCaptureStatusActive;
}


// Window placement, hot-set accounting and the capture decision: identical for a runtime
// holding one state and for one holding a state per sequence. Only what gets pre-touched
// differs, so only that is duplicated below.
LayerPlan CudaLocalityController::plan_and_window(const StateSegment* current, int current_count,
                                                 bool has_next_recurrent_layer,
                                                 const RecurrentGeometry& geometry,
                                                 LayerActions* actions,
                                                 cudaError_t* err) noexcept {
    *err = cudaSuccess;
    const auto& cfg = planner_.config();
    const StateSegment* chosen = select_window_segment(current, current_count, cfg.window_target);
    WindowRegion region{};
    if (chosen) region = resolve_window_region(*chosen, cfg.window_scope, cfg.prefetch_distance);

    const auto plan = planner_.plan_for_layer(region.bytes, has_next_recurrent_layer,
                                              geometry, layer_index_);
    ++layer_index_;
    const bool capturing = graph_capture_active();
    ++stats_.layers;
    // Count what the name says. This used to key off hit_ratio_reduced, which
    // HotSetPolicy::Fixed never sets - so the counter read zero under precisely the policy
    // that ignores oversubscription hardest, and a run could look healthy while every layer
    // was over budget.
    if (plan.hot_set_oversubscribed) ++stats_.hot_set_oversubscribed;
    if (plan.hit_ratio_reduced) ++stats_.hit_ratio_reduced;
    if (actions) {
        *actions = LayerActions{};
        actions->plan = plan;
        actions->window_region = region.ptr;
        actions->window_region_bytes = plan.hot_window_bytes;
    }
    window_pending_node_attach_ = false;

    const bool want_window = plan.use_persisting_window && region.ptr && l2_set_aside_bytes_ > 0;
    if (want_window && capturing) {
        ++stats_.windows_deferred_to_caller;
        pending_window_ = make_access_policy_window(plan, const_cast<void*>(region.ptr));
        // Only arm the node attach when the caller asked for it. Under WindowAttach::Stream
        // the window is handed back and nothing of the runtime's is touched.
        window_pending_node_attach_ = planner_.config().window_attach != WindowAttach::Stream
                                   && !node_attach_disabled_;
        if (actions) {
            actions->window_requires_launch_attribute = true;
            actions->window = pending_window_;
        }
    } else if (want_window) {
        *err = set_access_policy_window(compute_stream_, const_cast<void*>(region.ptr),
                                       plan.hot_window_bytes, plan.hit_ratio);
        if (*err == cudaSuccess) {
            window_active_ = true;
            ++stats_.windows_applied;
            if (actions) {
                actions->window_applied_to_stream = true;
                actions->window = make_access_policy_window(plan, const_cast<void*>(region.ptr));
            }
        }
    } else if (window_active_) {
        *err = clear_access_policy_window(compute_stream_);
        if (*err == cudaSuccess) { window_active_ = false; ++stats_.windows_cleared; }
    }
    return plan;
}

cudaError_t CudaLocalityController::before_layer(const StateSegment* window_segments, int window_count,
                                                 const RowSet* next_rows, int next_row_sets,
                                                 bool has_next_recurrent_layer,
                                                 const RecurrentGeometry& geometry,
                                                 LayerActions* actions) noexcept {
    if (status_ != cudaSuccess) return status_;
    if (!compute_stream_ || !prefetch_stream_) return cudaErrorInvalidResourceHandle;

    cudaError_t err = cudaSuccess;
    const auto plan = plan_and_window(window_segments, window_count, has_next_recurrent_layer,
                                      geometry, actions, &err);
    if (err != cudaSuccess) return err;
    const bool capturing = graph_capture_active();

    if (!plan.prefetch_next || !next_rows || next_row_sets <= 0 || !scratch_ || !scratch_count_) {
        if (plan.prefetch_next) ++stats_.pre_touch_skipped;
        return cudaSuccess;
    }
    bool forked = false;
    for (int i = 0; i < next_row_sets; ++i) {
        const RowSet& set = next_rows[i];
        if (!set.device_bases || set.rows <= 0 || !set.bytes ||
            !pre_touch_covers(plan.pre_touch_coverage, set.kind)) continue;
        if (!forked && fork_event_) {
            auto e = cudaEventRecord(fork_event_, compute_stream_); if (e != cudaSuccess) return e;
            e = cudaStreamWaitEvent(prefetch_stream_, fork_event_, 0); if (e != cudaSuccess) return e;
            // The instant the prefetch stream joins the capture, a join is OWED. Set the flag
            // here, not after the loop: any early return between the fork and the end of the
            // loop would otherwise leave the capture with an unjoined fork and no record that
            // one is outstanding, and end_sequence() would emit nothing. The host runtime's
            // cudaStreamEndCapture then fails with cudaErrorStreamCaptureUnjoined and it may
            // proceed with a null graph.
            prefetch_fork_outstanding_ = true;
        }
        forked = true;
        auto e = pre_touch_rows_async(plan.pre_touch, set.device_bases, set.rows,
                                      set.byte_offset, set.bytes,
                                      scratch_, scratch_count_, prefetch_stream_);
        if (e != cudaSuccess) return e;
        ++stats_.pre_touch_segments;
        stats_.pre_touch_bytes += set.bytes * static_cast<std::uint64_t>(set.rows);
        if (actions) { ++actions->pre_touched_segments; actions->pre_touched_bytes += set.bytes * set.rows; }
    }
    if (!forked) { ++stats_.pre_touch_skipped; return cudaSuccess; }
    ++stats_.pre_touch_launches;
    prefetch_fork_outstanding_ = true;
    if (capturing && join_event_ && planner_.config().prefetch_join == PrefetchJoin::PerLayer) {
        auto e = cudaEventRecord(join_event_, prefetch_stream_); if (e != cudaSuccess) return e;
        e = cudaStreamWaitEvent(compute_stream_, join_event_, 0); if (e != cudaSuccess) return e;
        prefetch_fork_outstanding_ = false;
    }
    return cudaSuccess;
}

cudaError_t CudaLocalityController::before_layer(const StateSegment* current, int current_count,
                                                 const StateSegment* next, int next_count,
                                                 bool has_next_recurrent_layer,
                                                 const RecurrentGeometry& geometry,
                                                 LayerActions* actions) noexcept {
    if (status_ != cudaSuccess) return status_;
    if (!compute_stream_ || !prefetch_stream_) return cudaErrorInvalidResourceHandle;

    cudaError_t err = cudaSuccess;
    const auto plan = plan_and_window(current, current_count, has_next_recurrent_layer,
                                      geometry, actions, &err);
    if (err != cudaSuccess) return err;
    const bool capturing = graph_capture_active();

    if (!plan.prefetch_next || !next || next_count <= 0 || !scratch_ || !scratch_count_) {
        if (plan.prefetch_next) ++stats_.pre_touch_skipped;
        return cudaSuccess;
    }

    // One fork and one join for the whole layer, however many segments are walked: an
    // event pair per segment would cost more ordering than the second walk is worth.
    bool forked = false;
    for (int i = 0; i < next_count; ++i) {
        const StateSegment& seg = next[i];
        if (!seg.ptr || !seg.bytes || !pre_touch_covers(plan.pre_touch_coverage, seg.kind)) continue;
        if (!forked && fork_event_) {
            auto e = cudaEventRecord(fork_event_, compute_stream_); if (e != cudaSuccess) return e;
            e = cudaStreamWaitEvent(prefetch_stream_, fork_event_, 0); if (e != cudaSuccess) return e;
            // The instant the prefetch stream joins the capture, a join is OWED. Set the flag
            // here, not after the loop: any early return between the fork and the end of the
            // loop would otherwise leave the capture with an unjoined fork and no record that
            // one is outstanding, and end_sequence() would emit nothing. The host runtime's
            // cudaStreamEndCapture then fails with cudaErrorStreamCaptureUnjoined and it may
            // proceed with a null graph.
            prefetch_fork_outstanding_ = true;
        }
        forked = true;
        auto e = pre_touch_bytes_async(plan.pre_touch, seg.ptr, seg.bytes,
                                       scratch_, scratch_count_, prefetch_stream_);
        if (e != cudaSuccess) return e;
        ++stats_.pre_touch_segments;
        stats_.pre_touch_bytes += seg.bytes;
        if (actions) { ++actions->pre_touched_segments; actions->pre_touched_bytes += seg.bytes; }
    }
    if (!forked) { ++stats_.pre_touch_skipped; return cudaSuccess; }
    ++stats_.pre_touch_launches;
    prefetch_fork_outstanding_ = true;
    // Under TokenEnd the join is deferred to end_sequence(): one pair of graph nodes per
    // token instead of one per recurrent layer. The pre-touch then has no ordering against
    // the layer it warms, which is the trade being measured.
    if (capturing && join_event_ && planner_.config().prefetch_join == PrefetchJoin::PerLayer) {
        // An unjoined fork ends the capture invalid, so this is not optional under capture.
        auto e = cudaEventRecord(join_event_, prefetch_stream_); if (e != cudaSuccess) return e;
        e = cudaStreamWaitEvent(compute_stream_, join_event_, 0); if (e != cudaSuccess) return e;
        prefetch_fork_outstanding_ = false;
    }
    return cudaSuccess;
}

// Stream capture records kernel launches as graph nodes and hands back the nodes the last
// launch appended as the current capture dependencies. Setting the access-policy attribute
// on those nodes is what makes a persisting window survive into every graph replay -- and
// it needs no change to how the runtime launches its kernel.
cudaError_t CudaLocalityController::declare_geometry(const RecurrentGeometry& geometry) noexcept {
    if (status_ != cudaSuccess) return status_;
    if (planner_.config().set_aside_policy == SetAsidePolicy::Fixed) return cudaSuccess;

    const auto wanted = planner_.recommended_l2_set_aside(geometry);
    // Against the last REQUEST, not against what the driver granted. cudaDeviceSetLimit
    // rounds up - on an RTX 5090 a 15 MiB request comes back as 18 - so comparing a fresh
    // request against the granted bytes never matches for any target that is not exactly the
    // device maximum, and the L2 partition would be re-carved on every single token.
    if (wanted == l2_set_aside_requested_) { settle_count_ = 0; return cudaSuccess; }

    // A workload does not always declare the same geometry on every token. SparkInfer's
    // concurrent decode packs most tokens and runs the tail chunk unpacked, and the packed
    // path compacts the matrix state to bf16 - so the declared bytes per layer ALTERNATE, and
    // a policy that acted on each declaration would re-carve the device's L2 partition
    // several times a second, evicting on every change exactly the state it had just kept.
    // Measured on the MoE checkpoint at four sequences: 64 of 80 tokens packed, 16 not.
    //
    // So a new target has to hold for `kSettleTokens` consecutive declarations before it is
    // acted on. A real workload change settles in eight tokens; an alternation never wins.
    if (wanted != settle_target_) { settle_target_ = wanted; settle_count_ = 1; return cudaSuccess; }
    if (++settle_count_ < kSettleTokens) return cudaSuccess;

    // Never resize the device's L2 partition from inside a capture. It is not a stream
    // operation, it would not be recorded, and it would take effect at a moment that has
    // nothing to do with the replay it is supposed to serve.
    if (compute_stream_) {
        cudaStreamCaptureStatus capture = cudaStreamCaptureStatusNone;
        if (cudaStreamIsCapturing(compute_stream_, &capture) != cudaSuccess) {
            cudaGetLastError();
        } else if (capture != cudaStreamCaptureStatusNone) {
            return cudaSuccess;
        }
    }

    if (!wanted) {
        // The policy declined outright: give the reservation back rather than holding L2 the
        // plan will not use. `persist` then costs the hook and nothing else, which is the
        // honest thing for a policy that has decided this workload cannot pay.
        if (l2_set_aside_owned_) {
            if (configure_persisting_l2(device_, previous_l2_set_aside_, nullptr) != cudaSuccess)
                cudaGetLastError();
            l2_set_aside_owned_ = false;
        }
        l2_set_aside_bytes_ = 0;
        l2_set_aside_requested_ = 0;
        settle_count_ = 0;
        planner_.set_granted_l2_set_aside(0);
        return cudaSuccess;
    }

    if (!l2_set_aside_owned_ &&
        cudaDeviceGetLimit(&previous_l2_set_aside_, cudaLimitPersistingL2CacheSize) != cudaSuccess) {
        previous_l2_set_aside_ = 0; cudaGetLastError();
    }
    std::size_t granted = 0;
    if (configure_persisting_l2(device_, wanted, &granted) != cudaSuccess) {
        cudaGetLastError();
        return cudaSuccess;   // best-effort: a cache hint must never fail a decode step
    }
    l2_set_aside_owned_ = true;
    l2_set_aside_bytes_ = granted;
    l2_set_aside_requested_ = wanted;
    if (granted > l2_set_aside_peak_) l2_set_aside_peak_ = granted;
    settle_count_ = 0;
    // Budget the hot set against what the driver GAVE, not what we asked for. Skipping this
    // is what made every oversubscription decision wrong by the driver's rounding.
    planner_.set_granted_l2_set_aside(granted);
    return cudaSuccess;
}

cudaError_t CudaLocalityController::attach_window_to_captured_node() noexcept {
    if (status_ != cudaSuccess) return status_;
    if (!window_pending_node_attach_ || !compute_stream_) return cudaSuccess;
    window_pending_node_attach_ = false;

    cudaStreamCaptureStatus capture = cudaStreamCaptureStatusNone;
    unsigned long long id = 0;
    const cudaGraphNode_t* deps = nullptr;
    size_t dep_count = 0;
    cudaGraph_t graph = nullptr;
    // CUDA 13 dropped the _v2 spelling and redefined the plain name to the edge-data form,
    // which takes one more parameter. Both toolkits build from this one source.
#if CUDART_VERSION >= 13000
    const cudaGraphEdgeData* edges = nullptr;
    auto e = cudaStreamGetCaptureInfo(compute_stream_, &capture, &id, &graph, &deps, &edges, &dep_count);
#else
    auto e = cudaStreamGetCaptureInfo_v2(compute_stream_, &capture, &id, &graph, &deps, &dep_count);
#endif
    if (e != cudaSuccess || capture != cudaStreamCaptureStatusActive || !deps || !dep_count) {
        if (e != cudaSuccess) cudaGetLastError();
        ++stats_.window_attach_failures;
        return cudaSuccess;  // best-effort: never fail a decode step over a cache hint
    }

    // How many of the pending dependencies are kernel nodes. The hook fires immediately
    // after the runtime's own recurrent launch, so in the common case this is one -- but not
    // always: on the non-fused convolution branch the preceding launch is l2_norm_qk_kernel,
    // which reads no recurrent state, and marking it spends set-aside on memory it never
    // touches. Counting first is what lets Strict decline that case instead of guessing.
    std::size_t kernel_nodes = 0;
    for (size_t i = 0; i < dep_count; ++i) {
        cudaGraphNodeType type{};
        if (cudaGraphNodeGetType(deps[i], &type) != cudaSuccess) { cudaGetLastError(); continue; }
        if (type == cudaGraphNodeTypeKernel) ++kernel_nodes;
    }
    if (planner_.config().window_attach == WindowAttach::CaptureNodeStrict && kernel_nodes != 1) {
        ++stats_.window_attach_ambiguous;
        return cudaSuccess;   // a window on the wrong kernel is worse than no window
    }

    cudaKernelNodeAttrValue attr{};
    attr.accessPolicyWindow = pending_window_;
    bool attached = false;
    for (size_t i = 0; i < dep_count; ++i) {
        cudaGraphNodeType type{};
        if (cudaGraphNodeGetType(deps[i], &type) != cudaSuccess) { cudaGetLastError(); continue; }
        if (type != cudaGraphNodeTypeKernel) continue;
        if (cudaGraphKernelNodeSetAttribute(deps[i], cudaKernelNodeAttributeAccessPolicyWindow,
                                            &attr) != cudaSuccess) { cudaGetLastError(); continue; }
        attached = true;
        ++stats_.window_nodes_attached;
    }
    if (attached) ++stats_.windows_attached_to_node; else ++stats_.window_attach_failures;

    // Setting an attribute on a node of a graph that is still being captured is not a
    // documented operation, and it does not always survive. cudaStreamIsCapturing reports
    // an invalidated capture, which is the only way to find out before the runtime's
    // cudaStreamEndCapture fails and it falls back to a slower path - by which point the
    // throughput number for that run is measuring the fallback, not the policy.
    cudaStreamCaptureStatus after = cudaStreamCaptureStatusNone;
    if (cudaStreamIsCapturing(compute_stream_, &after) != cudaSuccess) { cudaGetLastError(); return cudaSuccess; }
    if (after == cudaStreamCaptureStatusInvalidated) {
        ++stats_.capture_invalidations;
        node_attach_disabled_ = true;   // once is evidence; twice is vandalism
    }
    return cudaSuccess;
}

cudaError_t CudaLocalityController::before_layer(void* current_state, std::size_t current_state_bytes,
                                                  const float* next_state, std::size_t next_state_count,
                                                  bool has_next_recurrent_layer,
                                                  std::size_t concurrently_hot_bytes,
                                                  LayerActions* actions) noexcept {
    if (status_ != cudaSuccess) return status_;
    if (!compute_stream_ || !prefetch_stream_) return cudaErrorInvalidResourceHandle;

    const auto plan = planner_.plan_for_layer(current_state_bytes, has_next_recurrent_layer, concurrently_hot_bytes, layer_index_);
    ++layer_index_;
    const bool capturing = graph_capture_active();
    ++stats_.layers;
    if (plan.hot_set_oversubscribed) ++stats_.hot_set_oversubscribed;
    if (plan.hit_ratio_reduced) ++stats_.hit_ratio_reduced;
    if (actions) {
        *actions = LayerActions{};
        actions->plan = plan;
        actions->window_region = current_state;
        actions->window_region_bytes = plan.hot_window_bytes;
    }
    window_pending_node_attach_ = false;

    const bool want_window = plan.use_persisting_window && l2_set_aside_bytes_ > 0;
    if (want_window && capturing) {
        // A stream access-policy window is host-side state, not a captured operation, so
        // setting it here would leave the graph without it and the hint would vanish for
        // every replay. Hand the window back instead; the caller attaches it to the
        // kernel launch config or graph node, which is the only path a graph records.
        ++stats_.windows_deferred_to_caller;
        pending_window_ = make_access_policy_window(plan, current_state);
        // Same rule as the segment path: node mutation is opt-in. Without this check the
        // documented-safe WindowAttach::Stream default was silently ignored on this
        // overload, and a consumer that asked for the safe path still got a graph mutated
        // mid-capture.
        window_pending_node_attach_ = planner_.config().window_attach != WindowAttach::Stream
                                   && !node_attach_disabled_;
        if (actions) {
            actions->window_requires_launch_attribute = true;
            actions->window = pending_window_;
        }
    } else if (want_window) {
        auto e = set_access_policy_window(compute_stream_, current_state, plan.hot_window_bytes, plan.hit_ratio);
        if (e != cudaSuccess) return e;
        window_active_ = true;
        ++stats_.windows_applied;
        if (actions) { actions->window_applied_to_stream = true; actions->window = make_access_policy_window(plan, current_state); }
    } else if (window_active_) {
        auto e = clear_access_policy_window(compute_stream_);
        if (e != cudaSuccess) return e;
        window_active_ = false;
        ++stats_.windows_cleared;
    }

    const bool can_pre_touch = plan.prefetch_next && next_state && next_state_count && scratch_ && scratch_count_;
    if (plan.prefetch_next && !can_pre_touch) ++stats_.pre_touch_skipped;
    if (can_pre_touch) {
        // Fork the prefetch stream off the compute stream so the pre-touch is ordered
        // behind the state writes it reads, instead of racing them. Under graph capture
        // the fork must also be rejoined or the capture ends invalid.
        if (fork_event_) {
            auto e = cudaEventRecord(fork_event_, compute_stream_); if (e != cudaSuccess) return e;
            e = cudaStreamWaitEvent(prefetch_stream_, fork_event_, 0); if (e != cudaSuccess) return e;
            prefetch_fork_outstanding_ = true;   // a join is owed from here on; see above
        }
        auto e = pre_touch_async(plan.pre_touch, next_state, next_state_count, scratch_, scratch_count_, prefetch_stream_);
        if (e != cudaSuccess) return e;
        ++stats_.pre_touch_launches;
        ++stats_.pre_touch_segments;
        stats_.pre_touch_bytes += static_cast<std::uint64_t>(next_state_count) * sizeof(float);
        if (actions) { actions->pre_touched_segments = 1; actions->pre_touched_bytes = next_state_count * sizeof(float); }
        prefetch_fork_outstanding_ = true;
        if (capturing && join_event_ && planner_.config().prefetch_join == PrefetchJoin::PerLayer) {
            e = cudaEventRecord(join_event_, prefetch_stream_); if (e != cudaSuccess) return e;
            e = cudaStreamWaitEvent(compute_stream_, join_event_, 0); if (e != cudaSuccess) return e;
            prefetch_fork_outstanding_ = false;
        }
    }
    return cudaSuccess;
}

// The window is scoped to the recurrent layer. Leaving it bound would apply
// persisting/streaming policy to every following non-recurrent kernel on the same
// stream, against a state pointer those layers never touch.
void CudaLocalityController::begin_sequence() noexcept { layer_index_ = 0; }

cudaError_t CudaLocalityController::end_sequence() noexcept {
    if (status_ != cudaSuccess) return status_;
    if (!prefetch_fork_outstanding_) return cudaSuccess;
    prefetch_fork_outstanding_ = false;
    if (!join_event_ || !compute_stream_ || !prefetch_stream_) return cudaSuccess;
    auto e = cudaEventRecord(join_event_, prefetch_stream_); if (e != cudaSuccess) return e;
    return cudaStreamWaitEvent(compute_stream_, join_event_, 0);
}

cudaError_t CudaLocalityController::after_layer() noexcept {
    if (status_ != cudaSuccess) return status_;
    if (!window_active_) return cudaSuccess;
    auto e = clear_access_policy_window(compute_stream_);
    if (e == cudaSuccess) { window_active_ = false; ++stats_.windows_cleared; }
    return e;
}

cudaError_t CudaLocalityController::before_streaming_region(void* ptr, std::size_t bytes) noexcept {
    if (status_ != cudaSuccess) return status_;
    if (!compute_stream_) return cudaErrorInvalidResourceHandle;
    if (!ptr || !bytes) return cudaErrorInvalidValue;
    if (graph_capture_active()) return cudaSuccess;  // caller owns the node attribute
    // The device caps an access-policy window (128 MiB on an RTX 5090) and rejects anything
    // larger outright. The persisting path already clamps; a weight buffer is exactly the
    // sort of region that exceeds it, so hinting the leading window is right and failing is
    // not.
    const auto limit = planner_.caps().access_policy_max_window_bytes;
    if (limit) bytes = std::min(bytes, limit);
    auto e = set_streaming_policy_window(compute_stream_, ptr, bytes);
    if (e != cudaSuccess) return e;
    window_active_ = true;
    ++stats_.streaming_windows_applied;
    return cudaSuccess;
}

cudaError_t CudaLocalityController::after_streaming_region() noexcept { return after_layer(); }

cudaError_t CudaLocalityController::reset() noexcept {
    if (status_ != cudaSuccess) return status_;
    if (compute_stream_ && window_active_) {
        auto e = clear_access_policy_window(compute_stream_);
        if (e != cudaSuccess) return e;
        window_active_ = false; ++stats_.windows_cleared;
    }
    // Outstanding pre-touch reads the caller's state buffer; let them finish before the
    // caller is free to release it.
    if (prefetch_stream_ && !graph_capture_active()) {
        auto e = cudaStreamSynchronize(prefetch_stream_);
        if (e != cudaSuccess) return e;
    }
    // The header promises this "releases the L2 set-aside", and until now it did not: it
    // only evicted persisting LINES. Honour the documented contract - a caller that calls
    // reset() is telling us it is done with the device, and leaving a device-wide
    // reservation behind after that is a leak with a performance cost for everyone else.
    auto e = cudaCtxResetPersistingL2Cache();
    if (l2_set_aside_owned_) {
        const auto restore = configure_persisting_l2(device_, previous_l2_set_aside_, nullptr);
        if (e == cudaSuccess) e = restore;
        l2_set_aside_owned_ = false;
        l2_set_aside_bytes_ = 0;
    }
    // Let go of the caller's streams. They are BORROWED - the runtime owns them and is free
    // to destroy them the moment it has told us it is done, which is what reset() means.
    // Holding them past that point is how the destructor came to call cudaStreamIsCapturing
    // on a destroyed stream and segfault inside libcuda at process exit. After reset() the
    // controller holds nothing of the caller's; initialize() and bind_streams() revive it.
    compute_stream_ = prefetch_stream_ = nullptr;
    window_active_ = false;
    prefetch_fork_outstanding_ = false;
    window_pending_node_attach_ = false;
    return e;
}

} // namespace recurlocal
#endif
