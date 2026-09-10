#pragma once
#include <cstddef>
#include <cstdint>
#include "recurlocal/planner.h"

#if defined(RECURLOCAL_WITH_CUDA) || defined(RECURLLOCAL_WITH_CUDA)
#include <cuda_runtime_api.h>
namespace recurlocal {

// Non-throwing capability query. Preferred at an integration boundary: a runtime built
// with -fno-exceptions, or driven from a Python binding, cannot take a throw here.
cudaError_t query_device_caps(int device, DeviceCaps* out) noexcept;
// Throwing convenience for tools and tests.
DeviceCaps query_device_caps(int device);

cudaError_t configure_persisting_l2(int device, std::size_t requested_bytes, std::size_t* actual_bytes = nullptr) noexcept;
cudaError_t set_access_policy_window(cudaStream_t stream, void* ptr, std::size_t bytes, double hit_ratio) noexcept;
cudaError_t clear_access_policy_window(cudaStream_t stream) noexcept;
// Marks a region as streaming: it should pass through the cache without displacing
// anything that wants to stay. The counterpart to the persisting window, for the model
// weights and activations that share L2 with recurrent state.
cudaError_t set_streaming_policy_window(cudaStream_t stream, void* ptr, std::size_t bytes) noexcept;
// Strategy-selecting form; the 5-argument overload keeps the v0.1 scalar behaviour.
cudaError_t pre_touch_async(PreTouchStrategy strategy, const float* ptr, std::size_t count,
                            float* scratch, std::size_t scratch_count, cudaStream_t stream) noexcept;
cudaError_t pre_touch_async(const float* ptr, std::size_t count, float* scratch, std::size_t scratch_count, cudaStream_t stream) noexcept;
// Byte-oriented form. A hybrid model's recurrent state is not all fp32: Qwen3.8-27B's
// convolution window is bf16, and a pre-touch that could only be expressed in floats
// could not touch it at all. Nothing is interpreted, only read, so the element type is
// irrelevant to the walk.
cudaError_t pre_touch_bytes_async(PreTouchStrategy strategy, const void* ptr, std::size_t bytes,
                                  float* scratch, std::size_t scratch_count, cudaStream_t stream) noexcept;
// Row-major form, for concurrent decode. `device_bases` is a DEVICE array of `rows` base
// pointers - the shape a runtime already builds to drive its own batched kernels - and every
// row is walked from the same byte offset for the same length, in ONE launch. A launch per
// row would add `rows` kernel nodes per layer to a captured decode graph, which is what
// makes the obvious implementation unusable at the concurrency it exists for.
cudaError_t pre_touch_rows_async(PreTouchStrategy strategy, const void* const* device_bases,
                                 int rows, std::size_t byte_offset, std::size_t bytes,
                                 float* scratch, std::size_t scratch_count,
                                 cudaStream_t stream) noexcept;

// The window a plan asks for, as a value the caller can attach to its own launch
// (cudaLaunchAttributeAccessPolicyWindow) or to a graph kernel node
// (cudaKernelNodeAttributeAccessPolicyWindow). Under CUDA Graph capture that is the
// only path that works, because a stream attribute is host-side state and is not
// recorded into the graph.
cudaAccessPolicyWindow make_access_policy_window(const LayerPlan& plan, void* ptr) noexcept;

// What the controller did, and what the caller still has to do, for one layer.
struct LayerActions {
    LayerPlan plan{};
    // The controller installed the window on the compute stream; nothing more to do.
    bool window_applied_to_stream = false;
    // Graph capture is active. The window below was NOT applied and must be attached
    // by the caller to the recurrent kernel's launch config or graph node, or the
    // persisting hint is silently lost for the whole captured decode path.
    bool window_requires_launch_attribute = false;
    cudaAccessPolicyWindow window{};
    // The region the window covers, after WindowScope/WindowTarget resolution. Reported
    // because "which bytes did the policy actually protect" is otherwise invisible, and
    // under Allocation scope it is deliberately not the slice that was passed in.
    const void* window_region = nullptr;
    std::size_t window_region_bytes = 0;
    // Segments the pre-touch walked and their total size, under PreTouchCoverage.
    int pre_touched_segments = 0;
    std::size_t pre_touched_bytes = 0;
};

// Counters an integrator can read to explain why a policy is or is not helping.
// Cheap host-side counters only; no device synchronisation.
struct ControllerStats {
    std::uint64_t layers = 0;
    std::uint64_t windows_applied = 0;
    std::uint64_t windows_cleared = 0;
    std::uint64_t windows_deferred_to_caller = 0;  // graph capture was active
    std::uint64_t hot_set_oversubscribed = 0;      // the hot set exceeded the L2 budget
    std::uint64_t hit_ratio_reduced = 0;           // ...and the policy responded by backing off
    std::uint64_t pre_touch_launches = 0;
    std::uint64_t pre_touch_bytes = 0;
    std::uint64_t pre_touch_skipped = 0;           // no scratch, or no next state
    std::uint64_t streaming_windows_applied = 0;
    // Attach CALLS that marked at least one captured graph kernel node -- not the number of
    // nodes marked, which is `window_nodes_attached`. Under graph decode this is the only
    // count that can be non-zero for a persisting policy: a stream attribute is not recorded
    // into a graph, so windows_applied stays at 0 there however the policy is configured. A
    // run with windows_deferred_to_caller > 0 and this at 0 measured no persisting policy at
    // all, whatever the mode was called.
    std::uint64_t windows_attached_to_node = 0;
    std::uint64_t window_attach_failures = 0;
    // Captures that cudaStreamIsCapturing reported as INVALIDATED right after a node
    // attach. Non-zero means the locality layer broke the runtime's graph: whatever the
    // throughput number for that run was, it is measuring the runtime's fallback path.
    // The controller latches node attachment off after the first one.
    std::uint64_t capture_invalidations = 0;
    std::uint64_t pre_touch_segments = 0;
    // The controller was torn down while its compute stream was still capturing. That is a
    // caller bug (an exception unwinding mid-capture, usually), but it is one the library
    // must not make worse by spraying failing CUDA calls into the caller's context.
    std::uint64_t released_during_capture = 0;
    int compute_stream_priority = 0;
    int prefetch_stream_priority = 0;
    // Appended, not inserted. Fields are added at the END of this struct and nowhere else:
    // inserting one shifts every offset after it, and a consumer built against the older
    // header reads the wrong member with nothing in either build to notice. See
    // docs/STABILITY.md; the static_assert below is what makes the rule enforceable.
    //
    // Graph kernel NODES marked. Equal to windows_attached_to_node only when every attach
    // found exactly one kernel node pending; larger means the capture's dependency set held
    // more than one, and the window went onto kernels the hook did not fire for. The two
    // were reported as one number, and "48 of 48 nodes" was read off a counter counting calls.
    std::uint64_t window_nodes_attached = 0;
    // Attaches WindowAttach::CaptureNodeStrict declined because more than one kernel node was
    // pending, so which one the hook fired for was not decidable. Non-zero here, next to a
    // measurable difference against CaptureNode, is the size of the mis-attachment.
    std::uint64_t window_attach_ambiguous = 0;
};

// Thread-safety: one controller instance is not synchronised. A runtime decoding several
// sequences concurrently should hold one controller per compute stream, which is also the
// only way the hot-set accounting can be meaningful.
class CudaLocalityController {
public:
    CudaLocalityController() noexcept = default;
    // Never throws: construction failure is reported by status(), so this is safe to
    // build inside a runtime that does not use exceptions.
    CudaLocalityController(int device, PlannerConfig config) noexcept;
    ~CudaLocalityController();
    CudaLocalityController(const CudaLocalityController&) = delete;
    CudaLocalityController& operator=(const CudaLocalityController&) = delete;

    cudaError_t initialize(int device, PlannerConfig config) noexcept;
    cudaError_t status() const noexcept { return status_; }

    // The two streams must differ: a pre-touch issued on the compute stream is not a
    // prefetch, it is extra work on the critical path.
    //
    // Both streams are BORROWED: this class never destroys them and must never touch one
    // after the owner has destroyed it. Call reset() (or destroy the controller) BEFORE
    // destroying the streams you handed over — reset() drops the handles, after which the
    // controller touches neither. Getting this wrong is not a leak, it is a segfault inside
    // libcuda at whatever point the controller next probes the stream.
    cudaError_t bind_streams(cudaStream_t compute_stream, cudaStream_t prefetch_stream) noexcept;

    cudaError_t before_layer(void* current_state, std::size_t current_state_bytes,
                             const float* next_state, std::size_t next_state_count,
                             bool has_next_recurrent_layer,
                             std::size_t concurrently_hot_bytes = 0,
                             LayerActions* actions = nullptr) noexcept;

    // Multi-state form. A hybrid recurrent layer owns more than one mutable state - a
    // matrix state and a convolution window in Qwen3.5/3.8 - in separate allocations with
    // separate strides. Passing them as segments is what lets the hot set be counted over
    // all of them and the pre-touch walk all of them; the single-pointer form above can
    // only ever model the first, which is how the accounting came to be wrong.
    // `geometry` is what the planner cannot see from one layer's pointers, and without it
    // the hot-set model degrades to CurrentLayer and says so in the plan.
    cudaError_t before_layer(const StateSegment* current, int current_count,
                             const StateSegment* next, int next_count,
                             bool has_next_recurrent_layer,
                             const RecurrentGeometry& geometry,
                             LayerActions* actions = nullptr) noexcept;

    // One recurrent state, held once per concurrently decoding sequence, addressed through
    // a device-side array of per-sequence base pointers. `window_segments` is what the
    // persisting window may be placed on - host-visible allocations, typically the first
    // sequence's - while `next_rows` is what gets pre-touched, in one launch per set.
    struct RowSet {
        const void* const* device_bases = nullptr;  // DEVICE array of `rows` pointers
        int rows = 0;
        std::size_t byte_offset = 0;                // the next layer's slice within a row
        std::size_t bytes = 0;                      // slice size
        StateKind kind = StateKind::Other;
    };
    cudaError_t before_layer(const StateSegment* window_segments, int window_count,
                             const RowSet* next_rows, int next_row_sets,
                             bool has_next_recurrent_layer,
                             const RecurrentGeometry& geometry,
                             LayerActions* actions = nullptr) noexcept;

    // Attach the window the last before_layer() deferred to the kernel node(s) the
    // runtime has just recorded on the compute stream. Call immediately after launching
    // the recurrent kernel, while capture is still active.
    //
    // Production decode is captured into a CUDA graph, so a stream access-policy window is
    // set on host state the graph never records and vanishes from every replay. A graph
    // node attribute is recorded. cudaStreamGetCaptureInfo hands back the nodes the last
    // launch appended, which is enough to set the attribute on them without the runtime
    // having to convert its launch to cudaLaunchKernelEx.
    // No-op when nothing was deferred, so it is safe to call unconditionally.
    cudaError_t attach_window_to_captured_node() noexcept;

    cudaError_t after_layer() noexcept;

    // Cache QoS: bracket a non-recurrent region (attention or MoE weights streaming
    // through the same L2) so it is hinted as streaming instead of inheriting whatever
    // window the last recurrent layer left behind.
    cudaError_t before_streaming_region(void* ptr, std::size_t bytes) noexcept;
    cudaError_t after_streaming_region() noexcept;
    // Drops the window, waits for outstanding pre-touch work, and returns the device-wide
    // persisting-L2 set-aside to whatever it was before initialize() claimed it. Call this
    // when the runtime is done with locality control; the controller needs initialize()
    // again afterwards to reserve a set-aside once more.
    cudaError_t reset() noexcept;

    const LocalityPlanner& planner() const noexcept { return planner_; }
    std::size_t l2_set_aside_bytes() const noexcept { return l2_set_aside_bytes_; }
    const ControllerStats& stats() const noexcept { return stats_; }
    void reset_stats() noexcept { stats_ = ControllerStats{}; }
    // Restarts the per-layer counter the prefetch schedule is indexed by. Call at the top
    // of each token's layer walk.
    void begin_sequence() noexcept;

    // Re-size the L2 set-aside for a workload initialize() could not see.
    //
    // The set-aside is requested once, at initialize(), from PlannerConfig alone - and under
    // SetAsidePolicy::Fixed that is the whole story, because the size depends on nothing else.
    // The workload-aware policies size it from the recurrent footprint, which is not known
    // until the runtime declares its geometry, so they need somewhere to act. This is it.
    //
    // A no-op under Fixed, a no-op when the recomputed size is the one already installed, and
    // a no-op while the compute stream is capturing - cudaDeviceSetLimit is a device-wide
    // operation and a graph capture is not the place for one. Call it whenever the geometry
    // changes; calling it every token is cheap and correct.
    cudaError_t declare_geometry(const RecurrentGeometry& geometry) noexcept;
    // Closes the token's layer walk. Under PrefetchJoin::TokenEnd this is where the single
    // join is emitted, and under graph capture it is not optional: a fork that is never
    // rejoined ends the capture invalid. A no-op under PerLayer.
    cudaError_t end_sequence() noexcept;
    bool graph_capture_active() const noexcept;

private:
    cudaError_t release() noexcept;
    // Shared by both before_layer() forms: window placement and hot-set accounting do not
    // depend on whether the state is one allocation or one per sequence.
    LayerPlan plan_and_window(const StateSegment* current, int current_count,
                              bool has_next_recurrent_layer, const RecurrentGeometry& geometry,
                              LayerActions* actions, cudaError_t* err) noexcept;

    int device_ = -1;
    cudaError_t status_ = cudaErrorNotPermitted;  // "not initialised" until initialize() runs
    LocalityPlanner planner_{};
    cudaStream_t compute_stream_ = nullptr;
    cudaStream_t prefetch_stream_ = nullptr;
    cudaEvent_t fork_event_ = nullptr;
    cudaEvent_t join_event_ = nullptr;
    float* scratch_ = nullptr;
    std::size_t scratch_count_ = 0;
    std::size_t l2_set_aside_bytes_ = 0;
    // What was last ASKED for, as distinct from what the driver granted. The driver rounds
    // requests up, so a policy that re-derives its target every token needs the request to
    // compare against or it re-carves the L2 partition on every one of them.
    std::size_t l2_set_aside_requested_ = 0;
    // What the device's persisting-L2 limit was before we touched it. The limit is
    // device-wide and context-lifetime: without restoring it, merely constructing a
    // controller carves a permanent hole out of L2 for every other kernel in the process,
    // including after the controller is gone.
    std::size_t previous_l2_set_aside_ = 0;
    bool l2_set_aside_owned_ = false;
    bool window_active_ = false;
    // A pre-touch fork is outstanding on the prefetch stream and has not been rejoined.
    bool prefetch_fork_outstanding_ = false;
    // Set by before_layer() under capture, consumed by attach_window_to_captured_node().
    bool window_pending_node_attach_ = false;
    // Latched after the first capture this controller invalidated. One is a measurement;
    // continuing to do it is vandalism.
    bool node_attach_disabled_ = false;
    cudaAccessPolicyWindow pending_window_{};
    int layer_index_ = 0;
    ControllerStats stats_{};
};

} // namespace recurlocal
#endif
