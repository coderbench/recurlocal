#pragma once
#include <cstddef>
#include "recurlocal/version.h"

namespace recurlocal {

enum class LocalityMode { Baseline, Persist, Prefetch, Combined };

// How the next layer's state is touched. Each strategy must be read-only and produce the
// same (ignored) result; they differ only in how the bytes are moved, which is the point.
// Adding one is the narrowest useful contribution to this project: a new kernel, a new
// enumerator, and it is measurable on its own against every other strategy.
enum class PreTouchStrategy {
    Scalar,     // one float per thread; the v0.1 reference
    Vec4,       // 128-bit loads, a quarter of the memory instructions
    Vec4Ldcg,   // 128-bit loads cached at L2, bypassing L1
    PtxL2,      // prefetch.global.L2 instructions; no data enters registers at all
    WarpTile,   // each warp walks a contiguous tile instead of a global stride
    Partial     // touch only the leading half of the state
};

// How the prefetch distance varies with depth. A single global constant assumes every
// recurrent layer has the same amount of compute ahead of it to hide the walk behind,
// which is false in a hybrid model where attention layers sit between them.
enum class PrefetchSchedule {
    Uniform,      // the same distance everywhere; the v0.1 assumption
    Ramp,         // reach further as depth grows and more work is queued ahead
    Alternating,  // prefetch every other layer, halving pre-touch traffic
    Sparse        // prefetch every fourth layer, but twice as far
};

// How much recurrent state is counted as competing for the L2 set-aside.
//
// This is the accounting, not the policy: HotSetPolicy decides what to do once the hot set
// exceeds the budget, and it can only be as right as the number it is handed. The v0.1
// model counted the layer about to run plus whatever the caller declared, which is what
// made `persist` measure -11% at four concurrent sequences while the planner reported no
// oversubscription at all (docs/OPTIMIZATION-SURFACES.md). The reuse distance for a
// recurrent state is one whole token: layer 0's state is next read after every other layer
// has run. Anything that does not survive that interval was never resident.
enum class HotSetModel {
    CurrentLayer,    // this layer's window plus caller-declared bytes; the v0.1 control
    TokenFootprint,  // every recurrent layer's state, every sequence: what a token revisits
    ReuseWindow      // TokenFootprint plus the non-recurrent bytes streamed between two
                     // visits to the same state - everything that can evict it
};

// Which region a persisting window covers. A window over the slice of the layer about to
// run only helps if that slice is re-read before the window moves on, and under
// TokenFootprint accounting it is not: it is re-read a whole token later.
enum class WindowScope {
    Layer,       // the slice this layer will touch; the v0.1 assumption
    Allocation,  // the whole recurrent-state allocation the slice lives in
    Ahead        // this slice plus the next prefetch_distance slices
};

// Which recurrent state a persisting window protects. A hybrid model carries more than
// one: Qwen3.8-27B holds a 3 MiB fp32 matrix state and a 60 KiB bf16 convolution window per
// layer. Only one access-policy window can be bound at a time, so this is a choice, and the
// small state is the one whose whole allocation can actually fit in a set-aside.
enum class WindowTarget {
    Matrix,     // the large recurrent matrix state
    Conv,       // the small convolution window state
    Widest,     // whichever segment has the most bytes
    Narrowest   // whichever segment has the fewest
};

// Which recurrent states are pre-touched. v0.1 walked one buffer because it modelled one.
enum class PreTouchCoverage { Matrix, Conv, Both };

// How a persisting window reaches the kernel under CUDA Graph capture.
//
// A stream access-policy window is host-side state and is never recorded into a graph, so
// under capture the window has to reach the kernel some other way or the policy silently
// does not exist in any replay. There are two ways and they are not equivalent:
//
//   Stream      Apply on the stream when not capturing; under capture, hand the window back
//               in LayerActions and let the runtime attach it to its own launch
//               (cudaLaunchKernelEx) or graph node. Nothing is mutated behind the runtime's
//               back. Costs an integration change at every recurrent kernel launch site.
//
//   CaptureNode Set the attribute on the kernel node the capture just recorded, found via
//               cudaStreamGetCaptureInfo. One line at the hook site and no launch changes -
//               but it modifies a graph that is still being captured, which CUDA does not
//               document as supported. Measured: works on SparkInfer's batch-1 decode
//               capture, and INVALIDATES its capture at 32 concurrent sequences, after
//               which the runtime falls back to per-row decode and loses 28% of aggregate
//               throughput for reasons that have nothing to do with cache policy.
//
// Default is Stream. A convenience that can cost a quarter of a runtime's throughput is not
// a default; the controller counts invalidations and latches CaptureNode off after the first.
enum class WindowAttach { Stream, CaptureNode };

// When the pre-touch stream is rejoined to the compute stream.
//
// This only matters under CUDA Graph capture, which is where production decode lives: a
// fork and a join are each recorded as graph nodes, and a hybrid model has one recurrent
// layer to pre-touch for nearly every layer it runs. SparkInfer measured its own single
// fork/join pair at ~0.89% of a decode step, so paying for one per recurrent layer is a
// cost the prefetch has to earn back before it wins anything.
enum class PrefetchJoin {
    PerLayer,  // join before the next layer runs; the v0.1 shape, and the safe one
    TokenEnd   // fork per layer, join once after the whole layer walk - half the nodes,
               // at the cost of no ordering between a pre-touch and the layer it warms
};

// What to do when the recurrent state that wants to be resident exceeds the L2 set-aside.
// This is the policy the overview calls the major v0.1 frontier: the shipped heuristic is
// one line, and nothing has ever measured it against an alternative.
enum class HotSetPolicy {
    Proportional,  // scale the requested hit ratio by the budget share (the v0.1 heuristic)
    Fixed,         // ask for the full hit ratio regardless; the naive control
    Sqrt,          // back off by sqrt(share) - gentler than proportional
    Cliff          // give up the window entirely rather than thrash a shared cache
};

// A recurrent model carries more than one mutable state per layer, in separate
// allocations with separate strides. Describing them as segments is what lets the planner
// account for all of them and the controller pre-touch all of them, instead of modelling
// the largest and silently ignoring the rest.
enum class StateKind { Matrix, Conv, Other };

struct StateSegment {
    // The slice this layer reads and writes.
    const void* ptr = nullptr;
    std::size_t bytes = 0;
    // The allocation the slice belongs to. WindowScope::Allocation needs it, and it is the
    // only way to clamp a widened window to memory the runtime actually owns.
    const void* base = nullptr;
    std::size_t base_bytes = 0;
    StateKind kind = StateKind::Other;
};

// What the runtime knows about its own recurrent state, which the planner cannot derive
// from a single layer's pointers. Without it the hot-set models below have nothing to
// count and the planner falls back to CurrentLayer accounting, reporting that it did.
struct RecurrentGeometry {
    int recurrent_layers = 0;                  // layers carrying recurrent state
    std::size_t bytes_per_layer = 0;           // all segments, one layer, one sequence
    int sequences = 1;                         // sequences decoding concurrently
    std::size_t streamed_bytes_per_token = 0;  // weights/activations through L2 per token
    bool valid() const noexcept { return recurrent_layers > 0 && bytes_per_layer > 0; }
};

struct DeviceCaps {
    std::size_t l2_bytes = 0;
    std::size_t persisting_l2_max_bytes = 0;
    std::size_t access_policy_max_window_bytes = 0;
};

struct PlannerConfig {
    LocalityMode mode = LocalityMode::Combined;
    double persisting_budget_fraction = 0.75;
    double hit_ratio = 0.70;
    // How many recurrent layers ahead to pre-touch. The runtime supplies the state this
    // many layers ahead; the planner only decides whether and how far.
    int prefetch_distance = 1;
    std::size_t max_hot_window_bytes = 0;
    PreTouchStrategy pre_touch = PreTouchStrategy::Vec4;
    HotSetPolicy hot_set_policy = HotSetPolicy::Proportional;
    PrefetchSchedule prefetch_schedule = PrefetchSchedule::Uniform;
    // Default to the v0.1 accounting so an existing caller's numbers do not move under it;
    // the corrected models are opted into and measured against this control.
    HotSetModel hot_set_model = HotSetModel::CurrentLayer;
    WindowScope window_scope = WindowScope::Layer;
    WindowTarget window_target = WindowTarget::Matrix;
    PreTouchCoverage pre_touch_coverage = PreTouchCoverage::Matrix;
    PrefetchJoin prefetch_join = PrefetchJoin::PerLayer;
    WindowAttach window_attach = WindowAttach::Stream;
    // Floor the backing-off policies apply; below this a window is not worth asking for.
    double min_hit_ratio = 0.05;
};

// Largest prefetch distance the planner will accept. Beyond a few layers the state is
// evicted again before its layer runs, so this is a sanity bound, not a policy.
inline constexpr int kMaxPrefetchDistance = 8;

// A hybrid model's recurrent layers are not contiguous: Qwen3.5/3.8 run
// `full_attn_interval - 1` recurrent layers then one full-attention layer. So "the layer N
// ahead" and "the recurrent layer N ahead" are different questions, and only the second one
// names state worth pre-touching. Getting it wrong warms the wrong buffer and nothing says
// so - the run is still correct, just pointless - which is why this is here and tested
// rather than open-coded in each adapter.
bool is_recurrent_layer(int layer, int full_attn_interval) noexcept;

// Absolute index of the recurrent layer `distance` recurrent layers after `layer`, or -1
// when the model runs out first.
int recurrent_layer_ahead(int layer, int distance, int layer_count,
                          int full_attn_interval) noexcept;

// Which of a layer's states a persisting window should go on, and how far it should reach.
// Pure address arithmetic and policy, deliberately on the CPU side: the v0.1 equivalent
// lived inside the CUDA controller and so could never be tested without a GPU, which is
// how a window that ran past the end of an allocation would have gone unnoticed.
// Returns nullptr when no segment is usable.
const StateSegment* select_window_segment(const StateSegment* segments, int count,
                                          WindowTarget target) noexcept;

struct WindowRegion {
    const void* ptr = nullptr;
    std::size_t bytes = 0;
};
// Never extends past the allocation the caller declared: a window over memory the runtime
// does not own is not a hint, it is a bug with a performance counter.
WindowRegion resolve_window_region(const StateSegment& segment, WindowScope scope,
                                   int prefetch_distance) noexcept;

bool pre_touch_covers(PreTouchCoverage coverage, StateKind kind) noexcept;

// nullptr when the config is usable, otherwise a static description of what is wrong.
// Lets an integrating runtime reject a bad config without provoking an exception.
const char* validate(const PlannerConfig& config) noexcept;

struct LayerPlan {
    std::size_t hot_window_bytes = 0;
    double hit_ratio = 0.0;
    bool use_persisting_window = false;
    bool prefetch_next = false;
    // Which layer ahead the caller should hand over when prefetch_next is set.
    int prefetch_distance = 0;
    PreTouchStrategy pre_touch = PreTouchStrategy::Scalar;
    // The hot set exceeded the L2 budget, whatever the policy then chose to do about it.
    bool hot_set_oversubscribed = false;
    // The requested hit ratio was cut because the hot set exceeded the L2 budget.
    // Surfaced because a persistently oversubscribed decode loop is the most likely
    // reason a locality policy fails to help, and it is otherwise invisible.
    bool hit_ratio_reduced = false;
    // The accounting behind the two flags above, reported so a null result can be read
    // rather than guessed at: how many bytes were counted, against what budget, and by
    // which model. `hot_set_model` is the model that was actually applied, which is
    // CurrentLayer whenever the caller supplied no geometry to apply a better one to.
    std::size_t hot_set_bytes = 0;
    std::size_t hot_set_budget_bytes = 0;
    HotSetModel hot_set_model = HotSetModel::CurrentLayer;
    // Resolved by the planner, applied by the controller, which owns the pointers.
    WindowScope window_scope = WindowScope::Layer;
    WindowTarget window_target = WindowTarget::Matrix;
    PreTouchCoverage pre_touch_coverage = PreTouchCoverage::Matrix;
};

class LocalityPlanner {
public:
    LocalityPlanner() noexcept = default;
    LocalityPlanner(DeviceCaps caps, PlannerConfig config);
    const DeviceCaps& caps() const noexcept { return caps_; }
    const PlannerConfig& config() const noexcept { return config_; }
    std::size_t recommended_l2_set_aside() const noexcept;
    // What the driver actually granted, once someone has asked it. cudaDeviceSetLimit does
    // not promise to honour the request: on an RTX 5090 a 15 MiB request comes back as
    // 18 MiB, and the device already has a non-zero default before anyone asks. Budgeting
    // the hot set against the number we WANTED rather than the one we GOT makes every
    // oversubscription decision wrong by the rounding. Zero means "not measured yet, fall
    // back to the recommendation".
    void set_granted_l2_set_aside(std::size_t bytes) noexcept { granted_l2_set_aside_ = bytes; }
    std::size_t granted_l2_set_aside() const noexcept { return granted_l2_set_aside_; }
    // The budget the hot set is actually compared against.
    std::size_t effective_l2_budget() const noexcept {
        return granted_l2_set_aside_ ? granted_l2_set_aside_ : recommended_l2_set_aside();
    }
    // The prefetch distance this layer index resolves to under the configured schedule,
    // 0 for "do not prefetch here". A runtime has to know it before calling into the
    // controller, because it is the runtime that must hand over the state that many
    // recurrent layers ahead.
    int distance_for_layer(int layer_index) const noexcept;
    // layer_index lets the schedule vary with depth; -1 means "unknown", which forces
    // Uniform so a caller that cannot supply it still gets defined behaviour.
    LayerPlan plan_for_layer(std::size_t current_state_bytes,
                             bool has_next_recurrent_layer,
                             std::size_t concurrently_hot_bytes = 0,
                             int layer_index = -1) const noexcept;
    // Geometry-aware form. Identical policy; the difference is that the hot-set models
    // other than CurrentLayer have something to count. `current_state_bytes` is the
    // region the controller resolved from WindowScope/WindowTarget, so the planner stays
    // free of pointer arithmetic and remains testable without a GPU.
    LayerPlan plan_for_layer(std::size_t current_state_bytes,
                             bool has_next_recurrent_layer,
                             const RecurrentGeometry& geometry,
                             int layer_index = -1) const noexcept;
private:
    // `additive` distinguishes "bytes competing with this layer's window" (the v0.1
    // caller-declared quantity) from "the whole live set, this layer's window included".
    LayerPlan plan_for_layer_impl(std::size_t current_state_bytes, bool has_next_recurrent_layer,
                                  std::size_t hot_bytes, bool additive,
                                  int layer_index) const noexcept;

    DeviceCaps caps_{};
    PlannerConfig config_{};
    std::size_t granted_l2_set_aside_ = 0;
};

const char* to_string(LocalityMode mode) noexcept;
LocalityMode parse_mode(const char* text);
const char* to_string(PreTouchStrategy strategy) noexcept;
PreTouchStrategy parse_pre_touch_strategy(const char* text);
const char* to_string(HotSetPolicy policy) noexcept;
HotSetPolicy parse_hot_set_policy(const char* text);
const char* to_string(PrefetchSchedule schedule) noexcept;
PrefetchSchedule parse_prefetch_schedule(const char* text);
const char* to_string(HotSetModel model) noexcept;
HotSetModel parse_hot_set_model(const char* text);
const char* to_string(WindowScope scope) noexcept;
WindowScope parse_window_scope(const char* text);
const char* to_string(WindowTarget target) noexcept;
WindowTarget parse_window_target(const char* text);
const char* to_string(PreTouchCoverage coverage) noexcept;
PreTouchCoverage parse_pre_touch_coverage(const char* text);
const char* to_string(PrefetchJoin join) noexcept;
PrefetchJoin parse_prefetch_join(const char* text);
const char* to_string(WindowAttach attach) noexcept;
WindowAttach parse_window_attach(const char* text);

} // namespace recurlocal
