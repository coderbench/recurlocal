#pragma once
#include <cstddef>   // std::size_t, offsetof
#include "tensortransit/version.h"

namespace tensortransit {

enum class LocalityMode : int { Baseline, Persist, Prefetch, Combined };

// How the next layer's state is touched. Each strategy must be read-only and produce the
// same (ignored) result; they differ only in how the bytes are moved, which is the point.
// Adding one is the narrowest useful contribution to this project: a new kernel, a new
// enumerator, and it is measurable on its own against every other strategy.
enum class PreTouchStrategy : int {
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
enum class PrefetchSchedule : int {
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
enum class HotSetModel : int {
    CurrentLayer,    // this layer's window plus caller-declared bytes; the v0.1 control
    TokenFootprint,  // every recurrent layer's state, every sequence: what a token revisits
    ReuseWindow      // TokenFootprint plus the non-recurrent bytes streamed between two
                     // visits to the same state - everything that can evict it
};

// Which region a persisting window covers. A window over the slice of the layer about to
// run only helps if that slice is re-read before the window moves on, and under
// TokenFootprint accounting it is not: it is re-read a whole token later.
enum class WindowScope : int {
    Layer,       // the slice this layer will touch; the v0.1 assumption
    Allocation,  // the whole recurrent-state allocation the slice lives in
    Ahead        // this slice plus the next prefetch_distance slices
};

// Which recurrent state a persisting window protects. A hybrid model carries more than
// one: Qwen3.8-27B holds a 3 MiB fp32 matrix state and a 60 KiB bf16 convolution window per
// layer. Only one access-policy window can be bound at a time, so this is a choice, and the
// small state is the one whose whole allocation can actually fit in a set-aside.
enum class WindowTarget : int {
    Matrix,     // the large recurrent matrix state
    Conv,       // the small convolution window state
    Widest,     // whichever segment has the most bytes
    Narrowest   // whichever segment has the fewest
};

// Which recurrent states are pre-touched. v0.1 walked one buffer because it modelled one.
enum class PreTouchCoverage : int { Matrix, Conv, Both };

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
//               cudaStreamGetCaptureInfo. One line at the hook site and no launch changes.
//               This header used to call that undocumented; it is not. CUDA's own
//               cudaStreamGetCaptureInfo documentation says "All operations other than
//               destroy and node removal are permitted on the graph while the capture
//               sequence is in progress" and blesses passing the driver-owned dependency
//               array straight to APIs that operate on the graph (cuda_runtime_api.h:2743
//               and :2755, unchanged since CUDA 11.3). tools/capture_attr_probe.cu reads the
//               window back off the finished graph at 1 to 128 nodes and finds it present
//               and correct every time.
//
//               Its real cost is precision: it marks EVERY kernel node the capture has
//               pending, which is not always the one the hook fired for.
//
//   CaptureNodeStrict
//               CaptureNode, except that it attaches only when the capture has recorded
//               EXACTLY ONE kernel node since the last attach - i.e. only when the node it
//               is about to mark is unambiguously the recurrent kernel the hook fired for.
//
//               CaptureNode marks EVERY kernel node in the capture's current dependency set,
//               and that set is not always one node. On the non-fused convolution branch the
//               launch immediately before the hook is `l2_norm_qk_kernel`, which never touches
//               convolution state: it inherits a persisting window over memory it does not
//               read, spending set-aside on nothing. Strict declines instead and counts it,
//               which turns a silent mis-attachment into a number.
//
// Default is Stream. A convenience that can cost a quarter of a runtime's throughput is not
// a default; the controller counts invalidations and latches node attachment off after the
// first one. New enumerators are appended, never inserted: see docs/STABILITY.md.
enum class WindowAttach : int { Stream, CaptureNode, CaptureNodeStrict };

// When the pre-touch stream is rejoined to the compute stream.
//
// This only matters under CUDA Graph capture, which is where production decode lives: a
// fork and a join are each recorded as graph nodes, and a hybrid model has one recurrent
// layer to pre-touch for nearly every layer it runs. SparkInfer measured its own single
// fork/join pair at ~0.89% of a decode step, so paying for one per recurrent layer is a
// cost the prefetch has to earn back before it wins anything.
enum class PrefetchJoin : int {
    PerLayer,  // join before the next layer runs; the v0.1 shape, and the safe one
    TokenEnd   // fork per layer, join once after the whole layer walk - half the nodes,
               // at the cost of no ordering between a pre-touch and the layer it warms
};

// How large an L2 set-aside to ask the driver for.
//
// The shipped behaviour reserves a constant fraction of the device's persisting-L2 capacity,
// chosen before anything is known about the workload. Measurement says the right fraction is
// not constant. On the sparse-MoE checkpoint `budget_fraction 1.00` is the best setting at
// batch 1 - +1.63% against +1.26% at the shipped 0.75 - and the WORST at four concurrent
// sequences, -1.29% against -0.46% (results/rtx5090-moe-scored.json). The set-aside is taken
// from the same L2 the weight and KV streams use, so it is a trade, and where it lands
// depends on how much of the recurrent footprint the reservation can actually hold.
//
// The planner already computes that footprint for the hot-set models, and the runtime already
// declares the sequence count, so the input exists; only the policy was missing.
enum class SetAsidePolicy : int {
    // `persisting_budget_fraction` of capacity, whatever the workload. THE CONTROL: this is
    // what every number in this repository was measured under, and it stays the default.
    Fixed,
    // Never reserve more than the footprint can use. Reserving 60 MiB to hold a 10 MiB
    // footprint takes 50 MiB from the cache the rest of the step streams through and buys
    // nothing with it. Parameter-free and unarguable in that direction; it is identical to
    // Fixed at fraction 1.0 on every arm measured so far, because on this device the
    // footprint has always been the larger of the two. It is the regime a device with more
    // persisting L2, or a smaller model, would be in.
    FitFootprint,
    // FitFootprint, plus: decline the reservation entirely when the fraction of the footprint
    // that could be held falls below `min_residency`. A window that cannot hold the working
    // set still costs the shared cache the bytes it reserved, and the return falls with the
    // share it can keep. The threshold is an EMPIRICAL parameter, not a derived one, and the
    // two measurements that bracket it disagree about where it sits - the dense model gains
    // +0.10% at a resident fraction of 0.41 while the MoE loses 0.46% at 0.48 - so it is
    // exposed as a swept axis rather than asserted. See docs/OPTIMIZATION-SURFACES.md.
    Residency
};

// What to do when the recurrent state that wants to be resident exceeds the L2 set-aside.
// This is the policy the overview calls the major v0.1 frontier: the shipped heuristic is
// one line, and nothing has ever measured it against an alternative.
enum class HotSetPolicy : int {
    Proportional,  // scale the requested hit ratio by the budget share (the v0.1 heuristic)
    Fixed,         // ask for the full hit ratio regardless; the naive control
    Sqrt,          // back off by sqrt(share) - gentler than proportional
    Cliff,         // give up the window entirely rather than thrash a shared cache
    // Admit whole layers until the budget is spent, and decline the window for the rest.
    //
    // Every policy above answers oversubscription by moving ONE dial - the hit ratio - for
    // EVERY layer alike. That models the cache as something that can keep 97% of a byte,
    // which it cannot: a line is resident or it is not. Where the footprint is a small
    // multiple of the set-aside, asking thirty layers for 97% of a window is a different
    // request from keeping twenty-nine of them whole, and the second is the one the hardware
    // can actually honour.
    //
    // It is inert where the footprint is many times the budget - admitting a fortieth of the
    // layers is not obviously better than thrashing all of them - and decisive where it is
    // close, which is the regime a sparse-MoE hybrid puts this library in.
    Quota
};

// A recurrent model carries more than one mutable state per layer, in separate
// allocations with separate strides. Describing them as segments is what lets the planner
// account for all of them and the controller pre-touch all of them, instead of modelling
// the largest and silently ignoring the rest.
enum class StateKind : int { Matrix, Conv, Other };

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
    // Read by SetAsidePolicy::Fixed only. The workload-aware policies size the set-aside from
    // the footprint instead; applying both would leave them unable to reach the setting the
    // workload wants without also changing the constant they exist to replace.
    double persisting_budget_fraction = 0.75;
    double hit_ratio = 0.70;
    // How many recurrent layers ahead to pre-touch. The runtime supplies the state this
    // many layers ahead; the planner only decides whether and how far.
    int prefetch_distance = 1;
    std::size_t max_hot_window_bytes = 0;
    PreTouchStrategy pre_touch = PreTouchStrategy::Vec4;
    HotSetPolicy hot_set_policy = HotSetPolicy::Proportional;
    // Default to the shipped constant-fraction reservation so an existing caller's numbers do
    // not move under it; the workload-aware rules are opted into and measured against it.
    SetAsidePolicy set_aside_policy = SetAsidePolicy::Fixed;
    // Resident fraction of the recurrent footprint below which SetAsidePolicy::Residency
    // declines to reserve anything at all. Only that policy reads it.
    //
    // 0.50 is not a guess and not a fit; it is the one value the measurements leave room for.
    // The persist family is measured to PAY at a resident fraction of 0.977 (sparse MoE,
    // batch 1, +1.26% at the shipped dials and +1.63% at their maximum) and measured NOT to
    // pay at 0.478 and below (the same model at four concurrent sequences, -0.46%, inside its
    // own 0.46% noise floor). Anything in (0.478, 0.977) declines everything known not to pay
    // and admits everything known to pay; the crossover inside that interval is unmeasured,
    // and 0.50 sits at its lower edge so the rule gives up as little as the evidence allows.
    //
    // The cost of that choice is stated rather than hidden: the DENSE checkpoint at batch 1
    // sits at 0.409 and measures a resolved +0.10%, so this policy declines a real if tiny
    // gain there. That is a falsifiable prediction, and `--axis min-residency` is how it gets
    // falsified.
    double min_residency = 0.50;
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
    // The policy that was actually APPLIED, which is not always the one configured.
    // HotSetPolicy::Quota rations by layer ordinal over a known number of recurrent layers;
    // a caller who supplies neither gets Fixed, and used to get it silently - selecting the
    // rationing policy had no effect and the telemetry said nothing had backed off. Same
    // idiom as `hot_set_model`: the plan reports what it did, not what it was asked for.
    HotSetPolicy hot_set_policy = HotSetPolicy::Proportional;
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
    // The set-aside `set_aside_policy` asks for once the workload is known. Identical to the
    // geometry-free form under SetAsidePolicy::Fixed, and identical to it under any policy
    // when the geometry is absent or degenerate - a rule that cannot see the workload must
    // not silently invent one. Pure arithmetic, so the policy is testable without a GPU.
    std::size_t recommended_l2_set_aside(const RecurrentGeometry& geometry) const noexcept;
    // Bytes that must be resident for a recurrent state to survive until its layer runs
    // again: every recurrent layer, every sequence. Saturates rather than wrapping. This is
    // what COMPETES for the cache, and it is what the hot-set models count.
    static std::size_t token_footprint_bytes(const RecurrentGeometry& geometry) noexcept;
    // Bytes a persisting window can actually cover, which is not the same number. An
    // access-policy window is an address range and the controller places ONE per layer, over
    // one sequence's slice - at concurrency the runtime hands over a device array of per-row
    // pointers for the pre-touch and a single host-nameable row for the window. So a byte of
    // set-aside beyond ONE sequence's footprint protects nothing, however many sequences are
    // in flight. Every recurrent layer, one sequence.
    static std::size_t windowed_footprint_bytes(const RecurrentGeometry& geometry) noexcept;
    // Fraction of that footprint the device's persisting capacity could hold, in [0,1].
    // Zero when there is no capacity or no footprint to hold.
    double achievable_residency(const RecurrentGeometry& geometry) const noexcept;
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
    // `ordinal_period` is how many recurrent layers the caller will walk, or 0 for unknown.
    // HotSetPolicy::Quota needs it: it spreads the admitted layers evenly over that many
    // ordinals, and computing the spread from a byte count instead let the pattern repeat.
    LayerPlan plan_for_layer_impl(std::size_t current_state_bytes, bool has_next_recurrent_layer,
                                  std::size_t hot_bytes, bool additive,
                                  int layer_index, std::size_t ordinal_period = 0) const noexcept;

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
const char* to_string(SetAsidePolicy policy) noexcept;
SetAsidePolicy parse_set_aside_policy(const char* text);
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

// The structs above cross the boundary by value, so their layout IS the ABI. docs/STABILITY.md
// declares it unstable between minor versions and append-only within one; these turn that rule
// into something the build can fail on rather than something a reviewer has to notice. A field
// inserted into one of PlannerConfig's padding holes at offset 4, 52 or 92 changes no size at
// all, so the offset checks are not redundant with the size ones.
//
// These do NOT catch every layout change, and the gap is worth knowing: a four-byte field
// appended into a struct's TAIL padding changes no size and moves no offset. StateSegment
// ends with a four-byte StateKind at offset 32 and has sizeof 40 - appending another
// four-byte enum leaves it at 40 and both checks below pass. There is no portable
// compile-time member count, so that case is enforced by review; docs/STABILITY.md section 4
// says so rather than leaving a reader to assume the build has it covered.
//
// Guarded on LP64 because the numbers are LP64 numbers. An exotic target should not get a hard
// error for being exotic; it gets no check, and docs/STABILITY.md says so.
#if defined(__LP64__) || defined(_WIN64)
static_assert(sizeof(PlannerConfig) == 104,
              "PlannerConfig layout changed - append at the end, and see docs/STABILITY.md");
static_assert(sizeof(LayerPlan) == 72,
              "LayerPlan layout changed - append at the end, and see docs/STABILITY.md");
static_assert(sizeof(DeviceCaps) == 24, "DeviceCaps layout changed - see docs/STABILITY.md");
static_assert(sizeof(RecurrentGeometry) == 32,
              "RecurrentGeometry layout changed - see docs/STABILITY.md");
static_assert(sizeof(StateSegment) == 40, "StateSegment layout changed - see docs/STABILITY.md");
static_assert(sizeof(WindowRegion) == 16, "WindowRegion layout changed - see docs/STABILITY.md");
static_assert(offsetof(DeviceCaps, access_policy_max_window_bytes) == 16,
              "DeviceCaps fields moved - see docs/STABILITY.md");
static_assert(offsetof(RecurrentGeometry, streamed_bytes_per_token) == 24,
              "RecurrentGeometry fields moved - see docs/STABILITY.md");
static_assert(offsetof(StateSegment, kind) == 32, "StateSegment fields moved - see docs/STABILITY.md");
#endif

} // namespace tensortransit
