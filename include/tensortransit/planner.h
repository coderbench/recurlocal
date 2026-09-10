#pragma once
#include <cstddef>
#include <cstdint>
#include <memory>

#include "tensortransit/device.h"
#include "tensortransit/graph.h"
#include "tensortransit/plan.h"
#include "tensortransit/tensor.h"

namespace tensortransit {

// A set of tensor roles, as a bitmask. What makes a policy *scoped*: "persist recurrent
// state only" and "persist everything" are the same planner with two different masks, which
// is what lets spec section 38's five comparison arms be one binary and one flag rather
// than five code paths that can drift apart.
struct RoleMask {
    std::uint32_t bits = 0;
    static constexpr std::uint32_t bit(TensorRole role) noexcept {
        return 1u << static_cast<int>(role);
    }
    static RoleMask none() noexcept { return RoleMask{0}; }
    static RoleMask all() noexcept { return RoleMask{0xFFFFFFFFu}; }
    static RoleMask of(TensorRole a) noexcept { return RoleMask{bit(a)}; }
    static RoleMask of(TensorRole a, TensorRole b) noexcept { return RoleMask{bit(a) | bit(b)}; }
    bool has(TensorRole role) const noexcept { return (bits & bit(role)) != 0; }
    RoleMask& add(TensorRole role) noexcept { bits |= bit(role); return *this; }
    RoleMask& remove(TensorRole role) noexcept { bits &= ~bit(role); return *this; }
    bool empty() const noexcept { return bits == 0; }
};

// How a scarce locality budget is divided when the tensors that want to be resident do not
// all fit. This is the project's central open question, generalized off recurrent state:
// RecurLocal's HotSetPolicy answered it for one role, and every one of its enumerators is
// still here because each was measured and none is dominated on every workload.
//
// A new rule is the narrowest useful contribution to this repository: one enumerator, one
// implementation, and it is measurable against every other on the same trace with no GPU.
enum class AdmissionRule : int {
    // Sort by modelled bytes saved per byte of budget, take greedily. The knapsack
    // relaxation, and the obvious default -- but it is scale-free, so it will spend the
    // whole budget on many tiny tensors when one large one carries most of the reuse.
    Density = 0,
    // Admit whole tensors at the full hit ratio until the budget is spent, decline the rest.
    // RecurLocal's HotSetPolicy::Quota. Models the cache as it actually is -- a line is
    // resident or it is not -- and beats hit-ratio shaving where the footprint is a small
    // multiple of the budget. Inert where it is many multiples.
    Quota = 1,
    // Admit everything and scale every hit ratio by the budget share. RecurLocal's v0.1
    // heuristic, kept as the CONTROL: it is what every published number in this repository
    // was measured under, and it models a cache that can keep 97% of a byte.
    Proportional = 2,
    // Nearest next use first, in the configured reuse metric. Belady-flavoured: it is what
    // an optimal cache would do if it evicted rather than admitted. Differs from Density
    // exactly when the soonest-needed tensor is not the densest one.
    ReuseOrder = 3,
    // Reserve a floor for each role that has reuse, then spend the remainder by Density.
    //
    // This is the rule the multi-tensor claim stands on. Density on a mixed workload is
    // winner-take-all: whichever role has the better bytes-saved-per-byte takes the entire
    // budget, which reproduces "recurrent-only" or "KV-only" by accident rather than
    // coordinating them. RoleFloor is the cheapest rule that cannot do that, and it is the
    // one spec section 38 asks to beat the naive independent policies. Whether it does is a
    // MEASUREMENT that has not been taken -- see docs/evaluation.md.
    RoleFloor = 4,
};

const char* to_string(AdmissionRule rule) noexcept;
bool parse_admission_rule(const char* text, AdmissionRule* out) noexcept;

// When a prefetch is issued relative to the consumer that needs the data.
enum class PrefetchTiming : int {
    // Issue `distance` kernels ahead, uniformly. The v0.1 assumption, and wrong in a hybrid
    // model where the amount of work between two recurrent layers is not constant.
    FixedDistance = 0,
    // Issue as early as the recorded time estimates say is needed to complete, and no
    // earlier. Needs KernelEvent::estimated_duration_ns; degrades to FixedDistance without it.
    BandwidthAware = 1,
    // Issue at the START of the window for everything, i.e. as early as possible. The naive
    // control: it maximises the chance the data has arrived and minimises the chance it is
    // still there.
    Eager = 2,
};

const char* to_string(PrefetchTiming timing) noexcept;
bool parse_prefetch_timing(const char* text, PrefetchTiming* out) noexcept;

// How long a persisting window stays bound.
//
// This is a hardware constraint made into a policy choice. CUDA binds ONE access-policy
// window to a stream (or to a launch, or to a graph node) at a time, so a plan cannot hold
// several regions marked at once through that mechanism -- the set-aside is device-wide but
// the window is singular. The two ways to live with that are measurably different and
// neither dominates:
enum class WindowBinding : int {
    // Bind before each kernel that reads the tensor and drop immediately after. Any number
    // of tensors can be admitted, because no two windows are ever live together -- but a
    // region is only marked while its own kernel runs, so residency between two uses rests
    // entirely on the set-aside not being re-carved. RecurLocal's v0.1 shape, and the shape
    // every published number in this repository was measured under.
    PerConsumer = 0,
    // Bind once before the first use and drop after the last. The region stays marked across
    // everything in between, which is what actually protects it from the weight stream --
    // but only ONE tensor can have it, so the planner must choose, and every other admitted
    // tensor falls back to PerConsumer. Unmeasured on hardware; see docs/evaluation.md.
    Sticky = 1,
};

const char* to_string(WindowBinding binding) noexcept;
bool parse_window_binding(const char* text, WindowBinding* out) noexcept;

// Which candidate keeps the window when more than one wants it across the same kernel and
// `max_windows_per_kernel` forces a choice. Under a LINEAR cost model two whole-tensor
// candidates have identical density -- saved/granted collapses to the hit ratio -- so
// density alone does not choose, and something has to. New enumerators are appended.
enum class WindowPreference : int {
    Densest = 0,       // most modelled bytes saved per byte of budget
    Widest = 1,        // the largest region. Reproduces RecurLocal's WindowTarget::Matrix on
                       // a model whose matrix state is the larger of the two segments, which
                       // is what every published number here was measured under.
    Narrowest = 2,     // the smallest region: the one most likely to be held WHOLE
    SoonestReuse = 3,  // nearest next use, in bytes of intervening traffic
};

const char* to_string(WindowPreference preference) noexcept;
bool parse_window_preference(const char* text, WindowPreference* out) noexcept;

struct TransitPlannerConfig {
    // --- what the planner is allowed to touch ---------------------------------------
    RoleMask persist_roles = RoleMask::all();
    RoleMask prefetch_roles = RoleMask::none();
    RoleMask stream_roles = RoleMask::none();

    // --- how it divides the budget ---------------------------------------------------
    AdmissionRule admission = AdmissionRule::Density;
    ReuseMetric reuse_metric = ReuseMetric::Bytes;
    // Fraction of the device's persisting capacity to ask for. The measured dial: on the
    // one model where the persist family pays at all, 0.25 -> 1.00 moved the gain
    // monotonically from +0.68% to +1.53%, and on a model whose footprint does not fit it
    // is flat. Not a constant anyone should trust across workloads.
    double budget_fraction = 0.75;
    double hit_ratio = 0.70;
    // Below this share of a tensor's bytes, a reservation is not worth the cache it takes
    // from everything else.
    double min_hit_ratio = 0.05;
    // RoleFloor only: share of the budget guaranteed to each role that has reuse, before
    // Density spends the rest. 0.0 makes RoleFloor identical to Density, which is the
    // control the mechanism has to beat.
    double role_floor_share = 0.25;
    // Decline a tensor whose next use is further away than this many times the budget: more
    // traffic runs in between than the cache can survive, so the reservation buys nothing
    // and costs everything else. 0 disables the check. This is the generalization of "the
    // reuse distance for a recurrent state is one whole token".
    double max_reuse_distance_budgets = 0.0;

    // How long an admitted window stays bound. Under Sticky the single densest admitted
    // candidate holds the window for the whole recorded step and the rest bind per consumer.
    WindowBinding window_binding = WindowBinding::PerConsumer;

    // How many tensors may hold a persisting window ACROSS ONE KERNEL.
    //
    // CUDA binds one access-policy window to a stream, to a launch, or to a graph node at a
    // time. A plan that marks two regions before one kernel is therefore not describing
    // something the hardware can do: the second window replaces the first, the first is
    // silently absent, and the plan's own telemetry counts two applied windows. That is a
    // null policy wearing a real one's counters, which is this repository's oldest failure
    // mode.
    //
    // 0 is unbounded -- the historical behaviour, what the golden plans were recorded under,
    // and correct for an integration that delivers per-tensor windows some other way. An
    // integration that delivers through the singular stream/node binding sets 1;
    // adapters/sparkinfer does.
    int max_windows_per_kernel = 0;
    WindowPreference window_preference = WindowPreference::Densest;

    // --- prefetch --------------------------------------------------------------------
    bool prefetch_enabled = false;
    PrefetchTiming prefetch_timing = PrefetchTiming::FixedDistance;
    int prefetch_distance = 1;  // kernels, under FixedDistance
    // Do not prefetch a region below this size: the fork/join is a permanent graph node
    // under capture and costs ~0.027% of a decode step each, so a small region cannot earn
    // it back. Measured, not guessed -- see docs/evaluation.md.
    std::size_t prefetch_min_bytes = 64 * 1024;
    // One join per token instead of one per prefetch. Halves the graph nodes at the cost of
    // no ordering between a prefetch and the kernel it warms. Worth 1.20 points on the one
    // model where it was measured, which is more than the locality it was overlapping.
    bool prefetch_join_at_end = false;

    // --- streaming hints -------------------------------------------------------------
    // Mark tensors in `stream_roles` that have no reuse as streaming, so they pass through
    // without evicting what does. The counterpart to Persist, and the other half of the
    // "shared L2 budget" the multi-tensor claim is about.
    std::size_t stream_min_bytes = 1024 * 1024;

    // --- limits ----------------------------------------------------------------------
    // Hard cap on plan size. A planner that emitted an action per tensor per kernel would
    // put its own cost on the critical path; spec section 80 budgets the whole layer at
    // under 0.5% of token latency.
    std::size_t max_actions = 4096;
};

// nullptr when usable, otherwise a static description of what is wrong.
const char* validate(const TransitPlannerConfig& config) noexcept;

// Everything a planner may look at. Spec section 18 writes this as three parameters; the
// registry is here as a fourth because a TransitAction carries a RESOLVED region, and
// resolving one needs the descriptor. Putting region arithmetic in the planner rather than
// the executor is deliberate: it is the part that has been wrong before, and here it is
// testable without a GPU.
struct PlanInput {
    const TransitGraph* graph = nullptr;
    const TensorRegistry* registry = nullptr;
    DeviceProfile device{};
    RuntimeState runtime{};
    bool valid() const noexcept { return graph != nullptr && registry != nullptr; }
};

// The competition surface. Different planners can be compared on one trace, in one process,
// with no hardware -- which is what makes the planner the part of this repository a
// contributor can move without a GPU.
class ITransitPlanner {
public:
    virtual ~ITransitPlanner() = default;
    virtual TransitPlan build_plan(const PlanInput& input) = 0;
    // Stable identifier, recorded in every plan and every evaluation artifact.
    virtual const char* name() const noexcept = 0;
};

// The five arms spec section 38 requires a multi-tensor result to be compared against.
// First-class rather than described, because a comparison whose arms are assembled by hand
// in a shell script is one where the arms drift apart between runs.
enum class PolicyPreset : int {
    // No actions at all. The true control: not "the planner with everything off", which
    // would still pay the hook's cost, but a plan that is empty.
    Baseline = 0,
    RecurrentOnly = 1,        // persist recurrent state, ignore everything else
    KVOnly = 2,               // persist KV, ignore everything else
    NaiveBothPersistent = 3,  // persist both, no budget arbitration between them
    Global = 4,               // one budget, RoleFloor arbitration: the claim under test
};

const char* to_string(PolicyPreset preset) noexcept;
bool parse_policy_preset(const char* text, PolicyPreset* out) noexcept;

// Returns the config for a preset, starting from `base` so the dials that are not part of
// the arm (hit ratio, budget fraction) stay identical across the comparison. An arm that
// differed in two things at once would not be a comparison.
TransitPlannerConfig preset_config(PolicyPreset preset, const TransitPlannerConfig& base);
// The planner name each preset runs on.
const char* preset_planner(PolicyPreset preset) noexcept;

// --- implementations ------------------------------------------------------------------
// Emits nothing. The control case (spec section 20): no persistence, no prefetch, no
// streaming hints, no extra synchronisation.
std::unique_ptr<ITransitPlanner> make_baseline_planner();
// RecurLocal's shipped policy, expressed against the Transit Graph: persist the recurrent
// state about to run, pre-touch the next one, clear after use. Exists to prove the old
// design fits the new abstraction and that the old numbers still reproduce.
std::unique_ptr<ITransitPlanner> make_recurrent_v0_planner(const TransitPlannerConfig& config);
// benefit = reuse_weight x urgency / bytes, greedy. Deliberately simple (spec section 22).
std::unique_ptr<ITransitPlanner> make_greedy_planner(const TransitPlannerConfig& config);
// The general one: an explicit budget, an AdmissionRule, and a decline record for
// everything that did not fit. This is the planner the multi-tensor claim runs on.
std::unique_ptr<ITransitPlanner> make_budgeted_planner(const TransitPlannerConfig& config);
// BudgetedPlanner plus request arbitration: per-request priority, and a budget divided
// across in-flight requests rather than across tensors alone.
std::unique_ptr<ITransitPlanner> make_concurrency_planner(const TransitPlannerConfig& config);

// By name: "baseline", "recurrent_v0", "greedy", "budgeted", "concurrency". nullptr for an
// unknown name -- a caller that silently fell back to baseline would report a measurement
// of the wrong planner.
std::unique_ptr<ITransitPlanner> make_planner(const char* name,
                                              const TransitPlannerConfig& config);
// Null-terminated list of the registered names, for a CLI's help text and for the sweep
// harness, so adding a planner does not need a second edit somewhere else to be reachable.
const char* const* planner_names() noexcept;

}  // namespace tensortransit
