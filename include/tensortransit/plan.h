#pragma once
#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include "tensortransit/device.h"
#include "tensortransit/graph.h"
#include "tensortransit/tensor.h"

namespace tensortransit {

// Schema version of a serialized TransitPlan. Bumped on any breaking change to the JSON
// below; see docs/STABILITY.md and schemas/plan.schema.json.
inline constexpr int kPlanSchemaVersion = 1;

// What the executor is being asked to do. The innovation is the coordination, not any one
// of these -- every action here exists in some form in some other system. What does not is
// one planner choosing among them for several tensor classes against one budget.
enum class TransitActionKind : int {
    Normal = 0,       // explicitly no policy: the control case, and NOT the same as absence
    Persist = 1,      // ask for cache residency over a region, at a hit ratio
    Stream = 2,       // ask for the opposite: pass through without displacing anything
    Prefetch = 3,     // move the bytes toward the cache before the consumer runs
    RotateWindow = 4, // move an existing window to a new region without re-reserving
    ClearPolicy = 5,  // drop a policy; every Persist must have one (spec section 32)
    RecordEvent = 6,  // fork point for overlap
    WaitEvent = 7,    // join point for overlap
};

const char* to_string(TransitActionKind kind) noexcept;
bool parse_transit_action_kind(const char* text, TransitActionKind* out) noexcept;

struct TransitAction {
    TransitActionKind kind = TransitActionKind::Normal;
    TensorId tensor = kInvalidTensorId;

    // WHEN. An action fires immediately before `before_kernel` runs, or immediately after
    // `after_kernel` has run. Exactly one is set for a placed action; a ClearPolicy with
    // only `after_kernel` is the normal shape.
    KernelId before_kernel = kInvalidKernelId;
    KernelId after_kernel = kInvalidKernelId;

    // WHAT. The region, resolved by the planner from the tensor descriptor -- so the
    // executor never has to widen or clamp anything, and so the region is inspectable in a
    // plan dump rather than computed inside a .cu file where no test can reach it.
    const void* ptr = nullptr;
    std::size_t bytes = 0;
    double hit_ratio = 0.0;  // Persist only

    int stream_id = 0;
    // Correlates a RecordEvent with the WaitEvent that joins it. 0 for actions that need no
    // pairing.
    std::uint32_t event_id = 0;

    // WHY, carried for observability only. No executor reads these, and no planner may read
    // them back: a plan is a value, and a value that had to be interpreted to be executed
    // would not be serializable.
    std::size_t expected_saved_bytes = 0;  // what the cost model thinks this action is worth
    TensorRole role = TensorRole::Unknown;
};

// Why a tensor did NOT get what it asked for. A plan that only lists what it did leaves the
// most useful question -- "why is my policy not helping" -- answerable only by guessing.
// Every RecurLocal evaluator guard exists because a confident number hid a null result, so
// the generalization records the declines as first-class output.
enum class DeclineReason : int {
    None = 0,
    NoReuse = 1,           // nothing reads it again: a cache cannot help
    BudgetExhausted = 2,   // it lost the knapsack to something denser
    TooLarge = 3,          // it alone exceeds the budget or the hardware window limit
    ReuseTooFar = 4,       // more traffic runs between uses than the cache can survive
    RoleExcluded = 5,      // the policy scope was not allowed to act on this role
    BelowMinHitRatio = 6,  // the share it could hold is not worth the reservation
    NotSupported = 7,      // the device cannot do it
};

const char* to_string(DeclineReason reason) noexcept;

struct TransitDecline {
    TensorId tensor = kInvalidTensorId;
    TensorRole role = TensorRole::Unknown;
    DeclineReason reason = DeclineReason::None;
    std::size_t bytes = 0;
    std::size_t forgone_saved_bytes = 0;  // what the cost model says the decline cost
};

// What the planner believes the plan is worth, and against what. Kept next to the plan
// because a prediction that is not recorded cannot later be compared with a measurement,
// and comparing them is the only way a cost model ever improves.
//
// NOTHING here is a measurement. `predicted_*` are outputs of a model whose assumptions are
// stated in docs/evaluation.md; the project's rule is that no predicted figure is ever
// published as a gain.
struct PlanCostModel {
    std::size_t step_traffic_bytes = 0;     // from the graph: what the window moves
    std::size_t removable_bytes = 0;        // from the graph: what a perfect cache removes
    // What a perfect policy could remove WITH THIS DEVICE'S BUDGET. Not the same number and
    // not a scaling of it: over a cyclic window every tensor is re-read next iteration, so
    // `removable_bytes` is the whole step and the unlimited-cache ceiling is a broken
    // question. This is the one that binds, and the one `ceiling_throughput_ratio()` uses.
    std::size_t bounded_removable_bytes = 0;
    std::size_t predicted_saved_bytes = 0;  // what THIS plan is modelled to remove
    std::size_t budget_bytes = 0;           // locality budget the plan was allowed
    std::size_t committed_bytes = 0;        // budget the plan actually spent
    std::size_t peak_live_bytes = 0;        // what would have to be resident to save it all

    // Share of the step's traffic this plan is modelled to remove, in [0,1].
    double predicted_traffic_share() const noexcept;
    // Throughput ratio that share implies: a step carrying f less traffic runs in (1-f) of
    // the time, so tok/s rise by f/(1-f). Quoting the share instead UNDERSTATES the ceiling,
    // and this project has made that mistake in the other direction too -- see
    // docs/evaluation.md. Returns a ratio (1.0194 for +1.94%), not a percentage.
    double predicted_throughput_ratio() const noexcept;
    // The ceiling a perfect policy could reach on this graph WITH THIS BUDGET. A plan whose
    // predicted gain approaches it is done; one far below it has room. Uses
    // `bounded_removable_bytes`; falls back to `removable_bytes` only when nobody set it.
    double ceiling_throughput_ratio() const noexcept;
    // True when the unlimited-cache figure has saturated, which over a cyclic window it
    // always does. Printing "+0.000%" there -- which is what 1/(1-f) at f=1 collapses to --
    // reads as "there is nothing here", and the honest answer is the opposite.
    bool ceiling_saturated() const noexcept;
};

class TransitPlan {
public:
    TransitPlan() = default;

    void add(const TransitAction& action) { actions_.push_back(action); }
    void decline(const TransitDecline& d) { declines_.push_back(d); }

    const std::vector<TransitAction>& actions() const noexcept { return actions_; }
    const std::vector<TransitDecline>& declines() const noexcept { return declines_; }
    std::vector<TransitAction>& actions() noexcept { return actions_; }

    const PlanCostModel& cost() const noexcept { return cost_; }
    PlanCostModel& cost() noexcept { return cost_; }

    const std::string& planner_name() const noexcept { return planner_name_; }
    void set_planner_name(std::string name) { planner_name_ = std::move(name); }

    // Actions that fire before this kernel / after this kernel, in plan order. The executor
    // calls these per kernel, so they are precomputed into an index by finalize().
    void finalize();
    bool finalized() const noexcept { return finalized_; }
    const TransitAction* const* before(KernelId kernel, int* count) const noexcept;
    const TransitAction* const* after(KernelId kernel, int* count) const noexcept;

    bool empty() const noexcept { return actions_.empty(); }
    std::size_t size() const noexcept { return actions_.size(); }
    std::size_t count(TransitActionKind kind) const noexcept;

    // nullptr when the plan is well formed, otherwise what is wrong with it. Checks the
    // rules that are cheap and that this project has actually been bitten by: a Persist with
    // no matching ClearPolicy (spec section 32 -- a lingering policy hurts later kernels), a
    // RecordEvent with no WaitEvent (an unjoined fork ends a CUDA graph capture INVALID),
    // an action placed on no kernel at all, a zero-byte region.
    const char* validate() const noexcept;

    // --- serialization ---------------------------------------------------------------
    // A plan must be inspectable and serializable (spec section 16). These are the whole
    // reason: a contributor can dump a plan, edit it, replay it, and diff two planners
    // without a GPU in the loop.
    std::string to_json() const;
    // Human form, the shape of spec section 17. Grouped by kernel, declines listed after.
    std::string to_text() const;
    // Stable 64-bit digest of the ACTIONS only -- not of the cost model, not of the name.
    // Two planners that emit the same actions have the same digest, which is how the
    // golden-plan tests assert behaviour without pinning a formatting choice, and how
    // `tensortransit compare` says "these two plans are the same plan".
    std::uint64_t digest() const noexcept;

    void clear() noexcept;

private:
    std::vector<TransitAction> actions_;
    std::vector<TransitDecline> declines_;
    PlanCostModel cost_{};
    std::string planner_name_;

    // Index built by finalize(): pointers into actions_, grouped by kernel.
    std::vector<const TransitAction*> before_index_;
    std::vector<const TransitAction*> after_index_;
    std::vector<std::uint64_t> before_keys_;   // kernel id, parallel to before_offsets_
    std::vector<std::uint32_t> before_offsets_;
    std::vector<std::uint64_t> after_keys_;
    std::vector<std::uint32_t> after_offsets_;
    bool finalized_ = false;
};

}  // namespace tensortransit
