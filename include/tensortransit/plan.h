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

// How a plan's predicted saving is computed. This is the single most consequential choice in
// the planner, and until 0.2.1 there was only one option and it was making the whole
// admission axis meaningless.
enum class CostModel : int {
    // saved(t) = reused_bytes(t) x (granted/bytes) x hit_ratio.
    //
    // Linear in the resident share, so total saving is `sum_i granted_i x density_i` subject
    // to `sum_i granted_i <= budget`. That is a fractional knapsack, greedy-on-density is
    // optimal for it, and therefore NO admission rule can beat Density under this model --
    // `role_floor` is monotonically worse as its floor grows, by construction rather than by
    // accident. Kept as the CONTROL: every predicted figure this repository published before
    // 0.2.1 was computed under it, and a model that cannot beat it has not earned its place.
    Linear = 0,
    // Linear, times a SURVIVAL factor carrying the three terms docs/evaluation.md named as
    // missing, in the functional form the hardware measurements in results/ actually support:
    //
    //     resident(t) = whole cache lines of  min(granted, bytes) x hit_ratio
    //     survival(t) = min(1, (resident(t) / reuse_distance_bytes(t)) ^ beta)
    //     saved(t)    = reused_bytes(t) x (resident(t)/bytes) x survival(t)
    //                   -  eta x (resident_total / L2) x step_traffic
    //
    //   whole-line residency  a line is resident or it is not, so a grant is quantised to
    //                         cache lines and a sub-line grant is worth nothing;
    //   survival              capacity against reuse distance, as a POWER LAW -- the classic
    //                         shape of a cache miss-ratio curve, and the one the data picks:
    //                         an exponential cannot fit a batch-1 dense arm and a batch-1 MoE
    //                         arm at once (it needs sigma to differ by 3.4x between them),
    //                         while beta = 0.1108 predicts both to within 0.006 points;
    //   interference          the reservation is taken from the same L2 the weight and KV
    //                         streams use, so it COSTS. This is the term that gets the SIGN
    //                         right on the concurrency arms, where the persist family
    //                         measures negative, and the only term under which a Stream
    //                         action -- which lowers the reuse distance -- is worth anything.
    //
    // The consequence is the whole point. `saved` goes as resident^(1+beta), which is
    // SUPERLINEAR: the objective is convex in the grant, its optimum is at a vertex, and
    // concentrating the budget on fewer tensors beats spreading it. Under the linear model
    // the same objective is a fractional knapsack whose optimum is greedy-on-density, so no
    // admission rule could beat Density and the whole axis measured nothing. Under this one
    // Density is not optimal, and `AdmissionRule::Survival` is the rule that exploits it.
    //
    // beta and eta are fitted, once, against every paired hardware measurement of the persist
    // arm in results/ -- two parameters against eight arms on two architectures, with the
    // residuals published. See eval/cost_model_fit.py.
    Residency = 1,
};

const char* to_string(CostModel model) noexcept;
bool parse_cost_model(const char* text, CostModel* out) noexcept;

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
    // Bytes the plan asks the persisting partition to hold, quantised to cache lines and
    // charged once per tensor. NOT `committed_bytes`: per-consumer binding commits the same
    // region several times, and the hit ratio means a committed region is not wholly
    // resident. This is the quantity the residency model's self-eviction term is about.
    std::size_t resident_bytes = 0;
    // Interference a Stream action takes out of the picture. Zero under the linear model,
    // which is the whole reason a Stream action was priced at zero there.
    std::size_t stream_relieved_bytes = 0;
    // The saving BEFORE the reservation's own cost is charged, and that cost. Kept apart so a
    // reader can see a plan whose policy works and whose set-aside is not worth it -- which
    // is what the concurrency arms measure, and a single net figure cannot say.
    std::size_t predicted_gross_saved_bytes = 0;
    std::size_t reservation_cost_bytes = 0;

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
    // Which model produced `predicted_saved_bytes`. Recorded next to the number because the
    // two models can differ by an order of magnitude on the same plan, and a prediction whose
    // model is not on the page is exactly the failure this project keeps having.
    CostModel cost_model = CostModel::Residency;

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
