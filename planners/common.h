#pragma once
// Shared machinery for the planner implementations. NOT a public header: it is not
// installed, and nothing outside planners/ may include it. See docs/STABILITY.md section 7.
#include <cstddef>
#include <cstdint>
#include <vector>

#include "tensortransit/planner.h"

namespace tensortransit {
namespace detail {

// One tensor's case for a share of the locality budget, with everything a rule needs to
// compare it against another tensor of a different role and a different size.
struct Candidate {
    TensorId tensor = kInvalidTensorId;
    TensorRole role = TensorRole::Unknown;
    const TensorDesc* desc = nullptr;
    const TensorProfile* profile = nullptr;

    // What a window would have to hold for this tensor to be resident.
    std::size_t bytes = 0;
    // Bytes of HBM traffic the reuse edges would remove at full residency. Already counts a
    // read-modify-write twice; see use_traffic() in transit_graph.cpp.
    std::size_t saved_bytes = 0;
    // Nearest reuse in the configured metric. Small is urgent.
    std::uint64_t distance = 0;
    // The reuse distance in BYTES specifically, kept regardless of the configured metric,
    // because the "can this survive to its next use" question is always a bytes question
    // even when the ordering rule is not.
    std::size_t reuse_bytes = 0;

    // Where the policy is installed and where it is dropped. Every Persist gets a matching
    // ClearPolicy from these two -- spec section 32, enforced by TransitPlan::validate().
    KernelId first_consumer = kInvalidKernelId;
    KernelId last_consumer = kInvalidKernelId;
    // The consumer kernels in order, for per-kernel binding and for prefetch placement.
    std::vector<KernelId> consumers;

    int request_id = -1;

    // Modelled bytes saved per byte of budget. The knapsack score, and the reason a small
    // frequently-reused tensor outranks a large one that is read twice.
    double density() const noexcept {
        return bytes ? static_cast<double>(saved_bytes) / static_cast<double>(bytes) : 0.0;
    }
    // 1/(1+distance): near uses are urgent, far ones are not. Bounded so a distance of zero
    // does not divide by anything.
    double urgency() const noexcept { return 1.0 / (1.0 + static_cast<double>(distance)); }
};

// Builds the candidate list from a graph, filtered to `roles`. Tensors with no reuse are
// excluded and reported through `declined` -- a cache cannot serve a read that never
// happens, and saying so explicitly is what turns "the plan did nothing" into a diagnosis.
std::vector<Candidate> build_candidates(const PlanInput& input, RoleMask roles,
                                        ReuseMetric metric,
                                        std::vector<TransitDecline>* declined);

// The locality budget this config asks for on this device, clamped to what the hardware
// will grant and to what the runtime says is already spent.
std::size_t resolve_budget(const TransitPlannerConfig& config, const PlanInput& input);

// What a plan is asking the persisting partition to hold, and what else is flowing through
// it. Everything the residency cost model needs that is not a property of one candidate --
// which is the whole point of that model: the terms couple.
struct CostContext {
    CostModel model = CostModel::Residency;
    std::size_t budget_bytes = 0;      // C: what the driver granted
    std::size_t resident_bytes = 0;    // what the whole plan asks to keep resident
    std::size_t stream_relieved_bytes = 0;  // interference a Stream action takes out of D
    std::size_t l2_bytes = 0;          // the whole cache the reservation is carved from
    std::size_t step_traffic_bytes = 0;
    double beta = 0.1100;
    double reservation_cost = 0.001;
    double stream_relief = 1.0;
    std::size_t cache_line_bytes = 128;

    static CostContext from(const TransitPlannerConfig& config, std::size_t budget,
                            const PlanInput& input) noexcept {
        CostContext out;
        out.model = config.cost_model;
        out.budget_bytes = budget;
        out.beta = config.residency_beta;
        out.reservation_cost = config.reservation_cost;
        out.stream_relief = config.stream_relief;
        out.cache_line_bytes = config.cache_line_bytes;
        out.l2_bytes = input.device.l2_bytes;
        out.step_traffic_bytes = input.graph ? input.graph->step_traffic_bytes() : 0;
        return out;
    }
};

// What the reservation costs the traffic it displaces, in the same byte currency as the
// saving. Zero under the linear model, which is why that model could never say a policy was
// not worth its set-aside.
std::size_t reservation_cost_bytes(const CostContext& context) noexcept;

// Bytes of `t` the hardware is actually being asked to keep, quantised to whole cache lines.
// A grant that cannot hold one line holds nothing -- there is no 30% of a byte, and pricing
// one is how the linear model came to flatten AdmissionRule::Quota to a thousandth of a point.
std::size_t resident_bytes(std::size_t granted, std::size_t tensor_bytes, double hit_ratio,
                           const CostContext& context) noexcept;

// The share of this candidate's admitted lines that survive to their next use, in [0,1].
// Exactly 1.0 under CostModel::Linear, which is what makes that model separable.
double survival(const Candidate& candidate, const CostContext& context) noexcept;

// Bytes the cost model says a tensor saves when `granted` bytes of budget hold it at
// `hit_ratio`.
//
// The model is deliberately the simplest defensible one: saving scales linearly with the
// share of the tensor that is resident and with the requested hit ratio. It assumes a
// perfect replacement policy within the set-aside, which makes every figure it produces an
// UPPER bound -- the same assumption every ceiling in this repository is quoted under, and
// the reason none of them may be reported as an expected gain.
std::size_t modelled_saving(const Candidate& candidate, std::size_t granted, double hit_ratio);
// The same, under an explicit cost model. The two-argument form above is the linear model and
// stays for callers that have no context to give.
std::size_t modelled_saving(const Candidate& candidate, std::size_t granted, double hit_ratio,
                            const CostContext& context);

// Emits Persist/ClearPolicy for an admitted candidate, and accumulates the cost model.
// `binding` decides whether the window is bound per consumer or once for the whole window.
void emit_persist(TransitPlan* plan, const Candidate& candidate, std::size_t granted,
                  double hit_ratio, bool sticky);

// Emits Prefetch (and, when joining at the end, the fork/join event pair) for candidates
// whose role is in `config.prefetch_roles`.
void emit_prefetch(TransitPlan* plan, const std::vector<Candidate>& candidates,
                   const TransitPlannerConfig& config, const PlanInput& input);

// Emits Stream hints for large tensors that have no reuse -- the other half of a shared
// cache budget. A weight stream that displaces the state a policy just paid to keep is the
// failure this exists to prevent.
// `budget` is what a persisting policy has to work with, which is what decides whether a
// reused tensor could ever be kept -- and therefore whether telling it to stream is free.
void emit_stream_hints(TransitPlan* plan, const PlanInput& input,
                       const TransitPlannerConfig& config, std::size_t budget);

// Fills in the graph-derived parts of the cost model and finalizes the plan.
// `scope` is the roles the planner was allowed to act on, so the ceiling recorded next to
// the prediction is the ceiling for THAT policy rather than for an unrelated one.
// `config` is read for the plan-level constraints that are not any one candidate's business:
// today, how many windows one kernel may carry (max_windows_per_kernel).
void finish_plan(TransitPlan* plan, const PlanInput& input, std::size_t budget,
                 const std::vector<TransitDecline>& declined, RoleMask scope,
                 const TransitPlannerConfig& config);

}  // namespace detail
}  // namespace tensortransit
