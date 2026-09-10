#include "planners/common.h"

#include <algorithm>

namespace tensortransit {
namespace {

// Spec section 22's planner, implemented exactly as written:
//
//     benefit = reuse_weight x urgency / tensor_size
//     urgency = 1 / (1 + next_use_distance)
//
// It is intentionally simple and it is intentionally kept, because it is the control that
// BudgetedPlanner's admission rules have to beat. Its known weakness is that it is
// scale-free in the wrong direction: dividing by size makes it prefer many small tensors,
// so on a workload where one large tensor carries most of the reuse it spends the budget
// everywhere except where the traffic is.
class GreedyPlanner final : public ITransitPlanner {
public:
    explicit GreedyPlanner(const TransitPlannerConfig& config) : config_(config) {}

    TransitPlan build_plan(const PlanInput& input) override {
        TransitPlan plan;
        plan.set_planner_name(name());
        if (!input.valid()) return plan;

        std::vector<TransitDecline> declined;
        auto candidates = detail::build_candidates(input, config_.persist_roles,
                                                   config_.reuse_metric, &declined);
        const std::size_t budget = detail::resolve_budget(config_, input);

        std::stable_sort(candidates.begin(), candidates.end(),
                         [](const detail::Candidate& a, const detail::Candidate& b) {
                             return a.density() * a.urgency() > b.density() * b.urgency();
                         });

        std::size_t remaining = budget;
        for (const detail::Candidate& candidate : candidates) {
            if (!remaining) {
                declined.push_back(TransitDecline{candidate.tensor, candidate.role,
                                                  DeclineReason::BudgetExhausted, candidate.bytes,
                                                  candidate.saved_bytes});
                continue;
            }
            const std::size_t granted = std::min(remaining, candidate.bytes);
            const double share = static_cast<double>(granted) / static_cast<double>(candidate.bytes);
            const double hit = config_.hit_ratio * share;
            if (hit < config_.min_hit_ratio) {
                declined.push_back(TransitDecline{candidate.tensor, candidate.role,
                                                  DeclineReason::BelowMinHitRatio, candidate.bytes,
                                                  candidate.saved_bytes});
                continue;
            }
            detail::emit_persist(&plan, candidate, granted, config_.hit_ratio, false);
            remaining -= granted;
        }

        detail::emit_prefetch(&plan, candidates, config_, input);
        detail::emit_stream_hints(&plan, input, config_, budget);
        detail::finish_plan(&plan, input, budget, declined, config_.persist_roles, config_);
        return plan;
    }

    const char* name() const noexcept override { return "greedy"; }

private:
    TransitPlannerConfig config_;
};

}  // namespace

std::unique_ptr<ITransitPlanner> make_greedy_planner(const TransitPlannerConfig& config) {
    return std::unique_ptr<ITransitPlanner>(new GreedyPlanner(config));
}

}  // namespace tensortransit
