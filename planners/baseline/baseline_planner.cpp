#include "planners/common.h"

namespace tensortransit {
namespace {

// The true control (spec section 20).
//
// It emits NOTHING: no persistence, no prefetch, no streaming hint, no extra
// synchronisation. That is deliberately not the same thing as "the general planner with
// every dial at zero" -- that would still walk the candidate list and still pay whatever the
// hook costs, and an A/B against it would be measuring a planner against itself.
//
// It does still fill in the cost model, because the graph-derived parts of it (step traffic,
// removable bytes, peak live set) are properties of the WORKLOAD and not of the plan. Those
// are the numbers a contributor should read before writing a policy at all: if
// `removable_bytes / step_traffic_bytes` is under the significance floor, no planner in this
// repository can help, and finding that out costs one command and no GPU.
class BaselinePlanner final : public ITransitPlanner {
public:
    TransitPlan build_plan(const PlanInput& input) override {
        TransitPlan plan;
        plan.set_planner_name(name());
        if (!input.valid()) return plan;
        // Declines are still reported: "nothing was done, and here is what was on the table"
        // is a more useful control than an empty document.
        std::vector<TransitDecline> declined;
        detail::build_candidates(input, RoleMask::none(), ReuseMetric::Bytes, &declined);
        detail::finish_plan(&plan, input, 0, declined, RoleMask::all());
        return plan;
    }
    const char* name() const noexcept override { return "baseline"; }
};

}  // namespace

std::unique_ptr<ITransitPlanner> make_baseline_planner() {
    return std::unique_ptr<ITransitPlanner>(new BaselinePlanner());
}

}  // namespace tensortransit
