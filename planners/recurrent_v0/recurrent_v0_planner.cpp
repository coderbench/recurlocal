#include <algorithm>

#include "planners/common.h"
#include "tensortransit/recurrent.h"

namespace tensortransit {
namespace {

// RecurLocal's shipped policy, expressed against the Transit Graph.
//
// This planner exists to answer one question and it is an architectural one: does the
// existing recurrent-state design fit the new abstraction naturally, or was the abstraction
// built around a different shape? Spec section 21 and section 77.
//
// The answer had to be checkable, not asserted, so this does NOT reimplement the v0.1
// heuristic. It CALLS it: it builds the same LocalityPlanner from the same DeviceCaps and
// asks it for the same LayerPlan, then translates the answer into actions. A
// reimplementation would be free to drift from the policy every published number in this
// repository was measured under, and nothing in a diff would show it. Here, if the recurrent
// policy changes, this planner changes with it by construction.
class RecurrentV0Planner final : public ITransitPlanner {
public:
    explicit RecurrentV0Planner(const TransitPlannerConfig& config) : config_(config) {}

    TransitPlan build_plan(const PlanInput& input) override {
        TransitPlan plan;
        plan.set_planner_name(name());
        if (!input.valid()) return plan;

        // Only recurrent state, whatever the caller's mask says. This planner IS the
        // single-role policy; a version of it that touched KV would not be the thing whose
        // equivalence with v0.1 is being asserted.
        std::vector<TransitDecline> declined;
        auto candidates = detail::build_candidates(input, RoleMask::of(TensorRole::RecurrentState),
                                                   config_.reuse_metric, &declined);
        const std::size_t budget = detail::resolve_budget(config_, input);

        // The v0.1 planner, with the v0.1 accounting. `TokenFootprint` rather than the
        // shipped `CurrentLayer` default because the graph HAS the footprint -- counting one
        // layer's state and ignoring that every other layer's is equally live is precisely
        // the accounting error the Transit Graph exists to make impossible.
        DeviceCaps caps{};
        caps.l2_bytes = input.device.l2_bytes;
        caps.persisting_l2_max_bytes = input.device.persisting_l2_max_bytes;
        caps.access_policy_max_window_bytes = input.device.access_policy_max_window_bytes;

        PlannerConfig legacy{};
        legacy.mode = config_.prefetch_enabled ? LocalityMode::Combined : LocalityMode::Persist;
        legacy.persisting_budget_fraction = config_.budget_fraction;
        legacy.hit_ratio = config_.hit_ratio;
        legacy.min_hit_ratio = config_.min_hit_ratio;
        legacy.prefetch_distance = config_.prefetch_distance;
        legacy.hot_set_model = HotSetModel::TokenFootprint;
        legacy.hot_set_policy = HotSetPolicy::Proportional;
        if (validate(legacy) != nullptr) return plan;  // never throw out of a planner
        LocalityPlanner legacy_planner(caps, legacy);
        if (input.runtime.granted_budget_bytes)
            legacy_planner.set_granted_l2_set_aside(input.runtime.granted_budget_bytes);

        // The geometry the v0.1 hot-set models count, read off the graph rather than
        // declared by the caller: this is the migration in one statement.
        RecurrentGeometry geometry{};
        geometry.recurrent_layers = static_cast<int>(candidates.size());
        if (!candidates.empty()) {
            std::size_t total = 0;
            for (const auto& candidate : candidates) total += candidate.bytes;
            geometry.bytes_per_layer = total / candidates.size();
        }
        geometry.sequences = input.runtime.active_requests > 0 ? input.runtime.active_requests : 1;
        geometry.streamed_bytes_per_token = input.graph->step_traffic_bytes();

        int layer_index = 0;
        for (const detail::Candidate& candidate : candidates) {
            const LayerPlan layer = legacy_planner.plan_for_layer(
                candidate.bytes, /*has_next_recurrent_layer=*/true, geometry, layer_index++);
            if (!layer.use_persisting_window || layer.hot_window_bytes == 0) {
                declined.push_back(TransitDecline{candidate.tensor, candidate.role,
                                                  layer.hot_set_oversubscribed
                                                      ? DeclineReason::BudgetExhausted
                                                      : DeclineReason::BelowMinHitRatio,
                                                  candidate.bytes, candidate.saved_bytes});
                continue;
            }
            // The v0.1 shape is one window per layer, bound before the recurrent kernel and
            // dropped after it.
            detail::emit_persist(&plan, candidate, layer.hot_window_bytes, layer.hit_ratio,
                                 /*sticky=*/false);
        }

        detail::emit_prefetch(&plan, candidates, config_, input);
        detail::finish_plan(&plan, input, budget, declined, RoleMask::of(TensorRole::RecurrentState));
        return plan;
    }

    const char* name() const noexcept override { return "recurrent_v0"; }

private:
    TransitPlannerConfig config_;
};

}  // namespace

std::unique_ptr<ITransitPlanner> make_recurrent_v0_planner(const TransitPlannerConfig& config) {
    return std::unique_ptr<ITransitPlanner>(new RecurrentV0Planner(config));
}

}  // namespace tensortransit
