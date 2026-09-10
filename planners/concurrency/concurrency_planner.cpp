#include "planners/common.h"

#include <algorithm>
#include <vector>

namespace tensortransit {
namespace {

// Locality as a shared resource between REQUESTS, not only between tensors.
//
// At concurrency the budget question changes shape. Model weights are read once per decode
// step however many sequences are in flight; request-local state is read once per sequence.
// So the share of traffic a cache could address grows with concurrency while the share it
// can HOLD shrinks -- the footprint grows linearly and the cache does not. That is measured,
// not theorised: on the one model where a persisting policy pays at batch 1 it is negative
// from four sequences on, and the crossover sits at about half residency.
//
// The consequence for a planner is that at high concurrency, spreading the budget evenly
// across requests gives every one of them a share too small to hold anything, and the
// reservation costs the weight stream more than the residency returns. Concentrating it on
// fewer requests keeps those requests' state actually resident. Both are implemented and
// neither is asserted to win -- `share_policy` is the axis.
enum class ShareRule { Even, Priority, Concentrate };

class ConcurrencyPlanner final : public ITransitPlanner {
public:
    explicit ConcurrencyPlanner(const TransitPlannerConfig& config) : config_(config) {}

    TransitPlan build_plan(const PlanInput& input) override {
        TransitPlan plan;
        plan.set_planner_name(name());
        if (!input.valid()) return plan;

        std::vector<TransitDecline> declined;
        auto candidates = detail::build_candidates(input, config_.persist_roles,
                                                   config_.reuse_metric, &declined);
        const std::size_t budget = detail::resolve_budget(config_, input);
        if (!budget || candidates.empty()) {
            for (const auto& candidate : candidates)
                declined.push_back(TransitDecline{candidate.tensor, candidate.role,
                                                  DeclineReason::BudgetExhausted, candidate.bytes,
                                                  candidate.saved_bytes});
            detail::emit_stream_hints(&plan, input, config_, budget);
            detail::finish_plan(&plan, input, budget, declined, config_.persist_roles, config_);
            return plan;
        }

        // Group candidates by request. Request -1 (model-global tensors) is one group and
        // is never starved by the request arbitration: it is shared by everyone.
        std::vector<int> requests;
        for (const auto& candidate : candidates)
            if (std::find(requests.begin(), requests.end(), candidate.request_id) == requests.end())
                requests.push_back(candidate.request_id);
        std::sort(requests.begin(), requests.end());

        // How many requests the budget can actually serve. A share below what one request's
        // resident set needs buys nothing for anybody, so admitting all of them is the
        // failure mode this computes its way out of rather than walking into.
        std::size_t admitted_requests = requests.size();
        if (rule() == ShareRule::Concentrate && !requests.empty()) {
            std::size_t per_request = 0;
            for (const auto& candidate : candidates)
                if (candidate.request_id == requests.front()) per_request += candidate.bytes;
            if (per_request) {
                admitted_requests = budget / per_request;
                if (admitted_requests == 0) admitted_requests = 1;
                admitted_requests = std::min(admitted_requests, requests.size());
            }
        }

        const std::size_t share = requests.empty()
                                      ? budget
                                      : budget / std::max<std::size_t>(admitted_requests, 1);

        std::vector<detail::Candidate> emitted;
        for (std::size_t r = 0; r < requests.size(); ++r) {
            const bool admitted = r < admitted_requests;
            std::size_t remaining = admitted ? share : 0;
            std::vector<std::size_t> order;
            for (std::size_t i = 0; i < candidates.size(); ++i)
                if (candidates[i].request_id == requests[r]) order.push_back(i);
            std::stable_sort(order.begin(), order.end(), [&](std::size_t a, std::size_t b) {
                // benefit x priority x urgency / bytes, the spec section 24 score. Priority
                // is carried on the descriptor; absent one, every request is equal and this
                // degrades to the section 22 score.
                const double sa = candidates[a].density() * candidates[a].urgency();
                const double sb = candidates[b].density() * candidates[b].urgency();
                return sa > sb;
            });
            for (const std::size_t i : order) {
                const detail::Candidate& candidate = candidates[i];
                const std::size_t granted = std::min(remaining, candidate.bytes);
                const double hit_share = candidate.bytes
                                             ? static_cast<double>(granted) /
                                                   static_cast<double>(candidate.bytes)
                                             : 0.0;
                if (!granted || config_.hit_ratio * hit_share < config_.min_hit_ratio) {
                    declined.push_back(TransitDecline{candidate.tensor, candidate.role,
                                                      DeclineReason::BudgetExhausted,
                                                      candidate.bytes, candidate.saved_bytes});
                    continue;
                }
                detail::emit_persist(&plan, candidate, granted, config_.hit_ratio, false);
                emitted.push_back(candidate);
                remaining -= granted;
            }
        }

        detail::emit_prefetch(&plan, emitted, config_, input);
        detail::emit_stream_hints(&plan, input, config_, budget);
        detail::finish_plan(&plan, input, budget, declined, config_.persist_roles, config_);
        return plan;
    }

    const char* name() const noexcept override { return "concurrency"; }

private:
    // Derived rather than configured, for now: `role_floor_share` at its default spreads,
    // and a caller that sets it to zero is asking for concentration. A dedicated enumerator
    // belongs here once there is a measurement to choose between them -- adding one before
    // that would be a dial nobody can set from evidence.
    ShareRule rule() const noexcept {
        return config_.role_floor_share > 0.0 ? ShareRule::Even : ShareRule::Concentrate;
    }
    TransitPlannerConfig config_;
};

}  // namespace

std::unique_ptr<ITransitPlanner> make_concurrency_planner(const TransitPlannerConfig& config) {
    return std::unique_ptr<ITransitPlanner>(new ConcurrencyPlanner(config));
}

}  // namespace tensortransit
