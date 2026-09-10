#include "planners/common.h"

#include <algorithm>
#include <vector>

namespace tensortransit {
namespace {

// One tensor's share of the budget, as decided by an AdmissionRule.
struct Grant {
    std::size_t index = 0;   // into the candidate vector
    std::size_t bytes = 0;   // budget bytes committed to it
    double hit_ratio = 0.0;
};

// The general planner: an explicit budget, a named rule for dividing it, and a decline
// record for everything that did not fit.
//
// This is the planner the multi-tensor claim runs on. Every arm of spec section 38's
// comparison -- recurrent-only, KV-only, naive both-persistent, global -- is THIS planner
// with a different RoleMask and a different AdmissionRule, which is what stops the arms of a
// comparison drifting apart between runs.
class BudgetedPlanner final : public ITransitPlanner {
public:
    explicit BudgetedPlanner(const TransitPlannerConfig& config) : config_(config) {}

    TransitPlan build_plan(const PlanInput& input) override {
        TransitPlan plan;
        plan.set_planner_name(name());
        if (!input.valid()) return plan;

        std::vector<TransitDecline> declined;
        auto candidates = detail::build_candidates(input, config_.persist_roles,
                                                   config_.reuse_metric, &declined);
        const std::size_t budget = detail::resolve_budget(config_, input);
        if (!budget) {
            for (const auto& candidate : candidates)
                declined.push_back(TransitDecline{candidate.tensor, candidate.role,
                                                  input.device.supports_persisting_l2()
                                                      ? DeclineReason::BudgetExhausted
                                                      : DeclineReason::NotSupported,
                                                  candidate.bytes, candidate.saved_bytes});
            detail::emit_stream_hints(&plan, input, config_);
            detail::finish_plan(&plan, input, budget, declined, config_.persist_roles);
            return plan;
        }

        candidates = screen(std::move(candidates), input, budget, &declined);
        const std::vector<Grant> grants = admit(candidates, budget, &declined);

        // Sticky binding is singular in hardware, so at most one grant gets it: the first,
        // which every rule orders as its best. Handing it to two tensors would be a plan the
        // device cannot execute, and the executor would silently apply only the last one.
        bool sticky_taken = false;
        for (const Grant& grant : grants) {
            const bool sticky = config_.window_binding == WindowBinding::Sticky && !sticky_taken;
            if (sticky) sticky_taken = true;
            detail::emit_persist(&plan, candidates[grant.index], grant.bytes, grant.hit_ratio,
                                 sticky);
        }

        detail::emit_prefetch(&plan, candidates, config_, input);
        detail::emit_stream_hints(&plan, input, config_);
        detail::finish_plan(&plan, input, budget, declined, config_.persist_roles);
        return plan;
    }

    const char* name() const noexcept override { return "budgeted"; }

private:
    // Rejects what no rule should be allowed to admit, before any of them runs. Kept
    // separate so that every rule is screened identically -- a filter applied inside one
    // rule and not another would show up as that rule being better.
    std::vector<detail::Candidate> screen(std::vector<detail::Candidate> candidates,
                                          const PlanInput& input, std::size_t budget,
                                          std::vector<TransitDecline>* declined) const {
        std::vector<detail::Candidate> kept;
        const std::size_t hardware_limit = input.device.max_window_bytes();
        for (auto& candidate : candidates) {
            // A tensor larger than the largest window the hardware will honour cannot be
            // made resident whatever the budget. The driver would silently truncate the
            // request, and the telemetry would report a policy protecting several times
            // what it actually covers.
            if (hardware_limit && candidate.bytes > hardware_limit &&
                candidate.bytes > budget) {
                declined->push_back(TransitDecline{candidate.tensor, candidate.role,
                                                   DeclineReason::TooLarge, candidate.bytes,
                                                   candidate.saved_bytes});
                continue;
            }
            // More traffic runs between two uses than the cache can survive: the line is
            // gone before it pays off, and the reservation cost every other tensor the
            // bytes it held. This is the generalization of the finding that closed the
            // recurrent-only surface -- the reuse distance for a recurrent state is a whole
            // token, so the footprint that must survive is every layer of every sequence.
            if (config_.max_reuse_distance_budgets > 0.0 && budget) {
                const double budgets = static_cast<double>(candidate.reuse_bytes) /
                                       static_cast<double>(budget);
                if (budgets > config_.max_reuse_distance_budgets) {
                    declined->push_back(TransitDecline{candidate.tensor, candidate.role,
                                                       DeclineReason::ReuseTooFar,
                                                       candidate.bytes, candidate.saved_bytes});
                    continue;
                }
            }
            kept.push_back(std::move(candidate));
        }
        return kept;
    }

    std::vector<Grant> admit(std::vector<detail::Candidate>& candidates, std::size_t budget,
                             std::vector<TransitDecline>* declined) const {
        switch (config_.admission) {
            case AdmissionRule::Density:      return admit_ordered(candidates, budget, declined, by_density);
            case AdmissionRule::ReuseOrder:   return admit_ordered(candidates, budget, declined, by_urgency);
            case AdmissionRule::Quota:        return admit_quota(candidates, budget, declined);
            case AdmissionRule::Proportional: return admit_proportional(candidates, budget, declined);
            case AdmissionRule::RoleFloor:    return admit_role_floor(candidates, budget, declined);
        }
        return admit_ordered(candidates, budget, declined, by_density);
    }

    static bool by_density(const detail::Candidate& a, const detail::Candidate& b) {
        return a.density() > b.density();
    }
    static bool by_urgency(const detail::Candidate& a, const detail::Candidate& b) {
        // Nearest next use first, in the configured metric. Belady-flavoured, and it differs
        // from density exactly when the soonest-needed tensor is not the densest one.
        return a.distance < b.distance;
    }

    // Sort, then fill. The knapsack relaxation: greedy on a ratio is optimal for the
    // fractional problem and within a factor for the integral one, and a persisting window
    // IS fractional -- a partly-covered tensor gets a proportionally lower hit ratio.
    template <typename Less>
    std::vector<Grant> admit_ordered(std::vector<detail::Candidate>& candidates,
                                     std::size_t budget, std::vector<TransitDecline>* declined,
                                     Less less) const {
        std::vector<std::size_t> order(candidates.size());
        for (std::size_t i = 0; i < order.size(); ++i) order[i] = i;
        std::stable_sort(order.begin(), order.end(), [&](std::size_t a, std::size_t b) {
            return less(candidates[a], candidates[b]);
        });

        std::vector<Grant> grants;
        std::size_t remaining = budget;
        for (const std::size_t i : order) {
            const detail::Candidate& candidate = candidates[i];
            const std::size_t granted = std::min(remaining, candidate.bytes);
            if (!granted) {
                declined->push_back(TransitDecline{candidate.tensor, candidate.role,
                                                   DeclineReason::BudgetExhausted,
                                                   candidate.bytes, candidate.saved_bytes});
                continue;
            }
            const double share = static_cast<double>(granted) / static_cast<double>(candidate.bytes);
            if (config_.hit_ratio * share < config_.min_hit_ratio) {
                declined->push_back(TransitDecline{candidate.tensor, candidate.role,
                                                   DeclineReason::BelowMinHitRatio,
                                                   candidate.bytes, candidate.saved_bytes});
                continue;
            }
            grants.push_back(Grant{i, granted, config_.hit_ratio});
            remaining -= granted;
        }
        return grants;
    }

    // Admit WHOLE tensors at the full hit ratio until the budget is spent; decline the rest.
    //
    // Models the cache as it actually is: a line is resident or it is not. Where the
    // footprint is a small multiple of the budget, keeping twenty-nine tensors whole is a
    // different request from asking thirty for 97% each, and it is the one the hardware can
    // honour. Where the footprint is many multiples it is inert -- admitting a fortieth of
    // the tensors is not obviously better than thrashing all of them -- which is exactly
    // what its one measurement on hardware found.
    std::vector<Grant> admit_quota(std::vector<detail::Candidate>& candidates,
                                   std::size_t budget,
                                   std::vector<TransitDecline>* declined) const {
        std::vector<std::size_t> order(candidates.size());
        for (std::size_t i = 0; i < order.size(); ++i) order[i] = i;
        std::stable_sort(order.begin(), order.end(), [&](std::size_t a, std::size_t b) {
            return by_density(candidates[a], candidates[b]);
        });

        std::vector<Grant> grants;
        std::size_t remaining = budget;
        for (const std::size_t i : order) {
            const detail::Candidate& candidate = candidates[i];
            if (candidate.bytes > remaining) {
                // Whole or nothing. A partial grant here is what the other rules do, and the
                // difference between them is the mechanism under test.
                declined->push_back(TransitDecline{candidate.tensor, candidate.role,
                                                   DeclineReason::BudgetExhausted,
                                                   candidate.bytes, candidate.saved_bytes});
                continue;
            }
            grants.push_back(Grant{i, candidate.bytes, config_.hit_ratio});
            remaining -= candidate.bytes;
        }
        return grants;
    }

    // Admit everything and scale every hit ratio by the budget share. RecurLocal's v0.1
    // heuristic, kept because it is the CONTROL: every number this repository has published
    // was measured under it, and a rule that cannot beat it has not earned its enumerator.
    std::vector<Grant> admit_proportional(std::vector<detail::Candidate>& candidates,
                                          std::size_t budget,
                                          std::vector<TransitDecline>* declined) const {
        std::size_t wanted = 0;
        for (const auto& candidate : candidates) wanted += candidate.bytes;
        if (!wanted) return {};

        const double share = wanted <= budget
                                 ? 1.0
                                 : static_cast<double>(budget) / static_cast<double>(wanted);
        std::vector<Grant> grants;
        for (std::size_t i = 0; i < candidates.size(); ++i) {
            const detail::Candidate& candidate = candidates[i];
            const double hit = config_.hit_ratio * share;
            if (hit < config_.min_hit_ratio) {
                declined->push_back(TransitDecline{candidate.tensor, candidate.role,
                                                   DeclineReason::BelowMinHitRatio,
                                                   candidate.bytes, candidate.saved_bytes});
                continue;
            }
            const auto granted =
                static_cast<std::size_t>(static_cast<double>(candidate.bytes) * share);
            grants.push_back(Grant{i, granted ? granted : 1, hit});
        }
        return grants;
    }

    // Reserve a floor per role, then spend the remainder by density.
    //
    // This is the rule the whole multi-tensor claim rests on, and the reason it exists is a
    // specific failure of Density on a mixed workload: density is winner-take-all, so
    // whichever role has the better bytes-saved-per-byte takes the entire budget and the
    // "global" planner silently reproduces the recurrent-only or the KV-only arm. A planner
    // that does that has not coordinated anything -- it has picked a side, and a comparison
    // against the independent policies would be comparing one of them against itself.
    //
    // Whether the floor actually beats those arms on a real workload is a MEASUREMENT, and
    // it has not been taken. See docs/evaluation.md.
    std::vector<Grant> admit_role_floor(std::vector<detail::Candidate>& candidates,
                                        std::size_t budget,
                                        std::vector<TransitDecline>* declined) const {
        std::vector<TensorRole> roles;
        for (const auto& candidate : candidates)
            if (std::find(roles.begin(), roles.end(), candidate.role) == roles.end())
                roles.push_back(candidate.role);
        // One role is not a coordination problem; fall through to Density so the rule is
        // exactly its own control on a single-role workload rather than a slightly different
        // one. That equivalence is asserted by a test.
        if (roles.size() <= 1) return admit_ordered(candidates, budget, declined, by_density);

        const auto floor_per_role = static_cast<std::size_t>(
            static_cast<double>(budget) * config_.role_floor_share /
            static_cast<double>(roles.size()));

        std::vector<Grant> grants;
        std::vector<char> granted_flag(candidates.size(), 0);
        std::size_t remaining = budget;

        // Phase 1: each role gets its floor, spent on its own best candidates by density.
        for (const TensorRole role : roles) {
            std::vector<std::size_t> order;
            for (std::size_t i = 0; i < candidates.size(); ++i)
                if (candidates[i].role == role) order.push_back(i);
            std::stable_sort(order.begin(), order.end(), [&](std::size_t a, std::size_t b) {
                return by_density(candidates[a], candidates[b]);
            });
            std::size_t role_remaining = std::min(floor_per_role, remaining);
            for (const std::size_t i : order) {
                if (!role_remaining) break;
                const std::size_t granted = std::min(role_remaining, candidates[i].bytes);
                if (!granted) continue;
                grants.push_back(Grant{i, granted, config_.hit_ratio});
                granted_flag[i] = 1;
                role_remaining -= granted;
                remaining -= granted;
            }
        }

        // Phase 2: the rest by global density, topping up a partly-granted tensor before
        // starting a new one -- half a window is worth less than half of two windows only if
        // the saving is linear in residency, and it is not once a line is evicted.
        std::vector<std::size_t> order(candidates.size());
        for (std::size_t i = 0; i < order.size(); ++i) order[i] = i;
        std::stable_sort(order.begin(), order.end(), [&](std::size_t a, std::size_t b) {
            return by_density(candidates[a], candidates[b]);
        });
        for (const std::size_t i : order) {
            if (!remaining) break;
            std::size_t already = 0;
            for (const Grant& grant : grants)
                if (grant.index == i) already = grant.bytes;
            if (already >= candidates[i].bytes) continue;
            const std::size_t top_up = std::min(remaining, candidates[i].bytes - already);
            if (!top_up) continue;
            bool merged = false;
            for (Grant& grant : grants)
                if (grant.index == i) {
                    grant.bytes += top_up;
                    merged = true;
                    break;
                }
            if (!merged) {
                grants.push_back(Grant{i, top_up, config_.hit_ratio});
                granted_flag[i] = 1;
            }
            remaining -= top_up;
        }

        for (std::size_t i = 0; i < candidates.size(); ++i)
            if (!granted_flag[i])
                declined->push_back(TransitDecline{candidates[i].tensor, candidates[i].role,
                                                   DeclineReason::BudgetExhausted,
                                                   candidates[i].bytes,
                                                   candidates[i].saved_bytes});
        return grants;
    }

    TransitPlannerConfig config_;
};

}  // namespace

std::unique_ptr<ITransitPlanner> make_budgeted_planner(const TransitPlannerConfig& config) {
    return std::unique_ptr<ITransitPlanner>(new BudgetedPlanner(config));
}

}  // namespace tensortransit
