#include "planners/common.h"

#include <algorithm>

#include "tensortransit/ceiling.h"

namespace tensortransit {
namespace detail {

std::vector<Candidate> build_candidates(const PlanInput& input, RoleMask roles,
                                        ReuseMetric metric,
                                        std::vector<TransitDecline>* declined) {
    std::vector<Candidate> out;
    if (!input.valid()) return out;
    const TransitGraph& graph = *input.graph;
    const TensorRegistry& registry = *input.registry;

    for (const TensorProfile& profile : graph.profiles()) {
        const TensorDesc* desc = registry.find(profile.tensor);
        if (!desc) continue;

        if (!roles.has(profile.role)) {
            if (declined && profile.has_reuse())
                declined->push_back(TransitDecline{profile.tensor, profile.role,
                                                   DeclineReason::RoleExcluded, profile.bytes,
                                                   profile.reused_bytes});
            continue;
        }
        if (!profile.has_reuse()) {
            if (declined)
                declined->push_back(TransitDecline{profile.tensor, profile.role,
                                                   DeclineReason::NoReuse, profile.bytes, 0});
            continue;
        }

        Candidate candidate{};
        candidate.tensor = profile.tensor;
        candidate.role = profile.role;
        candidate.desc = desc;
        candidate.profile = &profile;
        candidate.bytes = profile.bytes;
        candidate.saved_bytes = profile.reused_bytes;
        candidate.reuse_bytes = profile.min_reuse_bytes;
        candidate.request_id = profile.request_id;

        // Nearest reuse under the configured metric. Not the mean: a planner deciding
        // whether something is worth keeping cares about the soonest moment it pays off.
        std::uint64_t nearest = kNoNextUse;
        for (const TransitEdge& edge : graph.edges()) {
            if (edge.tensor != profile.tensor) continue;
            nearest = std::min(nearest, edge.distance(metric));
        }
        candidate.distance = nearest == kNoNextUse ? 0 : nearest;

        for (const TensorUse& use : graph.uses()) {
            if (use.tensor != profile.tensor) continue;
            if (!has_read_traffic(use.access)) continue;
            candidate.consumers.push_back(use.kernel);
        }
        if (candidate.consumers.empty()) {
            if (declined)
                declined->push_back(TransitDecline{profile.tensor, profile.role,
                                                   DeclineReason::NoReuse, profile.bytes, 0});
            continue;
        }
        candidate.first_consumer = candidate.consumers.front();
        candidate.last_consumer = candidate.consumers.back();
        out.push_back(std::move(candidate));
    }
    return out;
}

std::size_t resolve_budget(const TransitPlannerConfig& config, const PlanInput& input) {
    const DeviceProfile& device = input.device;
    if (!device.supports_persisting_l2()) return 0;
    if (config.budget_fraction <= 0.0) return 0;

    // What the executor actually HOLDS beats what the config asked for. The driver rounds a
    // request up and the device has a non-zero default before anyone asks, so budgeting
    // against the request makes every admission decision wrong by the rounding.
    std::size_t budget = input.runtime.granted_budget_bytes;
    if (!budget) {
        budget = device.persisting_l2_max_bytes;
        if (config.budget_fraction < 1.0)
            budget = static_cast<std::size_t>(static_cast<double>(budget) * config.budget_fraction);
    }
    budget = std::min(budget, device.max_window_bytes());
    // Somebody else in this process already holds part of the partition.
    if (input.runtime.committed_budget_bytes >= budget) return 0;
    return budget - input.runtime.committed_budget_bytes;
}

std::size_t modelled_saving(const Candidate& candidate, std::size_t granted, double hit_ratio) {
    if (!candidate.bytes || !granted) return 0;
    double share = static_cast<double>(granted) / static_cast<double>(candidate.bytes);
    if (share > 1.0) share = 1.0;
    const double saved = static_cast<double>(candidate.saved_bytes) * share * hit_ratio;
    return saved <= 0.0 ? 0 : static_cast<std::size_t>(saved);
}

void emit_persist(TransitPlan* plan, const Candidate& candidate, std::size_t granted,
                  double hit_ratio, bool sticky) {
    const std::size_t region_bytes = std::min(granted, candidate.bytes);
    if (!region_bytes) return;

    TransitAction persist{};
    persist.kind = TransitActionKind::Persist;
    persist.tensor = candidate.tensor;
    persist.role = candidate.role;
    persist.ptr = candidate.desc->ptr;
    persist.bytes = region_bytes;
    persist.hit_ratio = hit_ratio;
    persist.expected_saved_bytes = modelled_saving(candidate, granted, hit_ratio);

    if (sticky) {
        // One binding for the whole recorded window: install before the first consumer and
        // drop after the last. Fewer actions on the critical path, and the region stays
        // marked across the kernels in between -- but only one window can be bound to a
        // stream at a time, so a plan that makes several tensors sticky is asking the
        // hardware for something it cannot do. BudgetedPlanner therefore makes at most the
        // top candidate sticky; the rest bind per consumer.
        persist.before_kernel = candidate.first_consumer;
        plan->add(persist);

        TransitAction clear{};
        clear.kind = TransitActionKind::ClearPolicy;
        clear.tensor = candidate.tensor;
        clear.role = candidate.role;
        clear.after_kernel = candidate.last_consumer;
        plan->add(clear);
        return;
    }

    // Per-consumer binding: the v0.1 shape. Bind immediately before each kernel that reads
    // the tensor and drop immediately after, so no policy outlives the kernel it was for.
    for (const KernelId consumer : candidate.consumers) {
        TransitAction bind = persist;
        bind.before_kernel = consumer;
        plan->add(bind);

        TransitAction clear{};
        clear.kind = TransitActionKind::ClearPolicy;
        clear.tensor = candidate.tensor;
        clear.role = candidate.role;
        clear.after_kernel = consumer;
        plan->add(clear);
    }
}

void emit_prefetch(TransitPlan* plan, const std::vector<Candidate>& candidates,
                   const TransitPlannerConfig& config, const PlanInput& input) {
    if (!config.prefetch_enabled || config.prefetch_roles.empty()) return;
    const TransitGraph& graph = *input.graph;
    const std::vector<KernelEvent>& kernels = graph.kernels();
    if (kernels.empty()) return;

    std::uint32_t next_event = 1;
    for (const Candidate& candidate : candidates) {
        if (!config.prefetch_roles.has(candidate.role)) continue;
        // A fork and a join are permanent graph nodes under capture, ~0.027% of a decode
        // step each. A region too small to earn that back must not be prefetched, and the
        // threshold is a measured one rather than a taste.
        if (candidate.bytes < config.prefetch_min_bytes) continue;

        for (const KernelId consumer : candidate.consumers) {
            // Find the kernel `distance` positions before the consumer.
            std::size_t consumer_index = 0;
            bool found = false;
            for (std::size_t i = 0; i < kernels.size(); ++i)
                if (kernels[i].id == consumer) {
                    consumer_index = i;
                    found = true;
                    break;
                }
            if (!found) continue;

            std::size_t issue_index = 0;
            switch (config.prefetch_timing) {
                case PrefetchTiming::Eager:
                    issue_index = 0;
                    break;
                case PrefetchTiming::BandwidthAware: {
                    // Walk back until the accumulated estimated duration covers the time the
                    // transfer needs at peak bandwidth. Degrades to FixedDistance when the
                    // trace carries no estimates, rather than issuing at index 0 and calling
                    // that "bandwidth aware".
                    const std::uint64_t bandwidth = input.device.peak_bandwidth_bytes_per_s;
                    if (!bandwidth || !kernels[consumer_index].estimated_duration_ns) {
                        issue_index = consumer_index >= static_cast<std::size_t>(config.prefetch_distance)
                                          ? consumer_index - static_cast<std::size_t>(config.prefetch_distance)
                                          : 0;
                        break;
                    }
                    const double need_ns = static_cast<double>(candidate.bytes) * 1e9 /
                                           static_cast<double>(bandwidth);
                    double have_ns = 0.0;
                    issue_index = consumer_index;
                    while (issue_index > 0 && have_ns < need_ns) {
                        --issue_index;
                        have_ns += static_cast<double>(kernels[issue_index].estimated_duration_ns);
                    }
                    break;
                }
                case PrefetchTiming::FixedDistance:
                default:
                    issue_index = consumer_index >= static_cast<std::size_t>(config.prefetch_distance)
                                      ? consumer_index - static_cast<std::size_t>(config.prefetch_distance)
                                      : 0;
                    break;
            }
            if (issue_index == consumer_index) continue;  // nowhere to hide it

            const std::uint32_t event_id = next_event++;
            TransitAction fork{};
            fork.kind = TransitActionKind::RecordEvent;
            fork.tensor = candidate.tensor;
            fork.role = candidate.role;
            fork.before_kernel = kernels[issue_index].id;
            fork.event_id = event_id;
            plan->add(fork);

            TransitAction prefetch{};
            prefetch.kind = TransitActionKind::Prefetch;
            prefetch.tensor = candidate.tensor;
            prefetch.role = candidate.role;
            prefetch.ptr = candidate.desc->ptr;
            prefetch.bytes = candidate.bytes;
            prefetch.before_kernel = kernels[issue_index].id;
            prefetch.event_id = event_id;
            prefetch.stream_id = 1;  // the prefetch stream; a pre-touch on the compute
                                     // stream is not a prefetch, it is extra critical path
            plan->add(prefetch);

            TransitAction join{};
            join.kind = TransitActionKind::WaitEvent;
            join.tensor = candidate.tensor;
            join.role = candidate.role;
            join.event_id = event_id;
            // One join per token instead of one per prefetch halves the graph nodes, at the
            // cost of no ordering between a prefetch and the kernel it warms. Worth 1.20
            // points on the one model where it was measured -- more than the locality it
            // was overlapping, which is why it is an axis and not a constant.
            join.before_kernel = config.prefetch_join_at_end ? kernels.back().id : consumer;
            plan->add(join);
        }
    }
}

void emit_stream_hints(TransitPlan* plan, const PlanInput& input,
                       const TransitPlannerConfig& config) {
    if (config.stream_roles.empty()) return;
    const TransitGraph& graph = *input.graph;
    const TensorRegistry& registry = *input.registry;

    for (const TensorProfile& profile : graph.profiles()) {
        if (!config.stream_roles.has(profile.role)) continue;
        // Only tensors with nothing to gain from residency. Marking a reused tensor as
        // streaming would actively evict the thing a policy just paid to keep.
        if (profile.has_reuse()) continue;
        if (profile.bytes < config.stream_min_bytes) continue;
        const TensorDesc* desc = registry.find(profile.tensor);
        if (!desc) continue;

        // Placed on the FIRST kernel that touches it and dropped after the last, so the hint
        // covers the streaming region and nothing else.
        KernelId first = kInvalidKernelId, last = kInvalidKernelId;
        for (const TensorUse& use : graph.uses()) {
            if (use.tensor != profile.tensor) continue;
            if (first == kInvalidKernelId) first = use.kernel;
            last = use.kernel;
        }
        if (first == kInvalidKernelId) continue;

        TransitAction stream{};
        stream.kind = TransitActionKind::Stream;
        stream.tensor = profile.tensor;
        stream.role = profile.role;
        stream.ptr = desc->ptr;
        stream.bytes = desc->bytes;
        stream.before_kernel = first;
        plan->add(stream);

        TransitAction clear{};
        clear.kind = TransitActionKind::ClearPolicy;
        clear.tensor = profile.tensor;
        clear.role = profile.role;
        clear.after_kernel = last;
        plan->add(clear);
    }
}

void finish_plan(TransitPlan* plan, const PlanInput& input, std::size_t budget,
                 const std::vector<TransitDecline>& declined, RoleMask scope) {
    for (const TransitDecline& d : declined) plan->decline(d);

    PlanCostModel& cost = plan->cost();
    cost.step_traffic_bytes = input.graph->step_traffic_bytes();
    cost.removable_bytes = input.graph->removable_bytes();
    cost.peak_live_bytes = input.graph->peak_live_bytes();
    cost.budget_bytes = budget;
    // What a PERFECT policy with this device's budget could have removed over the roles this
    // planner was allowed to act on. The honest denominator for "how much room is left":
    // comparing a plan against an unlimited-cache figure would say every plan is hopeless,
    // and comparing it against nothing at all is how a plan comes to look finished.
    cost.bounded_removable_bytes =
        compute_ceiling(*input.graph, input.device, scope).bounded_saved_bytes;

    std::size_t predicted = 0, committed = 0;
    for (const TransitAction& action : plan->actions()) {
        if (action.kind != TransitActionKind::Persist) continue;
        predicted += action.expected_saved_bytes;
        committed += action.bytes;
    }
    // Per-consumer binding emits one Persist per consumer over the SAME region, so summing
    // them would count one reservation several times and report a plan spending many times
    // its budget. Charge each tensor's region once.
    std::vector<TensorId> counted;
    committed = 0;
    predicted = 0;
    for (const TransitAction& action : plan->actions()) {
        if (action.kind != TransitActionKind::Persist) continue;
        if (std::find(counted.begin(), counted.end(), action.tensor) != counted.end()) continue;
        counted.push_back(action.tensor);
        committed += action.bytes;
        predicted += action.expected_saved_bytes;
    }
    cost.committed_bytes = committed;
    cost.predicted_saved_bytes = predicted;

    plan->finalize();
}

}  // namespace detail
}  // namespace tensortransit
