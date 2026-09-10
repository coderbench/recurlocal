#include "planners/common.h"

#include <algorithm>
#include <cmath>
#include <utility>

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

std::size_t resident_bytes(std::size_t granted, std::size_t tensor_bytes, double hit_ratio,
                           const CostContext& context) noexcept {
    if (!tensor_bytes || !granted || hit_ratio <= 0.0) return 0;
    const std::size_t covered = granted < tensor_bytes ? granted : tensor_bytes;
    double kept = static_cast<double>(covered) * hit_ratio;
    if (kept <= 0.0) return 0;
    if (context.model == CostModel::Linear) return static_cast<std::size_t>(kept);
    // Whole lines only. A grant below one line is not a small amount of residency, it is
    // none: the hardware cannot keep part of a line, and a model that says otherwise is
    // exactly the model that made every admission rule look identical.
    const std::size_t line = context.cache_line_bytes ? context.cache_line_bytes : 1;
    const std::size_t lines = static_cast<std::size_t>(kept) / line;
    return lines * line;
}

double survival(const Candidate& candidate, const CostContext& context) noexcept {
    if (context.model == CostModel::Linear) return 1.0;
    if (!context.budget_bytes) return 0.0;

    // Interference: everything else that flows through the cache between two uses of this
    // tensor. `reuse_bytes` is exactly that -- the bytes of OTHER traffic in between -- which
    // is why the graph precomputes reuse distance in three currencies rather than one. A
    // Stream hint on part of that traffic takes it out of the picture, and that is the only
    // way a Stream action is worth anything at all.
    double interfering = static_cast<double>(candidate.reuse_bytes);
    const double relieved = static_cast<double>(context.stream_relieved_bytes) *
                            context.stream_relief;
    interfering = interfering > relieved ? interfering - relieved : 0.0;
    if (interfering <= 0.0) return 1.0;

    // Capacity against reuse distance, as a POWER LAW: the classic shape of a cache
    // miss-ratio curve, and the shape the measurements pick. An exponential cannot fit the
    // dense and the MoE batch-1 arms at once -- it needs its coefficient to differ by 3.4x
    // between them -- while this one predicts both to within 0.006 points at beta = 0.11.
    //
    // `resident_here` is what THIS tensor holds, not what the plan holds in total, and that
    // is where the coupling lives: a budget spread over more tensors gives each of them less,
    // and the exponent makes the loss superlinear.
    const double resident_here = static_cast<double>(context.resident_bytes);
    if (resident_here <= 0.0) return 0.0;
    const double ratio = resident_here / interfering;
    if (ratio >= 1.0) return 1.0;
    if (context.beta <= 0.0) return 1.0;
    return std::pow(ratio, context.beta);
}

std::size_t reservation_cost_bytes(const CostContext& context) noexcept {
    if (context.model == CostModel::Linear) return 0;
    if (!context.l2_bytes || !context.step_traffic_bytes) return 0;
    const double share = static_cast<double>(context.resident_bytes) /
                         static_cast<double>(context.l2_bytes);
    const double cost = context.reservation_cost * share *
                        static_cast<double>(context.step_traffic_bytes);
    return cost <= 0.0 ? 0 : static_cast<std::size_t>(cost);
}

std::size_t modelled_saving(const Candidate& candidate, std::size_t granted, double hit_ratio,
                            const CostContext& context) {
    if (context.model == CostModel::Linear)
        return modelled_saving(candidate, granted, hit_ratio);
    if (!candidate.bytes || !granted) return 0;
    const std::size_t kept = resident_bytes(granted, candidate.bytes, hit_ratio, context);
    if (!kept) return 0;
    // This tensor's own residency drives its own survival, so the context is narrowed to it.
    CostContext mine = context;
    mine.resident_bytes = kept;
    const double share = static_cast<double>(kept) / static_cast<double>(candidate.bytes);
    const double saved = static_cast<double>(candidate.saved_bytes) * share *
                         survival(candidate, mine);
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
                       const TransitPlannerConfig& config, std::size_t budget) {
    if (config.stream_roles.empty()) return;
    const TransitGraph& graph = *input.graph;
    const TensorRegistry& registry = *input.registry;

    // Tensors this plan is already keeping. Telling one of them to stream would evict exactly
    // what the plan just paid to hold.
    std::vector<TensorId> persisted;
    for (const TransitAction& action : plan->actions())
        if (action.kind == TransitActionKind::Persist)
            persisted.push_back(action.tensor);

    for (const TensorProfile& profile : graph.profiles()) {
        if (!config.stream_roles.has(profile.role)) continue;
        if (std::find(persisted.begin(), persisted.end(), profile.tensor) != persisted.end())
            continue;
        if (profile.bytes < config.stream_min_bytes) continue;

        // "Has nothing to gain from residency" is the right rule, and `has_reuse()` is the
        // wrong test for it over a DECODE window.
        //
        // A decode token is one iteration of a loop, so the graph is cyclic and EVERY tensor
        // is read again next iteration -- including the weight stream. Under `has_reuse()`
        // alone, nothing in a decode graph is ever streamable, and the Stream action was
        // therefore dead code in the one regime this project exists for. It is not that the
        // weights are not re-read; it is that no cache this size can still be holding them
        // when they are.
        //
        // So the test is survival, not reuse: a tensor that could not be kept even if it were
        // given the WHOLE budget is one that will be re-fetched whatever anybody does, and
        // saying so is free. Under the linear model there is no survival term, so the old
        // rule is what remains -- which keeps every plan that model ever produced unchanged.
        if (profile.has_reuse()) {
            if (config.cost_model == CostModel::Linear) continue;
            // More traffic runs between two uses than the whole budget could hold, so the
            // line is gone before it comes back whatever anybody does. This is the same
            // criterion `screen()` declines a candidate on -- DeclineReason::ReuseTooFar --
            // read from the other side: a tensor that cannot be kept is one that can be told
            // to get out of the way for free.
            //
            // Parameter-free on purpose. A threshold on survival would need a constant, and
            // the power law is flat enough that any constant small enough to be defensible
            // would never fire.
            if (!budget) continue;
            if (profile.min_reuse_bytes == static_cast<std::size_t>(-1)) continue;
            if (profile.min_reuse_bytes <= budget) continue;
        }
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

namespace {

// Score used to break the tie when several tensors want the one window a kernel can carry.
double window_score(const TransitAction& action, const PlanInput& input,
                    WindowPreference preference) noexcept {
    switch (preference) {
        case WindowPreference::Widest:
            return static_cast<double>(action.bytes);
        case WindowPreference::Narrowest:
            return action.bytes ? -static_cast<double>(action.bytes) : 0.0;
        case WindowPreference::SoonestReuse: {
            const TensorProfile* profile = input.graph->profile(action.tensor);
            if (!profile || profile->min_reuse_bytes == static_cast<std::size_t>(-1)) return -1e300;
            // Nearest first, so negate: a small intervening-traffic figure is the urgent one.
            return -static_cast<double>(profile->min_reuse_bytes);
        }
        case WindowPreference::Densest:
        default:
            return action.bytes ? static_cast<double>(action.expected_saved_bytes) /
                                      static_cast<double>(action.bytes)
                                : 0.0;
    }
}

// Enforce TransitPlannerConfig::max_windows_per_kernel.
//
// CUDA binds ONE access-policy window to a stream, a launch, or a graph node at a time. A
// plan that marks two regions before one kernel is not describing something the hardware can
// do -- the second replaces the first, the first is silently absent, and the executor's
// telemetry counts two applied windows for one delivered policy. So the plan says which one
// it meant, here, where a CPU test can read it, rather than letting the driver pick by
// overwrite order.
//
// Dropped bindings become NotSupported declines, with the forgone saving attributed, so
// `tensortransit plan` shows the cost of the constraint instead of hiding it.
void cap_windows_per_kernel(TransitPlan* plan, const TransitPlannerConfig& config,
                            const PlanInput& input) {
    const int cap = config.max_windows_per_kernel;
    if (cap <= 0) return;

    std::vector<TransitAction>& actions = plan->actions();
    // Kernels that carry more Persist bindings than the hardware can deliver.
    // Persist AND Stream: a kernel node carries ONE access-policy window and both kinds set
    // it, so a plan that put one of each before the same kernel would have the second
    // silently replace the first.
    const auto windowed = [](TransitActionKind kind) {
        return kind == TransitActionKind::Persist || kind == TransitActionKind::Stream;
    };
    std::vector<KernelId> kernels;
    for (const TransitAction& a : actions)
        if (windowed(a.kind) &&
            std::find(kernels.begin(), kernels.end(), a.before_kernel) == kernels.end())
            kernels.push_back(a.before_kernel);

    // (tensor, kernel) pairs whose binding loses.
    std::vector<std::pair<TensorId, KernelId>> dropped;
    for (const KernelId kernel : kernels) {
        std::vector<std::size_t> here;
        for (std::size_t i = 0; i < actions.size(); ++i)
            if (windowed(actions[i].kind) && actions[i].before_kernel == kernel)
                here.push_back(i);
        if (static_cast<int>(here.size()) <= cap) continue;

        std::stable_sort(here.begin(), here.end(), [&](std::size_t l, std::size_t r) {
            return window_score(actions[l], input, config.window_preference) >
                   window_score(actions[r], input, config.window_preference);
        });
        for (std::size_t rank = static_cast<std::size_t>(cap); rank < here.size(); ++rank)
            dropped.push_back({actions[here[rank]].tensor, kernel});
    }
    if (dropped.empty()) return;

    // Tensors that kept at least one binding somewhere: those are admitted, just not here.
    std::vector<TensorId> survivors;
    for (const TransitAction& a : actions) {
        if (!windowed(a.kind)) continue;
        const bool lost = std::find(dropped.begin(), dropped.end(),
                                    std::make_pair(a.tensor, a.before_kernel)) != dropped.end();
        if (lost) continue;
        if (std::find(survivors.begin(), survivors.end(), a.tensor) == survivors.end())
            survivors.push_back(a.tensor);
    }

    std::vector<TransitAction> kept;
    kept.reserve(actions.size());
    std::vector<TensorId> declined_now;
    for (const TransitAction& a : actions) {
        const bool is_persist = windowed(a.kind);
        const bool is_clear = a.kind == TransitActionKind::ClearPolicy;
        const KernelId kernel = is_persist ? a.before_kernel : a.after_kernel;
        if ((is_persist || is_clear) &&
            std::find(dropped.begin(), dropped.end(), std::make_pair(a.tensor, kernel)) !=
                dropped.end()) {
            if (is_persist &&
                std::find(survivors.begin(), survivors.end(), a.tensor) == survivors.end() &&
                std::find(declined_now.begin(), declined_now.end(), a.tensor) ==
                    declined_now.end()) {
                declined_now.push_back(a.tensor);
                plan->decline(TransitDecline{a.tensor, a.role, DeclineReason::NotSupported,
                                             a.bytes, a.expected_saved_bytes});
            }
            continue;
        }
        kept.push_back(a);
    }
    actions.swap(kept);
}

}  // namespace

void finish_plan(TransitPlan* plan, const PlanInput& input, std::size_t budget,
                 const std::vector<TransitDecline>& declined, RoleMask scope,
                 const TransitPlannerConfig& config) {
    for (const TransitDecline& d : declined) plan->decline(d);
    // Before anything is costed: a binding the hardware cannot deliver must not be counted
    // as budget spent or as saving predicted.
    cap_windows_per_kernel(plan, config, input);

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

    // Re-price the finished plan under the configured cost model.
    //
    // Deliberately AFTER admission and in one place, because the residency model is not
    // separable: a candidate's survival depends on how much the WHOLE plan is asking the
    // partition to hold, which is not known while the admission rule is still deciding. An
    // admission rule that wants to decide using this price asks for it explicitly -- see
    // AdmissionRule::Survival -- and the difference between deciding with it and merely
    // reporting it is the difference between a rule and a label.
    //
    // Plan DIGESTS are unaffected: the digest covers actions, and this changes only what they
    // are predicted to be worth.
    if (config.cost_model != CostModel::Linear) {
        CostContext context = CostContext::from(config, budget, input);
        // What the plan asks to keep resident, charged ONCE per tensor: per-consumer binding
        // emits a Persist per consumer over the same region, and summing them would report a
        // partition several times oversubscribed by a plan that fits.
        std::vector<TensorId> seen;
        for (const TransitAction& action : plan->actions()) {
            if (action.kind != TransitActionKind::Persist) continue;
            if (std::find(seen.begin(), seen.end(), action.tensor) != seen.end()) continue;
            seen.push_back(action.tensor);
            const TensorProfile* profile = input.graph->profile(action.tensor);
            const std::size_t tensor_bytes = profile ? profile->bytes : action.bytes;
            context.resident_bytes += resident_bytes(action.bytes, tensor_bytes,
                                                     action.hit_ratio, context);
        }
        // A Stream hint is the only way the streaming half of the cache stops evicting the
        // persisting half, and under the linear model it was worth exactly nothing.
        for (const TransitAction& action : plan->actions())
            if (action.kind == TransitActionKind::Stream)
                context.stream_relieved_bytes += action.bytes;

        std::size_t repriced = 0;
        seen.clear();
        for (TransitAction& action : plan->actions()) {
            if (action.kind != TransitActionKind::Persist) continue;
            const TensorProfile* profile = input.graph->profile(action.tensor);
            if (!profile) continue;
            Candidate candidate{};
            candidate.tensor = action.tensor;
            candidate.role = action.role;
            candidate.profile = profile;
            candidate.bytes = profile->bytes;
            candidate.saved_bytes = profile->reused_bytes;
            candidate.reuse_bytes = profile->min_reuse_bytes == static_cast<std::size_t>(-1)
                                        ? 0
                                        : profile->min_reuse_bytes;
            action.expected_saved_bytes =
                modelled_saving(candidate, action.bytes, action.hit_ratio, context);
            if (std::find(seen.begin(), seen.end(), action.tensor) != seen.end()) continue;
            seen.push_back(action.tensor);
            repriced += action.expected_saved_bytes;
        }
        // What the reservation costs the traffic it displaces. Subtracted, not ignored: the
        // persist family measures NEGATIVE on the concurrency arms, and a model with only a
        // benefit term cannot express that -- which is why every prediction this repository
        // made before 0.2.1 had the wrong sign there.
        const std::size_t reservation = reservation_cost_bytes(context);
        cost.predicted_saved_bytes = repriced > reservation ? repriced - reservation : 0;
        cost.predicted_gross_saved_bytes = repriced;
        cost.reservation_cost_bytes = reservation;
        cost.resident_bytes = context.resident_bytes;
        cost.stream_relieved_bytes = context.stream_relieved_bytes;
    } else {
        cost.resident_bytes = committed;
    }
    cost.cost_model = config.cost_model;

    plan->finalize();
}

}  // namespace detail
}  // namespace tensortransit
