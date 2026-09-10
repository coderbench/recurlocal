#include "tensortransit/planner.h"

#include <cstring>

namespace tensortransit {

namespace {
struct RuleName { AdmissionRule rule; const char* name; };
constexpr RuleName kRules[] = {
    {AdmissionRule::Density, "density"},
    {AdmissionRule::Quota, "quota"},
    {AdmissionRule::Proportional, "proportional"},
    {AdmissionRule::ReuseOrder, "reuse_order"},
    {AdmissionRule::RoleFloor, "role_floor"},
};
struct TimingName { PrefetchTiming timing; const char* name; };
constexpr TimingName kTimings[] = {
    {PrefetchTiming::FixedDistance, "fixed_distance"},
    {PrefetchTiming::BandwidthAware, "bandwidth_aware"},
    {PrefetchTiming::Eager, "eager"},
};
struct BindingName { WindowBinding binding; const char* name; };
constexpr BindingName kBindings[] = {
    {WindowBinding::PerConsumer, "per_consumer"},
    {WindowBinding::Sticky, "sticky"},
};
struct PresetName { PolicyPreset preset; const char* name; };
constexpr PresetName kPresets[] = {
    {PolicyPreset::Baseline, "baseline"},
    {PolicyPreset::RecurrentOnly, "recurrent_only"},
    {PolicyPreset::KVOnly, "kv_only"},
    {PolicyPreset::NaiveBothPersistent, "naive_both"},
    {PolicyPreset::Global, "global"},
};

// Adding a planner means adding it here and nowhere else. A second edit somewhere far away
// is how a planner comes to exist, build, be tested, and be unreachable from the CLI and
// the sweep harness that are supposed to be measuring it.
const char* const kPlannerNames[] = {"baseline", "recurrent_v0", "greedy", "budgeted",
                                     "concurrency", nullptr};
}  // namespace

const char* to_string(AdmissionRule rule) noexcept {
    for (const auto& entry : kRules)
        if (entry.rule == rule) return entry.name;
    return "density";
}
bool parse_admission_rule(const char* text, AdmissionRule* out) noexcept {
    if (!text || !out) return false;
    for (const auto& entry : kRules)
        if (std::strcmp(text, entry.name) == 0) { *out = entry.rule; return true; }
    return false;
}
const char* to_string(PrefetchTiming timing) noexcept {
    for (const auto& entry : kTimings)
        if (entry.timing == timing) return entry.name;
    return "fixed_distance";
}
bool parse_prefetch_timing(const char* text, PrefetchTiming* out) noexcept {
    if (!text || !out) return false;
    for (const auto& entry : kTimings)
        if (std::strcmp(text, entry.name) == 0) { *out = entry.timing; return true; }
    return false;
}
const char* to_string(WindowBinding binding) noexcept {
    for (const auto& entry : kBindings)
        if (entry.binding == binding) return entry.name;
    return "per_consumer";
}
bool parse_window_binding(const char* text, WindowBinding* out) noexcept {
    if (!text || !out) return false;
    for (const auto& entry : kBindings)
        if (std::strcmp(text, entry.name) == 0) { *out = entry.binding; return true; }
    return false;
}
const char* to_string(PolicyPreset preset) noexcept {
    for (const auto& entry : kPresets)
        if (entry.preset == preset) return entry.name;
    return "baseline";
}
bool parse_policy_preset(const char* text, PolicyPreset* out) noexcept {
    if (!text || !out) return false;
    for (const auto& entry : kPresets)
        if (std::strcmp(text, entry.name) == 0) { *out = entry.preset; return true; }
    return false;
}

const char* validate(const TransitPlannerConfig& config) noexcept {
    if (!(config.budget_fraction >= 0.0 && config.budget_fraction <= 1.0))
        return "budget_fraction must be in [0,1]";
    if (!(config.hit_ratio > 0.0 && config.hit_ratio <= 1.0))
        return "hit_ratio must be in (0,1]";
    if (!(config.min_hit_ratio >= 0.0 && config.min_hit_ratio <= 1.0))
        return "min_hit_ratio must be in [0,1]";
    if (config.min_hit_ratio > config.hit_ratio)
        return "min_hit_ratio exceeds hit_ratio: every candidate would be declined";
    if (!(config.role_floor_share >= 0.0 && config.role_floor_share <= 1.0))
        return "role_floor_share must be in [0,1]";
    if (config.prefetch_distance < 0 || config.prefetch_distance > 64)
        return "prefetch_distance must be in [0,64]";
    if (!(config.max_reuse_distance_budgets >= 0.0))
        return "max_reuse_distance_budgets must be >= 0";
    if (config.max_actions == 0) return "max_actions must be non-zero";
    return nullptr;
}

TransitPlannerConfig preset_config(PolicyPreset preset, const TransitPlannerConfig& base) {
    // Every arm starts from the SAME base, so the dials that are not part of the comparison
    // -- hit ratio, budget fraction, prefetch settings -- are identical across the five.
    // An arm that differed in two things at once would not be a comparison of either.
    TransitPlannerConfig config = base;
    switch (preset) {
        case PolicyPreset::Baseline:
            config.persist_roles = RoleMask::none();
            config.prefetch_roles = RoleMask::none();
            config.stream_roles = RoleMask::none();
            config.prefetch_enabled = false;
            break;
        case PolicyPreset::RecurrentOnly:
            config.persist_roles = RoleMask::of(TensorRole::RecurrentState);
            config.admission = AdmissionRule::Density;
            break;
        case PolicyPreset::KVOnly:
            config.persist_roles = RoleMask::of(TensorRole::KVCache);
            config.admission = AdmissionRule::Density;
            break;
        case PolicyPreset::NaiveBothPersistent:
            // Both roles, and NO arbitration: every candidate asks for the full hit ratio
            // and the budget is divided by shaving. This is the straw man that has to be
            // beaten for "global coordination" to mean anything, and it is a straw man
            // people actually build -- it is what you get by turning two independent
            // policies on at once.
            config.persist_roles = RoleMask::of(TensorRole::RecurrentState, TensorRole::KVCache);
            config.admission = AdmissionRule::Proportional;
            break;
        case PolicyPreset::Global:
            config.persist_roles = RoleMask::of(TensorRole::RecurrentState, TensorRole::KVCache);
            config.admission = AdmissionRule::RoleFloor;
            // The global arm is the only one allowed to spend the OTHER half of a shared
            // cache budget: telling the weight stream to get out of the way is a
            // coordination action, and an arm that could not take it would be arguing
            // against itself.
            config.stream_roles = RoleMask::of(TensorRole::ModelWeight, TensorRole::ExpertWeight);
            break;
    }
    return config;
}

const char* preset_planner(PolicyPreset preset) noexcept {
    return preset == PolicyPreset::Baseline ? "baseline" : "budgeted";
}

std::unique_ptr<ITransitPlanner> make_planner(const char* name,
                                              const TransitPlannerConfig& config) {
    if (!name) return nullptr;
    if (std::strcmp(name, "baseline") == 0) return make_baseline_planner();
    if (std::strcmp(name, "recurrent_v0") == 0) return make_recurrent_v0_planner(config);
    if (std::strcmp(name, "greedy") == 0) return make_greedy_planner(config);
    if (std::strcmp(name, "budgeted") == 0) return make_budgeted_planner(config);
    if (std::strcmp(name, "concurrency") == 0) return make_concurrency_planner(config);
    // Deliberately not a fallback to baseline. A harness that silently planned with the
    // wrong planner would report a measurement of something nobody asked for, and this
    // project's whole history is of confident numbers that measured the wrong thing.
    return nullptr;
}

const char* const* planner_names() noexcept { return kPlannerNames; }

}  // namespace tensortransit
