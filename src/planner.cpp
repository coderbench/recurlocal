#include "recurlocal/planner.h"
#include <algorithm>
#include <stdexcept>
#include <string>

namespace recurlocal {

LocalityPlanner::LocalityPlanner(DeviceCaps caps, PlannerConfig config)
    : caps_(caps), config_(config) {
    if (config_.persisting_budget_fraction < 0.0 || config_.persisting_budget_fraction > 1.0)
        throw std::invalid_argument("persisting_budget_fraction must be in [0,1]");
    if (config_.hit_ratio < 0.0 || config_.hit_ratio > 1.0)
        throw std::invalid_argument("hit_ratio must be in [0,1]");
    if (config_.prefetch_distance < 0 || config_.prefetch_distance > 1)
        throw std::invalid_argument("v0 supports prefetch_distance 0 or 1");
}

std::size_t LocalityPlanner::recommended_l2_set_aside() const noexcept {
    if (!caps_.persisting_l2_max_bytes || config_.persisting_budget_fraction <= 0.0) return 0;
    const auto requested = static_cast<std::size_t>(
        static_cast<double>(caps_.persisting_l2_max_bytes) * config_.persisting_budget_fraction);
    return std::min(requested, caps_.persisting_l2_max_bytes);
}

LayerPlan LocalityPlanner::plan_for_layer(std::size_t current_state_bytes,
                                           bool has_next_recurrent_layer,
                                           std::size_t concurrently_hot_bytes) const noexcept {
    LayerPlan p{};
    const bool use_persist = config_.mode == LocalityMode::Persist || config_.mode == LocalityMode::Combined;
    const bool use_prefetch = config_.mode == LocalityMode::Prefetch || config_.mode == LocalityMode::Combined;

    if (use_persist && current_state_bytes && caps_.access_policy_max_window_bytes) {
        std::size_t window = std::min(current_state_bytes, caps_.access_policy_max_window_bytes);
        if (config_.max_hot_window_bytes) window = std::min(window, config_.max_hot_window_bytes);
        p.hot_window_bytes = window;
        p.use_persisting_window = window > 0;
        p.hit_ratio = config_.hit_ratio;
        const auto budget = recommended_l2_set_aside();
        if (budget && concurrently_hot_bytes + window > budget) {
            const double share = static_cast<double>(budget) / static_cast<double>(concurrently_hot_bytes + window);
            p.hit_ratio = std::clamp(config_.hit_ratio * share, 0.05, config_.hit_ratio);
        }
    }
    p.prefetch_next = use_prefetch && config_.prefetch_distance == 1 && has_next_recurrent_layer;
    return p;
}

const char* to_string(LocalityMode mode) noexcept {
    switch (mode) {
        case LocalityMode::Baseline: return "baseline";
        case LocalityMode::Persist: return "persist";
        case LocalityMode::Prefetch: return "prefetch";
        case LocalityMode::Combined: return "combined";
    }
    return "unknown";
}

LocalityMode parse_mode(const char* text) {
    if (!text) throw std::invalid_argument("mode is null");
    std::string s(text);
    if (s == "baseline") return LocalityMode::Baseline;
    if (s == "persist") return LocalityMode::Persist;
    if (s == "prefetch") return LocalityMode::Prefetch;
    if (s == "combined") return LocalityMode::Combined;
    throw std::invalid_argument("unknown mode: " + s);
}

} // namespace recurlocal
