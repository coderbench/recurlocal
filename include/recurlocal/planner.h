#pragma once
#include <cstddef>

namespace recurlocal {

enum class LocalityMode { Baseline, Persist, Prefetch, Combined };

struct DeviceCaps {
    std::size_t l2_bytes = 0;
    std::size_t persisting_l2_max_bytes = 0;
    std::size_t access_policy_max_window_bytes = 0;
};

struct PlannerConfig {
    LocalityMode mode = LocalityMode::Combined;
    double persisting_budget_fraction = 0.75;
    double hit_ratio = 0.70;
    int prefetch_distance = 1;
    std::size_t max_hot_window_bytes = 0;
};

struct LayerPlan {
    std::size_t hot_window_bytes = 0;
    double hit_ratio = 0.0;
    bool use_persisting_window = false;
    bool prefetch_next = false;
};

class LocalityPlanner {
public:
    LocalityPlanner(DeviceCaps caps, PlannerConfig config);
    const DeviceCaps& caps() const noexcept { return caps_; }
    const PlannerConfig& config() const noexcept { return config_; }
    std::size_t recommended_l2_set_aside() const noexcept;
    LayerPlan plan_for_layer(std::size_t current_state_bytes,
                             bool has_next_recurrent_layer,
                             std::size_t concurrently_hot_bytes = 0) const noexcept;
private:
    DeviceCaps caps_;
    PlannerConfig config_;
};

const char* to_string(LocalityMode mode) noexcept;
LocalityMode parse_mode(const char* text);

} // namespace recurlocal
