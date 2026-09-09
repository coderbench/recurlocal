#include "recurlocal/planner.h"
#include <iostream>
int main() {
    recurlocal::DeviceCaps caps{96ull*1024*1024, 64ull*1024*1024, 32ull*1024*1024};
    recurlocal::PlannerConfig cfg;
    recurlocal::LocalityPlanner p(caps, cfg);
    auto plan = p.plan_for_layer(3ull*1024*1024, true);
    std::cout << "RecurLocal 0.1.0\n"
              << "mode=" << recurlocal::to_string(cfg.mode) << "\n"
              << "example_set_aside_bytes=" << p.recommended_l2_set_aside() << "\n"
              << "example_hot_window_bytes=" << plan.hot_window_bytes << "\n"
              << "example_hit_ratio=" << plan.hit_ratio << "\n"
              << "example_prefetch_next=" << (plan.prefetch_next ? "true" : "false") << "\n";
}
