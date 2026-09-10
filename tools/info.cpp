#include <cstdlib>
#include <iostream>

#include "tensortransit/ceiling.h"
#include "tensortransit/recurrent.h"
#include "tensortransit/version.h"
#if defined(TENSORTRANSIT_WITH_CUDA) || defined(RECURLOCAL_WITH_CUDA) || \
    defined(RECURLLOCAL_WITH_CUDA)
#include "tensortransit/cuda_executor.h"
#include "tensortransit/cuda_recurrent.h"
#endif

int main(int argc, char** argv) {
    const int device = (argc > 1) ? std::atoi(argv[1]) : 0;

    // Fallback figures for a CPU-only build; clearly labelled so nobody mistakes them
    // for a measurement of the machine they are on.
    tensortransit::DeviceCaps caps{96ull * 1024 * 1024, 64ull * 1024 * 1024, 32ull * 1024 * 1024};
    const char* source = "illustrative (built without CUDA)";

#if defined(TENSORTRANSIT_WITH_CUDA) || defined(RECURLOCAL_WITH_CUDA) || \
    defined(RECURLLOCAL_WITH_CUDA)
    tensortransit::DeviceCaps queried{};
    if (tensortransit::query_device_caps(device, &queried) == cudaSuccess) {
        caps = queried;
        source = "queried from device";
    } else {
        source = "illustrative (no usable CUDA device)";
    }
    tensortransit::DeviceProfile profile{};
    if (tensortransit::query_device_profile(device, &profile) == cudaSuccess) {
        std::cout << "sm_count=" << profile.sm_count << "\n"
                  << "compute_capability=" << profile.major << "." << profile.minor << "\n"
                  << "global_memory_bytes=" << profile.global_memory_bytes << "\n"
                  << "peak_bandwidth_bytes_per_s=" << profile.peak_bandwidth_bytes_per_s << "\n";
    }
#else
    (void)device;
#endif

    tensortransit::PlannerConfig cfg;
    tensortransit::LocalityPlanner p(caps, cfg);
    const auto plan = p.plan_for_layer(3ull * 1024 * 1024, true);

    std::cout << "TensorTransit " << tensortransit::version_string() << "\n"
              << "device=" << device << "\n"
              << "caps_source=" << source << "\n"
              << "l2_bytes=" << caps.l2_bytes << "\n"
              << "persisting_l2_max_bytes=" << caps.persisting_l2_max_bytes << "\n"
              << "access_policy_max_window_bytes=" << caps.access_policy_max_window_bytes << "\n"
              << "mode=" << tensortransit::to_string(cfg.mode) << "\n"
              << "recommended_set_aside_bytes=" << p.recommended_l2_set_aside() << "\n"
              << "example_hot_window_bytes=" << plan.hot_window_bytes << "\n"
              << "example_hit_ratio=" << plan.hit_ratio << "\n"
              << "example_hit_ratio_reduced=" << (plan.hit_ratio_reduced ? "true" : "false") << "\n"
              << "example_prefetch_next=" << (plan.prefetch_next ? "true" : "false") << "\n";
}
