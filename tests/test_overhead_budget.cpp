// The planner-and-executor CPU budget, asserted rather than described.
//
// The specification budgets planner plus executor host time under 0.5% of token latency, and
// until this file existed nothing checked it. RuntimeStats::compile_ns and
// ExecutorStats::host_ns measured it; a budget nobody gates on is a wish.
//
// Two different failures live here and only one of them is about speed:
//
//   1. The layer is simply too slow per token.
//   2. The plan cache is not working, so the planner runs EVERY token. This is the single
//      most likely way a locality layer costs more than it saves, and it is invisible in a
//      throughput number -- the run is just a bit slower, for a reason no profiler was
//      pointed at. `reuse_rate` is what says so, and a steady decode loop must be at 1.0.
//
// The workload is the real one: the recorded geometry of Qwen3.8-27B, 64 layers with a
// full-attention interval of 4, so 48 recurrent layers carrying a 3 MiB matrix state and a
// 60 KiB convolution state, 16 attention layers carrying KV. That is what the adapter
// registers and records on every token; a synthetic eight-layer graph would pass this test
// while the real one blew the budget.
#include <chrono>
#include <cstdio>
#include <cstdint>
#include <vector>

#include "tensortransit/planner.h"
#include "tensortransit/runtime.h"
#include "tensortransit/trace.h"

using namespace tensortransit;

static int g_failures = 0;
#define CHECK(cond) do { if (!(cond)) { \
    std::printf("FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond); ++g_failures; } } while (0)

// Measured control-arm decode latency for the pinned model on the reference RTX 5090:
// 96.6678 tok/s at batch 1 (results/rtx5090-baseline-matrix.json), so 10.345 ms per token.
// Named here rather than assumed, because the budget is a SHARE of it.
static constexpr double kTokenLatencyNs = 10.344706e6;
static constexpr double kBudgetShare = 0.005;   // spec section 80

namespace {

struct DecodeStep {
    TensorRegistry registry;
    TransitGraph graph;
    std::vector<KernelId> kernels;

    // `sequences` scales the KV a step reads, exactly as a concurrent decode does.
    DecodeStep(int layers, int full_attn_interval, int sequences) {
        std::uintptr_t next = 1;
        const auto fake = [&](std::uintptr_t n) {
            return reinterpret_cast<const void*>(0x10000000ull + n * 0x1000ull);
        };
        for (int layer = 0; layer < layers; ++layer) {
            KernelEvent kernel{};
            kernel.id = static_cast<KernelId>(layer) + 1;
            kernel.order = static_cast<std::uint64_t>(layer);
            graph.record_kernel(kernel);
            kernels.push_back(kernel.id);

            const bool recurrent = ((layer + 1) % full_attn_interval) != 0;
            const auto use = [&](TensorId tensor, AccessKind access) {
                TensorUse u{};
                u.tensor = tensor;
                u.kernel = kernel.id;
                u.access = access;
                graph.record_use(u);
            };
            if (recurrent) {
                for (const std::size_t bytes : {3145728ull, 61440ull}) {
                    TensorDesc d{};
                    d.ptr = fake(next++);
                    d.bytes = static_cast<std::size_t>(bytes);
                    d.role = TensorRole::RecurrentState;
                    d.mutable_data = true;
                    d.request_local = true;
                    use(registry.register_tensor(d).id, AccessKind::ReadWrite);
                }
            } else {
                for (int half = 0; half < 2; ++half) {   // K and V
                    TensorDesc d{};
                    d.ptr = fake(next++);
                    d.bytes = static_cast<std::size_t>(sequences) * 512 * 1024;
                    d.role = TensorRole::KVCache;
                    d.mutable_data = true;
                    d.request_local = true;
                    use(registry.register_tensor(d).id, AccessKind::Read);
                }
            }
            TensorDesc weight{};
            weight.ptr = fake(next++);
            weight.bytes = 18500000000ull / static_cast<std::size_t>(layers);
            weight.role = TensorRole::ModelWeight;
            weight.model_global = true;
            use(registry.register_tensor(weight).id, AccessKind::Read);
        }
        graph.set_cyclic(true);
    }
};

std::uint64_t now_ns() {
    return static_cast<std::uint64_t>(
        std::chrono::duration_cast<std::chrono::nanoseconds>(
            std::chrono::steady_clock::now().time_since_epoch()).count());
}

struct Result {
    double planner_ns_per_token = 0.0;
    double executor_ns_per_token = 0.0;
    double wall_ns_per_token = 0.0;
    double reuse_rate = 0.0;
    std::uint64_t compiles = 0;
};

Result run(const char* planner_name, const TransitPlannerConfig& config, int sequences,
           int tokens) {
    DecodeStep workload(64, 4, sequences);
    TransitRuntime runtime;
    DeviceProfile device{};
    device_profile_by_name("rtx5090", &device);
    runtime.set_device_profile(device);
    if (!runtime.set_planner(planner_name, config)) return Result{};

    RecordingExecutor executor;
    runtime.set_executor(&executor);
    runtime.begin_recording();
    for (const KernelEvent& kernel : workload.graph.kernels()) runtime.record_kernel(kernel);
    for (const TensorUse& use : workload.graph.uses()) runtime.record_use(use);
    // The registry has to be the runtime's own, so re-register into it.
    workload.registry.for_each([&](const TensorHandle&, const TensorDesc& desc) {
        runtime.register_tensor(desc);
    });
    runtime.end_recording(/*cyclic=*/true);

    RuntimeState state{};
    state.active_requests = sequences;
    state.granted_budget_bytes = 50331648;

    const std::uint64_t wall_start = now_ns();
    for (int token = 0; token < tokens; ++token) {
        runtime.compile(state);        // the per-token call an adapter makes
        runtime.begin_step();
        for (const KernelId kernel : workload.kernels) {
            runtime.before_kernel(kernel);
            runtime.after_kernel(kernel);
        }
        runtime.end_step();
        executor.clear_records();
    }
    const std::uint64_t wall = now_ns() - wall_start;

    Result out;
    const double n = tokens;
    out.planner_ns_per_token = static_cast<double>(runtime.stats().compile_ns) / n;
    out.executor_ns_per_token = static_cast<double>(executor.stats().host_ns) / n;
    out.wall_ns_per_token = static_cast<double>(wall) / n;
    out.reuse_rate = runtime.stats().reuse_rate();
    out.compiles = runtime.stats().compiles;
    return out;
}

TransitPlannerConfig config_for(AdmissionRule rule) {
    TransitPlannerConfig config{};
    config.persist_roles = RoleMask::of(TensorRole::RecurrentState, TensorRole::KVCache);
    config.stream_roles = RoleMask::of(TensorRole::ModelWeight);
    config.admission = rule;
    config.max_windows_per_kernel = 1;
    config.window_preference = WindowPreference::Widest;
    return config;
}

}  // namespace

int main() {
    const double budget_ns = kTokenLatencyNs * kBudgetShare;
    std::printf("budget: %.0f ns per token (%.1f%% of %.3f ms)\n", budget_ns,
                kBudgetShare * 100.0, kTokenLatencyNs / 1e6);

    struct Case { const char* planner; AdmissionRule rule; int sequences; };
    const Case cases[] = {
        {"recurrent_v0", AdmissionRule::Density, 1},
        {"budgeted", AdmissionRule::Density, 1},
        {"budgeted", AdmissionRule::RoleFloor, 1},
        {"budgeted", AdmissionRule::Survival, 1},
        {"budgeted", AdmissionRule::Quota, 32},
        {"concurrency", AdmissionRule::Density, 32},
    };
    for (const Case& c : cases) {
        const Result r = run(c.planner, config_for(c.rule), c.sequences, 512);
        const double total = r.planner_ns_per_token + r.executor_ns_per_token;
        std::printf("  %-12s %-12s c=%-3d  planner %7.0f ns  executor %7.0f ns  "
                    "total %7.0f ns (%.4f%%)  wall %7.0f ns  reuse %.3f\n",
                    c.planner, to_string(c.rule), c.sequences,
                    r.planner_ns_per_token, r.executor_ns_per_token, total,
                    total / kTokenLatencyNs * 100.0, r.wall_ns_per_token, r.reuse_rate);

        // The budget itself, on the two counters the specification names...
        CHECK(total < budget_ns);
        // ...and on the WALL time of a whole bracketed token, which is the number that
        // actually lands on the critical path: the compile call, the 64 before/after pairs,
        // and everything between them. Reporting only the two counters would let the
        // bracketing itself grow without limit.
        CHECK(r.wall_ns_per_token < budget_ns);
        // And the reason it holds. A steady decode loop must compile ONCE: the plan cache is
        // what makes the planner's cost amortize to nothing, and a cache that missed every
        // token would blow the budget by three orders of magnitude while every other number
        // in the run looked normal.
        CHECK(r.compiles == 1);
        CHECK(r.reuse_rate > 0.99);
    }

    if (g_failures) {
        std::printf("%d overhead-budget check(s) failed\n", g_failures);
        std::printf("NOTE: this is a HOST-CPU measurement. A failure here is either the "
                    "layer genuinely getting slower or a build/machine far slower than the "
                    "reference; check `reuse` first -- below 1.0 means the plan cache is "
                    "missing, which is the failure this budget is really about.\n");
        return 1;
    }
    std::printf("overhead budget tests passed\n");
    return 0;
}
