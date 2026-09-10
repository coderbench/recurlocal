// Device-side contract for CudaTransitExecutor.
//
// Exits 0 with a message when no GPU is present, so this stays registered on a device-less
// CUDA build box -- the alternative is a suite that only exists on the one machine that can
// run it, which is how a device contract rots.
#include <cstdio>
#include <vector>

#include "tensortransit/cuda_executor.h"
#include "tensortransit/planner.h"
#include "tensortransit/runtime.h"

using namespace tensortransit;

static int g_failures = 0;
#define CHECK(cond) do { if (!(cond)) { \
    std::printf("FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond); ++g_failures; } } while (0)

static constexpr std::size_t MiB = 1024ull * 1024ull;

__global__ void touch(float* p, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) p[i] += 1.0f;
}

// A device-resident fixture: two "recurrent states" and one "weight" buffer, registered and
// recorded exactly as a runtime would.
struct Fixture {
    TransitRuntime runtime;
    float* state_a = nullptr;
    float* state_b = nullptr;
    float* weights = nullptr;
    float* scratch = nullptr;
    TensorHandle ha, hb, hw;

    bool init() {
        // Zeroed, not merely allocated. `touch` reads before it writes, so uninitialised
        // device memory is an uninitialised READ -- 924 of them under
        // `compute-sanitizer --tool initcheck`. The library was clean; the fixture was not,
        // and it went unnoticed for as long as the sanitizer script covered only the 0.1
        // controller. It covers this binary now, so the next one is caught the same way.
        if (cudaMalloc(&state_a, 4 * MiB) != cudaSuccess) return false;
        if (cudaMemset(state_a, 0, 4 * MiB) != cudaSuccess) return false;
        if (cudaMalloc(&state_b, 4 * MiB) != cudaSuccess) return false;
        if (cudaMemset(state_b, 0, 4 * MiB) != cudaSuccess) return false;
        if (cudaMalloc(&weights, 16 * MiB) != cudaSuccess) return false;
        if (cudaMemset(weights, 0, 16 * MiB) != cudaSuccess) return false;
        if (cudaMalloc(&scratch, 1024 * sizeof(float)) != cudaSuccess) return false;
        if (cudaMemset(scratch, 0, 1024 * sizeof(float)) != cudaSuccess) return false;

        DeviceProfile profile{};
        if (query_device_profile(0, &profile) != cudaSuccess) return false;
        runtime.set_device_profile(profile);

        ha = declare(state_a, 4 * MiB, TensorRole::RecurrentState);
        hb = declare(state_b, 4 * MiB, TensorRole::RecurrentState);
        hw = declare(weights, 16 * MiB, TensorRole::ModelWeight);

        runtime.begin_recording();
        record(1, hw.id, AccessKind::Read);
        record(2, ha.id, AccessKind::ReadWrite);
        record(3, hb.id, AccessKind::ReadWrite);
        runtime.end_recording(true);
        return true;
    }
    TensorHandle declare(void* ptr, std::size_t bytes, TensorRole role) {
        TensorDesc desc{};
        desc.ptr = ptr;
        desc.bytes = bytes;
        desc.role = role;
        desc.mutable_data = role == TensorRole::RecurrentState;
        desc.device = 0;
        return runtime.register_tensor(desc);
    }
    void record(std::uint64_t order, TensorId tensor, AccessKind access) {
        KernelEvent kernel{};
        kernel.id = order;
        kernel.order = order;
        runtime.record_kernel(kernel);
        TensorUse use{};
        use.tensor = tensor;
        use.kernel = kernel.id;
        use.access = access;
        runtime.record_use(use);
    }
    ~Fixture() {
        cudaFree(state_a);
        cudaFree(state_b);
        cudaFree(weights);
        cudaFree(scratch);
    }
};

static void test_the_set_aside_is_given_back(Fixture& f) {
    std::size_t before = 0;
    cudaDeviceGetLimit(&before, cudaLimitPersistingL2CacheSize);
    {
        CudaTransitExecutor executor;
        CHECK(executor.initialize(0, 8 * MiB) == cudaSuccess);
        CHECK(executor.set_aside_bytes() > 0);
        CHECK(executor.set_aside_peak_bytes() >= executor.set_aside_bytes());
    }
    std::size_t after = 0;
    cudaDeviceGetLimit(&after, cudaLimitPersistingL2CacheSize);
    // The limit is device-wide and context-lifetime. Without restoring it, merely
    // constructing an executor carves a permanent hole out of L2 for every other kernel in
    // the process, including after the executor is gone.
    CHECK(before == after);
    (void)f;
}

static void test_a_plan_reaches_the_device(Fixture& f) {
    cudaStream_t compute = nullptr, prefetch = nullptr;
    CHECK(cudaStreamCreate(&compute) == cudaSuccess);
    CHECK(cudaStreamCreate(&prefetch) == cudaSuccess);

    CudaTransitExecutor executor;
    CHECK(executor.initialize(0, 16 * MiB) == cudaSuccess);
    CHECK(executor.bind_streams(compute, prefetch) == cudaSuccess);
    executor.set_registry(&f.runtime.registry());
    executor.set_scratch(f.scratch, 1024);
    f.runtime.set_executor(&executor);

    TransitPlannerConfig config{};
    config.persist_roles = RoleMask::of(TensorRole::RecurrentState);
    config.budget_fraction = 1.0;
    CHECK(f.runtime.set_planner("budgeted", config));

    RuntimeState state{};
    state.active_requests = 1;
    const TransitPlan& plan = f.runtime.compile(state);
    CHECK(plan.validate() == nullptr);
    CHECK(plan.count(TransitActionKind::Persist) > 0);

    f.runtime.begin_step();
    for (const KernelEvent& kernel : f.runtime.graph().kernels()) {
        f.runtime.before_kernel(kernel.id);
        touch<<<64, 256, 0, compute>>>(f.state_a, 1024);
        f.runtime.after_kernel(kernel.id);
    }
    f.runtime.end_step();
    CHECK(cudaStreamSynchronize(compute) == cudaSuccess);

    // Outside capture the window goes on the stream, so this is the counter that must move.
    CHECK(executor.stats().persist_applied > 0);
    CHECK(executor.stats().clear_applied > 0);
    CHECK(executor.stats().actions_failed == 0);
    CHECK(executor.stats().stale_tensor_refs == 0);

    executor.release();
    f.runtime.set_executor(nullptr);
    cudaStreamDestroy(compute);
    cudaStreamDestroy(prefetch);
}

static void test_a_recycled_tensor_is_refused_not_written_over(Fixture& f) {
    cudaStream_t compute = nullptr, prefetch = nullptr;
    cudaStreamCreate(&compute);
    cudaStreamCreate(&prefetch);

    CudaTransitExecutor executor;
    executor.initialize(0, 16 * MiB);
    executor.bind_streams(compute, prefetch);
    executor.set_registry(&f.runtime.registry());
    executor.set_scratch(f.scratch, 1024);

    TransitPlannerConfig config{};
    config.persist_roles = RoleMask::of(TensorRole::RecurrentState);
    config.budget_fraction = 1.0;
    auto planner = make_planner("budgeted", config);
    PlanInput input{&f.runtime.graph(), &f.runtime.registry(), f.runtime.device_profile(),
                    RuntimeState{}};
    TransitPlan plan = planner->build_plan(input);
    plan.finalize();
    executor.set_plan(&plan);

    // The runtime frees the buffer under the compiled plan. Every address in the plan is
    // now a promise nobody is keeping, and placing a window on one is worse than nothing.
    TensorRegistry empty;
    executor.set_registry(&empty);

    executor.begin_step();
    for (const KernelEvent& kernel : f.runtime.graph().kernels()) {
        executor.before_kernel(kernel.id);
        executor.after_kernel(kernel.id);
    }
    executor.end_step();
    CHECK(executor.stats().stale_tensor_refs > 0);
    CHECK(executor.stats().persist_applied == 0);

    executor.release();
    cudaStreamDestroy(compute);
    cudaStreamDestroy(prefetch);
}

static void test_a_window_reaches_a_captured_graph_node(Fixture& f) {
    cudaStream_t compute = nullptr, prefetch = nullptr;
    cudaStreamCreate(&compute);
    cudaStreamCreate(&prefetch);

    CudaTransitExecutor executor;
    executor.initialize(0, 16 * MiB);
    executor.bind_streams(compute, prefetch);
    executor.set_registry(&f.runtime.registry());
    executor.set_scratch(f.scratch, 1024);

    TransitPlannerConfig config{};
    config.persist_roles = RoleMask::of(TensorRole::RecurrentState);
    config.budget_fraction = 1.0;
    auto planner = make_planner("budgeted", config);
    PlanInput input{&f.runtime.graph(), &f.runtime.registry(), f.runtime.device_profile(),
                    RuntimeState{}};
    TransitPlan plan = planner->build_plan(input);
    plan.finalize();
    executor.set_plan(&plan);

    cudaGraph_t graph = nullptr;
    CHECK(cudaStreamBeginCapture(compute, cudaStreamCaptureModeThreadLocal) == cudaSuccess);
    executor.begin_step();
    for (const KernelEvent& kernel : f.runtime.graph().kernels()) {
        executor.before_kernel(kernel.id);
        touch<<<64, 256, 0, compute>>>(f.state_a, 1024);
        // Under capture the window must go on the NODE. A stream attribute is host-side
        // state the graph never records, so without this the policy is absent from every
        // replay -- and `persist_applied` would still be non-zero, which is exactly the
        // null candidate this project has already measured once and published.
        executor.attach_window_to_captured_node();
        executor.after_kernel(kernel.id);
    }
    executor.end_step();
    CHECK(cudaStreamEndCapture(compute, &graph) == cudaSuccess);

    CHECK(executor.stats().persist_deferred > 0);
    CHECK(executor.stats().persist_attached_to_node > 0);
    CHECK(executor.stats().persist_nodes_attached >= executor.stats().persist_attached_to_node);
    // The stream path must NOT have been taken under capture.
    CHECK(executor.stats().persist_applied == 0);

    // And the attribute really is on the finished graph.
    if (graph) {
        std::size_t count = 0;
        cudaGraphGetNodes(graph, nullptr, &count);
        std::vector<cudaGraphNode_t> nodes(count);
        cudaGraphGetNodes(graph, nodes.data(), &count);
        bool found = false;
        for (cudaGraphNode_t node : nodes) {
            cudaGraphNodeType type = cudaGraphNodeTypeEmpty;
            if (cudaGraphNodeGetType(node, &type) != cudaSuccess) continue;
            if (type != cudaGraphNodeTypeKernel) continue;
            cudaKernelNodeAttrValue value{};
            if (cudaGraphKernelNodeGetAttribute(node, cudaKernelNodeAttributeAccessPolicyWindow,
                                                &value) != cudaSuccess)
                continue;
            if (value.accessPolicyWindow.num_bytes > 0) found = true;
        }
        CHECK(found);
        cudaGraphDestroy(graph);
    }

    executor.release();
    cudaStreamDestroy(compute);
    cudaStreamDestroy(prefetch);
}

int main() {
    int devices = 0;
    if (cudaGetDeviceCount(&devices) != cudaSuccess || devices == 0) {
        std::printf("no CUDA device; skipping the CUDA executor tests\n");
        return 0;
    }
    Fixture fixture;
    if (!fixture.init()) {
        std::printf("could not build the device fixture; skipping\n");
        return 0;
    }
    test_the_set_aside_is_given_back(fixture);
    test_a_plan_reaches_the_device(fixture);
    test_a_recycled_tensor_is_refused_not_written_over(fixture);
    test_a_window_reaches_a_captured_graph_node(fixture);

    if (g_failures) { std::printf("%d CUDA executor check(s) failed\n", g_failures); return 1; }
    std::printf("CUDA executor tests passed\n");
    return 0;
}
