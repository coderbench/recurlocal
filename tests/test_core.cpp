// Registry lifetime, graph reuse analysis, plan structure and serialization.
//
// Every check here is a property the planners depend on and that nothing else would notice
// breaking: a stale handle that still resolves, a reuse distance measured in the wrong
// currency, a wrap edge that is not closed. A planner fed any of those still produces a
// plan, and the plan is still valid, and it is still wrong.
#include <cstdio>
#include <cstring>
#include <iostream>
#include <string>

#include "tensortransit/executor.h"
#include "tensortransit/planner.h"
#include "tensortransit/runtime.h"
#include "tensortransit/trace.h"

using namespace tensortransit;

static int g_failures = 0;
// Counted, and printed on success. A suite that says only "passed" cannot be
// distinguished from one whose checks were all compiled out, and the repo manifest
// used to carry a hand-typed total that nothing regenerated.
static int g_checks = 0;
#define CHECK(cond) do { ++g_checks; if (!(cond)) { \
    std::printf("FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond); ++g_failures; } } while (0)

static constexpr std::size_t MiB = 1024ull * 1024ull;

// Distinct, never-dereferenced addresses. The registry keys on a pointer; nothing in the
// core reads through one.
static const void* fake(std::uintptr_t n) { return reinterpret_cast<const void*>(0x100000 + n * 0x1000); }

static TensorDesc desc(std::uintptr_t n, TensorRole role, std::size_t bytes) {
    TensorDesc d{};
    d.ptr = fake(n);
    d.bytes = bytes;
    d.role = role;
    d.mutable_data = role == TensorRole::RecurrentState || role == TensorRole::KVCache;
    return d;
}

// --- registry ---------------------------------------------------------------------------

static void test_a_handle_does_not_survive_the_memory_it_named() {
    TensorRegistry registry;
    const TensorHandle first = registry.register_tensor(desc(1, TensorRole::KVCache, 4096));
    CHECK(registry.resolve(first) != nullptr);

    registry.unregister_tensor(first);
    CHECK(registry.resolve(first) == nullptr);

    // The runtime frees a block and allocates another at the SAME address. This is the
    // failure TensorHandle exists for: without the generation, the stale handle would
    // resolve to the new tensor and a compiled plan would place a window on it.
    const TensorHandle second = registry.register_tensor(desc(1, TensorRole::Activation, 8192));
    CHECK(second.valid());
    CHECK(registry.resolve(second) != nullptr);
    CHECK(registry.resolve(first) == nullptr);
    CHECK(first != second);
}

static void test_a_stale_handle_cannot_unregister_the_new_occupant() {
    TensorRegistry registry;
    const TensorHandle first = registry.register_tensor(desc(1, TensorRole::KVCache, 4096));
    registry.unregister_tensor(first);
    const TensorHandle second = registry.register_tensor(desc(1, TensorRole::KVCache, 4096));

    registry.unregister_tensor(first);  // stale: must do nothing
    CHECK(registry.resolve(second) != nullptr);
    CHECK(registry.size() == 1);
}

static void test_redeclaring_the_same_tensor_does_not_grow_the_table() {
    TensorRegistry registry;
    const TensorDesc d = desc(1, TensorRole::RecurrentState, 3 * MiB);
    const TensorHandle first = registry.register_tensor(d);
    const std::uint64_t epoch = registry.epoch();

    // A decode loop re-declares its state every token. If that were a structural change the
    // plan cache would recompile on every one of them, and the table would grow forever.
    for (int i = 0; i < 100; ++i) CHECK(registry.register_tensor(d) == first);
    CHECK(registry.size() == 1);
    CHECK(registry.epoch() == epoch);

    // ...but the same ADDRESS with a different size is a different tensor, and must be.
    TensorDesc grown = d;
    grown.bytes = 4 * MiB;
    registry.register_tensor(grown);
    CHECK(registry.epoch() != epoch);
}

static void test_a_dead_slot_is_reused_rather_than_leaked() {
    TensorRegistry registry;
    for (int i = 0; i < 50; ++i) {
        const TensorHandle handle = registry.register_tensor(desc(1, TensorRole::Workspace, 1024));
        registry.unregister_tensor(handle);
    }
    CHECK(registry.empty());
    registry.register_tensor(desc(2, TensorRole::Workspace, 1024));
    CHECK(registry.size() == 1);
}

static void test_an_unusable_descriptor_is_refused_rather_than_stored() {
    TensorRegistry registry;
    CHECK(!registry.register_tensor(TensorDesc{}).valid());
    TensorDesc zero = desc(1, TensorRole::KVCache, 0);
    CHECK(!registry.register_tensor(zero).valid());
    CHECK(registry.empty());
}

// --- graph ------------------------------------------------------------------------------

// One decode token of a tiny hybrid: two recurrent layers, one KV read, one weight stream.
struct Fixture {
    TensorRegistry registry;
    TransitGraph graph;
    TensorHandle state_a, state_b, kv, weights;

    Fixture(bool cyclic = true) {
        state_a = registry.register_tensor(desc(1, TensorRole::RecurrentState, 3 * MiB));
        state_b = registry.register_tensor(desc(2, TensorRole::RecurrentState, 3 * MiB));
        kv = registry.register_tensor(desc(3, TensorRole::KVCache, 12 * MiB));
        weights = registry.register_tensor(desc(4, TensorRole::ModelWeight, 100 * MiB));

        add(1, weights.id, AccessKind::Read);
        add(2, state_a.id, AccessKind::ReadWrite);
        add(3, weights.id, AccessKind::Read);
        add(4, state_b.id, AccessKind::ReadWrite);
        add(5, kv.id, AccessKind::ReadWrite);
        graph.set_cyclic(cyclic);
        graph.build(registry);
    }
    void add(std::uint64_t id, TensorId tensor, AccessKind access) {
        KernelEvent kernel{};
        kernel.id = id;
        kernel.order = id;
        kernel.estimated_duration_ns = 1000;
        kernel.estimated_start_ns = id * 1000;
        TensorUse use{};
        use.tensor = tensor;
        use.access = access;
        graph.record(kernel, &use, 1);
    }
};

static void test_a_read_modify_write_moves_its_bytes_twice() {
    Fixture f;
    // 100 MiB weight read twice + (3+3+12) MiB read-modify-written once each, x2.
    const std::size_t expected = 2 * 100 * MiB + 2 * (3 * MiB) + 2 * (3 * MiB) + 2 * (12 * MiB);
    CHECK(f.graph.step_traffic_bytes() == expected);
}

static void test_a_decode_token_is_a_loop_and_the_state_edge_crosses_it() {
    // Each recurrent state is used ONCE per token. Without closing the loop it has no reuse
    // edge at all, every planner correctly declines it, and the whole recurrent surface
    // reads as unreachable -- which is a wrong answer that looks exactly like a right one.
    Fixture acyclic(false);
    CHECK(acyclic.graph.profile(acyclic.state_a.id)->reuse_count == 0);

    Fixture cyclic(true);
    const TensorProfile* profile = cyclic.graph.profile(cyclic.state_a.id);
    CHECK(profile != nullptr);
    CHECK(profile->reuse_count == 1);
    // The saving is the whole read-modify-write: 2 x 3 MiB.
    CHECK(profile->reused_bytes == 2 * 3 * MiB);
}

static void test_reuse_distance_in_bytes_counts_everything_in_between() {
    Fixture f;
    // The weight tensor is read at kernel 1 and again at kernel 3. Between them sits
    // state_a's read-modify-write: 2 x 3 MiB.
    const TransitEdge* edge = nullptr;
    for (const TransitEdge& e : f.graph.edges())
        if (e.tensor == f.weights.id && e.producer == 1 && e.consumer == 3) edge = &e;
    CHECK(edge != nullptr);
    if (edge) {
        CHECK(edge->reuse_bytes == 2 * 3 * MiB);
        CHECK(edge->reuse_kernels == 1);  // kernel 2 runs in between
        CHECK(edge->distance(ReuseMetric::Bytes) == 2 * 3 * MiB);
        CHECK(edge->distance(ReuseMetric::Ordinal) == 1);
        // The three metrics really do disagree, which is the whole reason there are three.
        CHECK(edge->distance(ReuseMetric::Bytes) != edge->distance(ReuseMetric::Ordinal));
    }
}

static void test_out_of_order_kernels_are_refused_rather_than_analysed() {
    TransitGraph graph;
    KernelEvent late{};
    late.id = 1;
    late.order = 10;
    CHECK(graph.record_kernel(late));
    KernelEvent early{};
    early.id = 2;
    early.order = 5;
    // A negative interval underflows to an enormous distance, which every planner reads as
    // "never reused" -- so the policy disables itself and nothing says why.
    CHECK(!graph.record_kernel(early));
}

static void test_a_use_of_an_unregistered_tensor_is_counted_not_silently_dropped() {
    TensorRegistry registry;
    TransitGraph graph;
    KernelEvent kernel{};
    kernel.id = 1;
    kernel.order = 1;
    TensorUse use{};
    use.tensor = 999;  // never registered
    use.access = AccessKind::Read;
    graph.record(kernel, &use, 1);
    graph.build(registry);
    CHECK(graph.unresolved_uses() == 1);
}

static void test_the_live_set_counts_each_tensor_once() {
    Fixture f;
    // At kernel 3, state_a's wrap edge and the weight edge are both live; state_a is one
    // tensor however many edges span the point.
    const std::size_t live = f.graph.live_bytes_at(3);
    CHECK(live <= 3 * MiB + 3 * MiB + 12 * MiB + 100 * MiB);
    CHECK(f.graph.peak_live_bytes() >= live);
}

// --- plan -------------------------------------------------------------------------------

static void test_a_persist_with_no_clear_is_rejected() {
    TransitPlan plan;
    TransitAction persist{};
    persist.kind = TransitActionKind::Persist;
    persist.tensor = 1;
    persist.before_kernel = 1;
    persist.bytes = 4096;
    persist.hit_ratio = 0.7;
    plan.add(persist);
    // Spec section 32: a policy with no lifetime keeps spending set-aside on a tensor
    // nothing is going to read, and the kernels that follow pay for it silently.
    CHECK(plan.validate() != nullptr);

    TransitAction clear{};
    clear.kind = TransitActionKind::ClearPolicy;
    clear.tensor = 1;
    clear.after_kernel = 1;
    plan.add(clear);
    CHECK(plan.validate() == nullptr);
}

static void test_an_unjoined_fork_is_rejected() {
    TransitPlan plan;
    TransitAction fork{};
    fork.kind = TransitActionKind::RecordEvent;
    fork.before_kernel = 1;
    fork.event_id = 7;
    plan.add(fork);
    // Under CUDA Graph capture an unjoined fork ends the capture INVALID and takes the
    // runtime's decode path with it.
    CHECK(plan.validate() != nullptr);

    TransitAction join{};
    join.kind = TransitActionKind::WaitEvent;
    join.before_kernel = 2;
    join.event_id = 7;
    plan.add(join);
    CHECK(plan.validate() == nullptr);
}

static void test_the_digest_ignores_the_name_and_the_cost_model() {
    TransitPlan a, b;
    TransitAction action{};
    action.kind = TransitActionKind::Stream;
    action.tensor = 4;
    action.before_kernel = 1;
    action.bytes = 1024;
    a.add(action);
    b.add(action);
    a.set_planner_name("greedy");
    b.set_planner_name("budgeted");
    a.cost().predicted_saved_bytes = 999;
    // Two planners that emit the same actions ARE the same plan; a digest that said
    // otherwise could not be used to assert behaviour in a golden test.
    CHECK(a.digest() == b.digest());
}

static void test_the_per_kernel_index_returns_actions_in_plan_order() {
    TransitPlan plan;
    for (int i = 0; i < 3; ++i) {
        TransitAction action{};
        action.kind = TransitActionKind::Persist;
        action.tensor = static_cast<TensorId>(i + 1);
        action.before_kernel = 5;
        action.bytes = 1024;
        action.hit_ratio = 0.5;
        plan.add(action);
        TransitAction clear{};
        clear.kind = TransitActionKind::ClearPolicy;
        clear.tensor = static_cast<TensorId>(i + 1);
        clear.after_kernel = 5;
        plan.add(clear);
    }
    plan.finalize();
    int count = 0;
    const TransitAction* const* before = plan.before(5, &count);
    CHECK(count == 3);
    if (before && count == 3)
        for (int i = 0; i < 3; ++i) CHECK(before[i]->tensor == static_cast<TensorId>(i + 1));
    plan.after(5, &count);
    CHECK(count == 3);
    plan.before(99, &count);
    CHECK(count == 0);
}

// --- trace round trip -------------------------------------------------------------------

static void test_a_trace_round_trips_its_reuse_structure() {
    Fixture f;
    TraceMetadata meta{};
    meta.model = "fixture";
    meta.runtime = "none";
    meta.phase = "decode";
    meta.active_requests = 1;
    meta.cyclic = true;
    const std::string json = write_trace(f.graph, f.registry, meta);

    TensorRegistry registry;
    TransitGraph graph;
    TraceMetadata read{};
    std::string error;
    CHECK(read_trace(json, &registry, &graph, &read, &error));
    if (!error.empty()) std::printf("  trace error: %s\n", error.c_str());
    CHECK(graph.step_traffic_bytes() == f.graph.step_traffic_bytes());
    CHECK(graph.removable_bytes() == f.graph.removable_bytes());
    CHECK(registry.size() == f.registry.size());
    CHECK(read.model == "fixture");
    // A trace carries no pointers, so everything it rebuilds is marked as non-executable.
    registry.for_each([&](const TensorHandle&, const TensorDesc& d) { CHECK(d.device == -1); });
}

static void test_a_malformed_trace_is_reported_rather_than_accepted() {
    TensorRegistry registry;
    TransitGraph graph;
    std::string error;
    CHECK(!read_trace("{", &registry, &graph, nullptr, &error));
    CHECK(!error.empty());
    error.clear();
    CHECK(!read_trace("{\"schema_version\":99,\"events\":[]}", &registry, &graph, nullptr, &error));
    CHECK(!error.empty());
    error.clear();
    // A use naming a tensor the table does not declare would otherwise become an
    // "unresolved use" and quietly shrink the graph.
    CHECK(!read_trace("{\"schema_version\":1,\"tensors\":[],\"events\":"
                      "[{\"kernel_id\":1,\"tensors\":[{\"id\":5}]}]}",
                      &registry, &graph, nullptr, &error));
    CHECK(!error.empty());
}

// A plan has to survive being written, shipped and read back, because that is the whole of the
// offline planning loop: trace -> planner -> plan.json -> replay. Two thirds of it existed and
// the third did not, so a plan could be dumped and read by a human but never fed back to an
// executor.
static void test_a_plan_round_trips_through_json() {
    Fixture f;
    TransitPlannerConfig config{};
    config.persist_roles = RoleMask::of(TensorRole::RecurrentState, TensorRole::KVCache);
    config.stream_roles = RoleMask::of(TensorRole::ModelWeight);
    PlanInput input{};
    input.graph = &f.graph;
    input.registry = &f.registry;
    device_profile_by_name("rtx5090", &input.device);
    const TransitPlan original = make_budgeted_planner(config)->build_plan(input);
    CHECK(!original.actions().empty());

    TransitPlan restored;
    std::string error;
    CHECK(read_plan(original.to_json(), &restored, &error));
    // The DIGEST is over the actions, so equality of digests is equality of what the plan
    // does. It is the only comparison worth making here: the cost model is carried too, but a
    // plan that came back with different actions is a different plan whatever it predicts.
    CHECK(restored.digest() == original.digest());
    CHECK(restored.actions().size() == original.actions().size());
    CHECK(restored.declines().size() == original.declines().size());
    CHECK(restored.planner_name() == original.planner_name());
    CHECK(restored.cost().predicted_saved_bytes == original.cost().predicted_saved_bytes);
    CHECK(restored.cost().cost_model == original.cost().cost_model);

    // It comes back with NULL regions -- a serialized plan carries no pointer, by design --
    // and rebinding against a live registry is what makes it executable.
    bool any_region = false;
    for (const TransitAction& action : restored.actions())
        if (action.ptr != nullptr) any_region = true;
    CHECK(!any_region);
    CHECK(restored.rebind(f.registry, &error));
    for (const TransitAction& action : restored.actions())
        if (action.kind == TransitActionKind::Persist) CHECK(action.ptr != nullptr);

    // A plan naming a tensor this registry does not know is a plan compiled against a
    // different model, and executing the rest of it would apply an arbitration made against
    // tensors that are not here.
    TensorRegistry empty;
    TransitPlan orphan;
    CHECK(read_plan(original.to_json(), &orphan, &error));
    CHECK(!orphan.rebind(empty, &error));
    CHECK(error.find("does not know") != std::string::npos);

    // And a plan from another schema is refused rather than guessed at.
    TransitPlan wrong;
    CHECK(!read_plan("{\"plan_schema_version\":99,\"actions\":[]}", &wrong, &error));
}

// --- runtime ----------------------------------------------------------------------------

static void test_a_steady_decode_loop_compiles_once() {
    TransitRuntime runtime;
    DeviceProfile device{};
    device_profile_by_name("rtx5090", &device);
    runtime.set_device_profile(device);

    TransitPlannerConfig config{};
    config.persist_roles = RoleMask::of(TensorRole::RecurrentState, TensorRole::KVCache);
    CHECK(runtime.set_planner("budgeted", config));

    const TensorHandle state = runtime.register_tensor(desc(1, TensorRole::RecurrentState, 3 * MiB));
    const TensorHandle kv = runtime.register_tensor(desc(2, TensorRole::KVCache, 12 * MiB));

    runtime.begin_recording();
    for (int i = 1; i <= 2; ++i) {
        KernelEvent kernel{};
        kernel.id = static_cast<KernelId>(i);
        kernel.order = static_cast<std::uint64_t>(i);
        runtime.record_kernel(kernel);
        TensorUse use{};
        use.tensor = (i == 1) ? state.id : kv.id;
        use.kernel = kernel.id;
        use.access = AccessKind::ReadWrite;
        CHECK(runtime.record_use(use));
    }
    runtime.end_recording(true);
    CHECK(runtime.graph().unresolved_uses() == 0);

    RuntimeState state_now{};
    state_now.active_requests = 1;
    runtime.compile(state_now);
    for (int i = 0; i < 32; ++i) runtime.compile(state_now);
    // Spec section 33: the plan is built once and reused across tokens. A loop that
    // recompiled every token would put the planner on the critical path, which is the
    // single most likely way this layer costs more than it saves.
    CHECK(runtime.stats().compiles == 1);
    CHECK(runtime.stats().plan_reuses == 32);
    CHECK(runtime.stats().reuse_rate() > 0.9);

    // Re-declaring the same tensors, as a decode loop does every token, must NOT invalidate.
    runtime.register_tensor(desc(1, TensorRole::RecurrentState, 3 * MiB));
    runtime.compile(state_now);
    CHECK(runtime.stats().compiles == 1);

    // Concurrency changing is a real change and must recompile.
    state_now.active_requests = 4;
    runtime.compile(state_now);
    CHECK(runtime.stats().compiles == 2);
    CHECK(runtime.stats().last_reason == RecompileReason::ConcurrencyChanged);

    // So is the runtime recycling memory under a compiled plan.
    runtime.unregister_tensor(kv);
    runtime.compile(state_now);
    CHECK(runtime.stats().compiles == 3);
    CHECK(runtime.stats().last_reason == RecompileReason::RegistryEpoch);
}

static void test_the_recording_executor_sees_a_balanced_plan() {
    Fixture f;
    TransitPlannerConfig config{};
    config.persist_roles = RoleMask::of(TensorRole::RecurrentState, TensorRole::KVCache);
    config.prefetch_roles = RoleMask::of(TensorRole::RecurrentState);
    config.prefetch_enabled = true;
    config.prefetch_min_bytes = 1024;
    auto planner = make_planner("budgeted", config);

    DeviceProfile device{};
    device_profile_by_name("rtx5090", &device);
    PlanInput input{&f.graph, &f.registry, device, RuntimeState{}};
    TransitPlan plan = planner->build_plan(input);
    plan.finalize();
    CHECK(plan.validate() == nullptr);

    RecordingExecutor executor;
    executor.set_plan(&plan);
    executor.begin_step();
    for (const KernelEvent& kernel : f.graph.kernels()) {
        executor.before_kernel(kernel.id);
        executor.after_kernel(kernel.id);
    }
    executor.end_step();
    // What actually FIRED must be balanced, not merely what was planned: a plan can be well
    // formed and still place its clear on a kernel the step never reaches.
    CHECK(executor.policies_balanced());
    CHECK(executor.stats().actions_applied > 0);
    CHECK(executor.stats().kernels == f.graph.kernels().size());
}

int main() {
    test_a_handle_does_not_survive_the_memory_it_named();
    test_a_stale_handle_cannot_unregister_the_new_occupant();
    test_redeclaring_the_same_tensor_does_not_grow_the_table();
    test_a_dead_slot_is_reused_rather_than_leaked();
    test_an_unusable_descriptor_is_refused_rather_than_stored();

    test_a_read_modify_write_moves_its_bytes_twice();
    test_a_decode_token_is_a_loop_and_the_state_edge_crosses_it();
    test_reuse_distance_in_bytes_counts_everything_in_between();
    test_out_of_order_kernels_are_refused_rather_than_analysed();
    test_a_use_of_an_unregistered_tensor_is_counted_not_silently_dropped();
    test_the_live_set_counts_each_tensor_once();

    test_a_persist_with_no_clear_is_rejected();
    test_an_unjoined_fork_is_rejected();
    test_the_digest_ignores_the_name_and_the_cost_model();
    test_the_per_kernel_index_returns_actions_in_plan_order();

    test_a_trace_round_trips_its_reuse_structure();
    test_a_plan_round_trips_through_json();
    test_a_malformed_trace_is_reported_rather_than_accepted();

    test_a_steady_decode_loop_compiles_once();
    test_the_recording_executor_sees_a_balanced_plan();

    if (g_failures) { std::cout << g_failures << " core check(s) failed\n"; return 1; }
    std::cout << "core tests passed (" << g_checks << " checks)\n";
    return 0;
}
