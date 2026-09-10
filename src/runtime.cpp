#include "tensortransit/runtime.h"

#include <chrono>
#include <cstdio>
#include <cstring>

namespace tensortransit {

namespace {
struct ReasonName { RecompileReason reason; const char* name; };
constexpr ReasonName kReasons[] = {
    {RecompileReason::None, "none"},
    {RecompileReason::FirstCompile, "first_compile"},
    {RecompileReason::RegistryEpoch, "registry_epoch"},
    {RecompileReason::GraphChanged, "graph_changed"},
    {RecompileReason::ConcurrencyChanged, "concurrency_changed"},
    {RecompileReason::PhaseChanged, "phase_changed"},
    {RecompileReason::DeviceChanged, "device_changed"},
    {RecompileReason::ConfigChanged, "config_changed"},
    {RecompileReason::Forced, "forced"},
};

constexpr std::uint64_t kFnvOffset = 1469598103934665603ull;
constexpr std::uint64_t kFnvPrime = 1099511628211ull;
template <typename T>
void hash_value(std::uint64_t* h, const T& value) noexcept {
    const auto* p = reinterpret_cast<const unsigned char*>(&value);
    for (std::size_t i = 0; i < sizeof(T); ++i) {
        *h ^= p[i];
        *h *= kFnvPrime;
    }
}

std::uint64_t now_ns() noexcept {
    return static_cast<std::uint64_t>(
        std::chrono::duration_cast<std::chrono::nanoseconds>(
            std::chrono::steady_clock::now().time_since_epoch())
            .count());
}

// Identity of the kernel sequence and its tensor demand. Two tokens of a steady decode loop
// hash the same; a change in batch shape or in which tensors a kernel touches does not.
std::uint64_t digest_graph(const TransitGraph& graph) noexcept {
    std::uint64_t h = kFnvOffset;
    for (const KernelEvent& kernel : graph.kernels()) {
        hash_value(&h, kernel.id);
        hash_value(&h, kernel.order);
        hash_value(&h, kernel.stream_id);
    }
    for (const TensorUse& use : graph.uses()) {
        hash_value(&h, use.tensor);
        hash_value(&h, use.kernel);
        hash_value(&h, use.access);
        hash_value(&h, use.bytes);
    }
    const bool cyclic = graph.cyclic();
    hash_value(&h, cyclic);
    return h;
}
}  // namespace

const char* to_string(RecompileReason reason) noexcept {
    for (const auto& entry : kReasons)
        if (entry.reason == reason) return entry.name;
    return "none";
}

bool operator==(const PlanKey& a, const PlanKey& b) noexcept {
    return a.registry_epoch == b.registry_epoch && a.graph_digest == b.graph_digest &&
           a.active_requests == b.active_requests && a.phase == b.phase &&
           a.device == b.device && a.config_digest == b.config_digest;
}

TransitRuntime::TransitRuntime() : planner_(make_baseline_planner()) {}
TransitRuntime::~TransitRuntime() = default;

void TransitRuntime::set_device_profile(const DeviceProfile& device) noexcept {
    if (std::memcmp(&device, &device_, sizeof(DeviceProfile)) != 0)
        invalidate_plan(RecompileReason::DeviceChanged);
    device_ = device;
}

void TransitRuntime::set_planner(std::unique_ptr<ITransitPlanner> planner) noexcept {
    planner_ = std::move(planner);
    // The plan is the planner's output; keeping it across a planner change would attribute
    // one planner's actions to another, which is exactly the mix-up an A/B cannot see.
    invalidate_plan(RecompileReason::ConfigChanged);
}

bool TransitRuntime::set_planner(const char* name, const TransitPlannerConfig& config) {
    auto planner = make_planner(name, config);
    if (!planner) return false;  // leave the previous planner in place, and say so
    std::uint64_t digest = kFnvOffset;
    hash_value(&digest, config);
    for (const char* p = name; *p; ++p) {
        digest ^= static_cast<unsigned char>(*p);
        digest *= kFnvPrime;
    }
    config_digest_ = digest;
    set_planner(std::move(planner));
    return true;
}

void TransitRuntime::set_executor(ITransitExecutor* executor) noexcept {
    executor_ = executor;
    if (executor_ && have_plan_) executor_->set_plan(&plan_);
}

TensorHandle TransitRuntime::register_tensor(const TensorDesc& desc) {
    return registry_.register_tensor(desc);
}
void TransitRuntime::unregister_tensor(TensorId id) noexcept { registry_.unregister_tensor(id); }
void TransitRuntime::unregister_tensor(const TensorHandle& handle) noexcept {
    registry_.unregister_tensor(handle);
}

void TransitRuntime::begin_recording() noexcept {
    graph_.clear();
    recording_ = true;
}

bool TransitRuntime::record_kernel(const KernelEvent& kernel) {
    if (!recording_) return false;
    const auto start = now_ns();
    const bool ok = graph_.record_kernel(kernel);
    stats_.record_ns += now_ns() - start;
    return ok;
}

bool TransitRuntime::record_use(const TensorUse& use) {
    if (!recording_) return false;
    const auto start = now_ns();
    const bool ok = graph_.record_use(use);
    stats_.record_ns += now_ns() - start;
    return ok;
}

void TransitRuntime::end_recording(bool cyclic) {
    recording_ = false;
    graph_.set_cyclic(cyclic);
    graph_.build(registry_);
    invalidate_plan(RecompileReason::GraphChanged);
}

PlanKey TransitRuntime::make_key(const RuntimeState& state) const noexcept {
    PlanKey key{};
    key.registry_epoch = registry_.epoch();
    key.graph_digest = digest_graph(graph_);
    key.active_requests = state.active_requests;
    key.phase = state.phase;
    key.device = device_.device;
    key.config_digest = config_digest_;
    return key;
}

const TransitPlan& TransitRuntime::compile(const RuntimeState& state) {
    const PlanKey key = make_key(state);
    if (have_plan_ && key == key_) {
        ++stats_.plan_reuses;
        return plan_;
    }
    // Name the reason before recompiling. "The plan is recompiling every token" is the
    // single most likely way this layer costs more than it saves, and a reason code is the
    // difference between diagnosing that from telemetry and needing a profiler.
    RecompileReason reason = RecompileReason::FirstCompile;
    if (have_plan_) {
        if (key.registry_epoch != key_.registry_epoch)      reason = RecompileReason::RegistryEpoch;
        else if (key.graph_digest != key_.graph_digest)     reason = RecompileReason::GraphChanged;
        else if (key.active_requests != key_.active_requests) reason = RecompileReason::ConcurrencyChanged;
        else if (key.phase != key_.phase)                   reason = RecompileReason::PhaseChanged;
        else if (key.device != key_.device)                 reason = RecompileReason::DeviceChanged;
        else                                                reason = RecompileReason::ConfigChanged;
    }
    stats_.last_reason = reason;
    return recompile(state);
}

const TransitPlan& TransitRuntime::recompile(const RuntimeState& state) {
    const auto start = now_ns();
    PlanInput input{};
    input.graph = &graph_;
    input.registry = &registry_;
    input.device = device_;
    input.runtime = state;

    if (planner_) {
        plan_ = planner_->build_plan(input);
    } else {
        plan_.clear();
    }
    if (!plan_.finalized()) plan_.finalize();
    key_ = make_key(state);
    have_plan_ = true;
    ++stats_.compiles;
    stats_.compile_ns += now_ns() - start;
    if (executor_) executor_->set_plan(&plan_);
    return plan_;
}

void TransitRuntime::invalidate_plan(RecompileReason reason) noexcept {
    have_plan_ = false;
    key_ = PlanKey{};
    stats_.last_reason = reason;
}

void TransitRuntime::begin_step() noexcept {
    ++stats_.steps;
    if (executor_) executor_->begin_step();
}
void TransitRuntime::before_kernel(KernelId kernel) noexcept {
    if (executor_) executor_->before_kernel(kernel);
}
void TransitRuntime::after_kernel(KernelId kernel) noexcept {
    if (executor_) executor_->after_kernel(kernel);
}
void TransitRuntime::end_step() noexcept {
    if (executor_) executor_->end_step();
}

void TransitRuntime::reset_stats() noexcept {
    stats_ = RuntimeStats{};
    if (executor_) executor_->reset_stats();
}

void TransitRuntime::reset() noexcept {
    registry_.clear();
    graph_.clear();
    plan_.clear();
    have_plan_ = false;
    recording_ = false;
    key_ = PlanKey{};
    stats_ = RuntimeStats{};
    if (executor_) {
        executor_->set_plan(nullptr);
        executor_->reset_stats();
    }
}

std::string TransitRuntime::stats_json() const {
    char buffer[1024];
    const ExecutorStats empty{};
    const ExecutorStats& ex = executor_ ? executor_->stats() : empty;
    std::snprintf(
        buffer, sizeof(buffer),
        "{\"tensortransit\":\"%s\",\"planner\":\"%s\",\"executor\":\"%s\","
        "\"compiles\":%llu,\"plan_reuses\":%llu,\"reuse_rate\":%.4f,\"steps\":%llu,"
        "\"compile_ns\":%llu,\"record_ns\":%llu,\"last_recompile_reason\":\"%s\","
        "\"plan_actions\":%zu,\"plan_declines\":%zu,\"predicted_saved_bytes\":%zu,"
        "\"actions_applied\":%llu,\"actions_failed\":%llu,\"persist_applied\":%llu,"
        "\"persist_deferred\":%llu,\"persist_attached_to_node\":%llu,"
        "\"persist_nodes_attached\":%llu,\"prefetch_applied\":%llu,"
        "\"stale_tensor_refs\":%llu,\"executor_host_ns\":%llu}",
        version_string(), planner_ ? planner_->name() : "none",
        executor_ ? executor_->name() : "none",
        static_cast<unsigned long long>(stats_.compiles),
        static_cast<unsigned long long>(stats_.plan_reuses), stats_.reuse_rate(),
        static_cast<unsigned long long>(stats_.steps),
        static_cast<unsigned long long>(stats_.compile_ns),
        static_cast<unsigned long long>(stats_.record_ns), to_string(stats_.last_reason),
        plan_.actions().size(), plan_.declines().size(), plan_.cost().predicted_saved_bytes,
        static_cast<unsigned long long>(ex.actions_applied),
        static_cast<unsigned long long>(ex.actions_failed),
        static_cast<unsigned long long>(ex.persist_applied),
        static_cast<unsigned long long>(ex.persist_deferred),
        static_cast<unsigned long long>(ex.persist_attached_to_node),
        static_cast<unsigned long long>(ex.persist_nodes_attached),
        static_cast<unsigned long long>(ex.prefetch_applied),
        static_cast<unsigned long long>(ex.stale_tensor_refs),
        static_cast<unsigned long long>(ex.host_ns));
    return std::string(buffer);
}

}  // namespace tensortransit
