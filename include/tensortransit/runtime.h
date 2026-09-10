#pragma once
#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>

#include "tensortransit/device.h"
#include "tensortransit/executor.h"
#include "tensortransit/graph.h"
#include "tensortransit/plan.h"
#include "tensortransit/planner.h"
#include "tensortransit/tensor.h"

namespace tensortransit {

// Why a compiled plan stopped being valid. Recorded rather than inferred: "the plan is
// recompiling every token" is the single most likely reason a locality layer costs more
// than it saves (spec section 80), and without a reason code the only way to find out is a
// profiler.
enum class RecompileReason : int {
    None = 0,
    FirstCompile = 1,
    RegistryEpoch = 2,     // tensors registered/unregistered: the address set moved
    GraphChanged = 3,      // the kernel sequence itself changed
    ConcurrencyChanged = 4,
    PhaseChanged = 5,
    DeviceChanged = 6,
    ConfigChanged = 7,
    Forced = 8,
};

const char* to_string(RecompileReason reason) noexcept;

// Identity of the situation a plan was compiled for. Two situations with the same key get
// the same plan; anything else is a recompile. Spec section 34 -- and NOT a KV cache.
struct PlanKey {
    std::uint64_t registry_epoch = 0;
    std::uint64_t graph_digest = 0;
    int active_requests = 0;
    RuntimePhase phase = RuntimePhase::Decode;
    int device = 0;
    std::uint64_t config_digest = 0;

    friend bool operator==(const PlanKey& a, const PlanKey& b) noexcept;
    friend bool operator!=(const PlanKey& a, const PlanKey& b) noexcept { return !(a == b); }
};

struct RuntimeStats {
    std::uint64_t compiles = 0;
    std::uint64_t plan_reuses = 0;
    std::uint64_t steps = 0;
    std::uint64_t compile_ns = 0;   // host time inside the planner
    std::uint64_t record_ns = 0;    // host time recording graph events
    RecompileReason last_reason = RecompileReason::None;

    // Share of steps that reused a plan rather than recompiling. Under 1.0 in a steady
    // decode loop means something is invalidating the cache every token, and that is a
    // defect however good the plan is.
    double reuse_rate() const noexcept {
        const auto total = compiles + plan_reuses;
        return total ? static_cast<double>(plan_reuses) / static_cast<double>(total) : 0.0;
    }
};

// The whole public surface a runtime needs (spec section 51). Deliberately small: register
// what you own, say what each kernel touches, compile once, bracket each kernel.
//
// Thread-safety: one instance is not synchronised. Hold one per device context / compute
// stream, which is also the only scope on which a shared locality budget means anything
// (spec section 82).
class TransitRuntime {
public:
    TransitRuntime();
    ~TransitRuntime();
    TransitRuntime(const TransitRuntime&) = delete;
    TransitRuntime& operator=(const TransitRuntime&) = delete;

    // --- setup ------------------------------------------------------------------------
    void set_device_profile(const DeviceProfile& device) noexcept;
    const DeviceProfile& device_profile() const noexcept { return device_; }
    // Replaces the planner. Invalidates any compiled plan, because the plan is the
    // planner's output and keeping it would attribute one planner's actions to another --
    // which is exactly the kind of mix-up an A/B harness cannot see.
    void set_planner(std::unique_ptr<ITransitPlanner> planner) noexcept;
    // Convenience: look the planner up by name and configure it. False for an unknown name,
    // leaving the previous planner in place.
    bool set_planner(const char* name, const TransitPlannerConfig& config);
    const ITransitPlanner* planner() const noexcept { return planner_.get(); }
    // Borrowed; must outlive the runtime. Null disables execution while leaving planning on,
    // which is how a plan is inspected without applying it.
    void set_executor(ITransitExecutor* executor) noexcept;
    ITransitExecutor* executor() const noexcept { return executor_; }

    // --- declaration ------------------------------------------------------------------
    TensorHandle register_tensor(const TensorDesc& desc);
    void unregister_tensor(TensorId id) noexcept;
    void unregister_tensor(const TensorHandle& handle) noexcept;
    const TensorRegistry& registry() const noexcept { return registry_; }

    // --- recording --------------------------------------------------------------------
    // Opens a recording window. Everything recorded until end_recording() is one iteration
    // of the loop the plan will be reused across.
    void begin_recording() noexcept;
    bool record_kernel(const KernelEvent& kernel);
    bool record_use(const TensorUse& use);
    // Closes the window and builds the graph. `cyclic` says the window is one iteration of a
    // loop, so the last use of a tensor is followed by its first -- true for decode, false
    // for prefill. Getting this wrong makes every recurrent state look unreused.
    void end_recording(bool cyclic);
    const TransitGraph& graph() const noexcept { return graph_; }

    // --- planning ---------------------------------------------------------------------
    // Compiles if the situation changed, otherwise reuses. Returns the plan in force.
    // Cheap enough to call at the top of every token: the common path is one key compare.
    const TransitPlan& compile(const RuntimeState& state);
    // Compiles unconditionally.
    const TransitPlan& recompile(const RuntimeState& state);
    const TransitPlan& plan() const noexcept { return plan_; }
    void invalidate_plan(RecompileReason reason) noexcept;

    // --- execution --------------------------------------------------------------------
    // Bracket each kernel. No-ops with no executor or no plan, so a runtime can leave the
    // calls in place permanently and turn the layer on from configuration.
    void begin_step() noexcept;
    void before_kernel(KernelId kernel) noexcept;
    void after_kernel(KernelId kernel) noexcept;
    void end_step() noexcept;

    const RuntimeStats& stats() const noexcept { return stats_; }
    void reset_stats() noexcept;
    // Everything: registry, graph, plan, stats. The tensors are not freed -- this library
    // never owns runtime memory.
    void reset() noexcept;

    // One JSON object describing the configuration and what actually happened. A runtime
    // with no JSON dependency can print it verbatim; the eval harness parses it.
    std::string stats_json() const;

private:
    PlanKey make_key(const RuntimeState& state) const noexcept;

    TensorRegistry registry_;
    TransitGraph graph_;
    TransitPlan plan_;
    DeviceProfile device_{};
    std::unique_ptr<ITransitPlanner> planner_;
    ITransitExecutor* executor_ = nullptr;
    PlanKey key_{};
    bool have_plan_ = false;
    bool recording_ = false;
    std::uint64_t config_digest_ = 0;
    RuntimeStats stats_{};
};

}  // namespace tensortransit
