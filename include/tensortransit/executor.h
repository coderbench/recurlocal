#pragma once
#include <cstddef>
#include <cstdint>
#include <vector>

#include "tensortransit/plan.h"

namespace tensortransit {

// What an executor did, in the currency an integrator can act on. Every counter here exists
// because its absence once let a confident number hide a null result: a plan that computes
// windows and never applies one is not a locality policy, it is hook overhead with a name.
//
// Appended, never inserted -- a consumer built against an older header would otherwise read
// the wrong member with nothing in either build to notice. See docs/STABILITY.md.
struct ExecutorStats {
    std::uint64_t steps = 0;
    std::uint64_t kernels = 0;
    std::uint64_t actions_applied = 0;
    std::uint64_t actions_skipped = 0;   // the backend declined; see the per-kind counts
    std::uint64_t actions_failed = 0;    // the backend tried and the API returned an error

    std::uint64_t persist_applied = 0;
    std::uint64_t persist_bytes = 0;
    std::uint64_t stream_applied = 0;
    std::uint64_t prefetch_applied = 0;
    std::uint64_t prefetch_bytes = 0;
    std::uint64_t clear_applied = 0;
    std::uint64_t events_recorded = 0;
    std::uint64_t events_waited = 0;

    // The single most important number in this struct. Under CUDA Graph decode a stream
    // access-policy window is host-side state the graph never records, so `persist_applied`
    // can be non-zero while the policy is absent from every replay. Only this counter says
    // the window reached a kernel that actually ran.
    std::uint64_t persist_deferred = 0;       // handed back for the caller to attach
    std::uint64_t persist_attached_to_node = 0;  // attach calls that marked >=1 kernel node
    std::uint64_t persist_nodes_attached = 0;    // kernel NODES marked; >= the line above

    // A plan that references a tensor the registry no longer knows -- the runtime recycled
    // the memory under a compiled plan. Non-zero means the plan cache's invalidation is not
    // keeping up, and the executor is right to refuse rather than place a window on whatever
    // is at that address now.
    std::uint64_t stale_tensor_refs = 0;

    // Host nanoseconds spent inside the executor. Spec section 80 budgets planner plus
    // executor under 0.5% of token latency, and a budget nobody measures is a wish.
    std::uint64_t host_ns = 0;

    // Captures this executor's own node attachment INVALIDATED. Non-zero means the run
    // measured the runtime's fallback path, whatever throughput it reported, and the
    // mechanism has latched itself off. It has to be a counter and not a log line: an
    // evaluator reads this struct, and "the number is fine but it is a number about a
    // different code path" is exactly the failure a locality harness cannot see.
    std::uint64_t capture_invalidations = 0;
    // release() reached while the compute stream was capturing. A caller bug (an exception
    // unwinding, usually), and one this library must not make worse by spraying failing CUDA
    // calls into a live capture -- so it declines, and says how often.
    std::uint64_t released_during_capture = 0;
    // Prefetch actions the backend declined: no prefetch stream, or no scratch. Separated
    // from actions_skipped so that "the prefetch arm applied no prefetch" is answerable
    // without subtracting two other counters.
    std::uint64_t prefetch_skipped = 0;
    // Streaming windows handed to the node attach because a capture was active. The same
    // story as `persist_deferred`, for the other half of a shared cache budget: a stream
    // attribute is host-side state a graph never records, so under capture a Stream action
    // that was applied to the stream would be absent from every replay.
    std::uint64_t stream_deferred = 0;

    bool applied_anything() const noexcept {
        return actions_applied != 0 || persist_attached_to_node != 0;
    }
};

// Applies a TransitPlan. Deliberately narrow: an executor may not decide anything. Every
// choice -- which region, which hit ratio, when to clear -- is in the plan, where it is
// serializable and testable. A backend that made policy decisions of its own would put them
// somewhere no CPU test could reach, which is how the region arithmetic came to be wrong the
// first time.
class ITransitExecutor {
public:
    virtual ~ITransitExecutor() = default;

    // Installs the plan. The executor holds a REFERENCE: the plan must outlive it, which is
    // what makes reuse across tokens free (spec section 33).
    virtual void set_plan(const TransitPlan* plan) noexcept = 0;
    virtual const TransitPlan* plan() const noexcept = 0;

    virtual void begin_step() noexcept = 0;
    virtual void before_kernel(KernelId kernel) noexcept = 0;
    virtual void after_kernel(KernelId kernel) noexcept = 0;
    // Closes the step. Under a plan with prefetch_join_at_end this is where the single join
    // is emitted, and under graph capture that is not optional: a fork that is never
    // rejoined ends the capture INVALID and takes the runtime's decode path with it.
    virtual void end_step() noexcept = 0;

    virtual const ExecutorStats& stats() const noexcept = 0;
    virtual void reset_stats() noexcept = 0;
    virtual const char* name() const noexcept = 0;
};

// An executor that applies nothing and records everything.
//
// Not a mock. It is how the plan/executor contract is tested at all on a machine with no
// GPU -- action ordering, clear-after-persist pairing, event pairing, per-kernel dispatch --
// and it is what `tensortransit plan --replay` runs, so a contributor can see exactly what
// their planner would do to a real trace before asking for hardware.
class RecordingExecutor final : public ITransitExecutor {
public:
    struct Record {
        TransitActionKind kind;
        TensorId tensor;
        KernelId kernel;
        bool before;  // fired before the kernel rather than after
        std::size_t bytes;
        double hit_ratio;
    };

    void set_plan(const TransitPlan* plan) noexcept override { plan_ = plan; }
    const TransitPlan* plan() const noexcept override { return plan_; }
    void begin_step() noexcept override;
    void before_kernel(KernelId kernel) noexcept override;
    void after_kernel(KernelId kernel) noexcept override;
    void end_step() noexcept override;
    const ExecutorStats& stats() const noexcept override { return stats_; }
    void reset_stats() noexcept override { stats_ = ExecutorStats{}; }
    const char* name() const noexcept override { return "recording"; }

    const std::vector<Record>& records() const noexcept { return records_; }
    void clear_records() noexcept { records_.clear(); }
    // True when every Persist was followed by a ClearPolicy on the same tensor before the
    // step ended, and every RecordEvent was joined. The two invariants whose violation is
    // invisible until it costs a runtime its graph capture or its cache.
    bool policies_balanced() const noexcept;

private:
    void dispatch(const TransitAction* const* actions, int count, KernelId kernel,
                  bool before) noexcept;

    const TransitPlan* plan_ = nullptr;
    std::vector<Record> records_;
    ExecutorStats stats_{};
};

}  // namespace tensortransit
