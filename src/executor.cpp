#include "tensortransit/executor.h"

#include <algorithm>
#include <chrono>

namespace tensortransit {

namespace {
std::uint64_t now_ns() noexcept {
    return static_cast<std::uint64_t>(
        std::chrono::duration_cast<std::chrono::nanoseconds>(
            std::chrono::steady_clock::now().time_since_epoch()).count());
}
}  // namespace

void RecordingExecutor::begin_step() noexcept {
    ++stats_.steps;
}

void RecordingExecutor::dispatch(const TransitAction* const* actions, int count, KernelId kernel,
                                 bool before) noexcept {
    // Timed for the same reason the CUDA executor is: spec section 80 budgets planner PLUS
    // executor under 0.5% of token latency, and the half that is measured only on hardware is
    // the half nobody can gate on in CI. What this measures is the dispatch itself -- the
    // per-kernel index lookup and the action walk -- which is on the decode critical path
    // whatever backend is underneath it.
    const auto start = now_ns();
    for (int i = 0; i < count; ++i) {
        const TransitAction& action = *actions[i];
        records_.push_back(Record{action.kind, action.tensor, kernel, before, action.bytes,
                                  action.hit_ratio});
        ++stats_.actions_applied;
        switch (action.kind) {
            case TransitActionKind::Persist:
                ++stats_.persist_applied;
                stats_.persist_bytes += action.bytes;
                break;
            case TransitActionKind::Stream:
                ++stats_.stream_applied;
                break;
            case TransitActionKind::Prefetch:
                ++stats_.prefetch_applied;
                stats_.prefetch_bytes += action.bytes;
                break;
            case TransitActionKind::ClearPolicy:
                ++stats_.clear_applied;
                break;
            case TransitActionKind::RecordEvent:
                ++stats_.events_recorded;
                break;
            case TransitActionKind::WaitEvent:
                ++stats_.events_waited;
                break;
            default:
                break;
        }
    }
    stats_.host_ns += now_ns() - start;
}

void RecordingExecutor::before_kernel(KernelId kernel) noexcept {
    ++stats_.kernels;
    if (!plan_) return;
    int count = 0;
    const TransitAction* const* actions = plan_->before(kernel, &count);
    if (actions) dispatch(actions, count, kernel, true);
}

void RecordingExecutor::after_kernel(KernelId kernel) noexcept {
    if (!plan_) return;
    int count = 0;
    const TransitAction* const* actions = plan_->after(kernel, &count);
    if (actions) dispatch(actions, count, kernel, false);
}

void RecordingExecutor::end_step() noexcept {}

bool RecordingExecutor::policies_balanced() const noexcept {
    // Every persisted tensor must be cleared, and every fork joined. Checked over what
    // ACTUALLY FIRED rather than over the plan, because a plan can be well formed and still
    // place its clear on a kernel the step never reaches -- which is the failure that leaves
    // a window bound over memory nothing is going to read.
    std::vector<TensorId> persisted;
    std::vector<TensorId> cleared;
    std::size_t forks = 0, joins = 0;
    for (const Record& record : records_) {
        switch (record.kind) {
            case TransitActionKind::Persist:
            case TransitActionKind::Stream:
                if (std::find(persisted.begin(), persisted.end(), record.tensor) == persisted.end())
                    persisted.push_back(record.tensor);
                break;
            case TransitActionKind::ClearPolicy:
                cleared.push_back(record.tensor);
                break;
            case TransitActionKind::RecordEvent: ++forks; break;
            case TransitActionKind::WaitEvent:   ++joins; break;
            default: break;
        }
    }
    if (forks != joins) return false;
    for (const TensorId tensor : persisted)
        if (std::find(cleared.begin(), cleared.end(), tensor) == cleared.end()) return false;
    return true;
}

}  // namespace tensortransit
