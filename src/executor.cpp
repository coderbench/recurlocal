#include "tensortransit/executor.h"

#include <algorithm>

namespace tensortransit {

void RecordingExecutor::begin_step() noexcept {
    ++stats_.steps;
}

void RecordingExecutor::dispatch(const TransitAction* const* actions, int count, KernelId kernel,
                                 bool before) noexcept {
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
