#pragma once
#include <cstddef>

#include "tensortransit/device.h"
#include "tensortransit/graph.h"
#include "tensortransit/planner.h"

namespace tensortransit {

// The arithmetic that decides whether to start.
//
// This is `eval/traffic_budget.py`'s calculation, moved into the core and derived from a
// Transit Graph instead of from a hand-written geometry file. That matters for one specific
// reason: pairing one model's decode rates with another model's state shape produces a
// confident wrong number with nothing in the output to show it, and a graph cannot make that
// mistake because the rates and the shape come from the same recording.
//
// Everything here is an UPPER bound under assumptions that are stated, not hidden:
//   * every resident byte hits,
//   * replacement within the budget is perfect,
//   * a saved byte of traffic converts to time at the step's average rate.
// A real policy reaches some fraction of one of these. None of them may be reported as a
// gain, and `basis` says so in every artifact that carries one.
struct CeilingReport {
    std::size_t step_traffic_bytes = 0;
    // What a cache of unlimited size would remove: every reuse edge served.
    std::size_t removable_bytes = 0;
    // Bytes that would have to be simultaneously resident to remove them all.
    std::size_t required_resident_bytes = 0;

    // The device's budget, and what a density-greedy fill of it actually reaches. This is
    // the number that binds: a persisting window cannot save traffic it cannot hold.
    std::size_t budget_bytes = 0;
    std::size_t bounded_saved_bytes = 0;
    std::size_t bounded_resident_bytes = 0;

    // True when the removable traffic is essentially the whole step -- which happens
    // whenever the recorded window is cyclic and every tensor is re-read next iteration.
    //
    // The threshold is a share of 0.99, not exact equality, and the reason is arithmetic:
    // f/(1-f) at f = 0.99991 is +1176098%, which is a number rather than a bound on
    // anything. Reported rather than clamped, because "+0.000%" -- what f/(1-f) collapses to
    // at exactly 1 -- reads as "there is nothing here" when the truth is the opposite.
    static constexpr double kSaturationShare = 0.99;
    bool unbounded_saturated = false;

    // Share of the step's traffic, in [0,1].
    double removable_share() const noexcept;
    double bounded_share() const noexcept;
    // Throughput ratios. A step carrying f less traffic runs in (1-f) of the time, so tok/s
    // rise by f/(1-f). Quoting the share instead understates the ceiling; this project has
    // made that mistake and it cost a submission a number the repo called a physical limit.
    double removable_ratio() const noexcept;   // 0 when saturated: see unbounded_saturated
    double bounded_ratio() const noexcept;
    // Fraction of the required resident set the budget can hold, in [0,1]. The single most
    // predictive number for whether a persisting policy will pay: measured to pay at 0.98
    // and measured not to pay at 0.48 and below.
    double resident_fraction() const noexcept;
};

// `roles` restricts the calculation to the tensor classes a policy is allowed to act on, so
// "what can a recurrent-only policy reach" and "what can any policy reach" are the same
// function with two masks.
CeilingReport compute_ceiling(const TransitGraph& graph, const DeviceProfile& device,
                              RoleMask roles);

}  // namespace tensortransit
