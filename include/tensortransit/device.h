#pragma once
#include <cstddef>
#include <cstdint>

#include "tensortransit/version.h"

namespace tensortransit {

// What the planner is allowed to know about the hardware. Never hardcode a GPU assumption:
// every field here is queried at runtime by the CUDA executor, and every one of them is
// zero-means-unknown so that a planner running on a trace from another machine degrades to
// a defensible plan instead of a confident wrong one.
struct DeviceProfile {
    int device = 0;

    std::size_t l2_bytes = 0;
    // What the driver will let us set aside for persisting accesses. NOT the same as
    // l2_bytes and typically well under it -- on an RTX 5090 it is 60 MiB of a 128 MiB L2.
    // Every ceiling this project quotes has this number in its numerator.
    std::size_t persisting_l2_max_bytes = 0;
    std::size_t access_policy_max_window_bytes = 0;

    int sm_count = 0;
    int major = 0;
    int minor = 0;
    std::size_t global_memory_bytes = 0;

    // Peak HBM bandwidth in bytes/second, 0 when unknown. Used only to turn a byte saving
    // into a time saving for the cost model, and reported alongside any figure derived from
    // it -- a bandwidth-derived estimate is a model, not a measurement.
    std::uint64_t peak_bandwidth_bytes_per_s = 0;

    bool supports_persisting_l2() const noexcept { return persisting_l2_max_bytes != 0; }
    // The largest window the hardware will honour over one address range, clamped to the
    // set-aside. A planner asking for more than this gets silently truncated by the driver,
    // which is how a policy comes to protect a fraction of what its telemetry claims.
    std::size_t max_window_bytes() const noexcept;
};

// The part of the picture that is not the hardware and not the graph: how many requests are
// in flight, what phase the runtime is in, and what it has already spent. Passed separately
// from the graph because it changes on a different timescale -- the graph is stable across
// tokens and this is not (spec section 81).
enum class RuntimePhase : int { Unknown = 0, Prefill = 1, Decode = 2, SpeculativeDraft = 3,
                                SpeculativeVerify = 4 };

const char* to_string(RuntimePhase phase) noexcept;
bool parse_runtime_phase(const char* text, RuntimePhase* out) noexcept;

struct RuntimeState {
    RuntimePhase phase = RuntimePhase::Decode;
    int active_requests = 1;
    // Bytes of locality budget already committed by someone else in this process. A second
    // planner instance that ignored this would double-spend the same set-aside.
    std::size_t committed_budget_bytes = 0;
    // True while the compute stream is capturing a CUDA graph. Some actions are illegal
    // there (a device-wide set-aside change) and one is mandatory (attaching a window to the
    // node rather than the stream), so the planner must be able to see it.
    bool graph_capture_active = false;
    // Set-aside the executor actually holds. The driver rounds a request up and the device
    // has a non-zero default before anyone asks, so budgeting against what was REQUESTED
    // rather than what was GRANTED makes every admission decision wrong by the rounding.
    std::size_t granted_budget_bytes = 0;
};

}  // namespace tensortransit
