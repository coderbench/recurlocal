#include "tensortransit/device.h"

#include <algorithm>
#include <cstring>

namespace tensortransit {

std::size_t DeviceProfile::max_window_bytes() const noexcept {
    std::size_t limit = access_policy_max_window_bytes;
    // Zero means "not reported", which must not read as "no window is allowed" -- a
    // fabricated or partially populated profile would then disable every policy silently.
    if (limit == 0) limit = persisting_l2_max_bytes;
    if (persisting_l2_max_bytes) limit = std::min(limit, persisting_l2_max_bytes);
    // A set-aside is carved OUT of L2, so it cannot exceed L2. A device whose two numbers
    // disagree (an emulator, a MIG slice, a stubbed query, a test fixture) would otherwise
    // get a window larger than its entire cache with nothing to say so.
    if (l2_bytes) limit = std::min(limit, l2_bytes);
    return limit;
}

namespace {
struct PhaseName { RuntimePhase phase; const char* name; };
constexpr PhaseName kPhases[] = {
    {RuntimePhase::Unknown, "unknown"},
    {RuntimePhase::Prefill, "prefill"},
    {RuntimePhase::Decode, "decode"},
    {RuntimePhase::SpeculativeDraft, "speculative_draft"},
    {RuntimePhase::SpeculativeVerify, "speculative_verify"},
};
}  // namespace

const char* to_string(RuntimePhase phase) noexcept {
    for (const auto& entry : kPhases)
        if (entry.phase == phase) return entry.name;
    return "unknown";
}

bool parse_runtime_phase(const char* text, RuntimePhase* out) noexcept {
    if (!text || !out) return false;
    for (const auto& entry : kPhases) {
        if (std::strcmp(text, entry.name) == 0) {
            *out = entry.phase;
            return true;
        }
    }
    return false;
}

}  // namespace tensortransit
