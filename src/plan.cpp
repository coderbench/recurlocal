#include "tensortransit/plan.h"

#include <algorithm>
#include <cstdio>
#include <cstring>

namespace tensortransit {

namespace {
struct KindName { TransitActionKind kind; const char* name; };
constexpr KindName kKinds[] = {
    {TransitActionKind::Normal, "normal"},
    {TransitActionKind::Persist, "persist"},
    {TransitActionKind::Stream, "stream"},
    {TransitActionKind::Prefetch, "prefetch"},
    {TransitActionKind::RotateWindow, "rotate_window"},
    {TransitActionKind::ClearPolicy, "clear_policy"},
    {TransitActionKind::RecordEvent, "record_event"},
    {TransitActionKind::WaitEvent, "wait_event"},
};
struct ReasonName { DeclineReason reason; const char* name; };
constexpr ReasonName kReasons[] = {
    {DeclineReason::None, "none"},
    {DeclineReason::NoReuse, "no_reuse"},
    {DeclineReason::BudgetExhausted, "budget_exhausted"},
    {DeclineReason::TooLarge, "too_large"},
    {DeclineReason::ReuseTooFar, "reuse_too_far"},
    {DeclineReason::RoleExcluded, "role_excluded"},
    {DeclineReason::BelowMinHitRatio, "below_min_hit_ratio"},
    {DeclineReason::NotSupported, "not_supported"},
};

void append_u64(std::string* out, std::uint64_t value) {
    char buffer[24];
    std::snprintf(buffer, sizeof(buffer), "%llu", static_cast<unsigned long long>(value));
    out->append(buffer);
}
void append_double(std::string* out, double value) {
    char buffer[32];
    std::snprintf(buffer, sizeof(buffer), "%.6g", value);
    out->append(buffer);
}
void append_escaped(std::string* out, const char* text) {
    out->push_back('"');
    for (const char* p = text; p && *p; ++p) {
        switch (*p) {
            case '"':  out->append("\\\""); break;
            case '\\': out->append("\\\\"); break;
            case '\n': out->append("\\n"); break;
            case '\r': out->append("\\r"); break;
            case '\t': out->append("\\t"); break;
            default:   out->push_back(*p); break;
        }
    }
    out->push_back('"');
}

// FNV-1a. Not cryptographic and does not need to be: it identifies a plan against other
// plans in the same process and in a golden test, and a collision there is a test that
// passes when it should fail -- not a security boundary.
constexpr std::uint64_t kFnvOffset = 1469598103934665603ull;
constexpr std::uint64_t kFnvPrime = 1099511628211ull;
void hash_bytes(std::uint64_t* h, const void* data, std::size_t bytes) noexcept {
    const auto* p = static_cast<const unsigned char*>(data);
    for (std::size_t i = 0; i < bytes; ++i) {
        *h ^= p[i];
        *h *= kFnvPrime;
    }
}
template <typename T>
void hash_value(std::uint64_t* h, const T& value) noexcept {
    hash_bytes(h, &value, sizeof(T));
}
}  // namespace

const char* to_string(TransitActionKind kind) noexcept {
    for (const auto& entry : kKinds)
        if (entry.kind == kind) return entry.name;
    return "normal";
}

bool parse_transit_action_kind(const char* text, TransitActionKind* out) noexcept {
    if (!text || !out) return false;
    for (const auto& entry : kKinds) {
        if (std::strcmp(text, entry.name) == 0) {
            *out = entry.kind;
            return true;
        }
    }
    return false;
}

const char* to_string(DeclineReason reason) noexcept {
    for (const auto& entry : kReasons)
        if (entry.reason == reason) return entry.name;
    return "none";
}

double PlanCostModel::predicted_traffic_share() const noexcept {
    if (!step_traffic_bytes) return 0.0;
    const double share = static_cast<double>(predicted_saved_bytes) /
                         static_cast<double>(step_traffic_bytes);
    return share < 0.0 ? 0.0 : (share > 1.0 ? 1.0 : share);
}

double PlanCostModel::predicted_throughput_ratio() const noexcept {
    const double f = predicted_traffic_share();
    if (f >= 1.0) return 0.0;  // saturated; the model has left its domain
    return 1.0 / (1.0 - f);
}

double PlanCostModel::ceiling_throughput_ratio() const noexcept {
    if (!step_traffic_bytes) return 1.0;
    const std::size_t bytes = bounded_removable_bytes ? bounded_removable_bytes : removable_bytes;
    double f = static_cast<double>(bytes) / static_cast<double>(step_traffic_bytes);
    if (f < 0.0) f = 0.0;
    if (f >= 1.0) return 0.0;
    return 1.0 / (1.0 - f);
}

bool PlanCostModel::ceiling_saturated() const noexcept {
    if (!step_traffic_bytes) return false;
    // Same threshold and same reason as CeilingReport::kSaturationShare: f/(1-f) just under
    // 1 is a number, not a bound.
    return static_cast<double>(removable_bytes) / static_cast<double>(step_traffic_bytes) > 0.99;
}

std::size_t TransitPlan::count(TransitActionKind kind) const noexcept {
    std::size_t total = 0;
    for (const TransitAction& action : actions_)
        if (action.kind == kind) ++total;
    return total;
}

void TransitPlan::finalize() {
    before_index_.clear();
    after_index_.clear();
    before_keys_.clear();
    before_offsets_.clear();
    after_keys_.clear();
    after_offsets_.clear();

    // Group by kernel while preserving plan order within a group. The executor's per-kernel
    // dispatch is then two binary searches and a contiguous walk, not a scan of the whole
    // plan -- which matters because that walk is on the decode critical path.
    const auto build = [&](bool before) {
        std::vector<const TransitAction*>& index = before ? before_index_ : after_index_;
        std::vector<std::uint64_t>& keys = before ? before_keys_ : after_keys_;
        std::vector<std::uint32_t>& offsets = before ? before_offsets_ : after_offsets_;

        std::vector<std::uint64_t> kernels;
        for (const TransitAction& action : actions_) {
            const KernelId kernel = before ? action.before_kernel : action.after_kernel;
            if (kernel == kInvalidKernelId) continue;
            if (std::find(kernels.begin(), kernels.end(), kernel) == kernels.end())
                kernels.push_back(kernel);
        }
        std::sort(kernels.begin(), kernels.end());
        offsets.push_back(0);
        for (const std::uint64_t kernel : kernels) {
            keys.push_back(kernel);
            for (const TransitAction& action : actions_) {
                const KernelId at = before ? action.before_kernel : action.after_kernel;
                if (at == kernel) index.push_back(&action);
            }
            offsets.push_back(static_cast<std::uint32_t>(index.size()));
        }
    };
    build(true);
    build(false);
    finalized_ = true;
}

const TransitAction* const* TransitPlan::before(KernelId kernel, int* count) const noexcept {
    *count = 0;
    if (!finalized_) return nullptr;
    const auto it = std::lower_bound(before_keys_.begin(), before_keys_.end(), kernel);
    if (it == before_keys_.end() || *it != kernel) return nullptr;
    const auto slot = static_cast<std::size_t>(it - before_keys_.begin());
    *count = static_cast<int>(before_offsets_[slot + 1] - before_offsets_[slot]);
    return before_index_.data() + before_offsets_[slot];
}

const TransitAction* const* TransitPlan::after(KernelId kernel, int* count) const noexcept {
    *count = 0;
    if (!finalized_) return nullptr;
    const auto it = std::lower_bound(after_keys_.begin(), after_keys_.end(), kernel);
    if (it == after_keys_.end() || *it != kernel) return nullptr;
    const auto slot = static_cast<std::size_t>(it - after_keys_.begin());
    *count = static_cast<int>(after_offsets_[slot + 1] - after_offsets_[slot]);
    return after_index_.data() + after_offsets_[slot];
}

const char* TransitPlan::validate() const noexcept {
    for (const TransitAction& action : actions_) {
        if (action.before_kernel == kInvalidKernelId && action.after_kernel == kInvalidKernelId)
            return "an action is placed on no kernel and can never fire";
        switch (action.kind) {
            case TransitActionKind::Persist:
            case TransitActionKind::Stream:
            case TransitActionKind::Prefetch:
            case TransitActionKind::RotateWindow:
                if (action.bytes == 0) return "a region action covers zero bytes";
                if (action.tensor == kInvalidTensorId) return "a region action names no tensor";
                break;
            default:
                break;
        }
        if (action.kind == TransitActionKind::Persist &&
            !(action.hit_ratio > 0.0 && action.hit_ratio <= 1.0))
            return "a persist action has a hit ratio outside (0,1]";
    }

    // Every persist must be cleared. Spec section 32: a policy with no lifetime keeps
    // spending set-aside on a tensor nothing is going to read, and the kernels that follow
    // pay for it with no counter anywhere saying why.
    for (const TransitAction& action : actions_) {
        if (action.kind != TransitActionKind::Persist) continue;
        bool cleared = false;
        for (const TransitAction& other : actions_) {
            if (other.kind == TransitActionKind::ClearPolicy && other.tensor == action.tensor) {
                cleared = true;
                break;
            }
        }
        if (!cleared) return "a persist action is never cleared (see spec section 32)";
    }

    // Every fork must be joined. Under CUDA Graph capture an unjoined fork does not merely
    // leak a stream -- it ends the capture INVALID and takes the runtime's whole decode path
    // with it, which is a failure mode this library must never be the cause of.
    for (const TransitAction& action : actions_) {
        if (action.kind != TransitActionKind::RecordEvent) continue;
        bool joined = false;
        for (const TransitAction& other : actions_)
            if (other.kind == TransitActionKind::WaitEvent && other.event_id == action.event_id) {
                joined = true;
                break;
            }
        if (!joined) return "a record_event has no matching wait_event: a fork is never joined";
    }
    return nullptr;
}

std::string TransitPlan::to_json() const {
    std::string out;
    out.reserve(256 + actions_.size() * 160);
    out += "{\"plan_schema_version\":";
    append_u64(&out, static_cast<std::uint64_t>(kPlanSchemaVersion));
    out += ",\"planner\":";
    append_escaped(&out, planner_name_.c_str());
    out += ",\"digest\":\"";
    {
        char buffer[24];
        std::snprintf(buffer, sizeof(buffer), "%016llx",
                      static_cast<unsigned long long>(digest()));
        out += buffer;
    }
    out += "\",\"cost_model\":{\"step_traffic_bytes\":";
    append_u64(&out, cost_.step_traffic_bytes);
    out += ",\"removable_bytes\":";
    append_u64(&out, cost_.removable_bytes);
    out += ",\"bounded_removable_bytes\":";
    append_u64(&out, cost_.bounded_removable_bytes);
    out += ",\"ceiling_saturated\":";
    out += cost_.ceiling_saturated() ? "true" : "false";
    out += ",\"predicted_saved_bytes\":";
    append_u64(&out, cost_.predicted_saved_bytes);
    out += ",\"budget_bytes\":";
    append_u64(&out, cost_.budget_bytes);
    out += ",\"committed_bytes\":";
    append_u64(&out, cost_.committed_bytes);
    out += ",\"peak_live_bytes\":";
    append_u64(&out, cost_.peak_live_bytes);
    out += ",\"predicted_traffic_share\":";
    append_double(&out, cost_.predicted_traffic_share());
    out += ",\"predicted_throughput_ratio\":";
    append_double(&out, cost_.predicted_throughput_ratio());
    out += ",\"ceiling_throughput_ratio\":";
    append_double(&out, cost_.ceiling_throughput_ratio());
    // Said in the artifact itself, not only in the docs. Anything downstream that reads a
    // predicted figure and prints it next to a measured one has to be made to say which is
    // which, and the cheapest place to enforce that is here.
    out += ",\"basis\":\"model\",\"note\":\"predicted_* are cost-model outputs, not measurements\"}";

    out += ",\"actions\":[";
    for (std::size_t i = 0; i < actions_.size(); ++i) {
        const TransitAction& action = actions_[i];
        if (i) out += ',';
        out += "{\"kind\":";
        append_escaped(&out, to_string(action.kind));
        out += ",\"tensor\":";
        append_u64(&out, action.tensor);
        out += ",\"role\":";
        append_escaped(&out, to_string(action.role));
        if (action.before_kernel != kInvalidKernelId) {
            out += ",\"before_kernel\":";
            append_u64(&out, action.before_kernel);
        }
        if (action.after_kernel != kInvalidKernelId) {
            out += ",\"after_kernel\":";
            append_u64(&out, action.after_kernel);
        }
        out += ",\"bytes\":";
        append_u64(&out, action.bytes);
        if (action.kind == TransitActionKind::Persist) {
            out += ",\"hit_ratio\":";
            append_double(&out, action.hit_ratio);
        }
        if (action.event_id) {
            out += ",\"event_id\":";
            append_u64(&out, action.event_id);
        }
        out += ",\"stream\":";
        append_u64(&out, static_cast<std::uint64_t>(action.stream_id));
        out += ",\"expected_saved_bytes\":";
        append_u64(&out, action.expected_saved_bytes);
        out += '}';
    }
    out += "],\"declines\":[";
    for (std::size_t i = 0; i < declines_.size(); ++i) {
        const TransitDecline& decline = declines_[i];
        if (i) out += ',';
        out += "{\"tensor\":";
        append_u64(&out, decline.tensor);
        out += ",\"role\":";
        append_escaped(&out, to_string(decline.role));
        out += ",\"reason\":";
        append_escaped(&out, to_string(decline.reason));
        out += ",\"bytes\":";
        append_u64(&out, decline.bytes);
        out += ",\"forgone_saved_bytes\":";
        append_u64(&out, decline.forgone_saved_bytes);
        out += '}';
    }
    out += "]}";
    return out;
}

std::string TransitPlan::to_text() const {
    std::string out;
    char buffer[192];
    std::snprintf(buffer, sizeof(buffer), "plan: %s  actions=%zu  declines=%zu  digest=%016llx\n",
                  planner_name_.empty() ? "(unnamed)" : planner_name_.c_str(), actions_.size(),
                  declines_.size(), static_cast<unsigned long long>(digest()));
    out += buffer;

    std::snprintf(buffer, sizeof(buffer),
                  "  budget %zu B, committed %zu B, peak live %zu B\n",
                  cost_.budget_bytes, cost_.committed_bytes, cost_.peak_live_bytes);
    out += buffer;
    std::snprintf(buffer, sizeof(buffer),
                  "  step traffic %zu B, removable %zu B, predicted saved %zu B\n",
                  cost_.step_traffic_bytes, cost_.removable_bytes, cost_.predicted_saved_bytes);
    out += buffer;
    std::snprintf(buffer, sizeof(buffer),
                  "  predicted %+.3f%% against a %+.3f%% ceiling on this device"
                  "  [MODEL, not a measurement]\n",
                  (cost_.predicted_throughput_ratio() - 1.0) * 100.0,
                  (cost_.ceiling_throughput_ratio() - 1.0) * 100.0);
    out += buffer;

    // Grouped by kernel, the shape of spec section 17: a contributor reads this to see what
    // their planner would actually do before asking anyone for a GPU.
    std::vector<std::uint64_t> kernels;
    for (const TransitAction& action : actions_) {
        const std::uint64_t key =
            action.before_kernel != kInvalidKernelId ? action.before_kernel : action.after_kernel;
        if (std::find(kernels.begin(), kernels.end(), key) == kernels.end()) kernels.push_back(key);
    }
    std::sort(kernels.begin(), kernels.end());
    for (const std::uint64_t kernel : kernels) {
        std::snprintf(buffer, sizeof(buffer), "kernel %llu:\n",
                      static_cast<unsigned long long>(kernel));
        out += buffer;
        for (const TransitAction& action : actions_) {
            const bool before = action.before_kernel == kernel;
            const bool after = action.after_kernel == kernel;
            if (!before && !after) continue;
            std::snprintf(buffer, sizeof(buffer), "  %-6s %-13s tensor %llu (%s) %zu B",
                          before ? "before" : "after", to_string(action.kind),
                          static_cast<unsigned long long>(action.tensor), to_string(action.role),
                          action.bytes);
            out += buffer;
            if (action.kind == TransitActionKind::Persist) {
                std::snprintf(buffer, sizeof(buffer), " hit_ratio %.2f", action.hit_ratio);
                out += buffer;
            }
            out += '\n';
        }
    }
    if (!declines_.empty()) {
        out += "declined:\n";
        for (const TransitDecline& decline : declines_) {
            std::snprintf(buffer, sizeof(buffer), "  tensor %llu (%s) %zu B: %s\n",
                          static_cast<unsigned long long>(decline.tensor),
                          to_string(decline.role), decline.bytes, to_string(decline.reason));
            out += buffer;
        }
    }
    return out;
}

std::uint64_t TransitPlan::digest() const noexcept {
    std::uint64_t h = kFnvOffset;
    // Actions only. Not the planner name, not the cost model: two planners that emit the
    // same actions ARE the same plan, and a golden test that pinned the name could not say
    // so. The pointer is excluded for the same reason a trace does not carry one -- it is
    // per-process and would make every digest unreproducible.
    for (const TransitAction& action : actions_) {
        hash_value(&h, action.kind);
        hash_value(&h, action.tensor);
        hash_value(&h, action.before_kernel);
        hash_value(&h, action.after_kernel);
        hash_value(&h, action.bytes);
        // Quantised: a hit ratio is a hint the driver rounds anyway, and hashing the raw
        // double would make a digest depend on the last bit of a floating-point division.
        const auto ratio = static_cast<std::uint32_t>(action.hit_ratio * 10000.0 + 0.5);
        hash_value(&h, ratio);
        hash_value(&h, action.event_id);
        hash_value(&h, action.stream_id);
    }
    return h;
}

void TransitPlan::clear() noexcept {
    actions_.clear();
    declines_.clear();
    cost_ = PlanCostModel{};
    planner_name_.clear();
    before_index_.clear();
    after_index_.clear();
    before_keys_.clear();
    before_offsets_.clear();
    after_keys_.clear();
    after_offsets_.clear();
    finalized_ = false;
}

}  // namespace tensortransit
