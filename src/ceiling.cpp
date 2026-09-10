#include "tensortransit/ceiling.h"

#include <algorithm>
#include <vector>

namespace tensortransit {

namespace {
double ratio_from_share(double share) noexcept {
    if (share <= 0.0) return 1.0;
    if (share >= 1.0) return 0.0;  // saturated; the caller must read unbounded_saturated
    return 1.0 / (1.0 - share);
}
}  // namespace

double CeilingReport::removable_share() const noexcept {
    if (!step_traffic_bytes) return 0.0;
    const double share =
        static_cast<double>(removable_bytes) / static_cast<double>(step_traffic_bytes);
    return share > 1.0 ? 1.0 : share;
}

double CeilingReport::bounded_share() const noexcept {
    if (!step_traffic_bytes) return 0.0;
    const double share =
        static_cast<double>(bounded_saved_bytes) / static_cast<double>(step_traffic_bytes);
    return share > 1.0 ? 1.0 : share;
}

double CeilingReport::removable_ratio() const noexcept { return ratio_from_share(removable_share()); }
double CeilingReport::bounded_ratio() const noexcept { return ratio_from_share(bounded_share()); }

double CeilingReport::resident_fraction() const noexcept {
    if (!required_resident_bytes) return 0.0;
    if (budget_bytes >= required_resident_bytes) return 1.0;
    return static_cast<double>(budget_bytes) / static_cast<double>(required_resident_bytes);
}

CeilingReport compute_ceiling(const TransitGraph& graph, const DeviceProfile& device,
                              RoleMask roles) {
    CeilingReport report{};
    report.step_traffic_bytes = graph.step_traffic_bytes();
    report.budget_bytes = std::min(device.persisting_l2_max_bytes, device.max_window_bytes());

    struct Item {
        std::size_t bytes;
        std::size_t saved;
        double density;
    };
    std::vector<Item> items;
    for (const TensorProfile& profile : graph.profiles()) {
        if (!roles.has(profile.role)) continue;
        if (!profile.has_reuse() || !profile.bytes) continue;
        report.removable_bytes += profile.reused_bytes;
        report.required_resident_bytes += profile.bytes;
        items.push_back(Item{profile.bytes, profile.reused_bytes,
                             static_cast<double>(profile.reused_bytes) /
                                 static_cast<double>(profile.bytes)});
    }
    report.unbounded_saturated =
        report.step_traffic_bytes != 0 &&
        report.removable_share() > CeilingReport::kSaturationShare;

    // The bounded figure is a density-greedy fill of the budget, not a proportional scaling
    // of the unbounded one. Those differ whenever the candidates have different densities:
    // a read-modify-written state saves two bytes per byte held and a streamed weight saves
    // one, so a budget spent on the state is worth twice as much as the same budget spread
    // evenly -- and reporting the even spread would understate what a policy can reach.
    std::sort(items.begin(), items.end(),
              [](const Item& a, const Item& b) { return a.density > b.density; });
    std::size_t remaining = report.budget_bytes;
    for (const Item& item : items) {
        if (!remaining) break;
        const std::size_t granted = std::min(remaining, item.bytes);
        // Partial residency saves proportionally. This is the same linear assumption the
        // planner's cost model makes, and it is stated in both places rather than in
        // neither.
        report.bounded_saved_bytes += static_cast<std::size_t>(
            static_cast<double>(item.saved) * static_cast<double>(granted) /
            static_cast<double>(item.bytes));
        report.bounded_resident_bytes += granted;
        remaining -= granted;
    }
    return report;
}

}  // namespace tensortransit
