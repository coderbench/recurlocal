#include "tensortransit/graph.h"

#include <algorithm>
#include <cstring>

namespace tensortransit {

namespace {
struct MetricName { ReuseMetric metric; const char* name; };
constexpr MetricName kMetrics[] = {
    {ReuseMetric::Ordinal, "ordinal"},
    {ReuseMetric::Bytes, "bytes"},
    {ReuseMetric::Time, "time"},
};

// Bytes one use actually moves between HBM and the cache.
//
// A ReadWrite counts TWICE: the value is read in and the modified value is written back.
// This is not a refinement, it is the factor of two in every ceiling this project has ever
// quoted -- "48 layers x (3 MiB + 60 KiB) x 2 = 294 MiB per token" is exactly this rule
// applied to a recurrent state that is read-modify-written once per layer per token. A
// model that counted a ReadWrite once would report half the removable traffic and halve
// every ceiling with it.
std::size_t use_traffic(std::size_t bytes, AccessKind access) noexcept {
    return access == AccessKind::ReadWrite ? bytes * 2 : bytes;
}
}  // namespace

const char* to_string(ReuseMetric metric) noexcept {
    for (const auto& entry : kMetrics)
        if (entry.metric == metric) return entry.name;
    return "bytes";
}

bool parse_reuse_metric(const char* text, ReuseMetric* out) noexcept {
    if (!text || !out) return false;
    for (const auto& entry : kMetrics) {
        if (std::strcmp(text, entry.name) == 0) {
            *out = entry.metric;
            return true;
        }
    }
    return false;
}

const KernelEvent* TransitGraph::find_kernel(KernelId id) const noexcept {
    for (const KernelEvent& kernel : kernels_)
        if (kernel.id == id) return &kernel;
    return nullptr;
}

bool TransitGraph::record_kernel(const KernelEvent& kernel) {
    if (kernel.id == kInvalidKernelId) return false;
    // Non-decreasing order is a hard precondition, not a nicety. An out-of-order stream
    // produces a "distance" that underflows to an enormous number, which every planner then
    // reads as "never reused" -- so the policy silently disables itself and the telemetry
    // says nothing was wrong.
    if (!kernels_.empty() && kernel.order < kernels_.back().order) return false;
    if (find_kernel(kernel.id)) return false;  // ids are identity; a duplicate is a caller bug
    kernels_.push_back(kernel);
    built_ = false;
    return true;
}

bool TransitGraph::record_use(const TensorUse& use) {
    if (use.tensor == kInvalidTensorId) return false;
    const KernelEvent* kernel = find_kernel(use.kernel);
    if (!kernel) return false;
    TensorUse copy = use;
    copy.order = kernel->order;
    copy.stream_id = kernel->stream_id;
    uses_.push_back(copy);
    built_ = false;
    return true;
}

bool TransitGraph::record(const KernelEvent& kernel, const TensorUse* uses, int use_count) {
    if (!record_kernel(kernel)) return false;
    for (int i = 0; i < use_count; ++i) {
        TensorUse use = uses[i];
        use.kernel = kernel.id;
        if (!record_use(use)) return false;
    }
    return true;
}

std::uint32_t TransitGraph::intern_label(const char* text) {
    if (!text || !*text) return 0;
    const auto index = static_cast<std::uint32_t>(labels_.size());
    const std::size_t length = std::strlen(text);
    labels_.insert(labels_.end(), text, text + length);
    labels_.push_back('\0');
    return index;
}

const char* TransitGraph::label(std::uint32_t index) const noexcept {
    if (index >= labels_.size()) return "";
    return labels_.data() + index;
}

void TransitGraph::set_cyclic(bool cyclic) noexcept {
    if (cyclic_ != cyclic) built_ = false;
    cyclic_ = cyclic;
}

void TransitGraph::build(const TensorRegistry& registry) {
    edges_.clear();
    profiles_.clear();
    step_traffic_ = 0;
    removable_ = 0;
    peak_live_ = 0;
    live_orders_.clear();
    live_values_.clear();
    unresolved_uses_ = 0;

    // Uses arrive in whatever order the runtime recorded them; the analysis needs them in
    // execution order. Stable so that two uses at the same order keep their recorded
    // sequence, which is what a fused kernel touching several tensors looks like.
    std::stable_sort(uses_.begin(), uses_.end(),
                     [](const TensorUse& a, const TensorUse& b) { return a.order < b.order; });

    // Resolve sizes once. A use of a tensor the registry does not know is dropped -- and
    // counted, because "the plan did nothing" and "half the uses named tensors nobody
    // registered" are different problems with the same symptom.
    const std::size_t n = uses_.size();
    std::vector<std::size_t> traffic(n, 0);
    std::vector<std::size_t> read_bytes(n, 0);
    std::vector<char> resolved(n, 0);
    for (std::size_t i = 0; i < n; ++i) {
        const TensorDesc* desc = registry.find(uses_[i].tensor);
        if (!desc) {
            ++unresolved_uses_;
            continue;
        }
        resolved[i] = 1;
        const std::size_t bytes = uses_[i].bytes ? uses_[i].bytes : desc->use_bytes();
        read_bytes[i] = bytes;
        traffic[i] = use_traffic(bytes, uses_[i].access);
        step_traffic_ += traffic[i];
    }

    // Prefix sums, so "traffic strictly between use i and use j" is one subtraction rather
    // than a scan. Without this, building a graph would be quadratic in the number of uses
    // and would land inside the per-token compile path the plan cache exists to avoid.
    std::vector<std::size_t> prefix(n + 1, 0);
    for (std::size_t i = 0; i < n; ++i) prefix[i + 1] = prefix[i] + traffic[i];

    std::vector<std::uint64_t> kernel_orders;
    kernel_orders.reserve(kernels_.size());
    for (const KernelEvent& kernel : kernels_) kernel_orders.push_back(kernel.order);
    std::sort(kernel_orders.begin(), kernel_orders.end());
    // Distinct orders, which is the axis the live set is answered on. Every query lands in
    // one of these intervals, so the whole curve is a prefix sum over them.
    std::vector<std::uint64_t> distinct_orders = kernel_orders;
    distinct_orders.erase(std::unique(distinct_orders.begin(), distinct_orders.end()),
                          distinct_orders.end());
    std::vector<long long> live_delta(distinct_orders.size() + 1, 0);
    const auto order_index = [&](std::uint64_t order) -> std::size_t {
        const auto it = std::lower_bound(distinct_orders.begin(), distinct_orders.end(), order);
        return static_cast<std::size_t>(it - distinct_orders.begin());
    };
    const auto kernels_between = [&](std::uint64_t lo, std::uint64_t hi) -> std::uint64_t {
        if (hi <= lo) return 0;
        const auto first = std::upper_bound(kernel_orders.begin(), kernel_orders.end(), lo);
        const auto last = std::lower_bound(kernel_orders.begin(), kernel_orders.end(), hi);
        return last > first ? static_cast<std::uint64_t>(last - first) : 0;
    };
    const auto end_ns = [&](std::uint64_t order) -> std::uint64_t {
        for (const KernelEvent& kernel : kernels_)
            if (kernel.order == order) return kernel.estimated_start_ns + kernel.estimated_duration_ns;
        return 0;
    };
    const auto start_ns = [&](std::uint64_t order) -> std::uint64_t {
        for (const KernelEvent& kernel : kernels_)
            if (kernel.order == order) return kernel.estimated_start_ns;
        return 0;
    };

    // Group use indices by tensor, preserving order.
    std::vector<TensorId> tensors;
    std::vector<std::vector<std::size_t>> per_tensor;
    for (std::size_t i = 0; i < n; ++i) {
        if (!resolved[i]) continue;
        const auto it = std::find(tensors.begin(), tensors.end(), uses_[i].tensor);
        if (it == tensors.end()) {
            tensors.push_back(uses_[i].tensor);
            per_tensor.push_back({i});
        } else {
            per_tensor[static_cast<std::size_t>(it - tensors.begin())].push_back(i);
        }
    }

    for (std::size_t t = 0; t < tensors.size(); ++t) {
        const TensorDesc* desc = registry.find(tensors[t]);
        if (!desc) continue;
        const std::vector<std::size_t>& idx = per_tensor[t];

        TensorProfile profile{};
        profile.tensor = tensors[t];
        profile.role = desc->role;
        profile.bytes = desc->bytes;
        profile.uses = static_cast<std::uint32_t>(idx.size());
        profile.first_order = uses_[idx.front()].order;
        profile.last_order = uses_[idx.back()].order;
        profile.mutable_data = desc->mutable_data;
        profile.request_local = desc->request_local;
        profile.request_id = desc->request_id;
        for (std::size_t k : idx)
            if (has_read_traffic(uses_[k].access)) ++profile.read_uses;

        const auto add_edge = [&](std::size_t producer_index, std::size_t consumer_index,
                                  bool wrap) {
            const TensorUse& producer = uses_[producer_index];
            const TensorUse& consumer = uses_[consumer_index];
            // Only a read can be served from cache. A write-only consumer has no read
            // traffic to remove, so residency buys it nothing and admitting it would spend
            // budget on a saving the hardware cannot make.
            if (!has_read_traffic(consumer.access)) return;

            TransitEdge edge{};
            edge.tensor = tensors[t];
            edge.producer = producer.kernel;
            edge.consumer = consumer.kernel;
            edge.bytes = traffic[consumer_index];
            edge.producer_access = producer.access;
            edge.consumer_access = consumer.access;

            if (!wrap) {
                edge.reuse_kernels = kernels_between(producer.order, consumer.order);
                edge.reuse_bytes = prefix[consumer_index] - prefix[producer_index + 1];
                const auto p_end = end_ns(producer.order);
                const auto c_start = start_ns(consumer.order);
                edge.reuse_ns = c_start > p_end ? c_start - p_end : 0;
            } else {
                // The edge that crosses the iteration boundary. For a decode token this is
                // the ONLY edge a recurrent state has -- layer i's state is next read at
                // layer i of the NEXT token -- so a graph that did not close the loop would
                // report the whole recurrent surface as unreused and every planner would
                // correctly decline to cache any of it.
                const std::size_t after = prefix[n] - prefix[producer_index + 1];
                const std::size_t before = prefix[consumer_index];
                edge.reuse_bytes = after + before;
                const std::uint64_t total_kernels = static_cast<std::uint64_t>(kernel_orders.size());
                const auto inside = kernels_between(consumer.order, producer.order);
                edge.reuse_kernels = total_kernels > inside + 1 ? total_kernels - inside - 1 : 0;
                edge.reuse_ns = 0;  // a wrap has no recorded gap; Time degrades, by design
            }

            edges_.push_back(edge);
            // The interval this edge keeps the tensor live over, accumulated as a delta.
            //
            // The intervals of ONE tensor are disjoint by construction -- an edge joins two
            // CONSECUTIVE uses, and the wrap edge covers only the tail and the head -- so no
            // merging is needed and each tensor is counted once at every point, which is what
            // the live set means. This replaces a query that resolved each edge's endpoints
            // by scanning every use, for every edge, for every profile, at every kernel:
            // O(kernels x profiles x edges x uses), on the COMPILE path the plan cache exists
            // to protect. It is now O(edges) to build and O(log orders) to query.
            const std::size_t p_index = order_index(producer.order);
            const std::size_t c_index = order_index(consumer.order);
            const auto live = static_cast<long long>(profile.bytes);
            if (!wrap) {
                if (c_index > p_index) {
                    live_delta[p_index] += live;
                    live_delta[c_index] -= live;
                }
            } else {
                live_delta[p_index] += live;
                live_delta[distinct_orders.size()] -= live;
                if (c_index > 0) {
                    live_delta[0] += live;
                    live_delta[c_index] -= live;
                }
            }
            ++profile.reuse_count;
            profile.reused_bytes += edge.bytes;
            profile.min_reuse_kernels = std::min(profile.min_reuse_kernels, edge.reuse_kernels);
            profile.min_reuse_bytes = std::min(profile.min_reuse_bytes, edge.reuse_bytes);
            profile.max_reuse_bytes = std::max(profile.max_reuse_bytes, edge.reuse_bytes);
            if (edge.reuse_ns) profile.min_reuse_ns = std::min(profile.min_reuse_ns, edge.reuse_ns);
        };

        for (std::size_t k = 0; k + 1 < idx.size(); ++k) add_edge(idx[k], idx[k + 1], false);
        if (cyclic_ && !idx.empty()) add_edge(idx.back(), idx.front(), true);

        if (profile.reuse_count == 0) {
            profile.min_reuse_bytes = static_cast<std::size_t>(-1);
            profile.min_reuse_kernels = kNoNextUse;
        }
        removable_ += profile.reused_bytes;
        profiles_.push_back(profile);
    }

    // Peak live set: the largest total of tensors that must be simultaneously resident for
    // every reuse edge crossing one point to hit. Compared against the device's persisting
    // capacity, this is the residency fraction the whole persist bound turns on.
    live_orders_ = std::move(distinct_orders);
    live_values_.assign(live_orders_.size(), 0);
    long long running = 0;
    for (std::size_t i = 0; i < live_orders_.size(); ++i) {
        running += live_delta[i];
        const auto value = running > 0 ? static_cast<std::size_t>(running) : 0u;
        live_values_[i] = value;
        peak_live_ = std::max(peak_live_, value);
    }

    built_ = true;
}

std::size_t TransitGraph::live_bytes_at(std::uint64_t order) const noexcept {
    // A lookup into the curve build() accumulated, not a re-derivation. Each tensor is
    // counted once at every point, which is what the live set means; the disjointness of one
    // tensor's own intervals is what makes that true without a merge step.
    if (live_orders_.empty()) return 0;
    const auto it = std::upper_bound(live_orders_.begin(), live_orders_.end(), order);
    if (it == live_orders_.begin()) return 0;   // before the first kernel: nothing is live yet
    return live_values_[static_cast<std::size_t>(it - live_orders_.begin()) - 1];
}

std::size_t TransitGraph::removable_bytes_in_role(TensorRole role) const noexcept {
    std::size_t total = 0;
    for (const TensorProfile& profile : profiles_)
        if (profile.role == role) total += profile.reused_bytes;
    return total;
}

const TensorProfile* TransitGraph::profile(TensorId tensor) const noexcept {
    for (const TensorProfile& p : profiles_)
        if (p.tensor == tensor) return &p;
    return nullptr;
}

std::uint64_t TransitGraph::next_use_order(TensorId tensor, std::uint64_t order) const noexcept {
    for (const TensorUse& use : uses_)
        if (use.tensor == tensor && use.order >= order) return use.order;
    return kNoNextUse;
}

const TransitEdge* TransitGraph::incoming_edge(TensorId tensor,
                                               std::uint64_t consumer_order) const noexcept {
    for (const TransitEdge& edge : edges_) {
        if (edge.tensor != tensor) continue;
        for (const TensorUse& use : uses_)
            if (use.kernel == edge.consumer && use.tensor == tensor && use.order == consumer_order)
                return &edge;
    }
    return nullptr;
}

const char* TransitGraph::validate() const noexcept {
    if (kernels_.empty()) return "graph has no kernels";
    for (std::size_t i = 1; i < kernels_.size(); ++i)
        if (kernels_[i].order < kernels_[i - 1].order) return "kernel order is not monotonic";
    for (const TensorUse& use : uses_)
        if (!find_kernel(use.kernel)) return "a use names a kernel that was never recorded";
    if (built_ && step_traffic_ == 0) return "graph moves no bytes: every use was unresolved";
    return nullptr;
}

void TransitGraph::clear() noexcept {
    kernels_.clear();
    uses_.clear();
    edges_.clear();
    profiles_.clear();
    labels_.assign(1, '\0');
    step_traffic_ = 0;
    removable_ = 0;
    peak_live_ = 0;
    live_orders_.clear();
    live_values_.clear();
    unresolved_uses_ = 0;
    built_ = false;
}

}  // namespace tensortransit
