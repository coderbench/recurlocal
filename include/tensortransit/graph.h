#pragma once
#include <cstddef>
#include <cstdint>
#include <vector>

#include "tensortransit/tensor.h"

namespace tensortransit {

using KernelId = std::uint64_t;
inline constexpr KernelId kInvalidKernelId = 0;

// One unit of execution the runtime is willing to bracket. TensorTransit does not need to
// know what a kernel computes -- only enough to place its tensor demand on a timeline.
struct KernelEvent {
    KernelId id = kInvalidKernelId;
    std::uint64_t order = 0;  // position in the recorded sequence; strictly increasing
    int stream_id = 0;

    // Optional. Zero means "not estimated", and every consumer must behave sensibly without
    // it -- ordering alone is enough for a first plan (spec section 12).
    std::uint64_t estimated_start_ns = 0;
    std::uint64_t estimated_duration_ns = 0;

    // Free-form label carried through to traces and plan dumps. Never parsed for meaning:
    // a planner that switched on a kernel name would stop being engine-independent.
    // Index into TransitGraph's label pool, or 0 for none.
    std::uint32_t label = 0;
};

struct TensorUse {
    TensorId tensor = kInvalidTensorId;
    KernelId kernel = kInvalidKernelId;
    AccessKind access = AccessKind::Read;
    std::uint64_t order = 0;  // copied from the kernel, so a use sorts without a lookup
    int stream_id = 0;
    // Bytes this particular use moves, when it is not the whole tensor. Zero means
    // TensorDesc::use_bytes(). An expert weight read by one token and not the next is the
    // case this exists for.
    std::size_t bytes = 0;
};

// How far apart two uses of the same tensor are. These are not three views of one number;
// they answer three different questions and they disagree, which is why the choice is an
// enumerated axis a contributor can sweep rather than a constant somebody picked.
//
//   Ordinal  How many kernels run in between. Cheap, engine-independent, and WRONG as a
//            residency predictor whenever kernels differ in size -- the v0 assumption.
//   Bytes    How much OTHER traffic flows between the two uses. This is what actually
//            evicts a line, and it is the currency the whole persist bound is written in:
//            a tensor survives to its next use only if the cache is bigger than this.
//   Time     How many nanoseconds in between, from the recorded estimates. This is the
//            currency PREFETCH cares about -- a prefetch must be issued far enough ahead to
//            complete, which is a question about time, not about bytes.
//
// Residency is a Bytes question and prefetch timing is a Time question, so a planner that
// does both reads both.
enum class ReuseMetric : int { Ordinal = 0, Bytes = 1, Time = 2 };

const char* to_string(ReuseMetric metric) noexcept;
bool parse_reuse_metric(const char* text, ReuseMetric* out) noexcept;

inline constexpr std::uint64_t kNoNextUse = static_cast<std::uint64_t>(-1);

// A future-use edge: tensor T written or read at `producer` is read again at `consumer`.
// This is the object the whole project is named after -- everything a planner can do is a
// consequence of knowing this edge exists before the consumer runs.
struct TransitEdge {
    TensorId tensor = kInvalidTensorId;
    KernelId producer = kInvalidKernelId;
    KernelId consumer = kInvalidKernelId;
    std::size_t bytes = 0;  // bytes the consumer will read
    AccessKind producer_access = AccessKind::Read;
    AccessKind consumer_access = AccessKind::Read;

    // The three currencies of section ReuseMetric, all precomputed because a planner that
    // recomputed one per candidate would be O(uses) inside its own inner loop.
    std::uint64_t reuse_kernels = 0;  // kernels strictly between the two uses
    std::size_t reuse_bytes = 0;      // bytes of OTHER tensor traffic in between
    std::uint64_t reuse_ns = 0;       // estimated nanoseconds in between, 0 if unestimated

    std::uint64_t distance(ReuseMetric metric) const noexcept {
        switch (metric) {
            case ReuseMetric::Ordinal: return reuse_kernels;
            case ReuseMetric::Bytes:   return static_cast<std::uint64_t>(reuse_bytes);
            case ReuseMetric::Time:    return reuse_ns;
        }
        return reuse_kernels;
    }
};

// What the graph knows about one tensor over the window it covers. A planner works from
// these rather than from raw uses: the decision "is this worth cache" is a function of how
// often it comes back and how much runs in between, not of the individual accesses.
struct TensorProfile {
    TensorId tensor = kInvalidTensorId;
    TensorRole role = TensorRole::Unknown;
    std::size_t bytes = 0;  // resident size: what a window would have to hold

    std::uint32_t uses = 0;
    std::uint32_t read_uses = 0;  // uses with read traffic; the only ones a cache can serve
    // Reuse edges: uses that are read and have an earlier use to be served from.
    std::uint32_t reuse_count = 0;
    std::size_t reused_bytes = 0;  // total bytes those reuse edges would read

    // Over the reuse edges only. kNoNextUse when there are none.
    std::uint64_t min_reuse_kernels = kNoNextUse;
    std::size_t min_reuse_bytes = static_cast<std::size_t>(-1);
    std::uint64_t min_reuse_ns = kNoNextUse;
    std::size_t max_reuse_bytes = 0;

    std::uint64_t first_order = 0;
    std::uint64_t last_order = 0;
    bool mutable_data = false;
    bool request_local = false;
    int request_id = -1;

    bool has_reuse() const noexcept { return reuse_count != 0; }
    // Bytes a perfect cache of unlimited size would save over this graph: every reuse read
    // served from cache. The numerator of every ceiling this project quotes.
    std::size_t removable_bytes() const noexcept { return reused_bytes; }
};

// The Transit Graph: recorded kernel order, recorded tensor demand, and the future-use
// structure derived from the two.
//
// Deliberately a passive data structure with an explicit build step. A graph that
// recomputed itself on every query would put its cost inside a decode loop, and the whole
// point of spec section 33 is that the analysis happens once and the plan is reused across
// tokens.
class TransitGraph {
public:
    TransitGraph() noexcept = default;

    // --- recording -------------------------------------------------------------------
    // Kernels must be recorded in non-decreasing `order`. Returns false and records nothing
    // when they are not: a graph built from an out-of-order stream produces reuse distances
    // that are silently negative-turned-huge, which reads as "never reused" and disables
    // every policy with nothing to say why.
    bool record_kernel(const KernelEvent& kernel);
    // `use.order` is filled in from the kernel when left at 0. Returns false when the kernel
    // has not been recorded.
    bool record_use(const TensorUse& use);
    // Convenience for the common shape: one kernel, its uses, in order.
    bool record(const KernelEvent& kernel, const TensorUse* uses, int use_count);

    // Interns a kernel label and returns the index for KernelEvent::label.
    std::uint32_t intern_label(const char* text);
    const char* label(std::uint32_t index) const noexcept;

    // --- analysis --------------------------------------------------------------------
    // Computes edges and profiles. Idempotent; must be called after the last record_* and
    // before any query below. `registry` supplies role and size for each tensor id; a use
    // of an id the registry does not know is dropped and counted in `unresolved_uses()`.
    void build(const TensorRegistry& registry);
    bool built() const noexcept { return built_; }

    // Treat the recorded window as one iteration of a loop, so the LAST use of a tensor has
    // the FIRST use as its next use. This is not a detail: a decode token is exactly such a
    // loop, and without it every recurrent state looks like it is never reused -- its reuse
    // edge is the one that crosses the token boundary. Off by default because a prefill
    // trace is not a loop and closing it there invents reuse that does not happen.
    void set_cyclic(bool cyclic) noexcept;
    bool cyclic() const noexcept { return cyclic_; }

    // --- queries ---------------------------------------------------------------------
    const std::vector<KernelEvent>& kernels() const noexcept { return kernels_; }
    const std::vector<TensorUse>& uses() const noexcept { return uses_; }
    const std::vector<TransitEdge>& edges() const noexcept { return edges_; }
    const std::vector<TensorProfile>& profiles() const noexcept { return profiles_; }
    const TensorProfile* profile(TensorId tensor) const noexcept;

    // Total bytes the recorded window moves. The denominator of every ceiling: a policy that
    // removes B bytes from a step of T bytes is worth B/(T-B) in throughput, not B/T.
    std::size_t step_traffic_bytes() const noexcept { return step_traffic_; }
    // Bytes a cache of unlimited size would remove -- summed over every reuse edge. The
    // numerator. Compare against step_traffic_bytes() BEFORE choosing a policy: this is the
    // arithmetic that closed the recurrent-only surface, and it costs nothing to run.
    std::size_t removable_bytes() const noexcept { return removable_; }
    std::size_t removable_bytes_in_role(TensorRole role) const noexcept;

    // Bytes that must be simultaneously resident for every reuse edge crossing `order` to
    // hit -- i.e. the live set at that point, counting each tensor once. This generalizes
    // RecurLocal's HotSetModel::TokenFootprint to any mix of roles.
    std::size_t live_bytes_at(std::uint64_t order) const noexcept;
    // The largest such value anywhere in the window. What a cache would have to be to hold
    // everything; compare against the device's persisting capacity to get residency.
    std::size_t peak_live_bytes() const noexcept { return peak_live_; }

    // The next use of `tensor` at or after `order`, or kNoNextUse.
    std::uint64_t next_use_order(TensorId tensor, std::uint64_t order) const noexcept;
    // The edge whose consumer is this use, or nullptr. How a planner asks "if I keep this
    // now, when does it pay off, and how much runs in between".
    const TransitEdge* incoming_edge(TensorId tensor, std::uint64_t consumer_order) const noexcept;

    std::uint64_t unresolved_uses() const noexcept { return unresolved_uses_; }
    // nullptr when the graph is usable, otherwise a static description of what is wrong.
    const char* validate() const noexcept;

    void clear() noexcept;

private:
    const KernelEvent* find_kernel(KernelId id) const noexcept;

    std::vector<KernelEvent> kernels_;
    std::vector<TensorUse> uses_;
    std::vector<TransitEdge> edges_;
    std::vector<TensorProfile> profiles_;
    std::vector<char> labels_{'\0'};  // index 0 is the empty label
    std::size_t step_traffic_ = 0;
    std::size_t removable_ = 0;
    std::size_t peak_live_ = 0;
    std::uint64_t unresolved_uses_ = 0;
    bool built_ = false;
    bool cyclic_ = false;
};

}  // namespace tensortransit
