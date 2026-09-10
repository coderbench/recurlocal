#pragma once
#include <cstddef>
#include <cstdint>
#include <vector>

#include "tensortransit/version.h"

namespace tensortransit {

using TensorId = std::uint64_t;
inline constexpr TensorId kInvalidTensorId = 0;  // 0 is never issued, so {} is "no tensor"

// What a tensor is FOR, which is the only thing that lets one planner reason about several
// kinds of data at once. The role is not a hint about size or lifetime -- those are declared
// separately and measured -- it is the semantic class a policy is allowed to discriminate on.
//
// Roles are append-only and their numeric values are ABI (docs/STABILITY.md section 2).
enum class TensorRole : int {
    Unknown = 0,
    RecurrentState = 1,     // GDN / Mamba / linear-attention state: small, mutable, revisited
    KVCache = 2,            // per-request, grows with context, read once per token per layer
    ModelWeight = 3,        // large, immutable, streamed
    ExpertWeight = 4,       // ModelWeight that only some tokens read
    Activation = 5,         // produced and consumed within a token
    Workspace = 6,          // scratch; no cross-kernel reuse by construction
    SpeculativeState = 7,   // draft/verify state whose future use is probabilistic
    MultimodalFeature = 8,  // vision/audio features reused across many text tokens
};

// How a kernel touches a tensor. Write-only first use of a buffer has no read traffic to
// save, which is why the planner needs this and not just "uses".
enum class AccessKind : int { Read = 0, Write = 1, ReadWrite = 2 };

const char* to_string(TensorRole role) noexcept;
// Returns false and leaves *out untouched on an unrecognised name, so a config parser can
// report the bad value rather than silently defaulting to Unknown.
bool parse_tensor_role(const char* text, TensorRole* out) noexcept;
const char* to_string(AccessKind kind) noexcept;
bool parse_access_kind(const char* text, AccessKind* out) noexcept;

// True when a read of this tensor could have been served from cache -- i.e. the access has
// read traffic at all. A pure Write is a store; there is nothing to make resident for it.
bool has_read_traffic(AccessKind kind) noexcept;

struct TensorDesc {
    const void* ptr = nullptr;
    std::size_t bytes = 0;
    TensorRole role = TensorRole::Unknown;

    // The allocation `ptr` lives inside, when the tensor is a slice of a larger buffer. A
    // window widened past the allocation the runtime owns is not a hint, it is a bug with a
    // performance counter -- so every widening is clamped to this.
    // Zero base_bytes means "the slice is the allocation".
    const void* base = nullptr;
    std::size_t base_bytes = 0;

    bool mutable_data = false;   // written during decode; a persisted line must be written back
    bool request_local = false;  // one per in-flight request: footprint scales with concurrency
    bool model_global = false;   // one per model: footprint is concurrency-independent

    int device = 0;
    // Which request this belongs to, for the concurrency planner. -1 is "not request scoped".
    int request_id = -1;

    // Bytes the runtime will actually move for one use of this tensor, when that differs
    // from `bytes`. An expert weight is 100% of its bytes for the 8 experts a token selects
    // and 0% for the other 248; a KV block is read whole. Zero means "same as bytes".
    std::size_t bytes_per_use = 0;

    std::size_t use_bytes() const noexcept { return bytes_per_use ? bytes_per_use : bytes; }
    // The widest region a window over this tensor may legally cover.
    const void* allocation() const noexcept { return base ? base : ptr; }
    std::size_t allocation_bytes() const noexcept { return base_bytes ? base_bytes : bytes; }
    bool valid() const noexcept { return ptr != nullptr && bytes != 0; }
};

// Identity that survives a runtime recycling its memory.
//
// A TensorId alone cannot: a runtime frees a KV block and allocates another at the same
// address, and a plan compiled against the first would place a window on the second with
// nothing anywhere to notice. The generation is bumped on every unregister, so a handle
// taken before the recycle resolves to nothing afterwards. This is spec section 11, and it
// is the reason `resolve()` returns a pointer that can be null rather than a reference.
struct TensorHandle {
    TensorId id = kInvalidTensorId;
    std::uint64_t generation = 0;
    bool valid() const noexcept { return id != kInvalidTensorId; }
    friend bool operator==(const TensorHandle& a, const TensorHandle& b) noexcept {
        return a.id == b.id && a.generation == b.generation;
    }
    friend bool operator!=(const TensorHandle& a, const TensorHandle& b) noexcept {
        return !(a == b);
    }
};

// A non-owning directory of the tensors a runtime is willing to have coordinated.
//
// TensorTransit must never free runtime-owned model memory and never dereference a tensor
// pointer on the host: everything here is address arithmetic and bookkeeping. The registry
// is the only place that maps an address to a role, which is what makes every planner below
// engine-independent -- a planner sees roles, sizes and reuse, never a model.
//
// Thread-safety: one registry is not synchronised. Hold one per device context, which is
// also the only scope on which the locality budget it feeds means anything.
class TensorRegistry {
public:
    TensorRegistry() noexcept = default;

    // Registers `desc` and returns a handle, or an invalid handle when the descriptor is
    // unusable (null pointer or zero bytes). Never throws on the failure path.
    //
    // Re-registering an address that is already live UPDATES that entry and returns the SAME
    // handle rather than issuing a new id. A runtime that re-declares its state every token
    // -- which is the normal shape, because the addresses can move between tokens -- would
    // otherwise grow this table without bound over a long-running server, and defeat the plan
    // cache by changing the tensor set on every token when nothing had actually changed.
    TensorHandle register_tensor(const TensorDesc& desc);

    // Invalidates the handle. Later lookups of the same id fail even if the runtime
    // immediately reallocates the same address for something else.
    void unregister_tensor(TensorId id) noexcept;
    void unregister_tensor(const TensorHandle& handle) noexcept;

    // nullptr when the handle is stale or unknown. The returned pointer is invalidated by
    // the next register_tensor().
    const TensorDesc* resolve(const TensorHandle& handle) const noexcept;
    // Same, ignoring generation. For a caller that has an id from a serialized plan and has
    // separately established that the plan is still current.
    const TensorDesc* find(TensorId id) const noexcept;
    // The live handle for an address, or an invalid handle. Lets a runtime that only has a
    // pointer at a hook site avoid threading handles through its own call stack.
    TensorHandle find_by_pointer(const void* ptr) const noexcept;

    // Live entries only.
    std::size_t size() const noexcept { return live_; }
    bool empty() const noexcept { return live_ == 0; }
    // Total bytes of live tensors in a role. This is the raw footprint, not what competes
    // for cache over any particular interval -- TransitGraph computes that.
    std::size_t bytes_in_role(TensorRole role) const noexcept;
    std::size_t live_in_role(TensorRole role) const noexcept;

    // Drops every entry. Generations still advance, so handles taken before a reset never
    // resolve afterwards -- a reset is exactly the event that invalidates them.
    void clear() noexcept;

    // Iteration for planners and for the trace writer. Entries are visited in id order and
    // dead slots are skipped. `fn` must not mutate the registry.
    template <typename Fn>
    void for_each(Fn&& fn) const {
        for (const Slot& slot : slots_)
            if (slot.live) fn(TensorHandle{slot.id, slot.generation}, slot.desc);
    }

    // Monotonic counter of structural changes: a registration of a NEW id, an unregister, or
    // a clear. Re-registering an existing address does not bump it unless the descriptor
    // actually changed. This is what the plan cache keys on to know a compiled plan is stale
    // without having to diff the whole table.
    std::uint64_t epoch() const noexcept { return epoch_; }

private:
    struct Slot {
        TensorDesc desc{};
        TensorId id = kInvalidTensorId;
        std::uint64_t generation = 0;
        bool live = false;
    };
    // Index of the live slot holding `ptr`, or -1. Linear: the table is tens of entries for a
    // model's state and the lookup happens once per registration, not per kernel.
    std::ptrdiff_t index_of_pointer(const void* ptr) const noexcept;
    std::ptrdiff_t index_of_id(TensorId id) const noexcept;

    std::vector<Slot> slots_;
    TensorId next_id_ = 1;  // 0 is reserved for kInvalidTensorId
    std::size_t live_ = 0;
    std::uint64_t epoch_ = 0;
};

}  // namespace tensortransit
