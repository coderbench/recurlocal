#include "tensortransit/tensor.h"

#include <cstring>

namespace tensortransit {

const char* version_string() noexcept { return TENSORTRANSIT_VERSION_STRING; }
int version_number() noexcept { return TENSORTRANSIT_VERSION_NUMBER; }

namespace {
struct RoleName { TensorRole role; const char* name; };
// The table is the single source of truth for both directions, so a role added to the enum
// and not to the table fails the round-trip test rather than stringifying as "unknown".
constexpr RoleName kRoles[] = {
    {TensorRole::Unknown, "unknown"},
    {TensorRole::RecurrentState, "recurrent_state"},
    {TensorRole::KVCache, "kv_cache"},
    {TensorRole::ModelWeight, "model_weight"},
    {TensorRole::ExpertWeight, "expert_weight"},
    {TensorRole::Activation, "activation"},
    {TensorRole::Workspace, "workspace"},
    {TensorRole::SpeculativeState, "speculative_state"},
    {TensorRole::MultimodalFeature, "multimodal_feature"},
};
struct AccessName { AccessKind kind; const char* name; };
constexpr AccessName kAccess[] = {
    {AccessKind::Read, "read"},
    {AccessKind::Write, "write"},
    {AccessKind::ReadWrite, "read_write"},
};
}  // namespace

const char* to_string(TensorRole role) noexcept {
    for (const auto& entry : kRoles)
        if (entry.role == role) return entry.name;
    return "unknown";
}

bool parse_tensor_role(const char* text, TensorRole* out) noexcept {
    if (!text || !out) return false;
    for (const auto& entry : kRoles) {
        if (std::strcmp(text, entry.name) == 0) {
            *out = entry.role;
            return true;
        }
    }
    return false;
}

const char* to_string(AccessKind kind) noexcept {
    for (const auto& entry : kAccess)
        if (entry.kind == kind) return entry.name;
    return "read";
}

bool parse_access_kind(const char* text, AccessKind* out) noexcept {
    if (!text || !out) return false;
    for (const auto& entry : kAccess) {
        if (std::strcmp(text, entry.name) == 0) {
            *out = entry.kind;
            return true;
        }
    }
    return false;
}

bool has_read_traffic(AccessKind kind) noexcept {
    return kind == AccessKind::Read || kind == AccessKind::ReadWrite;
}

std::ptrdiff_t TensorRegistry::index_of_pointer(const void* ptr) const noexcept {
    if (!ptr) return -1;
    for (std::size_t i = 0; i < slots_.size(); ++i)
        if (slots_[i].live && slots_[i].desc.ptr == ptr) return static_cast<std::ptrdiff_t>(i);
    return -1;
}

std::ptrdiff_t TensorRegistry::index_of_id(TensorId id) const noexcept {
    if (id == kInvalidTensorId) return -1;
    for (std::size_t i = 0; i < slots_.size(); ++i)
        if (slots_[i].id == id) return static_cast<std::ptrdiff_t>(i);
    return -1;
}

namespace {
// Two descriptors describe the same thing when every field a planner reads is equal. Used
// only to decide whether re-registering an address is a structural change, so it must
// compare everything the plan could depend on -- a bytes change with the same pointer is a
// different tensor as far as a window is concerned.
bool same_desc(const TensorDesc& a, const TensorDesc& b) noexcept {
    return a.ptr == b.ptr && a.bytes == b.bytes && a.role == b.role && a.base == b.base &&
           a.base_bytes == b.base_bytes && a.mutable_data == b.mutable_data &&
           a.request_local == b.request_local && a.model_global == b.model_global &&
           a.device == b.device && a.request_id == b.request_id &&
           a.bytes_per_use == b.bytes_per_use;
}
}  // namespace

TensorHandle TensorRegistry::register_tensor(const TensorDesc& desc) {
    if (!desc.valid()) return TensorHandle{};

    // Re-registration of a live address updates in place. A decode loop that re-declares its
    // state every token must not grow this table without bound, and must not look like a
    // structural change to the plan cache when nothing about the tensor moved.
    const auto existing = index_of_pointer(desc.ptr);
    if (existing >= 0) {
        Slot& slot = slots_[static_cast<std::size_t>(existing)];
        if (!same_desc(slot.desc, desc)) {
            slot.desc = desc;
            ++epoch_;  // same address, different tensor: any compiled plan is stale
        }
        return TensorHandle{slot.id, slot.generation};
    }

    // Reuse a dead slot so a churning workload does not grow the vector forever. The
    // generation on that slot has already been advanced by unregister_tensor(), so a handle
    // to the old occupant cannot resolve to the new one.
    for (Slot& slot : slots_) {
        if (!slot.live) {
            slot.desc = desc;
            slot.id = next_id_++;
            slot.live = true;
            ++live_;
            ++epoch_;
            return TensorHandle{slot.id, slot.generation};
        }
    }

    Slot slot{};
    slot.desc = desc;
    slot.id = next_id_++;
    slot.generation = 1;
    slot.live = true;
    slots_.push_back(slot);
    ++live_;
    ++epoch_;
    return TensorHandle{slot.id, slot.generation};
}

void TensorRegistry::unregister_tensor(TensorId id) noexcept {
    const auto index = index_of_id(id);
    if (index < 0) return;
    Slot& slot = slots_[static_cast<std::size_t>(index)];
    if (!slot.live) return;
    slot.live = false;
    slot.desc = TensorDesc{};
    // The generation advances on the SLOT, so the next occupant of this slot is a different
    // generation and a handle taken before the recycle resolves to nothing. This is the
    // whole point of TensorHandle; without it a freed-and-reallocated address silently
    // inherits a compiled plan's window.
    ++slot.generation;
    slot.id = kInvalidTensorId;
    --live_;
    ++epoch_;
}

void TensorRegistry::unregister_tensor(const TensorHandle& handle) noexcept {
    const auto index = index_of_id(handle.id);
    if (index < 0) return;
    // A stale handle must not unregister whatever now occupies the slot.
    if (slots_[static_cast<std::size_t>(index)].generation != handle.generation) return;
    unregister_tensor(handle.id);
}

const TensorDesc* TensorRegistry::resolve(const TensorHandle& handle) const noexcept {
    if (!handle.valid()) return nullptr;
    const auto index = index_of_id(handle.id);
    if (index < 0) return nullptr;
    const Slot& slot = slots_[static_cast<std::size_t>(index)];
    if (!slot.live || slot.generation != handle.generation) return nullptr;
    return &slot.desc;
}

const TensorDesc* TensorRegistry::find(TensorId id) const noexcept {
    const auto index = index_of_id(id);
    if (index < 0) return nullptr;
    const Slot& slot = slots_[static_cast<std::size_t>(index)];
    return slot.live ? &slot.desc : nullptr;
}

TensorHandle TensorRegistry::find_by_pointer(const void* ptr) const noexcept {
    const auto index = index_of_pointer(ptr);
    if (index < 0) return TensorHandle{};
    const Slot& slot = slots_[static_cast<std::size_t>(index)];
    return TensorHandle{slot.id, slot.generation};
}

std::size_t TensorRegistry::bytes_in_role(TensorRole role) const noexcept {
    std::size_t total = 0;
    for (const Slot& slot : slots_)
        if (slot.live && slot.desc.role == role) total += slot.desc.bytes;
    return total;
}

std::size_t TensorRegistry::live_in_role(TensorRole role) const noexcept {
    std::size_t count = 0;
    for (const Slot& slot : slots_)
        if (slot.live && slot.desc.role == role) ++count;
    return count;
}

void TensorRegistry::clear() noexcept {
    for (Slot& slot : slots_) {
        if (slot.live) ++slot.generation;
        slot.live = false;
        slot.id = kInvalidTensorId;
        slot.desc = TensorDesc{};
    }
    live_ = 0;
    ++epoch_;
}

}  // namespace tensortransit
