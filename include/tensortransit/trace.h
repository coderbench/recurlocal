#pragma once
#include <string>

#include "tensortransit/device.h"
#include "tensortransit/graph.h"
#include "tensortransit/plan.h"
#include "tensortransit/tensor.h"

namespace tensortransit {

// Schema version of a serialized trace. See schemas/trace.schema.json.
inline constexpr int kTraceSchemaVersion = 1;

// Provenance for a trace. Everything here is technical.
//
// Trace privacy (spec section 45) is enforced by construction rather than by policy: there
// is no field on this struct, and no field in the writer, that can carry a prompt, a token,
// a request body or any other application data. A trace records tensor ids, sizes, roles,
// kernel ids, ordering and timing -- and it cannot record anything else, because there is
// nowhere to put it.
struct TraceMetadata {
    int schema_version = kTraceSchemaVersion;
    int device = 0;
    std::string device_name;   // "NVIDIA GeForce RTX 5090"
    std::string model;         // "Qwen3.8-27B"
    std::string model_digest;  // checkpoint digest, so a trace names the weights it saw
    std::string runtime;       // "SparkInfer"
    std::string runtime_commit;
    std::string phase;         // "decode"
    int active_requests = 1;
    bool cyclic = true;        // the recorded window is one iteration of a loop
    std::string notes;
};

// Serializes the registry and the graph. The registry is written as a tensor table (id,
// role, bytes, flags) and the graph as an ordered kernel list with each kernel's uses --
// the shape of spec section 44.
//
// Pointers are NOT written. A trace is meant to be replayed on another machine and a device
// address from another process is meaningless there; the offline planner works in bytes and
// roles, and only the live path resolves regions. That is also why a trace cannot leak an
// address space layout.
std::string write_trace(const TransitGraph& graph, const TensorRegistry& registry,
                        const TraceMetadata& meta);

// Reads a serialized TransitPlan back.
//
// The other two thirds of the offline planning loop already existed: the CLI plans from a
// trace and dumps the plan. This is what lets that plan be fed back to an executor -- which
// is what makes optimizing without touching a runtime possible at all.
//
// Declared HERE rather than in plan.h because it needs the same dependency-free JSON reader
// `read_trace` uses, and a second copy of that parser would be a second place for a
// malformed document to be accepted quietly.
//
// The plan comes back with NULL regions: a serialized plan carries no pointer, by design.
// Call TransitPlan::rebind() against a live registry before executing one.
bool read_plan(const std::string& json, TransitPlan* plan, std::string* error);
bool read_plan_file(const std::string& path, TransitPlan* plan, std::string* error);

// Rebuilds a registry and a graph from a trace. The registry entries carry synthetic
// pointers: distinct, non-null, never dereferenced, and never handed to a driver -- a plan
// built from a trace is for inspection and comparison, not for execution. `read_trace` sets
// `TensorDesc::device` to -1 to mark exactly that, and the CUDA executor refuses such a
// tensor rather than placing a window on a fabricated address.
//
// Returns false and fills `error` on a malformed document. Never throws: a CLI reading an
// arbitrary file from a contributor must report a bad file, not crash on one.
bool read_trace(const std::string& json, TensorRegistry* registry, TransitGraph* graph,
                TraceMetadata* meta, std::string* error);

// Convenience wrappers. False on an I/O error, with the reason in `error`.
bool write_trace_file(const std::string& path, const TransitGraph& graph,
                      const TensorRegistry& registry, const TraceMetadata& meta,
                      std::string* error);
bool read_trace_file(const std::string& path, TensorRegistry* registry, TransitGraph* graph,
                     TraceMetadata* meta, std::string* error);

// A trace is not a device profile, so a plan built offline needs one from somewhere. These
// name the devices this project has actually measured on, so a contributor without hardware
// can plan against the real numbers rather than invent them. Unknown name returns false.
bool device_profile_by_name(const char* name, DeviceProfile* out) noexcept;
const char* const* known_device_names() noexcept;

}  // namespace tensortransit
