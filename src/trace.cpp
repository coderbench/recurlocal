#include "tensortransit/trace.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <map>
#include <sstream>

namespace tensortransit {

namespace {

// --- a minimal JSON reader -------------------------------------------------------------
// Deliberately small and dependency-free: this library is embedded by inference runtimes,
// and obliging one of them to link a JSON library so that an OFFLINE tool can read a trace
// would be a poor trade. It parses the subset the trace schema uses and refuses everything
// else with a position, rather than accepting a malformed document and producing a graph
// that is quietly wrong.
struct Value;
using Object = std::map<std::string, Value>;
using Array = std::vector<Value>;

struct Value {
    enum class Type { Null, Bool, Number, String, Array, Object } type = Type::Null;
    bool boolean = false;
    double number = 0.0;
    std::string text;
    Array array;
    Object object;

    const Value* find(const char* key) const {
        if (type != Type::Object) return nullptr;
        const auto it = object.find(key);
        return it == object.end() ? nullptr : &it->second;
    }
    std::uint64_t as_u64(std::uint64_t fallback = 0) const {
        return type == Type::Number ? static_cast<std::uint64_t>(number) : fallback;
    }
    std::string as_string(const char* fallback = "") const {
        return type == Type::String ? text : std::string(fallback);
    }
    bool as_bool(bool fallback = false) const {
        return type == Type::Bool ? boolean : fallback;
    }
};

class Parser {
public:
    explicit Parser(const std::string& text) : text_(text) {}

    bool parse(Value* out) {
        skip();
        if (!value(out)) return false;
        skip();
        if (pos_ != text_.size()) return fail("trailing content after the document");
        return true;
    }
    const std::string& error() const { return error_; }

private:
    bool fail(const char* what) {
        char buffer[160];
        std::snprintf(buffer, sizeof(buffer), "%s at byte %zu", what, pos_);
        error_ = buffer;
        return false;
    }
    void skip() {
        while (pos_ < text_.size() &&
               (text_[pos_] == ' ' || text_[pos_] == '\t' || text_[pos_] == '\n' ||
                text_[pos_] == '\r'))
            ++pos_;
    }
    bool literal(const char* word) {
        const std::size_t length = std::strlen(word);
        if (text_.compare(pos_, length, word) != 0) return false;
        pos_ += length;
        return true;
    }
    bool value(Value* out) {
        skip();
        if (pos_ >= text_.size()) return fail("unexpected end of document");
        const char c = text_[pos_];
        if (c == '{') return object(out);
        if (c == '[') return array(out);
        if (c == '"') {
            out->type = Value::Type::String;
            return string(&out->text);
        }
        if (literal("true"))  { out->type = Value::Type::Bool; out->boolean = true;  return true; }
        if (literal("false")) { out->type = Value::Type::Bool; out->boolean = false; return true; }
        if (literal("null"))  { out->type = Value::Type::Null; return true; }
        return number(out);
    }
    bool string(std::string* out) {
        if (text_[pos_] != '"') return fail("expected a string");
        ++pos_;
        out->clear();
        while (pos_ < text_.size()) {
            const char c = text_[pos_++];
            if (c == '"') return true;
            if (c != '\\') { out->push_back(c); continue; }
            if (pos_ >= text_.size()) return fail("unterminated escape");
            switch (text_[pos_++]) {
                case '"':  out->push_back('"');  break;
                case '\\': out->push_back('\\'); break;
                case '/':  out->push_back('/');  break;
                case 'n':  out->push_back('\n'); break;
                case 't':  out->push_back('\t'); break;
                case 'r':  out->push_back('\r'); break;
                case 'b':  out->push_back('\b'); break;
                case 'f':  out->push_back('\f'); break;
                case 'u':
                    // The schema is ASCII; a \u escape is legal JSON but nothing this
                    // writer produces, so it is refused rather than mis-decoded.
                    return fail("\\u escapes are not supported in a trace");
                default: return fail("unknown escape");
            }
        }
        return fail("unterminated string");
    }
    bool number(Value* out) {
        const std::size_t start = pos_;
        if (pos_ < text_.size() && (text_[pos_] == '-' || text_[pos_] == '+')) ++pos_;
        bool any = false;
        while (pos_ < text_.size() &&
               ((text_[pos_] >= '0' && text_[pos_] <= '9') || text_[pos_] == '.' ||
                text_[pos_] == 'e' || text_[pos_] == 'E' || text_[pos_] == '-' ||
                text_[pos_] == '+')) {
            ++pos_;
            any = true;
        }
        if (!any) return fail("expected a value");
        out->type = Value::Type::Number;
        out->number = std::strtod(text_.substr(start, pos_ - start).c_str(), nullptr);
        return true;
    }
    bool array(Value* out) {
        out->type = Value::Type::Array;
        ++pos_;  // '['
        skip();
        if (pos_ < text_.size() && text_[pos_] == ']') { ++pos_; return true; }
        for (;;) {
            Value element;
            if (!value(&element)) return false;
            out->array.push_back(std::move(element));
            skip();
            if (pos_ >= text_.size()) return fail("unterminated array");
            if (text_[pos_] == ',') { ++pos_; continue; }
            if (text_[pos_] == ']') { ++pos_; return true; }
            return fail("expected ',' or ']'");
        }
    }
    bool object(Value* out) {
        out->type = Value::Type::Object;
        ++pos_;  // '{'
        skip();
        if (pos_ < text_.size() && text_[pos_] == '}') { ++pos_; return true; }
        for (;;) {
            skip();
            std::string key;
            if (pos_ >= text_.size() || text_[pos_] != '"') return fail("expected a key");
            if (!string(&key)) return false;
            skip();
            if (pos_ >= text_.size() || text_[pos_] != ':') return fail("expected ':'");
            ++pos_;
            Value element;
            if (!value(&element)) return false;
            out->object[key] = std::move(element);
            skip();
            if (pos_ >= text_.size()) return fail("unterminated object");
            if (text_[pos_] == ',') { ++pos_; continue; }
            if (text_[pos_] == '}') { ++pos_; return true; }
            return fail("expected ',' or '}'");
        }
    }

    const std::string& text_;
    std::size_t pos_ = 0;
    std::string error_;
};

void emit_string(std::string* out, const std::string& text) {
    out->push_back('"');
    for (const char c : text) {
        switch (c) {
            case '"':  out->append("\\\""); break;
            case '\\': out->append("\\\\"); break;
            case '\n': out->append("\\n"); break;
            case '\r': out->append("\\r"); break;
            case '\t': out->append("\\t"); break;
            default:
                // Anything non-ASCII is dropped rather than emitted raw: a trace is meant to
                // be machine-readable everywhere, and this writer's own reader is ASCII.
                if (static_cast<unsigned char>(c) >= 0x20 && static_cast<unsigned char>(c) < 0x7f)
                    out->push_back(c);
                break;
        }
    }
    out->push_back('"');
}

void emit_u64(std::string* out, std::uint64_t value) {
    char buffer[24];
    std::snprintf(buffer, sizeof(buffer), "%llu", static_cast<unsigned long long>(value));
    out->append(buffer);
}

struct KnownDevice {
    const char* name;
    DeviceProfile profile;
};

// Devices this project has actually measured on, so a contributor without hardware plans
// against real numbers instead of inventing them. Every field here was read off the device
// by `tensortransit inspect --device`, not off a spec sheet.
const KnownDevice kDevices[] = {
    // Read off the device by `tensortransit_info` on 2026-09-10, driver 595.84, CUDA 13.3.
    // Note l2_bytes: the device reports 96 MiB, not the 128 MiB a spec sheet gives -- which is
    // why this table exists at all rather than a constant somebody typed from a datasheet.
    {"rtx5090",
     DeviceProfile{/*device*/ 0,
                   /*l2_bytes*/ 100663296ull,
                   /*persisting_l2_max_bytes*/ 62914560ull,
                   /*access_policy_max_window_bytes*/ 134217728ull,
                   /*sm_count*/ 170,
                   /*major*/ 12,
                   /*minor*/ 0,
                   /*global_memory_bytes*/ 33670758400ull,
                   /*peak_bandwidth_bytes_per_s*/ 1792128000000ull}},
};
const char* const kDeviceNames[] = {"rtx5090", nullptr};

}  // namespace

std::string write_trace(const TransitGraph& graph, const TensorRegistry& registry,
                        const TraceMetadata& meta) {
    std::string out;
    out.reserve(1024 + graph.uses().size() * 96);
    out += "{\"schema_version\":";
    emit_u64(&out, static_cast<std::uint64_t>(kTraceSchemaVersion));
    out += ",\"device\":";
    emit_u64(&out, static_cast<std::uint64_t>(meta.device));
    out += ",\"device_name\":";
    emit_string(&out, meta.device_name);
    out += ",\"model\":";
    emit_string(&out, meta.model);
    out += ",\"model_digest\":";
    emit_string(&out, meta.model_digest);
    out += ",\"runtime\":";
    emit_string(&out, meta.runtime);
    out += ",\"runtime_commit\":";
    emit_string(&out, meta.runtime_commit);
    out += ",\"phase\":";
    emit_string(&out, meta.phase);
    out += ",\"active_requests\":";
    emit_u64(&out, static_cast<std::uint64_t>(meta.active_requests));
    out += ",\"cyclic\":";
    out += graph.cyclic() ? "true" : "false";
    out += ",\"notes\":";
    emit_string(&out, meta.notes);

    out += ",\"tensors\":[";
    bool first = true;
    registry.for_each([&](const TensorHandle& handle, const TensorDesc& desc) {
        if (!first) out += ',';
        first = false;
        out += "{\"id\":";
        emit_u64(&out, handle.id);
        out += ",\"role\":";
        emit_string(&out, to_string(desc.role));
        out += ",\"bytes\":";
        emit_u64(&out, desc.bytes);
        if (desc.bytes_per_use) {
            out += ",\"bytes_per_use\":";
            emit_u64(&out, desc.bytes_per_use);
        }
        out += ",\"mutable\":";
        out += desc.mutable_data ? "true" : "false";
        out += ",\"request_local\":";
        out += desc.request_local ? "true" : "false";
        out += ",\"model_global\":";
        out += desc.model_global ? "true" : "false";
        out += ",\"request_id\":";
        {
            char buffer[16];
            std::snprintf(buffer, sizeof(buffer), "%d", desc.request_id);
            out += buffer;
        }
        // No pointer. A device address from another process is meaningless on the machine
        // that reads this, and writing one would make a trace carry an address-space layout
        // for no benefit at all.
        out += '}';
    });
    out += "],\"events\":[";

    for (std::size_t k = 0; k < graph.kernels().size(); ++k) {
        const KernelEvent& kernel = graph.kernels()[k];
        if (k) out += ',';
        out += "{\"kernel_id\":";
        emit_u64(&out, kernel.id);
        out += ",\"name\":";
        emit_string(&out, graph.label(kernel.label));
        out += ",\"stream\":";
        emit_u64(&out, static_cast<std::uint64_t>(kernel.stream_id));
        out += ",\"order\":";
        emit_u64(&out, kernel.order);
        if (kernel.estimated_start_ns) {
            out += ",\"start_ns\":";
            emit_u64(&out, kernel.estimated_start_ns);
        }
        if (kernel.estimated_duration_ns) {
            out += ",\"duration_ns\":";
            emit_u64(&out, kernel.estimated_duration_ns);
        }
        out += ",\"tensors\":[";
        bool first_use = true;
        for (const TensorUse& use : graph.uses()) {
            if (use.kernel != kernel.id) continue;
            if (!first_use) out += ',';
            first_use = false;
            out += "{\"id\":";
            emit_u64(&out, use.tensor);
            out += ",\"access\":";
            emit_string(&out, to_string(use.access));
            if (use.bytes) {
                out += ",\"bytes\":";
                emit_u64(&out, use.bytes);
            }
            out += '}';
        }
        out += "]}";
    }
    out += "]}";
    return out;
}

bool read_trace(const std::string& json, TensorRegistry* registry, TransitGraph* graph,
                TraceMetadata* meta, std::string* error) {
    if (!registry || !graph) {
        if (error) *error = "read_trace needs a registry and a graph";
        return false;
    }
    Value root;
    Parser parser(json);
    if (!parser.parse(&root)) {
        if (error) *error = parser.error();
        return false;
    }
    if (root.type != Value::Type::Object) {
        if (error) *error = "the document is not an object";
        return false;
    }
    const Value* version = root.find("schema_version");
    if (!version || version->as_u64() != static_cast<std::uint64_t>(kTraceSchemaVersion)) {
        if (error) {
            char buffer[128];
            std::snprintf(buffer, sizeof(buffer),
                          "schema_version is %llu; this build reads %d",
                          static_cast<unsigned long long>(version ? version->as_u64() : 0),
                          kTraceSchemaVersion);
            *error = buffer;
        }
        return false;
    }

    registry->clear();
    graph->clear();

    if (meta) {
        meta->schema_version = kTraceSchemaVersion;
        if (const Value* v = root.find("device")) meta->device = static_cast<int>(v->as_u64());
        if (const Value* v = root.find("device_name")) meta->device_name = v->as_string();
        if (const Value* v = root.find("model")) meta->model = v->as_string();
        if (const Value* v = root.find("model_digest")) meta->model_digest = v->as_string();
        if (const Value* v = root.find("runtime")) meta->runtime = v->as_string();
        if (const Value* v = root.find("runtime_commit")) meta->runtime_commit = v->as_string();
        if (const Value* v = root.find("phase")) meta->phase = v->as_string();
        if (const Value* v = root.find("active_requests"))
            meta->active_requests = static_cast<int>(v->as_u64());
        if (const Value* v = root.find("notes")) meta->notes = v->as_string();
        if (const Value* v = root.find("cyclic")) meta->cyclic = v->as_bool(true);
    }

    // The trace's tensor ids are the document's own, and the registry issues its own. The
    // map is what keeps a use pointing at the right tensor across that renumbering; without
    // it a trace whose ids do not start at 1 would silently attach every use to the wrong
    // tensor and produce a graph that looks plausible.
    std::map<std::uint64_t, TensorId> id_map;
    const Value* tensors = root.find("tensors");
    if (tensors && tensors->type == Value::Type::Array) {
        std::uintptr_t synthetic = 0x1000;
        for (const Value& entry : tensors->array) {
            TensorDesc desc{};
            const Value* id = entry.find("id");
            if (!id) continue;
            // A synthetic, distinct, never-dereferenced address. It exists because the
            // registry keys on a pointer; -1 for the device marks the tensor as
            // trace-derived so the CUDA executor refuses to place a window on it.
            desc.ptr = reinterpret_cast<const void*>(synthetic);
            synthetic += 0x1000;
            desc.device = -1;
            if (const Value* v = entry.find("bytes")) desc.bytes = v->as_u64();
            if (const Value* v = entry.find("bytes_per_use")) desc.bytes_per_use = v->as_u64();
            if (const Value* v = entry.find("role")) {
                TensorRole role = TensorRole::Unknown;
                if (!parse_tensor_role(v->as_string().c_str(), &role)) {
                    if (error) *error = "unknown tensor role \"" + v->as_string() + "\"";
                    return false;
                }
                desc.role = role;
            }
            if (const Value* v = entry.find("mutable")) desc.mutable_data = v->as_bool();
            if (const Value* v = entry.find("request_local")) desc.request_local = v->as_bool();
            if (const Value* v = entry.find("model_global")) desc.model_global = v->as_bool();
            if (const Value* v = entry.find("request_id"))
                desc.request_id = static_cast<int>(static_cast<std::int64_t>(v->number));
            if (!desc.bytes) continue;
            const TensorHandle handle = registry->register_tensor(desc);
            id_map[id->as_u64()] = handle.id;
        }
    }

    const Value* events = root.find("events");
    if (!events || events->type != Value::Type::Array) {
        if (error) *error = "the document has no events array";
        return false;
    }
    std::uint64_t order = 0;
    for (const Value& event : events->array) {
        KernelEvent kernel{};
        const Value* id = event.find("kernel_id");
        if (!id) {
            if (error) *error = "an event has no kernel_id";
            return false;
        }
        kernel.id = id->as_u64();
        if (const Value* v = event.find("order")) kernel.order = v->as_u64();
        else kernel.order = order;
        order = kernel.order + 1;
        if (const Value* v = event.find("stream")) kernel.stream_id = static_cast<int>(v->as_u64());
        if (const Value* v = event.find("start_ns")) kernel.estimated_start_ns = v->as_u64();
        if (const Value* v = event.find("duration_ns")) kernel.estimated_duration_ns = v->as_u64();
        if (const Value* v = event.find("name"))
            kernel.label = graph->intern_label(v->as_string().c_str());
        if (!graph->record_kernel(kernel)) {
            if (error) *error = "events are not in non-decreasing order, or a kernel id repeats";
            return false;
        }
        const Value* uses = event.find("tensors");
        if (!uses || uses->type != Value::Type::Array) continue;
        for (const Value& entry : uses->array) {
            const Value* tensor_id = entry.find("id");
            if (!tensor_id) continue;
            const auto it = id_map.find(tensor_id->as_u64());
            if (it == id_map.end()) {
                if (error) {
                    char buffer[128];
                    std::snprintf(buffer, sizeof(buffer),
                                  "a use names tensor %llu, which the tensor table does not declare",
                                  static_cast<unsigned long long>(tensor_id->as_u64()));
                    *error = buffer;
                }
                return false;
            }
            TensorUse use{};
            use.tensor = it->second;
            use.kernel = kernel.id;
            if (const Value* v = entry.find("access")) {
                AccessKind access = AccessKind::Read;
                if (!parse_access_kind(v->as_string().c_str(), &access)) {
                    if (error) *error = "unknown access kind \"" + v->as_string() + "\"";
                    return false;
                }
                use.access = access;
            }
            if (const Value* v = entry.find("bytes")) use.bytes = v->as_u64();
            graph->record_use(use);
        }
    }

    graph->set_cyclic(meta ? meta->cyclic : true);
    graph->build(*registry);
    return true;
}

bool read_plan(const std::string& json, TransitPlan* plan, std::string* error) {
    if (!plan) return false;
    plan->clear();
    Value root;
    Parser parser(json);
    if (!parser.parse(&root)) {
        if (error) *error = parser.error();
        return false;
    }
    if (root.type != Value::Type::Object) {
        if (error) *error = "a plan document must be a JSON object";
        return false;
    }
    const Value* version = root.find("plan_schema_version");
    if (!version || version->as_u64() != static_cast<std::uint64_t>(kPlanSchemaVersion)) {
        if (error) {
            char buffer[288];
            std::snprintf(buffer, sizeof(buffer),
                          "plan_schema_version %llu, expected %d. A plan from another schema "
                          "is not a plan this build can execute, and guessing which fields "
                          "still mean the same thing is how a policy comes to be applied to "
                          "the wrong region.",
                          version ? (unsigned long long)version->as_u64() : 0ull,
                          kPlanSchemaVersion);
            *error = buffer;
        }
        return false;
    }
    if (const Value* name = root.find("planner"))
        plan->set_planner_name(name->as_string());

    const Value* actions = root.find("actions");
    if (actions && actions->type == Value::Type::Array) {
        for (const Value& item : actions->array) {
            TransitAction action{};
            const Value* kind = item.find("kind");
            if (!kind || !parse_transit_action_kind(kind->as_string().c_str(), &action.kind)) {
                if (error) *error = "action with an unknown `kind`";
                return false;
            }
            if (const Value* v = item.find("tensor")) action.tensor = v->as_u64();
            if (const Value* v = item.find("role")) {
                TensorRole role{};
                if (parse_tensor_role(v->as_string().c_str(), &role)) action.role = role;
            }
            if (const Value* v = item.find("before_kernel")) action.before_kernel = v->as_u64();
            if (const Value* v = item.find("after_kernel")) action.after_kernel = v->as_u64();
            if (const Value* v = item.find("bytes"))
                action.bytes = static_cast<std::size_t>(v->as_u64());
            if (const Value* v = item.find("hit_ratio"))
                action.hit_ratio = v->type == Value::Type::Number ? v->number : 0.0;
            if (const Value* v = item.find("stream"))
                action.stream_id = static_cast<int>(v->as_u64());
            if (const Value* v = item.find("event"))
                action.event_id = static_cast<std::uint32_t>(v->as_u64());
            if (const Value* v = item.find("expected_saved_bytes"))
                action.expected_saved_bytes = static_cast<std::size_t>(v->as_u64());
            // No `ptr`, ever. A serialized plan carries no address, so this one comes back
            // with null regions and TransitPlan::rebind() against a live registry is what
            // makes it executable. That is not an inconvenience, it is the property that
            // stops a file deciding what memory a driver touches.
            plan->add(action);
        }
    }

    const Value* declines = root.find("declines");
    if (declines && declines->type == Value::Type::Array) {
        for (const Value& item : declines->array) {
            TransitDecline decline{};
            if (const Value* v = item.find("tensor")) decline.tensor = v->as_u64();
            if (const Value* v = item.find("role")) {
                TensorRole role{};
                if (parse_tensor_role(v->as_string().c_str(), &role)) decline.role = role;
            }
            if (const Value* v = item.find("reason")) {
                DeclineReason reason{};
                if (parse_decline_reason(v->as_string().c_str(), &reason)) decline.reason = reason;
            }
            if (const Value* v = item.find("bytes"))
                decline.bytes = static_cast<std::size_t>(v->as_u64());
            if (const Value* v = item.find("forgone_saved_bytes"))
                decline.forgone_saved_bytes = static_cast<std::size_t>(v->as_u64());
            plan->decline(decline);
        }
    }

    if (const Value* cost = root.find("cost_model")) {
        PlanCostModel& out = plan->cost();
        const auto u64 = [&](const char* key, std::size_t* field) {
            if (const Value* v = cost->find(key)) *field = static_cast<std::size_t>(v->as_u64());
        };
        u64("step_traffic_bytes", &out.step_traffic_bytes);
        u64("removable_bytes", &out.removable_bytes);
        u64("bounded_removable_bytes", &out.bounded_removable_bytes);
        u64("predicted_saved_bytes", &out.predicted_saved_bytes);
        u64("predicted_gross_saved_bytes", &out.predicted_gross_saved_bytes);
        u64("reservation_cost_bytes", &out.reservation_cost_bytes);
        u64("budget_bytes", &out.budget_bytes);
        u64("committed_bytes", &out.committed_bytes);
        u64("peak_live_bytes", &out.peak_live_bytes);
        u64("resident_bytes", &out.resident_bytes);
        u64("stream_relieved_bytes", &out.stream_relieved_bytes);
        if (const Value* v = cost->find("cost_model")) {
            CostModel model{};
            if (parse_cost_model(v->as_string().c_str(), &model)) out.cost_model = model;
        }
    }

    plan->finalize();
    if (const char* problem = plan->validate()) {
        if (error) *error = std::string("the plan in this file is not well formed: ") + problem;
        return false;
    }
    return true;
}

bool read_plan_file(const std::string& path, TransitPlan* plan, std::string* error) {
    std::ifstream in(path, std::ios::binary);
    if (!in) {
        if (error) *error = "cannot open " + path;
        return false;
    }
    std::string text((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
    return read_plan(text, plan, error);
}

bool write_trace_file(const std::string& path, const TransitGraph& graph,
                      const TensorRegistry& registry, const TraceMetadata& meta,
                      std::string* error) {
    std::ofstream file(path);
    if (!file) {
        if (error) *error = "cannot open " + path + " for writing";
        return false;
    }
    file << write_trace(graph, registry, meta);
    if (!file) {
        if (error) *error = "write failed on " + path;
        return false;
    }
    return true;
}

bool read_trace_file(const std::string& path, TensorRegistry* registry, TransitGraph* graph,
                     TraceMetadata* meta, std::string* error) {
    std::ifstream file(path);
    if (!file) {
        if (error) *error = "cannot open " + path;
        return false;
    }
    std::ostringstream buffer;
    buffer << file.rdbuf();
    return read_trace(buffer.str(), registry, graph, meta, error);
}

bool device_profile_by_name(const char* name, DeviceProfile* out) noexcept {
    if (!name || !out) return false;
    for (const KnownDevice& device : kDevices) {
        if (std::strcmp(name, device.name) == 0) {
            *out = device.profile;
            return true;
        }
    }
    return false;
}

const char* const* known_device_names() noexcept { return kDeviceNames; }

}  // namespace tensortransit
