#!/usr/bin/env python3
"""Produce the one measurement that decides this project: real end-to-end decode.

Everything else RecurLocal measures is explanation. Sections 17, 21 and 28 say the score is
real serving throughput on a pinned model and runtime, and until this script existed there
was nothing that could produce a `real-result.json` for `decide.py --real` to read.

Shape of the experiment, and why:

  one binary          The hook is inert unless RECURLOCAL names a mode, so control and
                      candidate are the same executable with a different environment. Two
                      separately linked binaries cannot separate the policy from the link.

  interleaved         Graphics clocks cannot be pinned on the measurement box, so absolute
                      numbers drift with temperature over minutes. control/candidate pairs
                      run back to back and are compared within the pair; the medians of the
                      per-pair ratios are what get reported.

  token-exact gate    The exact-locality track forbids changing model output. The gate is a
                      greedy replay: identical prompt, identical token ids, compared as a
                      list and not as a score. A candidate that fails it is not scored.

  noise floor         Control repeats are compared against each other to get the run-to-run
                      spread of the measurement itself. A gain inside that spread is
                      reported as unresolved rather than as a win.

Usage:
    eval/real_eval.py --binary .../qwen3_gguf_bench --generate .../qwen3_gguf_generate \\
                      --model /path/to/checkpoint --output real-result.json \\
                      --candidate RECURLOCAL=combined RECURLOCAL_PREFETCH_DISTANCE=1
"""
import argparse, fcntl, json, os, re, statistics, subprocess, sys, time
from datetime import datetime, timezone
from pathlib import Path

SCHEMA_VERSION = 1

# Section 44's matrix. Batch 1 is the only arm a single-sequence decode benchmark can fill;
# the concurrency arms need a runtime path that decodes several sequences at once, and the
# runner refuses to invent them (see --concurrency).
DEFAULT_WEIGHTS = {"batch1": 0.40, "concurrency4": 0.20,
                   "concurrency16": 0.20, "concurrency32": 0.20}


# The model is 21 GB of VRAM and the driver does not always have it back by the time the
# next process asks for it, so a back-to-back arm can fail to load for a reason that has
# nothing to do with what is being measured. Settle, then retry once; a second failure is
# real and is reported.
SETTLE_SECONDS = 0.0 if os.environ.get('RECURLOCAL_EVAL_FAST') else 3.0
# Two eval processes on one GPU do not merely go slower: the second fails to load 21 GB of
# weights, the harness retries into the same contention, and what comes out is a number for a
# run that never happened. That is not hypothetical - it happened twice while this harness was
# being built, and both times the failure looked like a result. An advisory lock is cheap and
# turns a corrupt measurement into a wait.
GPU_LOCK_PATH = os.environ.get("RECURLOCAL_EVAL_LOCK", "/tmp/recurlocal-eval.lock")


class GpuLock:
    def __init__(self, path=GPU_LOCK_PATH, verbose=True):
        self.path, self.verbose, self.fh = path, verbose, None

    def __enter__(self):
        self.fh = open(self.path, "w")
        try:
            fcntl.flock(self.fh, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            if self.verbose:
                print(f">> another eval holds {self.path}; waiting rather than racing it for VRAM",
                      flush=True)
            fcntl.flock(self.fh, fcntl.LOCK_EX)
        return self

    def __exit__(self, *exc):
        if self.fh:
            fcntl.flock(self.fh, fcntl.LOCK_UN)
            self.fh.close()
        return False
LOAD_FAILURE_MARKERS = ("[FAIL] load", "out of memory", "cudaErrorMemoryAllocation")


# Every name the adapter reads. The control arm must have all of them scrubbed from the
# inherited environment, not merely left unset by the caller: an operator with
# `export RECURLOCAL=combined` in their shell would otherwise run a hooked "control", and the
# harness would report ~0% for a comparison of the candidate against itself.
#
# BOTH prefixes, and this is not belt-and-braces. As of 0.2.0 the adapter reads
# TENSORTRANSIT_X in preference to RECURLOCAL_X, so a scrub that knew only the old prefix
# would leave exactly the hole this guard exists to close -- and it would leave it silently,
# because a contaminated control produces a plausible number rather than an error.
ADAPTER_ENV_PREFIXES = ("TENSORTRANSIT", "RECURLOCAL")
# The 0.1 name, kept because it is referenced from docs and from run_from_base.sh.
ADAPTER_ENV_PREFIX = "RECURLOCAL"


def scrubbed_environment(base=None):
    """A copy of `base` with every adapter-controlled name removed.

    Exposed as a function rather than inlined so eval/test_decide.py can assert it, which is
    the only way a guard like this stays correct: nothing about a contaminated control looks
    wrong in the output."""
    env = dict(os.environ if base is None else base)
    for key in [k for k in env if k.startswith(ADAPTER_ENV_PREFIXES)]:
        del env[key]
    return env


def run(cmd, env_extra, timeout=1800, scrub_adapter_env=False):
    env = scrubbed_environment() if scrub_adapter_env else dict(os.environ)
    env.update(env_extra)
    t0 = time.time()
    time.sleep(SETTLE_SECONDS)
    p = subprocess.run(cmd, env=env, capture_output=True, text=True, timeout=timeout)
    out = p.stdout + p.stderr
    if p.returncode != 0 and any(m in out for m in LOAD_FAILURE_MARKERS):
        time.sleep(SETTLE_SECONDS * 4)
        p = subprocess.run(cmd, env=env, capture_output=True, text=True, timeout=timeout)
        out = p.stdout + p.stderr
    return p.returncode, out, time.time() - t0


def parse_sweep(text):
    """The bench prints one SWEEP_JSON line per process: {ctx: {decode_tps, prefill_pp}}."""
    m = re.search(r"SWEEP_JSON (\{.*\})", text)
    if not m:
        return None
    return json.loads(m.group(1))


def parse_adapter_stats(text):
    """The hook's own JSON, printed to stderr. Absent for the control, which is the point:
    it is how a result file records that the control really was unhooked."""
    m = re.search(r"(?:TENSORTRANSIT|RECURLOCAL)_STATS (\{.*\})", text)
    return json.loads(m.group(1)) if m else None


def declared_mode(env_extra):
    """The mode this arm ASKED for, under either spelling.

    One reader for both names, because the two guards below used to spell it differently:
    `is_control` read TENSORTRANSIT then RECURLOCAL, and the null-candidate guard read only
    RECURLOCAL. An arm configured `TENSORTRANSIT=baseline` -- the hook installed with no
    window, which is how the hook's own cost is measured and is one of the five arms the
    specification names -- therefore looked like a policy arm to that guard, and was refused as
    a NULL CANDIDATE for doing exactly what it was asked to do. Two spellings of one variable
    is a compatibility shim (docs/STABILITY.md section 6a); two READINGS of it is a bug.
    """
    return (env_extra or {}).get("TENSORTRANSIT",
                                 (env_extra or {}).get("RECURLOCAL", "off"))


def is_control(env_extra):
    """A control arm is one that names no mode under EITHER spelling."""
    return declared_mode(env_extra) in ("off", "0")


def is_baseline_arm(env_extra):
    """The hook with no window. It is SUPPOSED to apply no policy, so the null-candidate guard
    must not fire on it -- and a preset or planner named `baseline` says the same thing."""
    return (declared_mode(env_extra) == "baseline"
            or (env_extra or {}).get("TENSORTRANSIT_PRESET") == "baseline"
            or (env_extra or {}).get("RECURLOCAL_PRESET") == "baseline"
            or (env_extra or {}).get("TENSORTRANSIT_PLANNER") == "baseline"
            or (env_extra or {}).get("RECURLOCAL_PLANNER") == "baseline")


def policy_applied(stats):
    """Did this run actually DO anything, or merely load?

    A window that is computed and handed back is not a window that reached the kernel. Under
    CUDA-Graph decode with the safe WindowAttach::Stream, `persist` defers every window to a
    runtime that does not attach it: windows_applied 0, windows_attached_to_node 0,
    pre_touch_launches 0, and 192 windows_deferred_to_caller. That configuration is a null
    candidate — it measures the hook's overhead against the control and nothing else — and
    the first version of this harness scored exactly that as a go/no-go verdict because the
    guard only checked that the hook had initialised.
    """
    st = stats.get("stats", {}) if stats else {}
    return (st.get("windows_applied", 0) + st.get("windows_attached_to_node", 0)
            + st.get("pre_touch_launches", 0)) > 0


def hook_ran(stats):
    """Did the adapter ever initialise in this process?

    ever_initialised, not initialised: the stats snapshot is printed at process exit, by
    which time the runtime's model destructor has already called shutdown() and cleared the
    live flag. Older adapters do not emit the field, so fall back to the live one.

    One definition, used by both the guard that refuses an unhooked candidate and the
    correctness record. They disagreed once: the record read the live flag and so said
    candidate_hook_active=false on every run that worked, which reads as "the token-exact
    gate compared the control against itself" - the opposite of what the gate proved.
    """
    if not stats:
        return False
    return bool(stats.get("ever_initialised", stats.get("initialised")))


def declined_every_candidate(stats):
    """Did the PLANNER decide to do nothing, or did the plumbing fail?

    Both look identical in the 0.1 counters -- windows_applied 0, attached 0, pre_touch 0 --
    and they are opposite facts about a run. The transit engine's decline census tells them
    apart, and the case is not hypothetical: `naive_both`, which persists both tensor classes
    with no arbitration, declines every one of 512 candidates as `below_min_hit_ratio` at
    concurrency 4, because a budget shared among that many gives each less than the configured
    floor. That is the arm doing exactly what the specification says it should do badly, and
    refusing it as a NULL CANDIDATE would have taken the other four arms down with it.

    The plumbing failure keeps its guard: a window that was computed and handed back
    (`windows_deferred_to_caller > 0`) never reached a kernel, which is the incident the
    NULL CANDIDATE check exists for.
    """
    st = (stats or {}).get("stats", {})
    if st.get("windows_deferred_to_caller", 0):
        return None
    declines = ((stats or {}).get("transit") or {}).get("declines")
    if not isinstance(declines, dict):
        return None                      # the v0 engine keeps no census; no opinion
    total = sum(v for v in declines.values() if isinstance(v, (int, float)))
    if total <= 0:
        return None
    named = {k: v for k, v in declines.items() if v and k != "none"}
    return {"declined": total, "reasons": named} if named else None


def require_hook_engaged(text, env_extra, label):
    """A candidate run that produced no telemetry ran the control.

    The adapter turns itself off on a configuration it cannot parse, which is the right
    behaviour for a runtime and the wrong one for a measurement: the run would complete,
    the number would look fine, and it would be a number for the unhooked binary carrying
    the candidate's label. Every arm that asked for a mode has to prove it got one."""
    if is_control(env_extra):
        return None
    stats = parse_adapter_stats(text)
    if stats is None:
        raise SystemExit(f"{label}: candidate asked for {declared_mode(env_extra)!r} but the "
                         "run emitted no RECURLOCAL_STATS line. The binary is unhooked, or the "
                         "adapter refused the configuration. Refusing to report it as a candidate.\n"
                         + text[-2000:])
    if not hook_ran(stats) or stats.get("broken"):
        raise SystemExit(f"{label}: the hook did not initialise (ever_initialised="
                         f"{stats.get('ever_initialised')}, initialised={stats.get('initialised')}, "
                         f"broken={stats.get('broken')}, "
                         f"config_error={stats.get('config_error')}).\n" + text[-2000:])
    if stats.get("stats", {}).get("layers", 0) == 0:
        raise SystemExit(f"{label}: the hook initialised but bracketed no recurrent layer. "
                         "Either the model is not hybrid or the hook site was not reached.")
    if not is_baseline_arm(env_extra) and not policy_applied(stats):
        st = stats.get("stats", {})
        decided = declined_every_candidate(stats)
        if decided:
            # An EMPTY PLAN, not a null candidate. Said out loud, because a run that applied
            # nothing is worth a line in the log whichever of the two it was.
            print(f"    {label}: empty plan -- the planner declined all "
                  f"{decided['declined']} candidates ("
                  + ", ".join(f"{k}={v}" for k, v in sorted(decided["reasons"].items()))
                  + "). Measured as the arm it is.", file=sys.stderr, flush=True)
            return stats
        raise SystemExit(
            f"{label}: NULL CANDIDATE. The hook loaded and bracketed "
            f"{st.get('layers')} layers but applied no policy at all — "
            f"windows_applied={st.get('windows_applied')}, "
            f"windows_attached_to_node={st.get('windows_attached_to_node')}, "
            f"pre_touch_launches={st.get('pre_touch_launches')}, "
            f"windows_deferred_to_caller={st.get('windows_deferred_to_caller')}.\n"
            "Deferred windows are computed and handed back; unless the runtime attaches them "
            "they never reach a kernel. Scoring this would report the hook's overhead as a "
            "locality result. Use RECURLOCAL=baseline if measuring overhead is the intent.")
    return stats


# Which tensor families a preset is ABOUT. An arm that names one and registered none of it did
# not measure a weak mechanism; it measured the absence of one, and the two are
# indistinguishable from outside the run. This is the same class of guard as NULL CANDIDATE and
# it exists for the same reason: through 0.2.0 no adapter exposed KV to the registry at all, so
# every KV-scoped arm would have reported the recurrent policy's number under a KV label.
PRESET_TENSOR_FAMILIES = {
    "kv_only": ("kv_tensors",),
    "recurrent_only": ("recurrent_tensors",),
    "naive_both": ("kv_tensors", "recurrent_tensors"),
    "global": ("kv_tensors", "recurrent_tensors"),
}
FAMILY_ENGLISH = {"kv_tensors": "KV cache", "recurrent_tensors": "recurrent state",
                  "weight_tensors": "model weights"}


def declared_preset(env_extra):
    return ((env_extra or {}).get("TENSORTRANSIT_PRESET")
            or (env_extra or {}).get("RECURLOCAL_PRESET")
            or (env_extra or {}).get("TENSORTRANSIT_PLANNER")
            or (env_extra or {}).get("RECURLOCAL_PLANNER") or "")


def require_registered_families(stats, env_extra, label):
    """An arm scoped to a tensor family the registry never saw is unmeasurable, not weak.

    Returns a record for the artifact -- including the mechanisms this configuration could NOT
    have exercised, which is not a failure but must not be silently forgotten. The `Stream`
    action needs a ModelWeight tensor, and the adapter registers one only when the operator
    declares TENSORTRANSIT_STREAMED_BYTES_PER_TOKEN. A `global` sweep run without it measures
    role-floor arbitration alone -- which is exactly what the 0.2.1 arms sweep did, and the
    result was read as evidence about coordination until the counters said otherwise.
    """
    registry = (stats or {}).get("registry")
    if registry is None:
        return None                      # the v0 engine, or a build older than this counter
    preset = declared_preset(env_extra)
    required = PRESET_TENSOR_FAMILIES.get(preset, ())
    absent = [family for family in required if not registry.get(family)]
    if absent:
        raise SystemExit(
            f"{label}: UNMEASURABLE ARM. The configuration asks for preset {preset!r}, which "
            f"is about {' and '.join(FAMILY_ENGLISH[f] for f in required)}, and the graph it "
            f"planned over registered no "
            f"{' and no '.join(FAMILY_ENGLISH[f] for f in absent)} at all "
            f"({', '.join(f'{k}={v}' for k, v in sorted(registry.items()))}). Whatever this "
            f"run measured, it is not that preset. A runtime that does not declare its KV "
            f"pools cannot be used to score a KV-scoped planner.")
    untested = []
    if preset in ("global", "naive_both") and not registry.get("weight_tensors"):
        untested.append("stream")
    return {"registry": registry, "preset": preset,
            "untested_mechanisms": untested,
            # Said in English because a zero in a counter is easy to read past.
            "note": ("no ModelWeight tensor was registered, so no Stream action could be "
                     "taken; set TENSORTRANSIT_STREAMED_BYTES_PER_TOKEN to test it"
                     if untested else "")}


# Below this share of the DECODE STEPS a concurrency arm was asked for going through the
# runtime's packed path, the run did not measure concurrent decode at all -- it measured the
# single-sequence path executed once per row. A tail step of one row always falls through, so
# the normal figure is high but not 1.0; a collapse is near zero, not a few percent short.
PACKED_SHARE_MIN = 0.50

# The denominator is `max_new`, not the adapter's `tokens`, and that is a correction rather
# than a preference.
#
# `tokens` counts every step the hook brackets, and at long context most of them are PREFILL
# chunks: at ctx4096 a c=16 run brackets 133 steps of which 63 are decode, so a run whose every
# decode step batched sixteen rows reported a 47.4% "packed share" and was refused. The first
# full TTF-1 matrix lost four cells that way and its receipt read -99.5%. Two of the four were
# exactly this -- `max_rows_seen` 16 and 32, every decode step packed, wrong denominator. One
# decode step advances every live row, so the number of decode steps a concurrency arm should
# produce is `max_new`: known to the harness, and immune to how long the prefill was.
#
# The other two were real: `max_rows_seen` 0, not one batched decode step at any decode length,
# in the candidate AND in a baseline probe that ran no policy. That is the unambiguous test and
# it is checked first.
MIN_ROWS_FOR_CONCURRENCY = 2


def packed_path_used(stats, concurrency, max_new=None):
    """Did the runtime actually batch, or did it fall back to decoding one row at a time?

    This is the 32-sequence cliff, made visible. SparkInfer declines a packed forward for
    reasons that have nothing to do with locality -- a row set that moved, a tail chunk of one
    row, a weight quantisation its multi-row GEMV refuses at that width -- and when it does,
    aggregate throughput falls by anything from a third to a factor of five while every other
    number in the run looks normal. A median over repeats then turns one collapsed run into a
    plausible-looking 'result' for whatever configuration happened to be running.

    The adapter already counts what settles it: `tokens_packed`, and `max_rows_seen`. Returns a
    record for the artifact, and whether the arm is usable.

    Two questions, and conflating them cost this repository four cells of a ten-cell matrix:

      did concurrent decode EVER happen        `max_rows_seen >= 2`. Unambiguous. Zero means
                                               not one batched step, at any decode length.
      did MOST of it happen                    `tokens_packed` against the decode steps the
                                               arm was asked for, which is `max_new`. NOT
                                               against the adapter's `tokens`, which counts
                                               prefill chunks and at ctx4096 is dominated by
                                               them.

    Limitation, stated rather than hidden: the control arm is unhooked by construction, so it
    emits no telemetry and a collapse THERE is still invisible. `frontier/runner.py` answers
    that from the control's own concurrency scaling, which needs no telemetry at all.
    """
    if not stats or concurrency < 2:
        return None
    st = stats.get("stats", {})
    tokens = st.get("tokens", 0) or 0
    packed = st.get("tokens_packed", 0) or 0
    rows = st.get("max_rows_seen", 0) or 0
    expected = int(max_new) if max_new else 0
    # Share of the DECODE steps this arm asked for. One decode step advances every live row, so
    # `max_new` is how many there should be.
    decode_share = (packed / expected) if expected else None
    batched_at_all = rows >= MIN_ROWS_FOR_CONCURRENCY
    usable = batched_at_all and (decode_share is None or decode_share >= PACKED_SHARE_MIN)
    return {"tokens": tokens, "tokens_packed": packed,
            # Kept for continuity with every result file written before this correction, and
            # named for what it is: contaminated by prefill at long context.
            "packed_share_of_all_steps": (packed / tokens) if tokens else 0.0,
            "packed_share": decode_share,
            "decode_steps_expected": expected,
            "max_rows_seen": rows,
            "batched_at_all": batched_at_all,
            "layers_packed": st.get("layers_packed", 0),
            "concurrency_asked": concurrency,
            "used_packed_path": usable}


def require_packed_path(stats, concurrency, label, max_new=None):
    """A concurrency measurement that ran the per-row path is not a concurrency measurement."""
    rec = packed_path_used(stats, concurrency, max_new)
    if rec is None or rec["used_packed_path"]:
        return rec
    if not rec["batched_at_all"]:
        why = (f"max_rows_seen={rec['max_rows_seen']}: not one decode step batched more than "
               f"a single row")
    else:
        why = (f"the runtime packed {rec['tokens_packed']} of the "
               f"{rec['decode_steps_expected']} decode steps this arm asked for "
               f"({(rec['packed_share'] or 0) * 100:.1f}%), max_rows_seen="
               f"{rec['max_rows_seen']}")
    raise SystemExit(
        f"{label}: RUNTIME FELL OFF THE BATCHED DECODE PATH. Asked for {concurrency} "
        f"concurrent sequences; {why}. Aggregate throughput from a run that decoded one row at "
        "a time is not this workload's number, and averaging it with runs that did batch "
        "produces a gain for a measurement that never happened. Re-run the arm.")


def parse_output_ids(text):
    m = re.search(r"OUTPUT_IDS:((?: -?\d+)+)", text)
    return [int(x) for x in m.group(1).split()] if m else None


def median(xs):
    return statistics.median(xs) if xs else 0.0


def geomean(xs):
    if not xs:
        return 1.0
    prod = 1.0
    for x in xs:
        prod *= x
    return prod ** (1.0 / len(xs))


def batch1_arm(per_context, ctxs):
    """Combine the per-context results into the single batch-1 workload decide.py scores.

    One estimator, everywhere. This used to be a ratio *of medians*
    (median(candidate)/median(baseline)) while every other arm — and this module's own
    documented method — used the median *of paired ratios*. On tight data the two agree; on
    the drifting clocks this harness is designed for they do not, and the arm that disagreed
    carried the highest weight in the verdict. Controls of [100, 110, 120] against candidates
    of [105, 110, 132] are +5.0% paired and +0.0% as a ratio of medians: same measurements,
    opposite go/no-go.

    Pairing is the whole point of interleaving. Discarding it throws away the only thing that
    makes a same-box comparison valid.

    Lives out here, not inline in main(), so it can be tested without a GPU and a 21 GB
    checkpoint — which is why the inconsistency survived as long as it did.
    """
    ratios, refs = [], []
    for c in ctxs:
        arm = per_context[str(c)]
        if arm["paired_ratios"]:
            ratios.append(median(arm["paired_ratios"]))
        if arm["baseline_tps"] > 0:
            refs.append(arm["baseline_tps"])
    ratio = geomean(ratios) if ratios else 1.0
    baseline_ref = geomean(refs) if refs else 0.0
    return ratio, baseline_ref


def resolution(gain_pct, *spreads):
    """Is this workload's difference bigger than the run-to-run spread that produced it?

    A median hides an outlier instead of reporting one. At 32 concurrent sequences a no-op
    configuration measured -14.8% here, from two repeats that differed by 32% -- an effect a
    policy doing nothing cannot have. The number was not wrong, it was unresolved, and the
    only thing that distinguishes those is publishing the spread beside the gain."""
    floor = max([s for s in spreads if s is not None], default=None)
    if floor is None:
        return {"resolved": None, "floor_pct": None,
                "note": "fewer than two repeats; no spread to compare against"}
    return {"resolved": abs(gain_pct) > floor, "floor_pct": floor,
            "note": "resolved=false means the difference is inside the run-to-run spread of "
                    "the runs that produced it, whichever way it points"}


def rel_spread_pct(xs):
    """Peak-to-peak spread as a fraction of the median: the noise floor of this measurement
    on this box, measured rather than assumed."""
    if len(xs) < 2:
        return None
    mid = median(xs)
    return (max(xs) - min(xs)) / mid * 100.0 if mid else None


def measure(binary, model, tokens, ctxs, env_extra, label, verbose):
    env = dict(env_extra)
    env["SPARKINFER_BENCH_SWEEP_CTXS"] = ",".join(str(c) for c in ctxs)
    code, out, secs = run([binary, model, str(tokens), "sweep"], env,
                          scrub_adapter_env=is_control(env_extra))
    if code != 0:
        raise SystemExit(f"{label}: benchmark exited {code}\n{out[-4000:]}")
    sweep = parse_sweep(out)
    if not sweep:
        raise SystemExit(f"{label}: no SWEEP_JSON in output\n{out[-4000:]}")
    if verbose:
        print(f"    {label}: " + "  ".join(f"ctx{c}={sweep[str(c)]['decode_tps']:.2f}" for c in ctxs)
              + f"   ({secs:.0f}s)", flush=True)
    stats = require_hook_engaged(out, env_extra, label)
    require_registered_families(stats, env_extra, label)
    return sweep, stats, out


# Below this share of the tokens a concurrency arm was asked for, the run did not measure the
# workload -- it measured whatever survived. 0.90 rather than 1.00 because the harness cannot
# know the long-prefill request's exact contribution, and a legitimate run has been seen to
# come in a few tokens under.
MIN_COMPLETED_TOKEN_SHARE = 0.90


def require_requests_completed(out, concurrency, max_new, label):
    """A concurrency arm that lost requests did not run slower; it ran less.

    Measured on the dense checkpoint at 32 sequences: two of six identical isolated runs
    printed `[qwen35] malloc: out of memory` and `[warn] request error: device out of memory`
    for most of their requests, completed 320 and 384 decode tokens instead of 2056, and
    reported 142.9 and 164.8 tok/s against a healthy 925. Per-token latency was UNCHANGED
    through it -- 20.77 and 24.26 ms against 19.29 -- so nothing about the decode path was
    slow. The aggregate rate is tokens over wall time, so an arm that dropped four fifths of
    its requests reports a collapse that is really a failure, and averaging it into a
    control/candidate ratio corrupts the ratio.

    This is the cause of the intermittent 32-sequence collapse this repository has carried as
    unexplained for three releases, and it was invisible to every guard here: the hook ran,
    the packed path was used, `agg_tok_s` parsed fine. The token count is what says so.
    """
    tokens = re.search(r"decode_tokens=(\d+)", out)
    if not tokens:
        return None                      # older bench build; nothing to check against
    got = int(tokens.group(1))
    want = concurrency * max_new
    oom = out.count("out of memory")
    if got >= want * MIN_COMPLETED_TOKEN_SHARE:
        return {"decode_tokens": got, "expected_tokens": want, "out_of_memory_warnings": oom}
    raise SystemExit(
        f"{label}: REQUESTS DID NOT COMPLETE. The bench decoded {got} tokens where this "
        f"workload asks for about {want} ({got / want * 100:.0f}%), and the runtime printed "
        f"{oom} out-of-memory message(s). An arm that lost requests did not run slower, it "
        f"ran less: aggregate tok/s is tokens over wall time, so the number is a failure "
        f"reported as a slowdown. Re-run the arm in isolation, and give the device time to "
        f"release the previous arm's memory before it starts.")


def measure_concurrent(binary, model, concurrency, prompt_len, max_new, long_prefill,
                       env_extra, label, verbose):
    """Aggregate decode throughput under continuous batching.

    A different code path from the single-sequence bench, and the one the locality question
    is actually about: model weights are read once per step however many sequences are in
    flight, so the recurrent share of decode traffic grows with concurrency while the weight
    share does not."""
    code, out, secs = run([binary, model, str(concurrency), str(prompt_len), str(max_new),
                           str(long_prefill)], env_extra,
                          scrub_adapter_env=is_control(env_extra))
    if code != 0:
        raise SystemExit(f"{label}: cb bench exited {code}\n{out[-4000:]}")
    m = re.search(r"agg_tok_s=([0-9.]+)", out)
    itl = re.search(r"mean_itl_ms=([0-9.]+)", out)
    if not m:
        raise SystemExit(f"{label}: no agg_tok_s in output\n{out[-4000:]}")
    tps = float(m.group(1))
    require_requests_completed(out, concurrency, max_new, label)
    stats = require_hook_engaged(out, env_extra, label)
    require_registered_families(stats, env_extra, label)
    packing = require_packed_path(stats, concurrency, label, max_new)
    if verbose:
        pk = (f"  packed={(packing['packed_share'] or 0) * 100:.0f}% of decode "
              f"rows={packing['max_rows_seen']}") if packing else ""
        print(f"    {label}: agg={tps:.1f} tok/s  itl={itl.group(1) if itl else '?'} ms{pk}   ({secs:.0f}s)",
              flush=True)
    return tps, stats, out


def greedy_replay(generate, model, prompt_ids, max_new, env_extra, label):
    code, out, _ = run([generate, model, str(max_new), *[str(t) for t in prompt_ids]], env_extra,
                       scrub_adapter_env=is_control(env_extra))
    if code != 0:
        raise SystemExit(f"{label}: generate exited {code}\n{out[-4000:]}")
    ids = parse_output_ids(out)
    if ids is None:
        raise SystemExit(f"{label}: no OUTPUT_IDS in output\n{out[-4000:]}")
    return ids, require_hook_engaged(out, env_extra, label)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--binary", required=True, help="qwen3_gguf_bench from the patched build")
    ap.add_argument("--generate", help="qwen3_gguf_generate, for the token-exact gate")
    ap.add_argument("--model", required=True, help="checkpoint directory or GGUF")
    ap.add_argument("--output", type=Path, default=Path("real-result.json"))
    ap.add_argument("--tokens", type=int, default=128, help="decode tokens timed per context")
    ap.add_argument("--contexts", default="128,4096,16384",
                    help="prefill depths to time decode at; the batch-1 arm is their geometric mean")
    ap.add_argument("--repeats", type=int, default=3, help="interleaved control/candidate pairs")
    ap.add_argument("--gate-tokens", type=int, default=64, help="tokens compared by the exactness gate")
    ap.add_argument("--gate-prompt", default="9707,3837,1879,13,25001,752,911",
                    help="prompt token ids for the greedy replay gate")
    ap.add_argument("--gate-control-replays", type=int, default=3, metavar="N",
                    help="unhooked control replays that must ALL agree before the runtime is "
                         "called reproducible. One agreeing pair is one sample, not evidence: "
                         "a runtime that forks half its replays certifies as reproducible half "
                         "the time. Minimum 2.")
    ap.add_argument("--candidate", nargs="+", required=True, metavar="KEY=VALUE",
                    help="environment that turns the hook on, e.g. RECURLOCAL=combined")
    ap.add_argument("--control", nargs="*", default=[], metavar="KEY=VALUE",
                    help="environment for the control arm; empty means the unhooked runtime")
    ap.add_argument("--warmup-runs", type=int, default=0, metavar="N",
                    help="discard this many control measurements before the first counted "
                         "pair. The first process of a run can be measurably slower than the "
                         "ones after it, which lands entirely in the first pair and inflates "
                         "the noise floor the arm is judged against. Defaults to 0 so every "
                         "result published before this flag existed reproduces exactly.")
    ap.add_argument("--cb-binary", help="qwen3_gguf_cb_bench, for the concurrency arms")
    ap.add_argument("--concurrency", default="",
                    help="comma-separated concurrencies to measure, e.g. 4,16,32; needs --cb-binary")
    ap.add_argument("--cb-prompt-len", type=int, default=128)
    ap.add_argument("--cb-max-new", type=int, default=64)
    ap.add_argument("--cb-long-prefill", type=int, default=4096)
    ap.add_argument("--label", default="", help="short name for this candidate, recorded in the result")
    ap.add_argument("--note", default="", help="free text recorded with the result")
    ap.add_argument("--quiet", action="store_true")
    a = ap.parse_args()

    def as_env(pairs):
        env = {}
        for item in pairs:
            if "=" not in item:
                raise SystemExit(f"--candidate/--control entries must be KEY=VALUE, got {item!r}")
            k, v = item.split("=", 1)
            env[k] = v
        return env

    cand_env = as_env(a.candidate)
    ctrl_env = as_env(a.control)
    # The control must not carry the hook. Otherwise the comparison is between two
    # RecurLocal configurations and the result file would still call one of them "baseline".
    if not is_control(ctrl_env):
        raise SystemExit("the control arm must have the hook off; it is the unhooked runtime")
    cand_env.setdefault("RECURLOCAL_STATS", "1")
    ctxs = [int(c) for c in a.contexts.split(",") if c.strip()]
    verbose = not a.quiet

    provenance = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "gpu": None, "driver": None,
        "model": a.model,
        "binary": str(Path(a.binary).resolve()),
        "candidate_env": cand_env,
        "control_env": ctrl_env,
        "tokens_per_measurement": a.tokens,
        "contexts": ctxs,
        "repeats": a.repeats,
        "warmup_runs_discarded": a.warmup_runs,
        "label": a.label,
        "note": a.note,
    }
    pin = Path(__file__).resolve().parent.parent / "adapters" / "sparkinfer" / "pin.json"
    if pin.exists():
        doc = json.loads(pin.read_text())
        provenance["runtime_repository"] = doc.get("repository")
        provenance["runtime_commit"] = doc.get("commit")
        provenance["model_pin"] = doc.get("model", {}).get("repository")
    try:
        provenance["recurlocal_commit"] = subprocess.run(
            ["git", "-C", str(Path(__file__).resolve().parent.parent), "rev-parse", "HEAD"],
            capture_output=True, text=True).stdout.strip() or None
    except Exception:
        pass
    try:
        smi = subprocess.run(["nvidia-smi", "--query-gpu=name,driver_version",
                              "--format=csv,noheader"], capture_output=True, text=True).stdout.strip()
        if smi:
            provenance["gpu"], provenance["driver"] = [x.strip() for x in smi.split(",")[:2]]
    except Exception:
        pass

    lock = GpuLock(verbose=verbose)
    lock.__enter__()
    # ---- correctness gate first: a candidate that changes the output is not scored, and
    # there is no reason to spend an hour timing it.
    correctness = {"output_identical": None, "method": "greedy replay, token-exact"}
    if a.generate:
        prompt = [int(x) for x in a.gate_prompt.replace(" ", "").split(",") if x]
        if verbose:
            print(">> token-exact greedy replay gate", flush=True)
        ctrl_ids, _ = greedy_replay(a.generate, a.model, prompt, a.gate_tokens, ctrl_env, "control")
        # The control against ITSELF, before the candidate is allowed to be blamed for anything.
        # A gate that compares one control run to one candidate run cannot tell "the policy
        # changed the output" from "this runtime is not reproducible on this model" -- and it
        # reported the second as the first: on a sparse-MoE checkpoint two unhooked controls
        # diverge at token 2, because a few ULP in the prefill flip a discrete top-k expert
        # choice, and the harness called that a candidate that changed model output. The
        # candidate had changed nothing; on the dense checkpoint the same binary is
        # bit-identical across control, control and candidate.
        #
        # And it takes MORE THAN ONE re-run to answer. Nondeterminism seeded by a few ULP does
        # not fork every replay -- it forks the ones where some argmax along the way happens to
        # be close. Measured on the sparse-MoE checkpoint at a 256-token prompt, five unhooked
        # single-token replays returned 8894, 8894, 25001, 25001, 8894: any single pair drawn
        # from that has a better-than-even chance of agreeing and certifying a runtime that is
        # not reproducible at all. One agreeing pair is not evidence of reproducibility; it is
        # one sample. So the control is replayed --gate-control-replays times and EVERY replay
        # must agree, which turns a coin flip into (1/2)^(n-1) and is the difference between a
        # gate that answers the question and one that usually answers it.
        ctrl_replays = [ctrl_ids]
        for r in range(1, max(2, a.gate_control_replays)):
            ids, _ = greedy_replay(a.generate, a.model, prompt, a.gate_tokens, ctrl_env,
                                   f"control (reproducibility {r + 1})")
            ctrl_replays.append(ids)
            # Stop at the first disagreement: the question is already answered, and the
            # remaining replays would only cost model loads to re-answer it.
            if ids != ctrl_ids:
                break
        runtime_reproducible = all(ids == ctrl_ids for ids in ctrl_replays[1:])
        cand_ids, gate_stats = greedy_replay(a.generate, a.model, prompt, a.gate_tokens, cand_env, "candidate")
        identical = ctrl_ids == cand_ids
        correctness.update(tokens_compared=len(ctrl_ids),
                           prompt_ids=prompt,
                           runtime_reproducible=runtime_reproducible,
                           control_replays=len(ctrl_replays),
                           control_replays_requested=max(2, a.gate_control_replays),
                           candidate_matches_control=identical,
                           first_divergence=None if identical else next(
                               (i for i, (x, y) in enumerate(zip(ctrl_ids, cand_ids)) if x != y),
                               min(len(ctrl_ids), len(cand_ids))))
        # `output_identical` is what decide.py gates on, so it must mean "the candidate did not
        # change the output" and nothing else. Where the control is not reproducible against
        # itself the question is unanswerable, so it is left None -- inconclusive, which
        # decide.py refuses to score -- rather than False, which accuses the candidate.
        correctness["output_identical"] = identical if runtime_reproducible else None
        if not runtime_reproducible:
            correctness["method"] = ("greedy replay, token-exact -- INCONCLUSIVE: two control "
                                     "runs of this runtime on this model disagree with each "
                                     "other, so a candidate/control difference cannot be "
                                     "attributed to the candidate. Not a candidate defect and "
                                     "not scorable; the exact-locality gate needs a runtime "
                                     "and checkpoint that are reproducible.")
            forked = next(ids for ids in ctrl_replays[1:] if ids != ctrl_ids)
            correctness["control_first_divergence"] = next(
                (i for i, (x, y) in enumerate(zip(ctrl_ids, forked)) if x != y),
                min(len(ctrl_ids), len(forked)))
            # Which replay it took to find out. A 2 here means the runtime forked immediately;
            # a 5 means four replays agreed before one did not, which is exactly the case a
            # single-pair check would have certified as reproducible.
            correctness["control_first_divergent_replay"] = len(ctrl_replays)
        if gate_stats:
            correctness["candidate_hook_active"] = hook_ran(gate_stats)
        if verbose:
            if not runtime_reproducible:
                print(f"    CONTROL IS NOT REPRODUCIBLE: unhooked replay "
                      f"{correctness['control_first_divergent_replay']} of "
                      f"{correctness['control_replays_requested']} diverges from replay 1 at "
                      f"token {correctness['control_first_divergence']}. The gate cannot "
                      "attribute a difference to the candidate; correctness is inconclusive, "
                      "not failed.", flush=True)
            else:
                print(f"    control reproducible over {len(ctrl_replays)} unhooked replays",
                      flush=True)
            print(f"    candidate vs control: {'identical' if identical else 'DIVERGED'} over "
                  f"{len(ctrl_ids)} tokens", flush=True)
    else:
        correctness["method"] = "not run (--generate not given)"

    # ---- timing, interleaved
    concurrencies = [int(c) for c in a.concurrency.split(",") if c.strip()]
    if concurrencies and not a.cb_binary:
        raise SystemExit("--concurrency needs --cb-binary (qwen3_gguf_cb_bench)")

    ctrl_runs, cand_runs = [], []
    cb_ctrl = {n: [] for n in concurrencies}
    cb_cand = {n: [] for n in concurrencies}
    adapter_stats = None
    cb_adapter_stats = {}
    # Discard the opening measurements before any pair is counted. The first process of a run
    # can be measurably slower than the ones after it, and because control always runs first
    # that slowness lands entirely in pair 1 -- inflating the noise floor the arm is judged
    # against, and with it the chance the arm cannot resolve at all.
    for w in range(a.warmup_runs):
        if verbose:
            print(f">> warm-up {w + 1}/{a.warmup_runs} (discarded)", flush=True)
        measure(a.binary, a.model, a.tokens, ctxs, ctrl_env, "warm-up", verbose)
        for n in concurrencies:
            measure_concurrent(a.cb_binary, a.model, n, a.cb_prompt_len, a.cb_max_new,
                               a.cb_long_prefill, ctrl_env, f"warm-up c={n}", verbose)

    for i in range(a.repeats):
        if verbose:
            print(f">> pair {i + 1}/{a.repeats}", flush=True)
        c, _, _ = measure(a.binary, a.model, a.tokens, ctxs, ctrl_env, "control", verbose)
        d, stats, _ = measure(a.binary, a.model, a.tokens, ctxs, cand_env, "candidate", verbose)
        ctrl_runs.append(c)
        cand_runs.append(d)
        if stats:
            adapter_stats = stats
        for n in concurrencies:
            # The adapter counts sequences from the runtime's own row count, so the candidate
            # environment does not have to be told the concurrency it is running at.
            b_tps, _, _ = measure_concurrent(a.cb_binary, a.model, n, a.cb_prompt_len,
                                             a.cb_max_new, a.cb_long_prefill, ctrl_env,
                                             f"control  c={n}", verbose)
            c_tps, cstats, _ = measure_concurrent(a.cb_binary, a.model, n, a.cb_prompt_len,
                                                  a.cb_max_new, a.cb_long_prefill, cand_env,
                                                  f"candidate c={n}", verbose)
            cb_ctrl[n].append(b_tps)
            cb_cand[n].append(c_tps)
            if cstats:
                cb_adapter_stats[str(n)] = cstats

    per_context = {}
    for ctx in ctxs:
        key = str(ctx)
        base = [r[key]["decode_tps"] for r in ctrl_runs]
        cand = [r[key]["decode_tps"] for r in cand_runs]
        # Ratio per pair, then the median of the ratios: a pair is the only comparison the
        # box supports, because both halves of it saw the same clock state.
        ratios = [d / b for b, d in zip(base, cand) if b > 0]
        gain = (median(ratios) - 1.0) * 100.0 if ratios else 0.0
        per_context[key] = {
            "baseline_tps": median(base), "candidate_tps": median(cand),
            "baseline_runs": base, "candidate_runs": cand,
            "paired_ratios": ratios,
            "gain_pct": gain,
            "baseline_spread_pct": rel_spread_pct(base),
            "candidate_spread_pct": rel_spread_pct(cand),
            "resolution": resolution(gain, rel_spread_pct(base), rel_spread_pct(cand),
                                     rel_spread_pct(ratios)),
        }

    # The batch-1 workload decide.py scores is the whole context sweep, combined as a
    # geometric mean so no single context can carry the arm — using the same paired-ratio
    # estimator as every other arm. See batch1_arm().
    batch1_ratio, baseline_ref = batch1_arm(per_context, ctxs)

    # The measurement's own noise floor: the worst per-context control spread. A gain under
    # it is not resolved by this experiment, whatever its sign.
    floors = [per_context[str(c)]["baseline_spread_pct"] for c in ctxs
              if per_context[str(c)]["baseline_spread_pct"] is not None]
    noise_floor = max(floors) if floors else None
    gain_pct = (batch1_ratio - 1.0) * 100.0

    workloads = {"batch1": {"weight": DEFAULT_WEIGHTS["batch1"] if concurrencies else 1.0,
                           "baseline_tps": baseline_ref,
                           "candidate_tps": baseline_ref * batch1_ratio}}
    concurrency_detail = {}
    for n in concurrencies:
        base, cand = cb_ctrl[n], cb_cand[n]
        ratios = [d / b for b, d in zip(base, cand) if b > 0]
        name = f"concurrency{n}"
        gain = (median(ratios) - 1.0) * 100.0 if ratios else 0.0
        stats_n = cb_adapter_stats.get(str(n), {})
        concurrency_detail[name] = {
            "baseline_runs": base, "candidate_runs": cand, "paired_ratios": ratios,
            "gain_pct": gain,
            "baseline_spread_pct": rel_spread_pct(base),
            "candidate_spread_pct": rel_spread_pct(cand),
            "resolution": resolution(gain, rel_spread_pct(base), rel_spread_pct(cand),
                                     rel_spread_pct(ratios)),
            # Proof the concurrency arm measured the concurrent code path. A runtime that
            # declined to batch would decode row by row through the single-sequence path and
            # produce a perfectly good number for the wrong workload.
            "sequences_seen_by_adapter": stats_n.get("geometry", {}).get("sequences"),
            "tokens_packed": stats_n.get("stats", {}).get("tokens_packed"),
            "tokens_total": stats_n.get("stats", {}).get("tokens"),
            "max_rows_seen": stats_n.get("stats", {}).get("max_rows_seen"),
            # The share the guard judged, kept beside the raw counts: an arm that passed at 88%
            # packed and one that passed at 51% are not equally trustworthy, and a reader
            # cannot tell which from the gain.
            "packing": packed_path_used(stats_n, n),
        }
        if name in DEFAULT_WEIGHTS:
            workloads[name] = {"weight": DEFAULT_WEIGHTS[name],
                               "baseline_tps": median(base),
                               "candidate_tps": median(base) * median(ratios) if ratios else median(cand)}
    measured = ["batch1"] + [f"concurrency{n}" for n in concurrencies if f"concurrency{n}" in DEFAULT_WEIGHTS]

    doc = {
        "schema_version": SCHEMA_VERSION,
        "what_this_is": "Real end-to-end decode on a pinned SparkInfer commit and checkpoint. "
                        "This is the metric sections 17, 21 and 28 make authoritative.",
        "provenance": provenance,
        "correctness": correctness,
        "workloads": workloads,
        "workload_coverage": {
            "measured": measured,
            "unmeasured": [k for k in DEFAULT_WEIGHTS if k not in measured],
            "why": "Section 44 weights batch 1 at 0.40 and concurrency at 0.60. An arm that was "
                   "not run is left out of the matrix rather than filled in, and the weights "
                   "that remain are renormalised by decide.py - so a partial run is reported as "
                   "a partial verdict instead of a full one with invented halves.",
        },
        "per_context": per_context,
        "per_concurrency": concurrency_detail,
        "measurement": {
            "noise_floor_pct": noise_floor,
            "gain_pct": gain_pct,
            "resolved": None if noise_floor is None else abs(gain_pct) > noise_floor,
            "note": "resolved=false means the difference is inside the control arm's own "
                    "run-to-run spread on this box; the axis is open, not settled.",
            # Named explicitly so a reader does not have to check every arm by hand. An
            # unresolved arm still carries its weight in the verdict below -- it is a
            # measurement, not a discard -- but it is not evidence of anything.
            "unresolved_workloads": sorted(
                [f"batch1/ctx{c}" for c in ctxs
                 if per_context[str(c)]["resolution"]["resolved"] is False]
                + [k for k, v in concurrency_detail.items()
                   if v["resolution"]["resolved"] is False]),
        },
        "adapter": adapter_stats,
    }
    lock.__exit__()
    a.output.write_text(json.dumps(doc, indent=2) + "\n")
    if verbose:
        print(f"\nbatch-1 decode: {gain_pct:+.2f}%  (noise floor "
              f"{'unknown' if noise_floor is None else f'{noise_floor:.2f}%'})")
        print(f"wrote {a.output}")
        print(f"score it: eval/decide.py --real {a.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
