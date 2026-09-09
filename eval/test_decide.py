#!/usr/bin/env python3
"""Tests for the go/no-go decision.

The bands decide whether the project continues, so a silent drift in a boundary changes
what every past and future verdict meant. These pin them exactly as the overview states
them.
"""
import json, math, subprocess, sys, tempfile, unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import real_eval
import traffic_budget
import decide as label  # noqa: E402

DECIDE_PY = Path(__file__).resolve().parent / "decide.py"


def real_doc(workloads, identical=True):
    return {"correctness": {"output_identical": identical, "method": "greedy replay"},
            "workloads": workloads}


def flat(ratio, weights=None):
    """Every workload improved by the same ratio, so the geometric mean is that ratio."""
    weights = weights or label.DEFAULT_WEIGHTS
    return {name: {"weight": w, "baseline_tps": 100.0, "candidate_tps": 100.0 * ratio}
            for name, w in weights.items()}


class ImpactBands(unittest.TestCase):
    def test_tier_boundaries(self):
        # (gain %, expected tier) straight from overview section 26.
        for gain, tier in [(0.0, "none"), (1.99, "none"), (2.0, "XS"), (3.99, "XS"),
                           (4.0, "S"), (6.99, "S"), (7.0, "M"), (9.99, "M"),
                           (10.0, "L"), (17.99, "L"), (18.0, "XL"), (40.0, "XL")]:
            with self.subTest(gain=gain):
                self.assertEqual(label.band(gain, label.IMPACT)[1], tier)

    def test_go_no_go_boundaries(self):
        # Overview section 21.
        for gain, decision in [(-5.0, "reject"), (1.99, "reject"), (2.0, "weak"), (3.99, "weak"),
                               (4.0, "promising"), (6.99, "promising"), (7.0, "strong"),
                               (9.99, "strong"), (10.0, "expand"), (25.0, "expand")]:
            with self.subTest(gain=gain):
                self.assertEqual(label.band(gain, label.GO_NO_GO)[1], decision)

    def test_section_37_minimum_is_promising(self):
        # "at minimum >= 4% real end-to-end would justify continued serious development"
        self.assertEqual(label.band(4.0, label.GO_NO_GO)[1], "promising")
        self.assertEqual(label.band(4.0, label.IMPACT)[1], "S")


class RealScoring(unittest.TestCase):
    def test_uniform_gain_maps_to_that_gain(self):
        out = label.score_real(real_doc(flat(1.10)))
        self.assertTrue(out["scored"])
        self.assertAlmostEqual(out["weighted_gain_pct"], 10.0, places=9)
        self.assertEqual(out["impact"], "L")
        self.assertEqual(out["verdict"], "expand")

    def test_weighted_geometric_mean(self):
        ratios = {"batch1": 1.10, "concurrency4": 1.05, "concurrency16": 1.02, "concurrency32": 0.99}
        doc = real_doc({n: {"weight": label.DEFAULT_WEIGHTS[n], "baseline_tps": 100.0,
                            "candidate_tps": 100.0 * r} for n, r in ratios.items()})
        out = label.score_real(doc)
        expected = math.exp(sum(label.DEFAULT_WEIGHTS[n] * math.log(r) for n, r in ratios.items()))
        self.assertAlmostEqual(out["weighted_ratio"], expected, places=12)
        self.assertEqual(out["impact"], "S")
        self.assertEqual(out["verdict"], "promising")

    def test_partial_matrix_renormalises_rather_than_inventing_arms(self):
        # real_eval.py leaves out a workload it did not run, so the weights that remain do
        # not sum to 1. They must be renormalised, not treated as if the missing arms scored
        # zero gain -- which would silently dilute every result by 60% of section 44's
        # weight and make a partial run look like a worse full one.
        doc = real_doc({"batch1": {"weight": 0.40, "baseline_tps": 100.0, "candidate_tps": 110.0},
                        "concurrency4": {"weight": 0.20, "baseline_tps": 100.0, "candidate_tps": 110.0}})
        out = label.score_real(doc)
        self.assertAlmostEqual(out["weighted_gain_pct"], 10.0, places=9)
        # And the single-arm case, which is what a batch-1-only run produces.
        only = real_doc({"batch1": {"weight": 1.0, "baseline_tps": 100.0, "candidate_tps": 104.0}})
        self.assertAlmostEqual(label.score_real(only)["weighted_gain_pct"], 4.0, places=9)

    def test_geometric_mean_does_not_let_one_win_mask_a_loss(self):
        # Arithmetic mean of 2.0 and 0.5 is 1.25; geometric is 1.0.
        doc = real_doc({"a": {"weight": 0.5, "baseline_tps": 100.0, "candidate_tps": 200.0},
                        "b": {"weight": 0.5, "baseline_tps": 100.0, "candidate_tps": 50.0}})
        out = label.score_real(doc, allow_regression=True)
        self.assertAlmostEqual(out["weighted_ratio"], 1.0, places=12)
        self.assertEqual(out["impact"], "none")

    def test_regression_guard_blocks(self):
        doc = real_doc({"batch1": {"weight": 0.5, "baseline_tps": 100.0, "candidate_tps": 130.0},
                        "concurrency32": {"weight": 0.5, "baseline_tps": 100.0, "candidate_tps": 97.0}})
        out = label.score_real(doc)
        self.assertFalse(out["scored"])
        self.assertEqual(out["verdict"], "REGRESSION")
        self.assertEqual(out["regressed_workloads"], ["concurrency32"])
        self.assertIsNone(out["impact"])

    def test_exactly_two_percent_regression_is_allowed(self):
        doc = real_doc({"batch1": {"weight": 0.5, "baseline_tps": 100.0, "candidate_tps": 130.0},
                        "concurrency32": {"weight": 0.5, "baseline_tps": 100.0, "candidate_tps": 98.0}})
        self.assertTrue(label.score_real(doc)["scored"])

    def test_regression_waiver_scores_but_records(self):
        doc = real_doc({"batch1": {"weight": 0.5, "baseline_tps": 100.0, "candidate_tps": 130.0},
                        "concurrency32": {"weight": 0.5, "baseline_tps": 100.0, "candidate_tps": 90.0}})
        out = label.score_real(doc, allow_regression=True)
        self.assertTrue(out["scored"])
        self.assertTrue(out["regression_waived"])
        self.assertEqual(out["regressed_workloads"], ["concurrency32"])

    def test_changed_output_is_rejected_however_fast(self):
        out = label.score_real(real_doc(flat(2.0), identical=False))
        self.assertFalse(out["scored"])
        self.assertEqual(out["verdict"], "REJECT")
        self.assertIsNone(out["impact"])

    def test_missing_correctness_is_rejected(self):
        out = label.score_real({"workloads": flat(1.10)})
        self.assertEqual(out["verdict"], "REJECT")

    def test_sub_two_percent_is_not_significant(self):
        out = label.score_real(real_doc(flat(1.015)))
        self.assertTrue(out["scored"])
        self.assertFalse(out["significant"])
        self.assertEqual(out["impact"], "none")

    def test_default_weights_apply_by_name(self):
        doc = real_doc({n: {"baseline_tps": 100.0, "candidate_tps": 110.0} for n in label.DEFAULT_WEIGHTS})
        out = label.score_real(doc)
        self.assertAlmostEqual(out["weighted_gain_pct"], 10.0, places=9)

    def test_bad_input_is_rejected_loudly(self):
        for bad in ({"correctness": {"output_identical": True}, "workloads": {}},
                    real_doc({"x": {"weight": 1.0, "baseline_tps": 0.0, "candidate_tps": 1.0}}),
                    real_doc({"x": {"weight": 1.0, "baseline_tps": 1.0, "candidate_tps": -1.0}}),
                    real_doc({"unknown_name": {"baseline_tps": 1.0, "candidate_tps": 2.0}}),
                    real_doc({"x": {"weight": 0.0, "baseline_tps": 1.0, "candidate_tps": 2.0}})):
            with self.subTest(bad=bad), self.assertRaises(SystemExit):
                label.score_real(bad)


class SyntheticRefusal(unittest.TestCase):
    BASE = {"correctness": "pass", "best_mode": "combined", "best_synthetic_gain_pct": 35.0,
            "stability": {"verdict": "stable", "max_rel_spread_pct": 0.4, "threshold_pct": 2.0}}

    def test_large_synthetic_gain_is_never_tiered(self):
        out = label.score_synthetic(dict(self.BASE))
        self.assertFalse(out["scored"])
        self.assertIsNone(out["impact"])
        self.assertEqual(out["verdict"], "SYNTHETIC-ONLY")
        self.assertEqual(out["best_synthetic_gain_pct"], 35.0)

    def test_unstable_run_is_not_a_result(self):
        doc = dict(self.BASE, stability={"verdict": "unstable", "max_rel_spread_pct": 19.0,
                                         "threshold_pct": 2.0})
        self.assertEqual(label.score_synthetic(doc)["verdict"], "UNSTABLE")

    def test_failed_correctness_is_rejected(self):
        self.assertEqual(label.score_synthetic(dict(self.BASE, correctness="fail"))["verdict"], "REJECT")


class CommandLine(unittest.TestCase):
    def run_label(self, args, doc):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "in.json"
            p.write_text(json.dumps(doc))
            r = subprocess.run([sys.executable, str(DECIDE_PY), *args, str(p)],
                               capture_output=True, text=True)
            return r.returncode, json.loads(r.stdout)

    def test_scored_improvement_exits_zero(self):
        code, out = self.run_label(["--real"], real_doc(flat(1.10)))
        self.assertEqual(code, 0)
        self.assertEqual(out["impact"], "L")

    def test_synthetic_exits_nonzero_so_it_cannot_be_mistaken_for_a_pass(self):
        code, out = self.run_label(["--synthetic"], SyntheticRefusal.BASE)
        self.assertEqual(code, 1)
        self.assertIsNone(out["impact"])

    def test_regression_exits_nonzero(self):
        code, _ = self.run_label(["--real"], real_doc(
            {"batch1": {"weight": 1.0, "baseline_tps": 100.0, "candidate_tps": 90.0}}))
        self.assertEqual(code, 1)



class RealEvalEstimator(unittest.TestCase):
    """The batch-1 arm carries weight 0.40, so the estimator behind it decides verdicts.

    These tests exist because that arm used a different estimator from every other arm and
    from the module's own documented method, and nothing could catch it: the computation was
    inline in main(), which needs a GPU and a 21 GB checkpoint to reach.
    """

    @staticmethod
    def _arm(base, cand):
        return {"baseline_tps": real_eval.median(base), "candidate_tps": real_eval.median(cand),
                "paired_ratios": [c / b for b, c in zip(base, cand)]}

    def test_pairing_is_not_discarded(self):
        # Drifting clocks: the pairing is the only thing that makes this comparison valid.
        per = {"128": self._arm([100.0, 110.0, 120.0], [105.0, 110.0, 132.0])}
        ratio, _ = real_eval.batch1_arm(per, [128])
        self.assertAlmostEqual(ratio, 1.05, places=9)          # median of [1.05, 1.00, 1.10]
        # The ratio-of-medians estimator this replaced would have returned 110/110 = 1.0 and
        # turned a 'promising' verdict into 'reject' on the same measurements.
        self.assertNotAlmostEqual(ratio, 1.0, places=3)

    def test_matches_the_per_context_number_it_is_built_from(self):
        per = {"128": self._arm([100.0, 100.0, 100.0], [101.0, 102.0, 103.0])}
        ratio, base = real_eval.batch1_arm(per, [128])
        self.assertAlmostEqual((ratio - 1.0) * 100.0, 2.0, places=9)   # the median pair, +2%
        self.assertAlmostEqual(base, 100.0, places=9)

    def test_contexts_combine_geometrically_so_none_can_carry_the_arm(self):
        per = {"128": self._arm([100.0], [200.0]), "4096": self._arm([100.0], [50.0])}
        ratio, _ = real_eval.batch1_arm(per, [128, 4096])
        self.assertAlmostEqual(ratio, 1.0, places=9)   # 2x and 0.5x cancel, as they must

    def test_degenerate_inputs_do_not_fabricate_a_gain(self):
        self.assertEqual(real_eval.batch1_arm({}, []), (1.0, 0.0))
        per = {"128": {"baseline_tps": 0.0, "candidate_tps": 0.0, "paired_ratios": []}}
        self.assertEqual(real_eval.batch1_arm(per, [128]), (1.0, 0.0))


class HarnessIntegrity(unittest.TestCase):
    """The harness is meant to arbitrate. These cover the ways it could quietly not."""

    def test_a_run_whose_arms_did_not_resolve_is_not_significant(self):
        # A complete matrix, so that resolution is the only thing under test here.
        doc = real_doc(flat(1.08))
        self.assertTrue(label.score_real(doc)["significant"])          # resolved: a real result
        doc["measurement"] = {"unresolved_workloads": ["batch1/ctx128"]}
        out = label.score_real(doc)
        self.assertFalse(out["significant"])                           # same number, no evidence
        self.assertEqual(out["unresolved_workloads"], ["batch1/ctx128"])
        self.assertIn("did not resolve", out["reason"])

    def test_a_candidate_that_applied_no_policy_is_a_null_candidate(self):
        # Deferred windows are computed and handed back; unless the runtime attaches them they
        # never reach a kernel. Scoring that reports hook overhead as a locality result.
        null = {"stats": {"windows_applied": 0, "windows_attached_to_node": 0,
                          "pre_touch_launches": 0, "windows_deferred_to_caller": 192}}
        self.assertFalse(real_eval.policy_applied(null))
        for live in ({"windows_applied": 1}, {"windows_attached_to_node": 1},
                     {"pre_touch_launches": 1}):
            self.assertTrue(real_eval.policy_applied({"stats": live}))

    def test_control_is_identified_by_asking_for_no_mode(self):
        self.assertTrue(real_eval.is_control({}))
        self.assertTrue(real_eval.is_control({"RECURLOCAL": "off"}))
        self.assertTrue(real_eval.is_control({"RECURLOCAL": "0"}))
        self.assertFalse(real_eval.is_control({"RECURLOCAL": "persist"}))
        self.assertFalse(real_eval.is_control({"RECURLOCAL": "baseline"}))

    def test_control_arm_scrubs_an_ambient_adapter_env(self):
        # An operator with RECURLOCAL exported in their shell would otherwise run a hooked
        # "control" and the harness would report ~0% for the candidate against itself.
        import os
        os.environ["RECURLOCAL"] = "combined"
        os.environ["RECURLOCAL_PREFETCH_DISTANCE"] = "6"
        try:
            code, out, _ = real_eval.run([sys.executable, "-c",
                                          "import os;print(os.environ.get('RECURLOCAL','<unset>'),"
                                          "os.environ.get('RECURLOCAL_PREFETCH_DISTANCE','<unset>'))"],
                                         {}, scrub_adapter_env=True)
            self.assertIn("<unset> <unset>", out)
        finally:
            os.environ.pop("RECURLOCAL", None)
            os.environ.pop("RECURLOCAL_PREFETCH_DISTANCE", None)



class ShippedArtifactIsScorable(unittest.TestCase):
    """The repository publishes results/rtx5090-real.json and the README tells you to score
    it. That command used to fail: the matrix is nested under "scored_result" and score_real
    only looked at the top level."""

    def test_the_committed_result_scores(self):
        path = Path(__file__).resolve().parent.parent / "results" / "rtx5090-real.json"
        if not path.exists():
            self.skipTest("no committed result in this checkout")
        out = label.score_real(json.loads(path.read_text()))
        self.assertIn(out["verdict"], {"reject", "weak", "promising", "strong", "expand",
                                       "REGRESSION", "REJECT"})
        self.assertIn("workloads", out)

    def test_both_shapes_score_identically(self):
        bare = real_doc({"batch1": {"weight": 1.0, "baseline_tps": 100.0, "candidate_tps": 105.0}})
        wrapped = {"what_this_is": "bundle", "batch1_axes": {}, "scored_result": bare}
        self.assertEqual(label.score_real(bare)["weighted_gain_pct"],
                         label.score_real(wrapped)["weighted_gain_pct"])

    def test_a_bare_doc_missing_workloads_still_errors(self):
        with self.assertRaises(SystemExit):
            label.score_real({"correctness": {"output_identical": True}})


class WorkloadCoverage(unittest.TestCase):
    """An omitted workload is not a neutral omission.

    Missing arms are renormalised away, so leaving one out removes it from the mean
    instead of averaging it in. docs/MINING.md points the competition at concurrency 32,
    which makes that the arm a submission profits most from not running. The verdict has
    to say what was not measured and refuse to call the result significant.
    """

    def _partial(self, ratio=1.10):
        return real_doc({name: {"weight": w, "baseline_tps": 100.0,
                                "candidate_tps": 100.0 * ratio}
                         for name, w in label.DEFAULT_WEIGHTS.items()
                         if name != "concurrency32"})

    def test_a_missing_arm_is_named_with_the_weight_it_took_with_it(self):
        out = label.score_real(self._partial())
        self.assertTrue(out["partial"])
        self.assertEqual(out["workload_coverage"]["missing"], ["concurrency32"])
        self.assertAlmostEqual(out["workload_coverage"]["missing_weight_share"], 0.20)

    def test_an_incomplete_matrix_cannot_be_significant_however_large_the_gain(self):
        out = label.score_real(self._partial(ratio=1.10))
        self.assertAlmostEqual(out["weighted_gain_pct"], 10.0, places=9)
        self.assertEqual(out["impact"], "L")          # still described...
        self.assertFalse(out["significant"])          # ...but not banked
        self.assertIn("concurrency32", out["reason"])

    def test_allow_partial_is_the_only_way_through_and_it_is_recorded(self):
        out = label.score_real(self._partial(), allow_partial=True)
        self.assertTrue(out["significant"])
        self.assertTrue(out["partial"])
        self.assertTrue(out["partial_waived"])

    def test_a_complete_matrix_is_not_flagged(self):
        out = label.score_real(real_doc(flat(1.10)))
        self.assertFalse(out["partial"])
        self.assertNotIn("workload_coverage", out)
        self.assertTrue(out["significant"])

    def test_coverage_is_derived_not_taken_from_the_document(self):
        # A submission that declares full coverage while omitting the arm is still partial:
        # the scorer counts the workloads it actually scored.
        doc = self._partial()
        doc["workload_coverage"] = {"measured": sorted(label.DEFAULT_WEIGHTS), "missing": []}
        out = label.score_real(doc)
        self.assertEqual(out["workload_coverage"]["missing"], ["concurrency32"])
        self.assertFalse(out["significant"])

    def test_a_custom_matrix_is_not_forced_onto_section_44_names(self):
        # A result that uses none of the section 44 names is a different matrix, not an
        # incomplete one, and must not be flagged as missing every arm.
        doc = real_doc({"my_workload": {"weight": 1.0, "baseline_tps": 100.0,
                                        "candidate_tps": 110.0}})
        out = label.score_real(doc)
        self.assertFalse(out["partial"])
        self.assertTrue(out["significant"])

    def test_the_cli_refuses_to_exit_zero_on_a_partial_matrix(self):
        with tempfile.TemporaryDirectory() as d:
            f = Path(d) / "real.json"
            f.write_text(json.dumps(self._partial()))
            blocked = subprocess.run([sys.executable, str(DECIDE_PY), "--real", str(f)],
                                     capture_output=True, text=True)
            self.assertNotEqual(blocked.returncode, 0)
            waived = subprocess.run([sys.executable, str(DECIDE_PY), "--real", str(f),
                                     "--allow-partial"], capture_output=True, text=True)
            self.assertEqual(waived.returncode, 0)


class MatrixCeiling(unittest.TestCase):
    """What is the most this repository can ever pay?

    Each arm has a physical ceiling, and the verdict is a weighted mean over all of them, so
    the number that decides whether the project is worth competing on is what a submission
    scores if it hits every ceiling at once. That is only meaningful if it is computed with
    the same weights and bands the scorer uses -- hence the agreement test below.
    """

    PIN = json.loads((Path(__file__).resolve().parent.parent /
                      "integrations" / "sparkinfer" / "pin.json").read_text())["model"]

    def _spec(self, names=("batch1", "concurrency4", "concurrency16", "concurrency32")):
        base = {"batch1": {"sequences": 1, "ms_per_token": 10.41, "state_bytes_scale": 1.0},
                "concurrency4": {"sequences": 4, "aggregate_tps": 329.43, "state_bytes_scale": 0.5},
                "concurrency16": {"sequences": 16, "aggregate_tps": 770.1, "state_bytes_scale": 0.5},
                "concurrency32": {"sequences": 32, "aggregate_tps": 928.0, "state_bytes_scale": 0.5}}
        return {"arms": {k: v for k, v in base.items() if k in names}}

    def _run(self, spec):
        import contextlib, io
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(io.StringIO()):
            traffic_budget.matrix_ceiling(self.PIN, spec, 1792.0)
        return json.loads(buf.getvalue())

    def test_the_ceiling_is_weighted_the_way_the_scorer_weights(self):
        # Feed each arm's ceiling to decide.py as a candidate that achieved exactly it. The
        # scorer's weighted gain must equal the ceiling tool's. If the two ever disagree,
        # the repository is advertising a ceiling in one currency and paying in another.
        out = self._run(self._spec())
        doc = real_doc({name: {"baseline_tps": 100.0,
                               "candidate_tps": 100.0 * (1 + arm["ceiling_pct"] / 100.0)}
                        for name, arm in out["arms"].items() if arm.get("measured")})
        self.assertAlmostEqual(label.score_real(doc, allow_partial=True)["weighted_gain_pct"],
                               out["best_possible_weighted_gain_pct"], places=9)

    def test_an_arm_with_no_measured_rate_is_excluded_and_named(self):
        out = self._run(self._spec(("batch1", "concurrency4", "concurrency16")))
        self.assertEqual(out["arms_without_a_measured_rate"], ["concurrency32"])
        self.assertAlmostEqual(out["weights_covered"], 0.80)
        self.assertFalse(out["arms"]["concurrency32"]["measured"])

    def test_aggregate_tps_is_converted_per_step_not_per_token(self):
        # One step advances every sequence, so 4 sequences at 329.43 aggregate tok/s is a
        # 12.14 ms step, not 3.04 ms. Getting this backwards inflates the ceiling 4x.
        out = self._run(self._spec(("concurrency4",)))
        self.assertAlmostEqual(out["arms"]["concurrency4"]["measured_ms_per_token"],
                               4 / 329.43 * 1000.0, places=6)

    def test_bands_are_the_scorers_bands(self):
        out = self._run(self._spec())
        best = out["best_possible_weighted_gain_pct"]
        self.assertEqual(out["best_possible_impact"], label.band(best, label.IMPACT)[1])
        self.assertEqual(out["best_possible_verdict"], label.band(best, label.GO_NO_GO)[1])

    def test_the_ceiling_is_in_throughput_terms_not_traffic_share(self):
        # A step carrying f less traffic runs in (1-f) of the time, so tok/s rise by
        # f/(1-f). The scorer measures candidate_tps/baseline_tps, so the ceiling has to be
        # quoted in that currency; quoting the raw share understates it, and a submission
        # that beat a published "ceiling" would discredit every other number here.
        out = self._run(self._spec())
        for name, arm in out["arms"].items():
            if not arm.get("measured"):
                continue
            with self.subTest(arm=name):
                f = arm["recurrent_share_of_traffic_pct"] / 100.0
                self.assertAlmostEqual(arm["ceiling_pct"], 100.0 * f / (1 - f), places=9)
                self.assertGreater(arm["ceiling_pct"], arm["recurrent_share_of_traffic_pct"])

    def test_removing_that_traffic_really_does_return_the_ceiling(self):
        # End to end, in the units that decide the verdict: take the measured step time,
        # subtract exactly the recurrent-state bytes, and the resulting tok/s must be the
        # ceiling the tool published.
        out = self._run(self._spec(("concurrency16",)))
        arm = out["arms"]["concurrency16"]
        total = arm["implied_total_bytes_per_token"]
        freed = arm["state_traffic_bytes_per_token"]
        gain = (total / (total - freed) - 1.0) * 100.0
        self.assertAlmostEqual(arm["ceiling_pct"], gain, places=9)

    def test_a_matrix_with_nothing_measured_is_an_error_not_a_zero_ceiling(self):
        with self.assertRaises(SystemExit):
            self._run({"arms": {}})

    def test_an_arm_without_a_rate_at_all_is_an_error(self):
        with self.assertRaises(SystemExit):
            self._run({"arms": {"batch1": {"sequences": 1}}})


if __name__ == "__main__":
    unittest.main(verbosity=2)
