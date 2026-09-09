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

if __name__ == "__main__":
    unittest.main(verbosity=2)
