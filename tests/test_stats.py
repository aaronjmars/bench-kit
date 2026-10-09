import math

import pytest
from conftest import manifest, samples

from bench_kit import stats


def test_t_interval_hand_computed():
    # [1, 2, 3, 4]: mean 2.5, sd 1.290994, se 0.645497, t(3) = 3.182, half 2.0540
    m, se, lo, hi = stats.t_interval([1, 2, 3, 4])
    assert m == 2.5
    assert se == pytest.approx(0.645497, abs=1e-6)
    assert lo == pytest.approx(0.44602, abs=1e-4)
    assert hi == pytest.approx(4.55398, abs=1e-4)


def test_wilson_hand_computed():
    # 8 of 10: textbook Wilson 95% = [0.4902, 0.9433]
    lo, hi = stats.wilson(8, 10)
    assert lo == pytest.approx(0.4902, abs=1e-4)
    assert hi == pytest.approx(0.9433, abs=1e-4)
    lo, hi = stats.wilson(0, 5)
    assert lo == 0.0 and hi == pytest.approx(0.4345, abs=1e-4)


def test_t_crit_table_edges():
    assert stats.t_crit(1) == 12.706
    assert stats.t_crit(30) == 2.042
    assert stats.t_crit(1000) == pytest.approx(1.96, abs=1e-3)


def test_paired_hand_computed():
    # diffs [1, 0, 2]: mean 1, sd 1, se 0.57735, t(2) = 4.303 -> [-1.4843, 3.4843]
    out = stats.paired({"t1": 1, "t2": 2, "t3": 3}, {"t1": 2, "t2": 2, "t3": 5})
    assert out["diff"] == 1
    ci = out["uncertainty"]["confidence_interval"]
    assert ci["lower"] == pytest.approx(-1.4843, abs=1e-4)
    assert ci["upper"] == pytest.approx(3.4843, abs=1e-4)
    assert (out["wins"], out["ties"], out["losses"]) == (2, 1, 0)
    assert out["sign_test_p"] == 0.5


def test_paired_lower_is_better_flips_wins():
    out = stats.paired({"a": 10, "b": 10}, {"a": 8, "b": 12}, lower_is_better=True)
    assert (out["wins"], out["losses"]) == (1, 1)


def test_sign_test_exact():
    assert stats.sign_test_p(6, 0) == pytest.approx(0.03125)
    assert stats.sign_test_p(5, 1) == pytest.approx(0.21875)
    assert stats.sign_test_p(0, 0) is None


def test_task_is_the_unit():
    rows = samples()
    m = manifest()
    a = stats.per_task(rows, "A", m["metric_config"]["score"], "zero")
    assert a == {"t1": 1.0, "t2": 2.0, "t3": 3.0}


def test_compute_fills_results_and_comparison():
    out = stats.apply(manifest(), samples())
    a = out["evaluation_results"]["A"]["score"]
    assert a["score"] == 2.0
    assert a["uncertainty"]["num_samples"] == 3
    assert a["uncertainty"]["confidence_interval"]["method"] == "t, clipped to metric range"  # [-0.48, 4.48] clipped at min_score 0
    # per_repeat: repeat 1 mean (1 + 1.5 + 3) / 3, repeat 2 (1 + 2.5 + 3) / 3
    assert a["per_repeat"] == [pytest.approx(1.8333, abs=1e-4), pytest.approx(2.1667, abs=1e-4)]
    c = out["comparisons"][0]
    assert c["diff"] == 1 and c["wins"] == 2
    assert out["errors"]["n"] == 0 and out["errors"]["of"] == 12
    assert out["token_usage"]["A"] == {"input_tokens": 60, "output_tokens": 30, "total_tokens": 90}


def test_binary_metric_uses_wilson_when_tasks_are_0_or_1():
    # B solved: t1 2,2 -> 1 ; t2 2,2 -> 1 ; t3 4,6 -> 1 : all 1 -> wilson(3, 3)
    out = stats.apply(manifest(), samples())
    ci = out["evaluation_results"]["B"]["solved"]["uncertainty"]["confidence_interval"]
    assert ci["method"] == "wilson"
    lo, hi = stats.wilson(3, 3)
    assert ci["lower"] == pytest.approx(lo, abs=1e-4) and ci["upper"] == 1.0


def test_pass_at_k_and_pass_hat_k():
    rows = [
        {"sample_id": "t", "subject": "A", "repeat": i, "evaluation": {"score": 0, "is_correct": c}, "error": None}
        for i, c in enumerate([True, False, False])
    ]
    at = {"metric_parameters": {"from": "pass_at_k", "k": 1}}
    assert stats.per_task(rows, "A", at, "zero")["t"] == pytest.approx(1 / 3)
    at3 = {"metric_parameters": {"from": "pass_at_k", "k": 3}}
    assert stats.per_task(rows, "A", at3, "zero")["t"] == 1.0
    hat3 = {"metric_parameters": {"from": "pass_hat_k", "k": 3}}
    assert stats.per_task(rows, "A", hat3, "zero")["t"] == 0.0
    hat1 = {"metric_parameters": {"from": "pass_hat_k", "k": 1}}
    assert stats.per_task(rows, "A", hat1, "zero")["t"] == pytest.approx(1 / 3)


def test_errors_counted_as_zero_or_excluded():
    rows = samples()
    rows[0] = {**rows[0], "evaluation": {"score": None, "is_correct": None}, "error": {"source": "infra", "message": "boom"}}
    cfg = manifest()["metric_config"]["score"]
    zero = stats.per_task(rows, "A", cfg, "zero")
    assert zero["t1"] == 0.5  # (0 + 1) / 2
    excl = stats.per_task(rows, "A", cfg, "excluded")
    assert excl["t1"] == 1.0
    out = stats.apply(manifest(), rows)
    assert out["errors"]["n"] == 1 and out["errors"]["by_source"] == {"infra": 1}


def test_single_task_has_no_interval():
    sd = stats.summarize([0.7])
    assert sd["score"] == 0.7 and "confidence_interval" not in sd["uncertainty"]
    assert math.isclose(stats.summarize([1, 1, 0, 1])["score"], 0.75)


def test_score_metric_with_0_1_values_uses_t_not_wilson():
    cfg = {"score_type": "continuous", "min_score": -2, "max_score": 1, "metric_parameters": {"from": "score"}}
    sd = stats.summarize([1, 1, 1, 0], cfg)
    assert sd["uncertainty"]["confidence_interval"]["method"].startswith("t")
    assert sd["uncertainty"]["confidence_interval"]["upper"] == 1  # clipped to max_score


def test_rate_with_repeats_is_clipped_to_0_1():
    cfg = {"metric_parameters": {"from": "is_correct"}}
    sd = stats.summarize([0, 0, 0, 1 / 3, 2 / 3], cfg)
    ci = sd["uncertainty"]["confidence_interval"]
    assert ci["lower"] == 0 and ci["method"] == "t, clipped to metric range"


def test_rate_all_zero_one_uses_wilson():
    sd = stats.summarize([1, 0, 1, 1], {"metric_parameters": {"from": "is_correct"}})
    lo, hi = stats.wilson(3, 4)
    assert sd["uncertainty"]["confidence_interval"] == {"lower": round(lo, 4), "upper": round(hi, 4), "confidence_level": 0.95, "method": "wilson"}
