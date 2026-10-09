"""Statistics per STANDARD.md section 7.

The task is the unit: repeats are averaged within a task first, then everything is
computed across tasks. Intervals are 95%.
- rate metrics (is_correct, pass@k, pass^k) with per-task values all 0 or 1: Wilson score interval
- otherwise: Student t interval on the task means (n - 1 df), clipped to the metric range
- A vs B: paired per-task differences, t interval, wins/ties/losses, exact sign test
"""

from __future__ import annotations

import math
from collections import defaultdict
from math import comb

Z95 = 1.959963984540054
# two-sided 95% t critical values, df 1..30
T95 = [
    12.706,
    4.303,
    3.182,
    2.776,
    2.571,
    2.447,
    2.365,
    2.306,
    2.262,
    2.228,
    2.201,
    2.179,
    2.160,
    2.145,
    2.131,
    2.120,
    2.110,
    2.101,
    2.093,
    2.086,
    2.080,
    2.074,
    2.069,
    2.064,
    2.060,
    2.056,
    2.052,
    2.048,
    2.045,
    2.042,
]
ROUND = 4
EPS = 1e-9


def t_crit(df: int) -> float:
    if df < 1:
        raise ValueError("df must be >= 1")
    if df <= 30:
        return T95[df - 1]
    if df <= 40:
        return 2.021
    if df <= 60:
        return 2.000
    if df <= 120:
        return 1.980
    return Z95


def r(x: float | None) -> float | None:
    return None if x is None else round(x, ROUND)


def mean(xs: list[float]) -> float:
    return sum(xs) / len(xs)


def stdev(xs: list[float]) -> float:
    m = mean(xs)
    return math.sqrt(sum((x - m) ** 2 for x in xs) / (len(xs) - 1))


def wilson(successes: float, n: int, z: float = Z95) -> tuple[float, float]:
    if n <= 0:
        raise ValueError("n must be > 0")
    p = successes / n
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return max(0.0, center - half), min(1.0, center + half)


def t_interval(xs: list[float]) -> tuple[float, float, float, float]:
    """mean, se, lower, upper. Needs len(xs) >= 2."""
    m, sd = mean(xs), stdev(xs)
    se = sd / math.sqrt(len(xs))
    h = t_crit(len(xs) - 1) * se
    return m, se, m - h, m + h


def is_binary(xs: list[float]) -> bool:
    return all(abs(x) < EPS or abs(x - 1) < EPS for x in xs)


def bounds(metric: dict) -> tuple[float | None, float | None]:
    """Range a metric's per-task values can take: [0, 1] for rates, else min_score / max_score."""
    if (metric.get("metric_parameters") or {}).get("from") in ("is_correct", "pass_at_k", "pass_hat_k"):
        return 0.0, 1.0
    return metric.get("min_score"), metric.get("max_score")


def is_rate(metric: dict) -> bool:
    return (metric.get("metric_parameters") or {}).get("from") in ("is_correct", "pass_at_k", "pass_hat_k")


def summarize(task_values: list[float], metric: dict | None = None) -> dict:
    """score_details for a list of per-task values.

    Rate metrics whose per-task values are all 0 or 1 get a Wilson interval; everything else a t interval,
    clipped to the metric's range when it has one.
    """
    metric = metric or {}
    n = len(task_values)
    if n == 0:
        return {"score": None, "uncertainty": {"num_samples": 0}}
    m = mean(task_values)
    unc: dict = {"num_samples": n}
    if n >= 2:
        sd = stdev(task_values)
        unc["standard_deviation"] = r(sd)
        unc["standard_error"] = {"value": r(sd / math.sqrt(n)), "method": "sd of task means / sqrt(n)"}
        if is_rate(metric) and is_binary(task_values):
            lo, hi = wilson(sum(task_values), n)
            method = "wilson"
        else:
            _, _, lo, hi = t_interval(task_values)
            method = "t"
            lo_b, hi_b = bounds(metric)
            if (lo_b is not None and lo < lo_b) or (hi_b is not None and hi > hi_b):
                lo = lo if lo_b is None else max(lo, lo_b)
                hi = hi if hi_b is None else min(hi, hi_b)
                method = "t, clipped to metric range"
        unc["confidence_interval"] = {"lower": r(lo), "upper": r(hi), "confidence_level": 0.95, "method": method}
    return {"score": r(m), "uncertainty": unc}


def sign_test_p(wins: int, losses: int) -> float | None:
    n = wins + losses
    if n == 0:
        return None
    k = min(wins, losses)
    p = sum(comb(n, i) for i in range(k + 1)) / 2**n
    return min(1.0, 2 * p)


def paired(base: dict[str, float], cand: dict[str, float], lower_is_better: bool = False) -> dict:
    """Paired comparison over tasks present in both. diff = candidate - baseline."""
    common = sorted(set(base) & set(cand))
    diffs = [cand[t] - base[t] for t in common]
    better = [(-d if lower_is_better else d) for d in diffs]
    wins = sum(1 for d in better if d > EPS)
    losses = sum(1 for d in better if d < -EPS)
    ties = len(diffs) - wins - losses
    out: dict = {
        "diff": None,
        "uncertainty": {"num_samples": len(diffs)},
        "wins": wins,
        "ties": ties,
        "losses": losses,
        "sign_test_p": r(sign_test_p(wins, losses)),
        "method": "paired by task, t on per-task differences",
    }
    if diffs:
        out["diff"] = r(mean(diffs))
    if len(diffs) >= 2:
        m, se, lo, hi = t_interval(diffs)
        out["uncertainty"].update(
            {
                "standard_error": {"value": r(se), "method": "sd of paired diffs / sqrt(n)"},
                "confidence_interval": {"lower": r(lo), "upper": r(hi), "confidence_level": 0.95, "method": "t"},
            }
        )
    return out


# ------------------------------------------------------------- samples -> per-task values


def _score(row: dict, counted_as: str) -> float | None:
    v = (row.get("evaluation") or {}).get("score")
    if v is None and row.get("error") and counted_as == "zero":
        return 0.0
    return v


def _correct(row: dict, counted_as: str) -> bool | None:
    v = (row.get("evaluation") or {}).get("is_correct")
    if v is None and row.get("error") and counted_as == "zero":
        return False
    return v


def usable(rows: list[dict], counted_as: str) -> list[dict]:
    return [x for x in rows if not (x.get("error") and counted_as == "excluded")]


def per_task(rows: list[dict], subject: str, metric: dict, counted_as: str) -> dict[str, float]:
    """Per-task value for one subject and one metric_config entry."""
    params = metric.get("metric_parameters") or {}
    src = params.get("from", "external")
    by_task: dict[str, list] = defaultdict(list)
    for x in usable(rows, counted_as):
        if x.get("subject") != subject:
            continue
        if src == "score":
            v = _score(x, counted_as)
        else:
            c = _correct(x, counted_as)
            v = None if c is None else (1.0 if c else 0.0)
        if v is not None:
            by_task[x["sample_id"]].append(v)
    out: dict[str, float] = {}
    for t, vs in by_task.items():
        if src in ("score", "is_correct"):
            out[t] = mean(vs)
        elif src in ("pass_at_k", "pass_hat_k"):
            k = int(params.get("k", 1))
            n, c = len(vs), int(round(sum(vs)))
            if n < k:
                continue
            out[t] = (1 - comb(n - c, k) / comb(n, k)) if src == "pass_at_k" else comb(c, k) / comb(n, k)
    return out


def per_repeat(rows: list[dict], subject: str, metric: dict, counted_as: str) -> list[float | None]:
    params = metric.get("metric_parameters") or {}
    src = params.get("from", "external")
    if src not in ("score", "is_correct"):
        return []
    reps = sorted({x["repeat"] for x in rows if x.get("subject") == subject})
    out = []
    for rep in reps:
        sub = [x for x in rows if x.get("subject") == subject and x["repeat"] == rep]
        vals = per_task(sub, subject, metric, counted_as)
        out.append(r(mean(list(vals.values()))) if vals else None)
    return out


def computable(metric: dict) -> bool:
    return (metric.get("metric_parameters") or {}).get("from", "external") != "external"


def compute(manifest: dict, rows: list[dict]) -> dict:
    """Return the fields stats owns: evaluation_results (computable metrics),
    comparisons (numbers), errors (n/of/by_source), token_usage."""
    counted_as = (manifest.get("errors") or {}).get("counted_as", "zero")
    mcfg = manifest.get("metric_config") or {}
    subjects = [s["name"] for s in manifest.get("subjects", [])]
    results: dict = {}
    task_vals: dict = {}
    for s in subjects:
        for mid, m in mcfg.items():
            if not computable(m):
                continue
            tv = per_task(rows, s, m, counted_as)
            task_vals[(s, mid)] = tv
            sd = summarize(list(tv.values()), m)
            pr = per_repeat(rows, s, m, counted_as)
            if len(pr) > 1:
                sd["per_repeat"] = pr
            results.setdefault(s, {})[mid] = sd
    comps = []
    for c in manifest.get("comparisons") or []:
        mid = c["metric_id"]
        m = mcfg.get(mid, {})
        new = dict(c)
        if computable(m):
            b = task_vals.get((c["baseline"], mid)) or {}
            a = task_vals.get((c["candidate"], mid)) or {}
            new.update(paired(b, a, bool(m.get("lower_is_better"))))
        comps.append(new)
    errs = [x for x in rows if x.get("error")]
    by_src: dict = defaultdict(int)
    for x in errs:
        by_src[x["error"].get("source", "infra")] += 1
    errors = {"n": len(errs), "of": len(rows), "by_source": dict(sorted(by_src.items()))}
    usage: dict = {}
    for x in rows:
        tu = x.get("token_usage")
        if not tu:
            continue
        u = usage.setdefault(x["subject"], {})
        for k, v in tu.items():
            if isinstance(v, int):
                u[k] = u.get(k, 0) + v
    return {"evaluation_results": results, "comparisons": comps, "errors": errors, "token_usage": usage}


def apply(manifest: dict, rows: list[dict]) -> dict:
    """Manifest with stats-owned fields refreshed. Human fields and external metrics are kept."""
    out = dict(manifest)
    got = compute(manifest, rows)
    er = {s: dict(v) for s, v in (manifest.get("evaluation_results") or {}).items()}
    for s, metrics in got["evaluation_results"].items():
        for mid, sd in metrics.items():
            prev = er.get(s, {}).get(mid) or {}
            if prev.get("note"):
                sd = {**sd, "note": prev["note"]}
            er.setdefault(s, {})[mid] = sd
    out["evaluation_results"] = er
    if manifest.get("comparisons") is not None:
        out["comparisons"] = got["comparisons"]
    errors = dict(manifest.get("errors") or {"counted_as": "zero"})
    errors.update(got["errors"])
    out["errors"] = errors
    if got["token_usage"]:
        tu = {}
        for s, u in got["token_usage"].items():
            u = dict(u)
            if "total_tokens" not in u:
                u["total_tokens"] = u.get("input_tokens", 0) + u.get("output_tokens", 0)
            tu[s] = u
        out["token_usage"] = tu
    return out
