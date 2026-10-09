"""Export one run to Every Eval Ever (EEE) 0.3.0: one aggregate file + one samples file per subject."""

from __future__ import annotations

import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path

from jsonschema import Draft7Validator

from .io import Run, load_schema

EEE_VERSION = "0.3.0"


def det_uuid(*parts: str) -> str:
    """Deterministic UUID shaped as version 4 (EEE file names require it)."""
    b = bytearray(hashlib.sha256("|".join(parts).encode()).digest()[:16])
    b[6] = (b[6] & 0x0F) | 0x40
    b[8] = (b[8] & 0x3F) | 0x80
    h = b.hex()
    return f"{h[:8]}-{h[8:12]}-{h[12:16]}-{h[16:20]}-{h[20:]}"


def slug(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "-", s).strip("-") or "unknown"


def epoch(ts: str | None) -> str:
    if not ts:
        return "0"
    t = datetime.fromisoformat(ts.replace("Z", "+00:00"))
    if t.tzinfo is None:
        t = t.replace(tzinfo=timezone.utc)
    return str(int(t.timestamp()))


def model_info(subject: dict) -> dict:
    mi = subject.get("model_info") or {}
    dev = mi.get("developer") or "unknown"
    mid = mi.get("id") or subject["name"]
    hf_id = mid if "/" in mid else f"{dev}/{mid}"
    out = {
        "name": mi.get("name") or mid,
        "id": hf_id,
        "developer": dev,
        "additional_details": {"deployment_type": "externally_managed", "model_availability": "unknown"},
    }
    if mi.get("inference_platform"):
        out["inference_platform"] = mi["inference_platform"]
    h = subject.get("harness")
    if h:
        out["additional_details"]["harness"] = h["name"] + (f"@{h['version']}" if h.get("version") else "")
    out["additional_details"]["subject"] = subject["name"]
    return out


def metric_config(mid: str, m: dict) -> dict:
    out = {"metric_id": mid, "metric_name": m["metric_name"], "lower_is_better": m["lower_is_better"], "score_type": m["score_type"]}
    for k in ("metric_unit", "evaluation_description"):
        if m.get(k):
            out[k] = m[k]
    if m["score_type"] == "continuous":
        out["min_score"] = m.get("min_score")
        out["max_score"] = m.get("max_score")
    if m.get("metric_parameters"):
        out["metric_parameters"] = m["metric_parameters"]
    return out


def score_details(sd: dict) -> dict:
    out = {"score": sd["score"]}
    unc = {k: v for k, v in (sd.get("uncertainty") or {}).items() if v is not None}
    if unc:
        out["uncertainty"] = unc
    return out


def export(run: Run, out_dir: Path, validate: bool = True, org: str | None = None) -> list[Path]:
    """org overrides the manifest's source_organization.name; EEE requires a value, so it falls back to "unknown"."""
    m = run.manifest
    src_org = m.get("source_organization") or {}
    org_name = org or src_org.get("name") or "unknown"
    bench = m["bench"]
    sd_src = m.get("source_data") or {}
    pm = m["primary_metric"]
    ev_name = lambda mid: f"{sd_src.get('dataset_name', bench)}:{mid}"  # noqa: E731
    written: list[Path] = []
    agg_v = Draft7Validator(load_schema("eee/eval.schema.json")) if validate else None
    inst_v = Draft7Validator(load_schema("eee/instance_level_eval.schema.json")) if validate else None
    retrieved = epoch(m.get("finished_at") or m.get("started_at"))
    for subj in m["subjects"]:
        mi = model_info(subj)
        uid = det_uuid(bench, m["id"], subj["name"])
        dev_dir, model_dir = slug(mi["developer"]), slug(mi["id"].split("/", 1)[-1])
        base = out_dir / "data" / slug(bench) / dev_dir / model_dir
        base.mkdir(parents=True, exist_ok=True)
        eval_id = f"{bench}/{m['id']}/{mi['id']}/{retrieved}"
        results = []
        for mid, cfg in (m.get("metric_config") or {}).items():
            sd = (m.get("evaluation_results") or {}).get(subj["name"], {}).get(mid)
            if not sd or sd.get("score") is None:
                continue
            results.append(
                {
                    "evaluation_result_id": f"{m['id']}:{subj['name']}:{mid}",
                    "evaluation_name": ev_name(mid),
                    "source_data": {
                        "dataset_name": sd_src.get("dataset_name", bench),
                        "source_type": "other",
                        "additional_details": {"version": str(sd_src.get("version", "")), "n_tasks": str(sd_src.get("n_completed", ""))},
                    },
                    "evaluation_timestamp": m.get("started_at"),
                    "metric_config": metric_config(mid, cfg),
                    "score_details": score_details(sd),
                }
            )
        rows = [x for x in (run.samples or []) if x.get("subject") == subj["name"]]
        agg = {
            "schema_version": EEE_VERSION,
            "evaluation_id": eval_id,
            "retrieved_timestamp": retrieved,
            "evaluation_timestamp": m.get("started_at"),
            "source_metadata": {
                "source_name": bench,
                "source_type": "evaluation_run",
                "source_organization_name": org_name,
                "evaluator_relationship": "third_party",
                "additional_details": {"run_id": m["id"], "verdict": str(m.get("verdict"))},
            },
            "model_info": mi,
            "eval_library": {
                "name": (m.get("eval_library") or {}).get("name", bench),
                "version": (m.get("eval_library") or {}).get("version") or str((m.get("revision") or {}).get("commit") or "unknown"),
            },
            "evaluation_results": results,
        }
        if src_org.get("url") and not org:
            agg["source_metadata"]["source_organization_url"] = src_org["url"]
        if rows:
            inst = []
            for x in rows:
                ev = x.get("evaluation") or {}
                tu = x.get("token_usage")
                inst_row = {
                    "schema_version": EEE_VERSION,
                    "evaluation_id": eval_id,
                    "model_id": mi["id"],
                    "evaluation_name": ev_name(pm),
                    "sample_id": f"{x['sample_id']}#{x['repeat']}",
                    "sample_hash": x.get("sample_hash"),
                    "interaction_type": m.get("interaction_type", "agentic"),
                    "input": {"raw": "", "reference": []},
                    "answer_attribution": [],
                    "output": None,
                    "messages": [],
                    "evaluation": {"score": ev.get("score") if ev.get("score") is not None else 0.0, "is_correct": bool(ev.get("is_correct"))},
                    "error": None if not x.get("error") else f"{x['error'].get('source')}: {x['error'].get('message')}",
                    "metadata": {"task_id": str(x["sample_id"]), "repeat": str(x["repeat"])},
                }
                if tu and all(k in tu for k in ("input_tokens", "output_tokens")):
                    inst_row["token_usage"] = {**tu, "total_tokens": tu.get("total_tokens", tu["input_tokens"] + tu["output_tokens"])}
                lat = (x.get("performance") or {}).get("latency_ms")
                if lat is not None:
                    inst_row["performance"] = {"latency_ms": lat}
                inst.append(inst_row)
            body = "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in inst)
            sp = base / f"{uid}_samples.jsonl"
            sp.write_text(body)
            written.append(sp)
            agg["detailed_evaluation_results"] = {
                "format": "jsonl",
                "file_path": f"data/{slug(bench)}/{dev_dir}/{model_dir}/{uid}_samples.jsonl",
                "hash_algorithm": "sha256",
                "checksum": hashlib.sha256(body.encode()).hexdigest(),
                "total_rows": len(inst),
            }
            if inst_v:
                for i, r in enumerate(inst, 1):
                    errs = list(inst_v.iter_errors(r))
                    if errs:
                        raise ValueError(f"EEE instance row {i} for {subj['name']} invalid: {errs[0].message}")
        if agg_v:
            errs = list(agg_v.iter_errors(agg))
            if errs:
                raise ValueError(f"EEE aggregate for {subj['name']} invalid: {errs[0].message}")
        ap = base / f"{uid}.json"
        ap.write_text(json.dumps(agg, indent=2, ensure_ascii=False) + "\n")
        written.append(ap)
    return written
