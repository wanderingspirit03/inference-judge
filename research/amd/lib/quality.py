"""The fixed quality bar of ./submit and ./check.  Pure and stdlib only.

An engine may use smaller weights (4-bit and the like), so its tokens need
not be the reference's.  It is judged against a fixed bar on two measures:

  A  from the walk of the judged answers (compare.measure_a): the share of
     positions whose engine token is in the reference's top k, for small k,
     and the mean gap per position in nats (the reference's logprob of its
     own choice minus its logprob of the engine's token).  It needs no
     logprobs from the engine.
  B  task accuracy: prompts with one checkable answer (a number, or a letter
     A-E), graded by rule; the engine and the 16-bit reference answer the
     same drawn tasks, and the bar is on the paired difference (engine minus
     reference), by its one-sided 95% bootstrap lower bound.

The limits live in config quality.bar; null means not set.  While any is not
set, the measures are recorded and the verdict is "quality bar not set": the
result is the old one (the agreement limit).  Once all are set, the bar
decides, and with quality.bar_replaces_agreement the agreement limit is
reported, not applied.  measurements/2026-11-17/NOTES.md says how the limits
are to be measured on the card.

Grading follows bench/quality2.py (extract_answer, is_correct): the last
number of the reply for math, the first letter pattern for mc.
"""
from __future__ import annotations

import hashlib
import random
import re

NOT_SET = "quality bar not set"

_NUM_RE = re.compile(r"-?\d[\d,]*(?:\.\d+)?")
_MC_PATTERNS = (re.compile(r"answer\s*(?:is)?\s*[:\-]?\s*\(?([A-E])\b", re.I),
                re.compile(r"^\s*\(?([A-E])\)?\s*[.:)]?\s*$"),
                re.compile(r"(?<![A-Za-z])([A-E])(?![A-Za-z])"))


def extract(kind: str, text: str):
    """The comparable answer in a reply: a float for math, a letter for mc."""
    text = text or ""
    if kind == "math":
        nums = _NUM_RE.findall(text)
        return float(nums[-1].replace(",", "")) if nums else None
    if kind == "mc":
        for pat in _MC_PATTERNS:
            m = pat.search(text)
            if m:
                return m.group(1).upper()
        return None
    raise ValueError(f"unknown task kind {kind!r}")


def correct(task: dict, text: str) -> bool:
    ans = extract(task["kind"], text)
    if ans is None:
        return False
    gold = task["answer"]
    if task["kind"] == "math":
        return abs(float(ans) - float(gold)) <= 1e-6 * max(1.0, abs(float(gold)))
    return str(ans).upper() == str(gold).upper()


def paired_lower_bound(diffs: list[int], seed: int, resamples: int = 2000) -> float:
    """One-sided 95% percentile-bootstrap lower bound of mean(diffs)."""
    if not diffs:
        return 0.0
    r = random.Random(seed)
    n = len(diffs)
    means = sorted(sum(diffs[r.randrange(n)] for _ in range(n)) / n for _ in range(resamples))
    return means[int(0.05 * (resamples - 1))]


def measure_b(tasks: list[dict], engine_texts: list[str | None], ref_texts: list[str], resamples: int = 2000) -> dict:
    """Task accuracy, paired.  engine_texts[i] is None when the engine's
    request failed (graded wrong).  The bootstrap's seed comes from the task
    ids, so the same tasks give the same bound."""
    rows, diffs = [], []
    for t, e, r in zip(tasks, engine_texts, ref_texts):
        ce, cr = (e is not None and correct(t, e)), correct(t, r)
        diffs.append(int(ce) - int(cr))
        rows.append({"id": t["id"], "kind": t["kind"], "engine_ok": ce, "reference_ok": cr,
                     "engine_answer": None if e is None else extract(t["kind"], e),
                     "reference_answer": extract(t["kind"], r), "engine_failed": e is None})
    n = len(rows)
    seed = int.from_bytes(hashlib.sha256("|".join(t["id"] for t in tasks).encode()).digest()[:8], "little")
    ce = sum(r["engine_ok"] for r in rows)
    cr = sum(r["reference_ok"] for r in rows)
    return {"tasks": n, "engine_correct": ce, "reference_correct": cr,
            "engine_acc": round(ce / n, 4) if n else None, "reference_acc": round(cr / n, 4) if n else None,
            "delta": round((ce - cr) / n, 4) if n else None,
            "delta_lb95": round(paired_lower_bound(diffs, seed, resamples), 4) if n else None,
            "engine_only": sum(1 for d in diffs if d > 0), "reference_only": sum(1 for d in diffs if d < 0),
            "engine_failed": sum(r["engine_failed"] for r in rows), "per_task": rows,
            "method": "paired: delta = engine accuracy - reference accuracy on the same tasks; "
                      "delta_lb95 = one-sided 95% percentile bootstrap lower bound"}


def bar(cfg: dict) -> dict:
    return (cfg.get("quality") or {}).get("bar") or {}


def topk_limits(cfg: dict) -> dict:
    return {k: v for k, v in (bar(cfg).get("min_topk_share") or {}).items() if not str(k).startswith("_")}


def missing(cfg: dict) -> list[str]:
    """The limits of the bar that are not set ([] when it is set)."""
    b = bar(cfg)
    out = []
    if not any(v is not None for v in topk_limits(cfg).values()):
        out.append("min_topk_share")
    for k in ("max_mean_gap", "min_task_delta_lb"):
        if b.get(k) is None:
            out.append(k)
    return out


def is_set(cfg: dict) -> bool:
    return not missing(cfg)


def replaces_agreement(cfg: dict) -> bool:
    return is_set(cfg) and bool((cfg.get("quality") or {}).get("bar_replaces_agreement", True))


def verdict(cfg: dict, qa: dict | None, qb: dict | None) -> dict:
    """{"bar": "set"|"not set", "verdict": "pass"|"fail"|NOT_SET,
    "reasons": [...], "a": qa, "b": qb}."""
    out = {"a": qa, "b": qb, "reasons": []}
    if not is_set(cfg):
        out.update({"bar": "not set", "verdict": NOT_SET, "not_set": missing(cfg)})
        return out
    b = bar(cfg)
    why = out["reasons"]
    if qa is None:
        why.append("measure A was not taken (no answer was walked)")
    else:
        if qa.get("answers_cut") or qa.get("answers_failed"):
            why.append(f"{qa.get('answers_cut', 0)} answer(s) cut at the parting limit and "
                       f"{qa.get('answers_failed', 0)} failed: measure A needs every answer walked to its end")
        for k, v in sorted(topk_limits(cfg).items()):
            if v is None:
                continue
            got = (qa.get("topk_share") or {}).get(str(k))
            if got is None or got < v:
                why.append(f"top-{k} share {got} < limit {v}")
        g = qa.get("mean_gap")
        if g is None or g > b["max_mean_gap"]:
            why.append(f"mean gap {g} nats > limit {b['max_mean_gap']}")
    if not qb or not qb.get("tasks"):
        why.append("measure B was not taken (no tasks were answered)")
    elif qb["delta_lb95"] < b["min_task_delta_lb"]:
        why.append(f"task accuracy: engine {qb['engine_correct']}/{qb['tasks']}, reference "
                   f"{qb['reference_correct']}/{qb['tasks']}; lower bound of the difference {qb['delta_lb95']} "
                   f"< limit {b['min_task_delta_lb']}")
    out.update({"bar": "set", "verdict": "fail" if why else "pass"})
    return out


def lines(q: dict | None) -> list[str]:
    """What ./submit and ./check print about quality."""
    if not q:
        return []
    out = []
    a, b = q.get("a"), q.get("b")
    if a:
        tk = ", ".join(f"top-{k} {v}" for k, v in (a.get("topk_share") or {}).items())
        out.append(f"quality A: strict agreement {a['agreement_strict']}; share in the reference's {tk}; "
                   f"mean gap {a['mean_gap']} nats per position ({a['walked']} of {a['positions']} positions walked, "
                   f"{a['outside_top']} outside the reference's top)")
    if b and b.get("tasks"):
        out.append(f"quality B: tasks right: engine {b['engine_correct']}/{b['tasks']}, reference "
                   f"{b['reference_correct']}/{b['tasks']} on the same tasks; difference {b['delta']:+.3f} "
                   f"(95% lower bound {b['delta_lb95']:+.3f})"
                   + (f"; {b['engine_failed']} engine request(s) failed" if b.get("engine_failed") else ""))
    elif b and b.get("note"):
        out.append(f"quality B: {b['note']}")
    if q.get("verdict") == NOT_SET:
        out.append(f"quality: {NOT_SET} (no pass or fail on quality yet; limits not set: {', '.join(q.get('not_set') or [])})")
    else:
        out.append(f"quality: {q['verdict'].upper()} against the fixed bar"
                   + ("".join(f"\n  {r}" for r in q.get("reasons") or [])))
    return out
