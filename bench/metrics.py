#!/usr/bin/env python3
"""Shared metric definitions.

Every number the benchmark reports is defined exactly once, here, and imported
by harness.py (which measures) and compare.py (which judges). If you want to
argue with a result, argue with this file.
"""
from __future__ import annotations

import json
import math
import zlib
from dataclasses import dataclass, field
from typing import Any, Iterable, Sequence

import numpy as np

METRICS_VERSION = "1.6"

# Request-level deadlines. Exceeding either marks the request as an error.
FIRST_CHUNK_TIMEOUT_S = 30.0
TOTAL_TIMEOUT_S = 60.0


@dataclass
class TurnRecord:
    """One streamed reply as the client saw it. Times are milliseconds from
    the moment the HTTP request was sent."""

    t_request_s: float = 0.0     # monotonic clock at request send
    ttft_ms: float | None = None
    first_tool_ms: float | None = None   # first *complete* tool call
    total_ms: float | None = None
    max_gap_ms: float = 0.0
    n_content_chunks: int = 0
    error: bool = False
    error_kind: str | None = None
    error_detail: str | None = None
    http_status: int | None = None
    usage_missing: bool = False  # usage was requested and the server omitted it
    tokens_in: int | None = None
    tokens_out: int | None = None
    cached_tokens: int | None = None
    served_by: str | None = None  # upstream provider (a router's 'provider' field)
    reply_chars: int = 0
    produced_tool_call: str | None = None
    tool_call_args: dict[str, Any] | None = None
    produced_tool_calls: list[dict[str, Any]] = field(default_factory=list)
    text: str = ""


def percentile(values: Sequence[float], q: float) -> float | None:
    """Linear-interpolated percentile. None for an empty sample."""
    vals = [v for v in values if v is not None and not math.isnan(v)]
    if not vals:
        return None
    return float(np.percentile(np.asarray(vals, dtype=float), q))


# --------------------------------------------------------------------------
# TPS metric (DESIGN 2.2). One tps_records.jsonl row per request; times are
# seconds on the harness's own monotonic clock, relative to the start of the
# invocation. Nothing here reads the server's account of itself except to
# cross-check it: the token count is the client's re-tokenisation.
# --------------------------------------------------------------------------
# |tokens_client - tokens_usage| / osl above this marks the request invalid.
TPS_USAGE_TOLERANCE = 0.01
# The SLO a level must meet for its goodput to count.
TPS_DEFAULT_SLO = {"p99_tpot_ms": 50.0, "p95_ttft_ms": 2000.0}


def tps_invalid_reason(record: dict, osl: int, length_tol: float = 0.0,
                       usage_tol: float = TPS_USAGE_TOLERANCE) -> str | None:
    """Why this request does not count, or None when it does.

    In order: the request errored; the harness had to reuse a prompt (a
    repeat is memoisable, so it is not a fresh measurement); the server did
    not stop on the length limit; the client's count is not osl; the server's
    usage disagrees with the client's count by more than usage_tol x osl.
    Missing usage is not a reason: usage is only ever a cross-check, and
    omitting it gains a server nothing because the length rule does not read
    it. `length_tol` (a fraction of osl) is 0 for the ws tokenizer; a real
    tokenizer re-tokenising decoded text may need a little.
    """
    if record.get("error_kind"):
        return f"error:{record['error_kind']}"
    if record.get("prompt_reused"):
        return "prompt_reused"
    if record.get("finish_reason") != "length":
        return f"finish_reason:{record.get('finish_reason')}"
    tokens = record.get("tokens_client")
    if tokens is None or abs(tokens - osl) > length_tol * osl:
        return f"length:{tokens}!={osl}"
    usage = record.get("tokens_usage")
    if usage is not None and abs(tokens - usage) / max(1, osl) > usage_tol:
        return f"usage_mismatch:{usage}"
    return None


def _reason_kind(reason: str) -> str:
    return reason.split(":", 1)[0]


def text_sanity(text: str) -> tuple[float, float]:
    """(distinct-4-gram ratio, compression ratio) of one reply.

    Distinct 4-grams over whitespace tokens: 1.0 for text that never repeats
    a 4-gram, near 0 for `a a a a`. Compression is raw bytes over zlib bytes:
    higher is more repetitive. Together they catch junk emitted at speed.
    """
    words = text.split()
    grams = [tuple(words[i:i + 4]) for i in range(len(words) - 3)]
    distinct = len(set(grams)) / len(grams) if grams else 1.0
    raw = text.encode("utf-8")
    compression = len(raw) / len(zlib.compress(raw, 9)) if raw else 1.0
    return distinct, compression


def _sanity(pairs: Sequence[tuple[float, float]]) -> dict[str, Any]:
    if not pairs:
        return {"n": 0, "distinct4": None, "compression": None,
                "distinct4_min": None, "compression_max": None}
    d4 = [p[0] for p in pairs]
    cr = [p[1] for p in pairs]
    return {"n": len(pairs), "distinct4": float(np.mean(d4)),
            "compression": float(np.mean(cr)),
            "distinct4_min": float(min(d4)), "compression_max": float(max(cr))}


def ngram_sanity(texts: Iterable[str]) -> dict[str, Any]:
    """Mean distinct-4-gram ratio and compression ratio over replies."""
    return _sanity([text_sanity(t) for t in texts])


def _tps_ttft_ms(r: dict) -> float | None:
    if r.get("t_first") is None or r.get("t_send") is None:
        return None
    return (float(r["t_first"]) - float(r["t_send"])) * 1000.0


def _tps_tpot_ms(r: dict) -> float | None:
    n = r.get("tokens_client") or 0
    if n < 2 or r.get("t_first") is None or r.get("t_last") is None:
        return None
    return (float(r["t_last"]) - float(r["t_first"])) * 1000.0 / (n - 1)


def _stream_rates(sent: Sequence[dict]) -> dict[str, Any]:
    """One stream's speed over `sent` (valid requests), not divided by chips.

    stream_tps is output tokens over send-to-last-token time, the prefill
    included: what the one user waits for. decode_tps is 1000 / mean TPOT:
    first token to last, the prefill left out. A server that holds its first
    tokens back and emits them in one piece raises the second and not the
    first, and its TTFT carries the delay; one that emits in bursts and
    finishes sooner raises the first.
    """
    tpot = [v for v in (_tps_tpot_ms(r) for r in sent) if v is not None]
    ttft = [v for v in (_tps_ttft_ms(r) for r in sent) if v is not None]
    whole = [(r.get("tokens_client") or 0, float(r["t_last"]) - float(r["t_send"]))
             for r in sent if r.get("t_last") is not None]
    busy = sum(t for _, t in whole)
    return {"n": len(sent),
            "decode_tps": 1000.0 / float(np.mean(tpot)) if tpot and np.mean(tpot) > 0
            else None,
            "stream_tps": sum(n for n, _ in whole) / busy if busy > 0 else None,
            "p50_tpot_ms": percentile(tpot, 50),
            "p50_ttft_ms": percentile(ttft, 50), "p95_ttft_ms": percentile(ttft, 95)}


def _balanced(rates: Sequence[float | None]) -> float | None:
    """The speed over an equal number of requests of every size: the harmonic
    mean, or None when a size has no measurement."""
    if not rates or any(not r for r in rates):
        return None
    return len(rates) / sum(1.0 / r for r in rates)


def _slo_met(summary: dict, slo: dict | None) -> bool:
    slo = TPS_DEFAULT_SLO if slo is None else slo
    tpot, ttft = summary.get("p99_tpot_ms"), summary.get("p95_ttft_ms")
    if tpot is None or ttft is None:
        return False
    return (tpot <= float(slo.get("p99_tpot_ms", math.inf))
            and ttft <= float(slo.get("p95_ttft_ms", math.inf)))


def _sanity_ok(summary: dict, slo: dict | None) -> bool:
    """Inside the sanity band, when one is given (R's band, from quality2)."""
    slo = slo or {}
    d4, cr = summary.get("distinct4"), summary.get("compression")
    if "min_distinct4" in slo and (d4 is None or d4 < float(slo["min_distinct4"])):
        return False
    if "max_compression" in slo and (cr is None
                                     or cr > float(slo["max_compression"])):
        return False
    return True


def tps_level_summary(records: Sequence[dict], window: tuple[float, float],
                      chips: int, slo: dict | None = None,
                      sizes: Sequence[int] | None = None) -> dict[str, Any]:
    """One concurrency (or rate) level, scored over its steady window.

    `records` are every request the level sent; `window` is (start_s, end_s)
    on the records' clock, i.e. after the ramp and before the drain.

    - n_invalid counts every record of the level, ramp and drain included:
      a server that misbehaves only while warming up still misbehaved.
    - Throughput counts the requests that *completed* inside the window:
      tps_per_chip from the client's own token counts, goodput from valid
      requests only (each exactly osl tokens, so osl x completed / window).
    - Latency percentiles are over the valid requests *sent* inside the
      window, so the ramp's thundering herd and the drain's tail are out.
    - decode_tps and stream_tps are one stream's speed over those same
      requests (_stream_rates).
    - `sizes` (a closed-loop mix of input sizes, records carrying `isl`):
      by_isl has every size on its own, decode_tps and stream_tps are balanced
      over the sizes, and stream_tps_per_chip (level x stream_tps / chips),
      the whole-answer speed, is what the level is worth to tps_headline
      instead of the goodput. A window holds whole requests, so at one stream
      the goodput moves by a request's worth with where the window happens to
      end; the per-request speed does not. The first token is held to the
      SLO's fixed ceiling only (p95 over the level, like any other target).
    """
    start, end = float(window[0]), float(window[1])
    span = max(1e-9, end - start)
    chips = max(1, int(chips))
    level = records[0].get("level") if records else None
    invalid = [r for r in records if not r.get("ok")]
    reasons: dict[str, int] = {}
    for r in invalid:
        kind = _reason_kind(r.get("invalid_reason") or "unknown")
        reasons[kind] = reasons.get(kind, 0) + 1
    done = [r for r in records if r.get("t_last") is not None
            and start <= float(r["t_last"]) <= end]
    good = [r for r in done if r.get("ok")]
    sent = [r for r in records if r.get("ok") and r.get("t_send") is not None
            and start <= float(r["t_send"]) <= end]
    ttft = [v for v in (_tps_ttft_ms(r) for r in sent) if v is not None]
    tpot = [v for v in (_tps_tpot_ms(r) for r in sent) if v is not None]
    sanity = _sanity([(r["distinct4"], r["compression"]) for r in done
                      if r.get("distinct4") is not None
                      and r.get("compression") is not None])
    out: dict[str, Any] = {
        "level": level,
        "window": [start, end],
        "window_s": span,
        "chips": chips,
        "goodput_tps_per_chip": sum(r.get("tokens_client") or 0
                                    for r in good) / span / chips,
        "tps_per_chip": sum(r.get("tokens_client") or 0
                            for r in done) / span / chips,
        "p50_ttft_ms": percentile(ttft, 50), "p95_ttft_ms": percentile(ttft, 95),
        "p99_ttft_ms": percentile(ttft, 99),
        "p50_tpot_ms": percentile(tpot, 50), "p99_tpot_ms": percentile(tpot, 99),
        "n": len(records), "n_window": len(done), "n_latency": len(sent),
        "n_invalid": len(invalid), "invalid_reasons": reasons,
        "distinct4": sanity["distinct4"], "compression": sanity["compression"],
    }
    rates = _stream_rates(sent)
    out["decode_tps"], out["stream_tps"] = rates["decode_tps"], rates["stream_tps"]
    if sizes:
        by = {str(k): _stream_rates([r for r in sent if r.get("isl") == k])
              for k in sizes}
        out["by_isl"] = by
        out["decode_tps"] = _balanced([v["decode_tps"] for v in by.values()])
        out["stream_tps"] = _balanced([v["stream_tps"] for v in by.values()])
        out["stream_tps_per_chip"] = (float(level or 0) * (out["stream_tps"] or 0.0)
                                      / chips)
    out["slo_met"] = _slo_met(out, slo)
    out["sanity_ok"] = _sanity_ok(out, slo)
    return out


def tps_level_value(summary: dict) -> float:
    """What a level is worth: whole-answer speed per chip for a mix of input
    sizes (the summary has by_isl), goodput per chip otherwise."""
    key = "stream_tps_per_chip" if "by_isl" in summary else "goodput_tps_per_chip"
    return float(summary.get(key) or 0.0)


def tps_headline_level(level_summaries: Sequence[dict],
                       slo: dict | None) -> dict | None:
    """The level tps_headline takes its number from, or None."""
    best = None
    for s in level_summaries:
        if s.get("n_invalid", 1) or not _slo_met(s, slo) or not _sanity_ok(s, slo):
            continue
        if tps_level_value(s) > (tps_level_value(best) if best else 0.0):
            best = s
    return best


def tps_headline(level_summaries: Sequence[dict], slo: dict | None) -> float:
    """Max goodput per chip over the levels that count, else 0. For a mix of
    input sizes the levels are worth their whole-answer speed per chip
    instead (tps_level_value).

    A level counts when it meets the SLO (re-judged here from its
    percentiles, so a summary cannot vouch for itself), has 0 invalid
    requests, and sits inside the sanity band when `slo` carries one
    (min_distinct4, max_compression).
    """
    best = tps_headline_level(level_summaries, slo)
    return tps_level_value(best) if best else 0.0


def read_jsonl(path: str) -> list[dict]:
    out = []
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def write_jsonl(path: str, rows: Iterable[dict]) -> None:
    with open(path, "w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
