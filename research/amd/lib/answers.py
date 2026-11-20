"""The engine's answers and their comparison with the reference, for ./check
(public prompts) and ./submit (prompts drawn fresh from the judge's pool).

Protocol 2: the engine answers every prompt once, one request at a time,
each request in the one shape of engine.Server.ask; those answers are timed
and judged.  Then the engine is stopped and the reference walks along each
answer (compare.walk_engine).  Speed is the judge's formula: the sum of the
answer tokens over the sum of whole-request seconds (prompt time included),
over those same requests, every token of which the walk checked.
"""
from __future__ import annotations

import json
from pathlib import Path

import compare
import quality
import tokmap

PROTOCOL = 2
FORMULA = "sum of answer tokens / sum of whole-request seconds, prompt time included"


def limits(cfg: dict, floor: float | None = None) -> compare.Limits:
    L = cfg["limits"]
    return compare.Limits(min_agreement=L["min_agreement"], tie_eps=L["tie_eps"], margin=L["margin"],
                          max_partings_per_item=L["max_partings_per_item"], floor=floor,
                          compare_stop=bool(L.get("compare_stop", True)))


def load_prompts(path: Path) -> list[dict]:
    with open(path) as f:
        return [json.loads(x) for x in f if x.strip()]


def items_of(sess, prompts: list[dict], max_tokens: int) -> list[dict]:
    """[{id, ids, max_tokens}]; ids from the row's prompt_token_ids, else the
    reference's tokenizer (cached; starts the reference only on a miss)."""
    return [{"id": p.get("id", f"p{i}"), "ids": sess.ids_for(p), "max_tokens": int(p.get("max_tokens") or max_tokens)}
            for i, p in enumerate(prompts, 1)]


def start_then_items(sess, server, prompts: list[dict], max_tokens: int, times: dict, log,
                     tasks: list[dict] | None = None, task_max_tokens: int = 512):
    """./check's order: when a prompt's token ids still need the reference's
    tokenizer, start ./serve once first and stop it, so that a serve that
    cannot start fails (engine.EngineFailed) before the reference is
    started.  Then the items, then ./serve is started for the answers.
    With tasks: returns (items, task items)."""
    def make():
        return items_of(sess, prompts, max_tokens), task_items(sess, tasks or [], task_max_tokens)
    if not all(sess.ids_cached(p) for p in prompts + (tasks or [])):
        times["engine_first_start_s"] = round(server.start(), 1)
        server.stop()
        log("check: serve starts; the reference now tokenizes the prompts once (cached after)")
        items, titems = make()
        sess.close()
    else:
        items, titems = make()
    times["engine_ready_s"] = round(server.start(), 1)
    return items if tasks is None else (items, titems)


def warm_shapes(items: list[dict], warm: list[list[int]]) -> list[list[int]]:
    """One warm-up prompt for each prompt length among `items`, cut from the
    warm-up rows (never a prompt that is judged)."""
    stream = [int(t) for w in warm for t in w]
    if not stream:
        return []
    out, off = [], 0
    for n in sorted({len(it["ids"]) for it in items}):
        rep = stream * (n // len(stream) + 2)
        s = off % len(stream)
        out.append(rep[s:s + n])
        off += 7919
    return out


def warm_up(server, items: list[dict], warm: list[list[int]], log) -> int:
    """Each warm-up shape twice (the second time is the cached-prefix case),
    in the judge's one request shape.  Untimed, unjudged."""
    shapes = warm_shapes(items, warm)
    mt = max((it["max_tokens"] for it in items), default=1)
    for ids in shapes:
        for _ in range(2):
            server.ask(ids, mt)
    if shapes:
        log(f"warm-up: {len(shapes)} prompt length(s), twice each")
    return 2 * len(shapes)


def engine_answers(cfg: dict, server, items: list[dict], log) -> list[dict]:
    """The engine's answer to each item, one request at a time.  A failed
    request is {"error": ...}; after a time-out or an exit, later items are
    not sent."""
    outs, dead = [], None
    for it in items:
        if dead:
            outs.append({"error": f"not sent: {dead}"})
            continue
        try:
            outs.append(server.ask(it["ids"], it["max_tokens"]))
        except Exception as e:  # noqa: BLE001 - the engine failed on this prompt
            outs.append({"error": f"request failed: {e}"[:300]})
            exited = getattr(server, "p", None) is not None and server.p.poll() is not None
            if exited or "timed out" in str(e).lower() or isinstance(e, TimeoutError):
                dead = (f"./serve exited with {server.p.returncode}" if exited
                        else f"the engine did not answer {it['id']} within {cfg['engine']['request_timeout_s']} s")
    return outs


def _counted(out: dict, stops: set) -> int:
    t = out.get("tokens") or []
    return len(t) - (1 if out.get("finish") == "stop" and t and t[-1] in stops else 0)


def speed(items: list[dict], outs: list[dict], stops: set) -> dict | None:
    """The judge's formula over the answered requests: sum of answer tokens
    (a final stop token not counted) / sum of whole-request seconds."""
    per = []
    for it, o in zip(items, outs):
        if "error" in o:
            return None
        n = _counted(o, stops)
        per.append({"id": it["id"], "prompt_tokens": o["prompt_tokens"], "tokens": n, "seconds": round(o["seconds"], 4),
                    "finish": o.get("finish")})
    toks = sum(p["tokens"] for p in per)
    secs = sum(p["seconds"] for p in per)
    rates = [p["tokens"] / p["seconds"] for p in per if p["seconds"] > 0]
    return {"tokens_per_s": round(toks / secs, 2) if secs else 0.0, "tokens": toks, "seconds": round(secs, 3),
            "requests": len(per), "per_request_min": round(min(rates), 2) if rates else None,
            "per_request_max": round(max(rates), 2) if rates else None, "per_request": per,
            "formula": "sum of answer tokens / sum of whole-request seconds (prompt time included)",
            "protocol": PROTOCOL}


def _set_rates(per: list[dict]) -> dict:
    """One pass over the prompts: whole-request tokens/s by the judge's
    formula, the summed prompt time, and decode tokens/s (tokens after the
    first over the request time after the prompt time)."""
    toks = sum(p["tokens"] for p in per)
    secs = sum(p["seconds"] for p in per)
    pre = [p["prompt_s"] for p in per if p.get("prompt_s") is not None]
    out = {"whole_tps": round(toks / secs, 2) if secs else None}
    if len(pre) == len(per):
        dec_t = sum(max(p["seconds"] - p["prompt_s"], 0.0) for p in per)
        dec_n = sum(max(p["tokens"] - 1, 0) for p in per)
        out.update({"prompt_s": round(sum(pre), 4), "decode_tps": round(dec_n / dec_t, 2) if dec_t > 0 else None})
    return out


def breakdown(server, items: list[dict], outs: list[dict], stops: set, repeats: int, log) -> dict:
    """The speed in parts, from requests apart from the judged ones (they do
    not change the score).  Each repeat r sends every prompt rotated (same
    length, not the cached prefix of any earlier request): rotated by 2r+1
    tokens with max_tokens=1 (the prompt time), then rotated by 2r+2 whole.  The judged answers are
    one more sample of the whole-request rate.  Returns the per-repeat rates,
    their min and max, and every request."""
    samples, reqs = [], []
    judged = speed(items, outs, stops)
    if judged:
        samples.append({"pass": "judged", "whole_tps": judged["tokens_per_s"]})
    for r in range(repeats):
        per = []
        for it in items:
            # two different rotations of the same length: the whole request must not find the
            # max_tokens=1 request's prompt in a prefix cache
            n = max(len(it["ids"]), 1)
            k1, k2 = (2 * r + 1) % n, (2 * r + 2) % n
            p1 = server.ask(it["ids"][k1:] + it["ids"][:k1], 1)
            ids = it["ids"][k2:] + it["ids"][:k2]
            o = server.ask(ids, it["max_tokens"])
            per.append({"id": it["id"], "repeat": r + 1, "prompt_tokens": len(ids), "prompt_s": round(p1["seconds"], 4),
                        "tokens": _counted(o, stops), "seconds": round(o["seconds"], 4)})
        reqs += per
        samples.append({"pass": f"repeat {r + 1}", **_set_rates(per)})
    def mm(key):
        v = [s[key] for s in samples if s.get(key) is not None]
        return {"min": min(v), "max": max(v)} if v else None
    res = {"samples": samples, "whole_tps": mm("whole_tps"), "decode_tps": mm("decode_tps"), "prompt_s": mm("prompt_s"),
           "requests": reqs, "repeats": repeats,
           "formula": "whole: sum of answer tokens / sum of whole-request seconds (prompt time included); "
                      "decode: sum of (tokens - 1) / sum of (request seconds - prompt seconds); "
                      "prompt: the summed seconds of max_tokens=1 requests"}
    log("speed: " + breakdown_line(res))
    return res


def breakdown_line(b: dict) -> str:
    def rng(x, unit, f="{:.1f}"):
        if not x:
            return "?"
        a, z = f.format(x["min"]), f.format(x["max"])
        return f"{a} {unit}" if a == z else f"{a}-{z} {unit}"
    return (f"whole request {rng(b.get('whole_tps'), 'tokens/s')}; decode {rng(b.get('decode_tps'), 'tokens/s')}; "
            f"prompt time {rng(b.get('prompt_s'), 's', '{:.2f}')} per pass of {len({r['id'] for r in b.get('requests', [])}) or '?'} "
            f"prompts (min-max over {len(b.get('samples', []))} passes)")


def speed_lines(sp: dict | None, per_request: bool = False) -> list[str]:
    """What ./submit and ./score print about a passed run's speed: the score
    (the judge's formula over the judged requests) and the parts from the
    extra requests."""
    if not sp:
        return []
    out = [f"score: {sp['tokens']} answer tokens / {sp['seconds']} s of whole requests = {sp['tokens_per_s']} tokens/s "
           f"over {sp['requests']} judged requests (per request {sp.get('per_request_min')}-{sp.get('per_request_max')})"]
    parts = sp.get("parts")
    if parts and "error" not in parts:
        out.append("parts, not the score: " + breakdown_line({**parts, "requests": sp.get("per_request") or []}))
    elif parts:
        out.append(f"parts: {parts['error']}")
    if per_request:
        for p in sp.get("per_request") or []:
            rate = p["tokens"] / p["seconds"] if p["seconds"] else 0.0
            out.append(f"  {p['id']}: {p['prompt_tokens']} prompt tokens, {p['tokens']} answer tokens in "
                       f"{p['seconds']:.3f} s = {rate:.1f} tokens/s ({p.get('finish')})")
    return out


def decided_fail(cfg: dict, rows: list[dict], rest: list[dict], lim: compare.Limits) -> str | None:
    """Why the run already FAILs whatever the answers not yet walked (rest)
    hold, or None.  A failed answer fails the run in every configuration.

    While the quality bar replaces the agreement limit, only what
    quality.verdict fails on whatever comes later: an answer cut at the
    parting limit, and a top-k share that cannot reach its limit even if
    every position of the answers left (at most max_tokens + 1 each) were in
    the reference's top k.  The mean gap and measure B can still move either
    way, so they never stop the walk.

    While the agreement limit applies: even if every answer left agreed on
    every position, the run's agreement would stay under the limit."""
    bad = [r for r in rows if not r["item"].ok]
    if bad:
        return f"answer {bad[0]['id']} failed, which fails the run"
    if not rest:
        return None
    pos = sum(r["item"].positions for r in rows)
    room = sum(int(it["max_tokens"]) + 1 for it in rest)
    if quality.replaces_agreement(cfg):
        cut = [r for r in rows if r["item"].cut]
        if cut:
            return (f"answer {cut[0]['id']} was cut at the parting limit; the quality bar needs every answer "
                    f"walked to its end, which fails the run")
        for k, v in sorted(quality.topk_limits(cfg).items(), key=lambda kv: int(kv[0])):
            if v is None:
                continue
            hits = sum(1 for r in rows for rk, _, _ in r["item"].ranks if rk is not None and rk <= int(k))
            best = (hits + room) / (pos + room)
            if best < float(v) - 1e-4:
                return (f"top-{k} share {hits} of {pos} positions over {len(rows)} answer(s); even with every "
                        f"position of the {len(rest)} left in the top {k} it would be at most {best:.4f}, "
                        f"under the bar's limit {v}")
        return None
    agree = sum(r["item"].compared + r["item"].close for r in rows)
    room = sum(int(it["max_tokens"]) + 1 for it in rest)
    best = (agree + room) / (pos + room)
    need = lim.required()
    if best < need - 1e-4:
        return (f"agreement {agree} of {pos} positions over {len(rows)} answer(s); even full agreement on the "
                f"{len(rest)} left would give at most {best:.4f}, under the limit {need}")
    return None


def walk(cfg: dict, sess, items: list[dict], outs: list[dict], lim: compare.Limits, log,
         stopped: list | None = None) -> list[dict]:
    """The reference walks along every answer.  One row per item with its
    ItemResult and a readable account of the first partings.

    With a list as `stopped` the walk ends as soon as the run's FAIL is
    decided (decided_fail): the answers left are not walked, the reason is
    appended to `stopped`, and only the walked answers have rows; the caller
    must then fail the run (./submit does), whatever the walked rows alone
    would give.
    A hopeless engine's walk took about 13 minutes on 2026-11-18."""
    tm = tokmap.load(cfg["model"]["dir"])
    stops = tm.special_ids
    rows = []
    for i, (it, o) in enumerate(zip(items, outs)):
        if stopped is not None and rows:
            why = decided_fail(cfg, rows, items[i:], lim)
            if why:
                log(f"  the walk stops after {len(rows)} of {len(items)} answers: {why}")
                stopped.append(f"the walk stopped after {len(rows)} of {len(items)} answers: {why}")
                break
        pid, ids, mt = it["id"], it["ids"], it["max_tokens"]
        if "error" in o:
            res = compare.ItemResult(pid, ok=False, reason=o["error"], positions=mt)
        else:
            toks = o["tokens"]
            bad = [t for t in toks if t not in tm.by_id]
            if bad:
                res = compare.ItemResult(pid, ok=False, positions=mt,
                                         reason=f"the answer has token id {bad[0]}, which is not in the model's vocabulary")
            else:
                first = compare.RefSeg.of(sess.answer(ids, mt))

                def resume(pos, ids=ids, toks=toks, mt=mt):
                    return compare.RefSeg.of(sess.answer(ids + toks[:pos + 1], mt - pos - 1))
                res = compare.walk_engine(pid, toks, o["lps"], o.get("finish"), mt, first, resume, lim, stops)
        notes = []
        for n in res.notes[:3]:
            txt = (lambda v: "(stop)" if v == "stop" else tm.text(v))
            notes.append({"pos": n["pos"], "ref": txt(n["ref"]), "engine": txt(n["eng"]), "gap": n["gap"]})
        agree = (res.compared + res.close) / res.positions if res.positions else 0.0
        rows.append({"id": pid, "item": res, "agreement": round(agree, 4), "partings": notes,
                     "engine_first_text": "".join(tm.text(t) for t in (o.get("tokens") or [])[:40])})
        log(f"  {pid}: agreement {agree:.3f}, {len(res.partings)} parting(s)" + ("" if res.ok else f", FAILED: {res.reason}"))
    return rows


def decide(cfg: dict, items: list, lim: compare.Limits, qb: dict | None) -> tuple[bool, list[str], dict, dict]:
    """PASS or FAIL with the quality bar: (ok, reasons, aggregate, quality).
    While the bar is not set the result is compare.judge's (the agreement
    limit) and quality only records the measures.  Once it is set, a FAIL
    on it fails the run; with quality.bar_replaces_agreement the agreement
    limit is then reported, not applied.  A failed request always fails."""
    ok, reasons, agg = compare.judge(items, lim)
    ks = tuple(int(k) for k in (cfg.get("quality") or {}).get("topk", compare.TOPK))
    q = quality.verdict(cfg, compare.measure_a(items, ks) if items else None, qb)
    if quality.replaces_agreement(cfg):
        reasons = [f"{it.id}: {it.reason}" for it in items if not it.ok] + ([] if items else ["no answers were compared"])
        agg["agreement_limit"] = "reported, not applied: the quality bar is set and replaces it"
    if q["verdict"] == "fail":
        reasons = reasons + [f"quality: {r}" for r in q["reasons"]]
    return (not reasons), reasons, agg, q


def task_items(sess, tasks: list[dict], max_tokens: int) -> list[dict]:
    """Measure B's tasks as items (token ids given or cached), each with its
    task row (answer, kind) for grading."""
    return [{"id": t.get("id", f"t{i}"), "ids": sess.ids_for(t), "max_tokens": int(t.get("max_tokens") or max_tokens),
             "task": t} for i, t in enumerate(tasks, 1)]


def text_of(tm, toks) -> str:
    return "".join(tm.text(t) for t in (toks or []) if t not in tm.special_ids)


def engine_task_texts(cfg: dict, server, titems: list[dict], tm, log) -> list[str | None]:
    """The engine's answers to the tasks, in the judge's one request shape,
    one at a time; not timed and not in the speed.  None for a failed
    request (graded wrong)."""
    outs = engine_answers(cfg, server, titems, log)
    texts = [None if "error" in o else text_of(tm, o.get("tokens")) for o in outs]
    log(f"tasks: the engine answered {sum(t is not None for t in texts)} of {len(titems)}")
    return texts


def reference_task_texts(sess, titems: list[dict], tm) -> list[str]:
    """The reference's greedy answers to the same tasks (cached like every
    reference answer)."""
    return [text_of(tm, sess.answer(it["ids"], it["max_tokens"])["tokens"]) for it in titems]


def why(row: dict, need: float) -> str | None:
    it = row["item"]
    if not it.ok:
        return it.reason
    if row["agreement"] < need:
        bits = []
        for p in row["partings"]:
            g = "not in the reference's top 20" if p["gap"] is None else f"{p['gap']:.3f} below the reference's choice"
            bits.append(f"at token {p['pos']} the engine chose {p['engine']!r}, the reference {p['ref']!r}; the engine's is {g}")
        extra = "; ".join(bits)
        cut = " (stopped after the parting limit)" if it.cut else ""
        return f"agreement {row['agreement']:.3f} < {need}{cut}" + (f"; {extra}" if extra else "")
    return None
