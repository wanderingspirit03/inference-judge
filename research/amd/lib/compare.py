"""The answer comparison of ./check and ./submit.  Pure and stdlib only, so it
runs and is tested on a laptop without a GPU.

An engine's answer is the token ids it chose greedily for a prompt, with the
logprob it gave each one.  A reference answer is the token ids the reference
chose greedily, with the top logprobs it saw at each position.

The engine does not have to choose the same tokens.  Two correct engines
round differently, and an engine may trade a little precision for speed.  So
the comparison is a quality measure, not an equality test.

The walk (walk_engine, protocol 2): the engine answers each prompt once, and
that one answer is both judged and timed.  The walk goes along the engine's
tokens: at each position, given the engine's tokens so far, would the
reference have chosen the engine's token?  The share of positions where it
would is the agreement.  Where it would not (a parting), the reference is
asked to go on from the engine's prefix, and the walk resumes there.  So
every token the engine returned is checked, and the engine never sees a
reference token.

A parting whose token is within `tie_eps` (0.125) of the reference's own
token, in the reference's logprobs, is a close call: stock's logits sit on a
grid of that step, so the two tokens are tied, and the position counts as
agreeing.  It still counts toward max_partings_per_item.  The strict
agreement, without that tolerance, is reported beside it.

Where the answers end (Limits.compare_stop, on by default; an assumption the
user has not confirmed, see measurements/2026-11-17/NOTES.md): an engine that
stops before max_tokens is compared at that position as if it chose a stop
token, and the reference, when it stops where the engine goes on, is compared
the same way.  With compare_stop off those positions are not counted.

compare_item is the walk of protocol 1 (the reference answers first, the
engine is resumed from the reference's prefix).  The commands no longer use
it: an engine could answer the first request badly and the resumes well, and
lose one position per answer.  It stays for the test that shows that.

PASS needs the agreement to be at least stock's agreement with itself
(`floor`, measured by check-refill stock-speed on fresh sets) minus `margin`,
or `min_agreement` while that is not measured.

On every position where the two chose the same token, the walk also records
how far the engine's logprob for that token is from the reference's; that is
reported, not judged.  The engine's logprobs are used for nothing else, so an
engine may send token ids only (lib/client.py).

Measure A of the quality bar (lib/quality.py) comes from the same walk, from
the reference's logprobs alone: at every walked position, the rank of the
engine's token in the reference's top logprobs (1 when it is the reference's
choice) and the gap, the reference's logprob of its own choice minus its
logprob of the engine's token (0 when they agree).  A token outside the
reference's top is counted with the smallest gap it can have (the reference's
best minus the last of its top) and marked outside.  Positions the walk never
reached (an answer cut at max_partings_per_item, a failed answer) are in no
top-k.

Everything that decides PASS or FAIL is in `Limits` and `judge()`.
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict


@dataclass(frozen=True)
class Limits:
    # PASS needs at least this share of all reference positions to agree,
    # while stock's own agreement (floor) is not measured.
    min_agreement: float = 0.95
    # Each parting costs one resume.  After this many in one answer the walk
    # stops, and the rest of that answer counts as not agreeing.
    max_partings_per_item: int = 40
    # A parting within this many nats of the reference's token, in the
    # reference's logprobs, is a close call and counts as agreeing.
    tie_eps: float = 0.125
    # When stock's agreement with itself is known (floor), PASS needs at
    # least floor - margin instead of min_agreement.
    margin: float = 0.03
    floor: float | None = None
    # Compare where the answers end (see the module text).
    compare_stop: bool = True

    def required(self) -> float:
        if self.floor is None:
            return round(self.min_agreement, 5)
        return round(self.floor - self.margin, 5)

    def is_close(self, gap: float | None) -> bool:
        return gap is not None and gap <= self.tie_eps + 1e-9


@dataclass
class Walk:
    """What one pass over an engine segment found."""
    compared: int = 0                 # positions where both chose the same token
    dlps: list = field(default_factory=list)
    stop: str = "end"                 # end | part | short | long
    pos: int = -1                     # reference position where the walk stopped
    gap: float | None = None          # reference's logprob gap between its token and the engine's
    engine_token: int | None = None


def walk(ref_tokens: list[int], ref_top: list[dict], eng_tokens: list[int],
         eng_lps: list[float | None], start: int) -> Walk:
    """Compare an engine segment that begins at reference position `start`.

    ref_top[j] maps token id -> logprob, the reference's top logprobs at j.
    eng_tokens[i] is the engine's token at reference position start + i.
    """
    w = Walk()
    n = len(ref_tokens)
    for i, e in enumerate(eng_tokens):
        j = start + i
        if j >= n:
            w.stop, w.pos, w.engine_token = "long", j, e
            return w
        r = ref_tokens[j]
        top = ref_top[j]
        if e == r:
            lp_ref = top.get(r)
            lp_eng = eng_lps[i] if i < len(eng_lps) else None
            if lp_ref is not None and lp_eng is not None:
                w.dlps.append(abs(float(lp_eng) - float(lp_ref)))
            w.compared += 1
            continue
        w.stop, w.pos, w.engine_token = "part", j, e
        lp_ref = top.get(r)
        lp_e = top.get(e)
        if lp_ref is not None and lp_e is not None:     # None: not even in the reference's top
            w.gap = float(lp_ref) - float(lp_e)
        return w
    if start + len(eng_tokens) < n:
        w.stop, w.pos = "short", start + len(eng_tokens)
    return w


@dataclass
class ItemResult:
    id: str
    ok: bool = True                   # False only when the engine failed, never for a parting
    reason: str = ""
    positions: int = 0                # reference positions this answer is judged on
    compared: int = 0                 # of those, where the engine chose the reference's token
    close: int = 0                    # partings that were close calls; they count as agreeing
    partings: list = field(default_factory=list)   # [(pos, gap or None)]
    notes: list = field(default_factory=list)      # [{pos, ref, eng, gap}]; ref/eng: a token id or "stop"
    cut: bool = False                 # the walk stopped at max_partings_per_item
    dlps: list = field(default_factory=list)
    mean_abs_dlp: float = 0.0
    max_abs_dlp: float = 0.0
    # Measure A: (rank or None when outside the reference's top, gap, outside) per walked position
    ranks: list = field(default_factory=list)

    def summary(self) -> dict:
        d = asdict(self)
        d.pop("dlps")
        d.pop("ranks")
        d["measure_a"] = measure_a([self])
        return d


def compare_item(item_id: str, ref_tokens: list[int], ref_top: list[dict],
                 first: tuple[list[int], list], resume, limits: Limits) -> ItemResult:
    """Walk the engine's first answer; after each parting call
    resume(pos) -> (tokens, lps) for the engine's answer that continues from
    the reference's tokens [0, pos] (inclusive), and walk that.  resume may
    raise; the item then fails with the error as its reason."""
    n = len(ref_tokens)
    res = ItemResult(item_id, positions=n)
    toks, lps = first
    start = 0
    while True:
        w = walk(ref_tokens, ref_top, toks, lps, start)
        res.compared += w.compared
        res.dlps.extend(w.dlps)
        if w.stop == "end":
            break
        gap = None if w.gap is None else round(w.gap, 5)
        res.partings.append((w.pos, gap))
        if w.stop == "part" and limits.is_close(gap):
            res.close += 1
        if w.stop == "long":          # the engine went on where the reference ended
            res.positions = n + 1
            break
        if w.pos + 1 >= n:
            break
        if len(res.partings) >= limits.max_partings_per_item:
            res.cut = True
            break
        try:
            toks, lps = resume(w.pos)
        except Exception as e:  # noqa: BLE001 - any failure of the engine fails the item
            res.ok, res.reason = False, f"resume after a parting at token {w.pos} failed: {e}"[:300]
            break
        start = w.pos + 1
    if res.dlps:
        res.mean_abs_dlp = sum(res.dlps) / len(res.dlps)
        res.max_abs_dlp = max(res.dlps)
    return res


class RefFailed(Exception):
    """The reference's answer cannot be walked (a fault of the judge's side,
    not of the engine)."""


@dataclass
class RefSeg:
    """A reference answer from some position on: its tokens, the top
    logprobs at each ({id: logprob}), why it ended, and the stop token's entry
    when it stopped ({"id", "top": {id: logprob}}, or None)."""
    tokens: list
    top: list
    finish: str | None = None
    stop: dict | None = None

    @classmethod
    def of(cls, doc: dict) -> "RefSeg":
        st = doc.get("stop")
        if st:
            st = {"id": int(st["id"]), "top": {int(k): float(v) for k, v in (st.get("top") or {}).items()}}
        return cls([int(t) for t in doc["tokens"]], [{int(k): float(v) for k, v in t.items()} for t in doc["top_logprobs"]],
                   doc.get("finish_reason"), st)


def _gap(top: dict, ref_token: int, eng_lp: float | None) -> float | None:
    """The reference's logprob of its own token minus its logprob of the
    engine's choice; None when either is not in its top."""
    lr = top.get(ref_token)
    if lr is None or eng_lp is None:
        return None
    return float(lr) - float(eng_lp)


def _best(top: dict, ids) -> float | None:
    got = [top[i] for i in ids if i in top]
    return max(got) if got else None


def _rank(top: dict, tok) -> tuple[int | None, float, bool]:
    """(rank of tok in the reference's top, 1 = its best; the gap best - tok;
    outside).  Outside the top: rank None and the smallest possible gap."""
    if not top:
        return None, 0.0, True
    best = max(top.values())
    lp = top.get(tok)
    if lp is None:
        return None, float(best - min(top.values())), True
    return 1 + sum(1 for v in top.values() if v > lp + 1e-9), float(best - lp), False


def _rank_stop(top: dict, stop_ids) -> tuple[int | None, float, bool]:
    """The rank of the best-scored stop token (where the engine stopped)."""
    got = [i for i in stop_ids if i in top]
    return _rank(top, max(got, key=lambda i: top[i]) if got else None)


def walk_engine(item_id: str, eng_tokens: list[int], eng_lps: list, eng_finish: str | None, max_tokens: int,
                first: RefSeg, resume, limits: Limits, stop_ids) -> ItemResult:
    """Walk the reference along the engine's one answer.

    first: the reference's answer to the prompt.  resume(pos) -> RefSeg: the
    reference's answer that continues from the engine's tokens [0, pos]
    (inclusive), at most max_tokens - pos - 1 tokens.  stop_ids: the ids that
    end an answer.  Raises RefFailed when the reference's answer is not
    usable; any exception of resume goes through."""
    stop_ids = set(stop_ids)
    e, lps = list(eng_tokens), list(eng_lps)
    res = ItemResult(item_id)
    stopped = eng_finish == "stop"
    if stopped and e and e[-1] in stop_ids:
        e.pop()
    lps = lps[:len(e)]
    m = len(e)
    if m > max_tokens:
        res.ok, res.reason, res.positions = False, f"the engine returned {m} tokens; {max_tokens} were asked", max_tokens
        return res
    if not stopped and m < max_tokens:
        res.ok, res.positions = False, max_tokens
        res.reason = (f"the engine returned {m} of {max_tokens} tokens and did not stop "
                      f"(finish_reason {eng_finish!r}; an answer ends at max_tokens or with finish_reason \"stop\")")
        return res
    inner = [t for t in e if t in stop_ids]
    if inner:
        res.ok, res.positions = False, max_tokens
        res.reason = f"the engine's answer has the stop token {inner[0]} inside it and went on"
        return res
    stop_pos = stopped and m < max_tokens and limits.compare_stop
    res.positions = m + (1 if stop_pos else 0)

    def part(pos, ref, eng, gap) -> None:
        g = None if gap is None else round(gap, 5)
        res.partings.append((pos, g))
        res.notes.append({"pos": pos, "ref": ref, "eng": eng, "gap": g})
        if limits.is_close(g):
            res.close += 1

    seg, s, j = first, 0, 0
    while True:
        k = j - s
        if j == m:                                     # the engine's answer ends here
            if stop_pos:
                if k < len(seg.tokens):                # the reference would have gone on
                    top = seg.top[k]
                    res.ranks.append(_rank_stop(top, stop_ids))
                    part(j, seg.tokens[k], "stop", _gap(top, seg.tokens[k], _best(top, stop_ids)))
                elif seg.finish == "stop":
                    res.compared += 1
                    res.ranks.append((1, 0.0, False))
                else:
                    raise RefFailed(f"{item_id}: the reference's answer ended at token {j} without stopping")
            break
        if k >= len(seg.tokens):                       # the reference stopped here, the engine went on
            if seg.finish != "stop":
                raise RefFailed(f"{item_id}: the reference's answer ended at token {j} without stopping")
            if limits.compare_stop:
                st = seg.stop or {"top": {}}
                lr = st["top"].get(st.get("id")) if st.get("id") is not None else _best(st["top"], stop_ids)
                res.ranks.append(_rank(st["top"], e[j]))
                part(j, "stop", e[j], None if lr is None or e[j] not in st["top"] else lr - st["top"][e[j]])
            else:
                res.positions -= 1
        elif e[j] == seg.tokens[k]:
            res.compared += 1
            res.ranks.append((1, 0.0, False))
            lr, le = seg.top[k].get(e[j]), (lps[j] if j < len(lps) else None)
            if lr is not None and le is not None:
                res.dlps.append(abs(float(le) - float(lr)))
            j += 1
            continue
        else:
            top = seg.top[k]
            res.ranks.append(_rank(top, e[j]))
            part(j, seg.tokens[k], e[j], _gap(top, seg.tokens[k], top.get(e[j])))
        if len(res.partings) >= limits.max_partings_per_item:
            res.cut = True
            break
        if j + 1 >= m and not stop_pos:
            break
        seg, s, j = resume(j), j + 1, j + 1
    if res.dlps:
        res.mean_abs_dlp = sum(res.dlps) / len(res.dlps)
        res.max_abs_dlp = max(res.dlps)
    return res


def quantile(xs: list[float], q: float) -> float:
    if not xs:
        return 0.0
    s = sorted(xs)
    k = min(len(s) - 1, max(0, int(round(q * (len(s) - 1)))))
    return s[k]


def aggregate(items: list[ItemResult]) -> dict:
    dl = [x for it in items for x in it.dlps]
    positions = sum(it.positions for it in items)
    compared = sum(it.compared for it in items)
    close = sum(it.close for it in items)
    gaps = [g for it in items for _, g in it.partings if g is not None]
    return {
        "items": len(items),
        "items_failed": sum(1 for it in items if not it.ok),
        "items_cut": sum(1 for it in items if it.cut),
        "positions": positions,
        "compared_tokens": compared,
        "agreement": round((compared + close) / positions, 5) if positions else 0.0,
        "agreement_strict": round(compared / positions, 5) if positions else 0.0,
        "close_calls": close,
        "partings": sum(len(it.partings) for it in items),
        "partings_outside_top": sum(1 for it in items for _, g in it.partings if g is None),
        "median_parting_gap": round(quantile(gaps, 0.5), 5),
        "max_parting_gap": round(max(gaps), 5) if gaps else 0.0,
        "max_partings_in_item": max((len(it.partings) for it in items), default=0),
        "mean_abs_dlp": round(sum(dl) / len(dl), 6) if dl else 0.0,
        "p99_abs_dlp": round(quantile(dl, 0.99), 6),
        "max_abs_dlp": round(max(dl), 6) if dl else 0.0,
    }


TOPK = (1, 2, 5)


def measure_a(items: list[ItemResult], ks=TOPK) -> dict:
    """Measure A over the answers (see the module text): strict agreement,
    the share of positions whose engine token is in the reference's top k,
    and the mean gap per walked position, in nats.  Every position an answer
    is judged on is in the denominator of the shares; one never walked is in
    no top-k.  The mean gap is over walked positions; a token outside the
    reference's top adds its smallest possible gap, so the mean is a lower
    bound when outside_top > 0."""
    positions = sum(it.positions for it in items)
    walked = [r for it in items for r in it.ranks]
    share = (lambda n: round(n / positions, 5) if positions else 0.0)
    return {"positions": positions, "walked": len(walked), "not_walked": max(positions - len(walked), 0),
            "agreement_strict": share(sum(it.compared for it in items)),
            "topk_share": {str(k): share(sum(1 for r, _, _ in walked if r is not None and r <= k)) for k in ks},
            "mean_gap": round(sum(g for _, g, _ in walked) / len(walked), 5) if walked else None,
            "outside_top": sum(1 for _, _, o in walked if o),
            "answers_cut": sum(1 for it in items if it.cut),
            "answers_failed": sum(1 for it in items if not it.ok)}


def judge(items: list[ItemResult], limits: Limits) -> tuple[bool, list[str], dict]:
    """PASS or FAIL for a whole check, the reasons for a FAIL, and the numbers."""
    agg = aggregate(items)
    need = limits.required()
    agg["required_agreement"] = need
    why = [f"{it.id}: {it.reason}" for it in items if not it.ok]
    if not items:
        why.append("no answers were compared")
    elif agg["agreement"] < need:
        if limits.floor is None:
            how = f"limit {need}; stock against itself not measured yet"
        else:
            how = f"limit {need} = stock against itself {limits.floor} minus {limits.margin}"
        why.append(f"agreement with stock {agg['agreement']} ({how})")
    return (not why), why, agg


def parse_token(t: str) -> int:
    """'token_id:123' -> 123 (the form return_tokens_as_token_ids gives)."""
    if not isinstance(t, str) or not t.startswith("token_id:"):
        raise ValueError(f"token {t!r} is not in the token_id:<n> form; send return_tokens_as_token_ids")
    return int(t[len("token_id:"):])
