"""A deterministic toy language model for the stand-in GPU machine.  Stdlib.

Vocabulary: the 256 byte values (ids 0-255, a byte-level BPE with no merges)
and one stop token (256).  The next-token distribution depends only on the
last two ids, so the reference and a test engine that share this file give
the same answers; `damage` adds noise for an engine that must fail.  A "~"
is always followed by a second "~" and "~~" by the stop token, which is
chosen nowhere else: a prompt ending in "~~" gets an answer that is only the
stop token, and answers that reach a "~" stop two tokens later.
"""
from __future__ import annotations

import hashlib
import math
import random

V = 257
EOS = 256
STOP_AFTER = (126, 126)                   # "~~": the next token is the stop token
_LETTERS = set(b"etaoinshrdlu ETAOINSHRDLU,.\n")
_cache: dict = {}


def logprobs(ids: list[int], damage: float = 0.0) -> list[float]:
    key = (tuple(ids[-2:]), damage)
    got = _cache.get(key)
    if got is not None:
        return got
    seed = int.from_bytes(hashlib.sha256(repr(key[0]).encode()).digest()[:8], "little")
    r = random.Random(seed)
    z = [r.gauss(0, 2.0) + (3.0 if i in _LETTERS else 0.0) for i in range(V)]
    z[EOS] -= 30.0
    if key[0] == STOP_AFTER:              # "~~" asks it to stop: answers can end before max_tokens
        z[EOS] += 60.0
    elif key[0][-1:] == (126,):           # a single "~" is followed by another
        z[126] += 60.0
    if damage:
        r2 = random.Random(seed ^ 0x5EED)
        z = [x + r2.gauss(0, damage) for x in z]
    m = max(z)
    lse = m + math.log(sum(math.exp(x - m) for x in z))
    out = [x - lse for x in z]
    if len(_cache) > 200000:
        _cache.clear()
    _cache[key] = out
    return out


def greedy(ids: list[int], n: int, damage: float = 0.0, ignore_eos: bool = True):
    """[(id, logprobs)] for n greedy steps."""
    ctx, out = list(ids), []
    for _ in range(n):
        lp = logprobs(ctx, damage)
        g = max(range(V), key=lp.__getitem__)
        out.append((g, lp))
        ctx.append(g)
        if g == EOS and not ignore_eos:
            break
    return out


def top(lp: list[float], k: int) -> list[int]:
    return sorted(range(V), key=lambda i: (-lp[i], i))[:k]


def encode(text: str) -> list[int]:
    return list(text.encode("utf-8"))


def decode(ids: list[int]) -> str:
    return bytes(i for i in ids if i < 256).decode("utf-8", "replace")
