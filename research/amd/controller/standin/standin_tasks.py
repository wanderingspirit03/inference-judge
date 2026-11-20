#!/usr/bin/env python3
"""Tasks for measure B on the stand-in GPU machine (the toy model of toylm.py,
which cannot do arithmetic).  Stdlib.

  standin_tasks.py N [SEED]   N task rows on stdout

A task is a prompt whose undamaged greedy next token is one letter A-E, with
a clear margin (more than 0.5 nats over the second choice), and max_tokens 1:
the reference answers each right, an engine with a little noise in its
logits (a quantised one) nearly always, a damaged one rarely.  The rows have
the shape of the judge's task pool: id, prompt_token_ids, answer, kind "mc",
max_tokens."""
from __future__ import annotations

import json
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.realpath(__file__)))
import toylm  # noqa: E402

CHARS = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 .,:;!?-"


def tasks(n: int, seed: int = 0) -> list[dict]:
    r = random.Random(seed)
    out, seen = [], set()
    while len(out) < n:
        tail = r.choice(CHARS) + r.choice(CHARS)
        if tail in seen:
            continue
        seen.add(tail)
        ids = toylm.encode(f"Pick one of A-E ({len(out)}): {tail}")
        lp = toylm.logprobs(ids)
        s = sorted(lp, reverse=True)
        g = max(range(toylm.V), key=lp.__getitem__)
        if 65 <= g <= 69 and s[0] - s[1] > 0.5:
            out.append({"id": f"standin-task-{seed}-{len(out) + 1:03d}", "prompt_token_ids": ids, "answer": chr(g),
                        "kind": "mc", "max_tokens": 1, "session": f"standin-task-{seed}-{len(out) + 1:03d}",
                        "bucket": "task"})
        if len(seen) > len(CHARS) ** 2 - 1:
            raise RuntimeError(f"only {len(out)} stand-in tasks exist for seed {seed}")
    return out


if __name__ == "__main__":
    for row in tasks(int(sys.argv[1]), int(sys.argv[2]) if len(sys.argv) > 2 else 0):
        print(json.dumps(row))
