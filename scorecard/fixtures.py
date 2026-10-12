"""Small deterministic inputs shared by the scenarios: a prompt pool for the
harness, launch summaries in the shape the verdict reads, and the calibration
those synthetic verdicts are held against."""
from __future__ import annotations

import json
import math
import random
from pathlib import Path

# The noise floor of the synthetic verdicts (2b, 4b): 1% between launches, so
# the smallest effect that counts is max(2%, 3.2 sigma_L) = 3.2%, as
# bench/stats.py derives it. Illustrative, not measured on any machine.
CALIBRATION = {"sigma_L": 0.01, "mde": 0.032, "n_launches": 10}


def prompt_pool(dest: Path, n: int = 4000, words: int = 200, seed: int = 11) -> Path:
    """JSONL of {"id","prompt"}: distinct prompts, so no two requests repeat."""
    rng = random.Random(seed)
    vocab = ("call booking account policy claim address order refund delivery "
             "appointment doctor table party name number date time reference "
             "payment card change cancel confirm transfer agent hold queue").split()
    dest.parent.mkdir(parents=True, exist_ok=True)
    with dest.open("w") as fh:
        for i in range(n):
            text = " ".join(rng.choice(vocab) for _ in range(words))
            fh.write(json.dumps({"id": f"p{i:05d}", "prompt": f"{i} {text}"}) + "\n")
    return dest


def launch(label: str, role: str, levels: dict[str, float], blocks: int = 2) -> dict:
    """One clean launch as `compare.py tps-launch` summarises it: tokens/s per
    concurrency level, the best level as the headline, no invalid request."""
    head = max(levels.values())
    return {"label": label, "role": role, "status": "ok", "headline": head,
            "value": math.log(head), "levels": dict(levels),
            "block_values": [math.log(head)] * blocks, "n_blocks": blocks,
            "n_usable": blocks, "n_stalled": 0, "n_invalid": 0, "invalid_reasons": {},
            "decode_tps": None, "stream_tps": None, "by_isl": {}}


def abba(incumbent: dict[str, float], candidate: dict[str, float]) -> list[dict]:
    """Four launches in the order the verdict wants: I C C I."""
    return [launch("I1", "incumbent", incumbent), launch("C1", "candidate", candidate),
            launch("C2", "candidate", candidate), launch("I2", "incumbent", incumbent)]
