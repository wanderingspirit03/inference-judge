#!/usr/bin/env python3
"""The speed pools of whole requests: one file per request, sent as written.

A request lives in real/<pool>/<size>/<name>.txt and is the whole user message:
what a person would type, with any document or code they would paste under it.
Nothing is cut to length and nothing is padded. <size> is the input size the
request is measured under (the target's metric.isl), and the request must fit
it: with the harness's reference line in front, at most <size> tokens and not
much less (BAND).

    python real_pool.py count real/main/1000/*.txt     # tokens and the band, per file
    python real_pool.py build --out .                  # writes the two .jsonl files
    python real_pool.py build --out . --check          # exit 1 if they differ

`count` needs the model's tokenizer (GEMMA_TOKENIZER_JSON=/path/tokenizer.json);
`build` does not: it copies the files into rows {"id", "isl", "prompt"}.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

ROOT = HERE / "real"
FILES = {"main": "gemma4-real-main.jsonl", "confirm": "gemma4-real-confirm.jsonl"}
SIZES = (100, 1000, 2000, 4000)

# The harness puts "Request reference: <uuid>." in front of every request.
# With the Gemma tokenizer that line is 29 to 42 tokens (200,000 of them
# counted; nine in ten are 36 to 40), never the same twice.
NONCE_TOKENS = (29, 48)       # the least seen, and the most plus a margin
# A request is at least (1 - BAND) of its size, the target's metric.isl_tol.
# An engine that pads a prompt up to the next power of two then gives every
# request of a size the same shape: the smallest size, 100, stays above 64.
BAND = 0.30

# How many requests each size needs. A request is sent under its own size
# only, and run.sh sends a candidate no request twice: the screen launch, the
# first candidate launch (the incumbent launch beside it answers the same
# draw) and the second each use up their own. One sweep is the ramp and one
# window; a launch is a warm sweep and metric.blocks more. On the first v6e-4
# session (2026-10-13, 65 s sweeps, about 4.3 s a request) a sweep sent 15 to
# 19 requests, which over four sizes is 4 or 5 of each, so a candidate run
# needs about 32 of each: 8 for the screen, 14 for the first draw, 10 before
# the second runs short. The number grows with the speed, because a faster
# server answers more requests in the same window: 64 carries a candidate
# twice as fast. A confirmation is one draw on the confirm pool (10 at that
# speed).
MIN_PER_SIZE = {"main": 64, "confirm": 24}


def bounds(size: int) -> tuple[int, int]:
    """The fewest and the most tokens a request file of this size may hold."""
    return (math.ceil(size * (1 - BAND)) - NONCE_TOKENS[0], size - NONCE_TOKENS[1])


def request_files(pool: str, root: Path = ROOT) -> list[tuple[int, Path]]:
    out = []
    for size in SIZES:
        out += [(size, p) for p in sorted((root / pool / str(size)).glob("*.txt"))]
    return out


def read_request(path: Path) -> str:
    return path.read_text(encoding="utf-8").strip() + "\n"


def rows(pool: str, root: Path = ROOT) -> list[dict]:
    return [{"id": f"real-{pool}-{size}-{path.stem}", "isl": size,
             "prompt": read_request(path)}
            for size, path in request_files(pool, root)]


def build(root: Path = ROOT) -> dict[str, str]:
    for pool in FILES:
        have = [size for size, _ in request_files(pool, root)]
        for size in SIZES:
            if have.count(size) < MIN_PER_SIZE[pool]:
                raise SystemExit(f"real/{pool}/{size} holds {have.count(size)} requests; "
                                 f"a run needs {MIN_PER_SIZE[pool]} (MIN_PER_SIZE)")
    return {name: "".join(json.dumps(r, ensure_ascii=False, sort_keys=True) + "\n"
                          for r in rows(pool, root))
            for pool, name in FILES.items()}


def check(out_dir: Path, root: Path = ROOT) -> list[str]:
    """The files in out_dir that are not a fresh build."""
    return [name for name, text in build(root).items()
            if not (out_dir / name).is_file()
            or (out_dir / name).read_text(encoding="utf-8") != text]


def _tokenizer():
    import tokenizer
    spec = os.environ.get("GEMMA_TOKENIZER_JSON", "")
    if not spec:
        raise SystemExit("count needs GEMMA_TOKENIZER_JSON=/path/to/tokenizer.json")
    return tokenizer.load_tokenizer(spec)


def cmd_count(paths: list[str]) -> int:
    tok, bad = _tokenizer(), 0
    for name in paths:
        path = Path(name)
        n = tok.count(read_request(path))
        try:
            lo, hi = bounds(int(path.parent.name))
        except ValueError:
            print(f"{n:6d}  (no size directory)  {path}")
            continue
        verdict = "ok" if lo <= n <= hi else ("TOO SHORT" if n < lo else "TOO LONG")
        bad += verdict != "ok"
        print(f"{n:6d}  {verdict:9s}  want {lo}..{hi}  {path}")
    return 1 if bad else 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("count")
    c.add_argument("files", nargs="+")
    b = sub.add_parser("build")
    b.add_argument("--out", default=str(HERE))
    b.add_argument("--check", action="store_true")
    args = ap.parse_args()
    if args.cmd == "count":
        return cmd_count(args.files)
    out_dir = Path(args.out)
    if args.check:
        differing = check(out_dir)
        for name in differing:
            print(f"differs from a fresh build: {name}")
        return 1 if differing else 0
    for name, text in build().items():
        (out_dir / name).write_text(text, encoding="utf-8")
        print(f"wrote {out_dir / name} ({text.count(chr(10))} requests)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
