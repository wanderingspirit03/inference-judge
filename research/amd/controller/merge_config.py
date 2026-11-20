#!/usr/bin/env python3
"""merge_config.py DEFAULTS TARGET: write TARGET from DEFAULTS when it does not
exist; otherwise add only the keys TARGET lacks.  A value already in TARGET
(the user's spending cap, for one) is never changed.  Some dicts are one value
as a whole (WHOLE): judge.draw's buckets are the operator's, and a default
bucket the operator left out is not added back."""
import json
import sys
from pathlib import Path

WHOLE = {("judge", "draw")}


def merge(dst: dict, src: dict, path: tuple = ()) -> int:
    n = 0
    for k, v in src.items():
        if k not in dst:
            dst[k] = v
            n += 1
        elif isinstance(v, dict) and isinstance(dst[k], dict) and path + (k,) not in WHOLE:
            n += merge(dst[k], v, path + (k,))
    return n


def main(a) -> int:
    src = json.loads(Path(a[0]).read_text())
    t = Path(a[1])
    if not t.exists():
        t.write_text(json.dumps(src, indent=1) + "\n")
        print(f"[install] {t} written")
        return 0
    dst = json.loads(t.read_text())
    n = merge(dst, src)
    if n:
        t.write_text(json.dumps(dst, indent=1) + "\n")
        print(f"[install] {t}: {n} missing key(s) added; existing values kept")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
