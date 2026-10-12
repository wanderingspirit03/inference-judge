#!/usr/bin/env python3
"""The table of one run_scorecard.py result.

    python scorecard/report.py [scorecard/results/latest.json]

Columns: row | tried against the judge | the judge must | the judge did |
result | evidence. `result` is PASS, FAIL, ERROR (the scenario crashed) or
SKIP (it could not run here, for example without Docker). One line per row;
a cell that does not fit is cut and ends in "~" (the JSON has it in full).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

HEAD = ["row", "tried against the judge", "the judge must", "the judge did",
        "result", "evidence"]
WIDTHS = (4, 64, 46, 64, 6, 10)
BAD = ("FAIL", "ERROR")


def verdict(row: dict) -> str:
    if row["passed"] is True:
        return "PASS"
    if row["passed"] is False:
        return "FAIL"
    return "ERROR" if row["observed"].startswith("ERROR") else "SKIP"


def summary(doc: dict) -> dict:
    counts: dict[str, int] = {}
    for r in doc["outcomes"]:
        counts[verdict(r)] = counts.get(verdict(r), 0) + 1
    return dict(sorted(counts.items()))


def table(doc: dict) -> str:
    rows = [HEAD] + [[r["scenario"], r["tried"], r["expected"], r["observed"],
                      verdict(r), r["tier"]] for r in doc["outcomes"]]
    rows = [[c if len(c) <= w else c[:w - 1] + "~"
             for c, w in zip((" ".join(str(c).split()) for c in r), WIDTHS)] for r in rows]
    widths = [max(len(r[i]) for r in rows) for i in range(len(HEAD))]
    fmt = lambda r: " | ".join(c.ljust(widths[i]) for i, c in enumerate(r)).rstrip()  # noqa: E731
    counts = summary(doc)
    return "\n".join([fmt(rows[0]), "-+-".join("-" * w for w in widths)]
                     + [fmt(r) for r in rows[1:]]
                     + ["", f"{len(doc['outcomes'])} rows: "
                        + ", ".join(f"{v} {k}" for k, v in counts.items())])


def main(argv: list[str]) -> int:
    if len(argv) > 1:
        print(__doc__, file=sys.stderr)
        return 2
    path = Path(argv[0]) if argv else Path(__file__).resolve().parent / "results" / "latest.json"
    doc = json.loads(path.read_text())
    print(table(doc))
    return 1 if any(k in BAD for k in summary(doc)) else 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
