"""Runs INSIDE the judge's interpreter with bench/ first on sys.path (judge.py
starts it the way run.sh starts bench code). Never imported by the suite.

    probe.py <command> <in.json> <out.json>

A command reads its arguments from in.json and writes a JSON result.
"""
from __future__ import annotations

import importlib
import json
import sys
from pathlib import Path

BENCH = Path(sys.path[0]).resolve()


def need(module: str):
    """Import a bench module and refuse one that came from anywhere else
    (the venv may carry an editable install of another checkout)."""
    mod = importlib.import_module(module)
    origin = Path(mod.__file__).resolve()
    if BENCH not in origin.parents:
        raise SystemExit(f"cross-load: {module} came from {origin}, not {BENCH}")
    return mod


# ------------------------------------------------------------------ commands --
def promotions(args: dict) -> dict:
    """stats.py per simulated candidate: calibrate on the reference launches,
    abba, then a second abba + confirm. A void session (drift) is not a
    promotion. Also tallies the step that stopped each one not promoted."""
    s = need("stats")
    out, sig, stops = [], [], {}
    for sess in args["sessions"]:
        cal = s.calibrate(sess["calibration"])
        sigma_l, mde = cal["sigma_L"], cal["mde"]
        sig.append(sigma_l)
        a = s.abba(*sess["abba"], sigma_L=sigma_l, mde=mde)
        stop = None
        if a.get("void_drift"):
            stop = "abba void_drift"
        elif not a.get("passed"):
            stop = "abba below mde" if 0 < a["delta"] < mde else "abba not significant"
        else:
            b = s.abba(*sess["confirm"], sigma_L=sigma_l, mde=mde)
            if b.get("void_drift"):
                stop = "confirm void_drift"
            elif not s.confirm(b["delta"], sigma_L=sigma_l, mde=mde).get("passed"):
                stop = "confirm not passed"
        if stop:
            stops[stop] = stops.get(stop, 0) + 1
        out.append(stop is None)
    srt = sorted(sig)
    return {"promoted": out, "calibration": {
        "sigma_L_p10_p50_p90": [srt[len(srt) // 10], srt[len(srt) // 2],
                                srt[9 * len(srt) // 10]] if srt else None,
        "n_sessions": len(sig)}, "stopped_at": stops,
        "stats_version": getattr(s, "STATS_VERSION", None)}


COMMANDS = {f.__name__: f for f in (promotions,)}


def main() -> int:
    cmd, src, dst = sys.argv[1], Path(sys.argv[2]), Path(sys.argv[3])
    dst.write_text(json.dumps(COMMANDS[cmd](json.loads(src.read_text())), default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
