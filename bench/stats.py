#!/usr/bin/env python3
"""Promotion statistics.

Pure functions over log-throughput values: higher is better. A latency caller
passes -log(p95). Every input that is a launch's headline may be a float or a
list of that launch's per-block values (the block mean is used).

Speed is judged against the incumbent, never against stored records: the
incumbent is re-launched next to the candidate so linear drift cancels.

  calibrate(launches)                 Step 0: sigma_L, sigma_B, MDE from R launches
  screen(cand, sentinel, tol)         Step 1: liberal screen against the sentinel
  abba(i1, c1, c2, i2, sigma_L, mde)  Step 2: I C C I, drift-cancelling, z-test
  ici(i1, c, i2, sigma_L, mde)        Step 2, cheaper: I C I (when sigma_L ~ sigma_B)
  confirm(delta', sigma_L, mde)       Step 3: held-out confirmation; the
                                      leaderboard records delta', not delta
  aa_false_promotion_rate(...)        the simulation the tests use

Definitions (one meaning everywhere):
  sigma_B  SD of a block's value around its launch mean (within a launch).
  sigma_L  SD of a launch's headline, i.e. of its block mean, between cold
           launches. It is the total: block noise averaged over the blocks is
           inside it. It is exactly the SE that ABBA's delta has
           (Var(delta) = 1/4 * 4 * sigma_L^2), the scale of the drift check
           |I1 - I2| > 3 * sqrt(2) * sigma_L, and the MDE's unit.
           run.sh's ABBA sends I1 and I2 different prompts (I C pairs share
           theirs), so its calibration gives every launch its own draw: the
           sigma_L it measures holds prompt set against prompt set as the
           drift check needs. For delta, where that part cancels inside each
           pair, it is then an upper bound: the test is no laxer for it.
  MDE      max(2%, 3.2 * sigma_L) in log units: z_0.01 + z_0.2 = 2.33 + 0.84,
           i.e. 80% power for the z-test alone. Because a pass also needs
           delta >= MDE, power at a true effect of exactly 1 MDE is ~50%; at
           2 MDE it is ~1 (tests/test_stats.py).

Default noise parameters. They come from repeated launches of one
configuration on a GPU server, measured as a p95 latency per block:

  * Three byte-identical launches about an hour apart: the pooled SD of a
    block's log value inside a launch was 0.027 at the headline load (0.040
    at twice the load, 0.075 at a quarter of it).
    -> DEFAULT_SIGMA_B = 0.03 (headline load, rounded up).
  * The SD of those three launches' means was 0.016 (2 df, so roughly +-50%
    relative error); two draws of another config on a loaded machine differed
    by 6.6%, a launch SD near 0.047.
    -> DEFAULT_SIGMA_L = 0.02, a middle value between 0.016 and 0.047.
  * The same byte-identical config measured 2.9% slower 16.5 h later.
    -> DEFAULT_DRIFT_PER_HOUR = 0.002 (0.0018 rounded up).
  * A stored incumbent record was typically several hours old when a
    candidate was judged against it (the leaderboard moved five times in
    16 h).
    -> DEFAULT_HOURS_SINCE_INCUMBENT = 6.
  * A bootstrap CI over one run's samples sees only within-run noise. Between
    identical draws its half-width was ~5%, so the SE it used for a
    difference is ~0.025 = sigma_B * sqrt(2 / n_blocks) at the defaults; the
    simulator uses that expression for the naive protocol (v1 below).

Measured with these defaults (tests/test_stats.py, 1e6 sims): A/A false
promotions are 0.10 under v1 (0.044 with no drift) against 2.3e-5 for
ABBA + confirmation when sigma_L is known, 1.5e-3 when sigma_L is estimated
from 6 calibration launches and 6.7e-4 from 10. An underestimated sigma_L
shrinks the MDE, so Step 0 needs about 10 launches to hold 1e-3.

These are latency numbers from one GPU server; throughput on other hardware
will have its own noise. They are the defaults for tests and the scorecard only. Real
decisions use calibrate() on the target's own Step-0 launches.
"""
from __future__ import annotations

import math
from statistics import NormalDist
from typing import Sequence, Union

import numpy as np

STATS_VERSION = "2.0"

MDE_FLOOR = 0.02            # no threshold may sit below 2%
MDE_SIGMAS = 3.2            # z_0.01 + z_0.2
DRIFT_SIGMAS = 3.0          # void when |I1 - I2| > 3 * sqrt(2) * sigma_L
CONFIRM_MDE_FRACTION = 0.5  # Step 3 needs delta' >= MDE / 2

DEFAULT_SIGMA_L = 0.02
DEFAULT_SIGMA_B = 0.03
DEFAULT_DRIFT_PER_HOUR = 0.002
DEFAULT_HOURS_SINCE_INCUMBENT = 6.0
DEFAULT_N_BLOCKS = 3
DEFAULT_LAUNCH_SPACING_H = 50 / 60   # compile + warmup sweep + sweep
DEFAULT_CONFIRM_GAP_H = 6.0          # Step 3 runs at least 6 h after Step 2

PROTOCOLS = ("v1_ci_low", "v2_abba_confirm", "v2_ici_confirm")

Launch = Union[float, Sequence[float]]


def _z(alpha: float) -> float:
    return NormalDist().inv_cdf(1.0 - alpha)


def _value(x: Launch) -> float:
    if isinstance(x, (int, float)):
        return float(x)
    vals = [float(v) for v in x]
    if not vals:
        raise ValueError("a launch needs at least one block value")
    return sum(vals) / len(vals)


def _mde(sigma_L):
    return np.maximum(MDE_FLOOR, MDE_SIGMAS * sigma_L)


# --------------------------------------------------------------------------
# Step 0
# --------------------------------------------------------------------------
def calibrate(launches: list[list[float]]) -> dict:
    """Noise floor from >= 2 cold launches of one config (R or the sentinel).

    launches[i] is launch i's per-block log values. sigma_B pools the
    within-launch variance over launches with >= 2 blocks (None if none has).
    sigma_L is the SD of the launch means (see the module docstring).
    """
    if len(launches) < 2:
        raise ValueError("calibrate needs at least two launches")
    blocks = [np.asarray(launch, dtype=float) for launch in launches]
    if any(b.size == 0 for b in blocks):
        raise ValueError("every launch needs at least one block value")
    means = np.array([b.mean() for b in blocks])
    sigma_L = float(np.std(means, ddof=1))
    ss = sum(float(((b - b.mean()) ** 2).sum()) for b in blocks if b.size > 1)
    df = sum(b.size - 1 for b in blocks if b.size > 1)
    sigma_B = math.sqrt(ss / df) if df else None
    return {"sigma_L": sigma_L, "sigma_B": sigma_B,
            "mde": float(_mde(sigma_L)), "n_launches": len(launches),
            "n_blocks": [int(b.size) for b in blocks],
            "stats_version": STATS_VERSION}


# --------------------------------------------------------------------------
# Steps 1-3
# --------------------------------------------------------------------------
def screen(cand: float, sentinel: float, tol: float = 0.02) -> bool:
    """Liberal Step 1: the candidate is no more than `tol` below the
    incumbent's latest sentinel value. Both are log values."""
    return bool(cand >= sentinel + math.log(1.0 - tol))


def _decide(delta, se, drift_gap, sigma_L, mde, z_crit):
    """The one decision rule, vectorised. Used by abba(), ici() and the
    simulator, so the simulated protocol is the shipped protocol."""
    void = np.abs(drift_gap) > DRIFT_SIGMAS * math.sqrt(2.0) * sigma_L
    significant = delta - z_crit * se > 0
    big_enough = delta >= mde
    return void, significant, big_enough, significant & big_enough & ~void


def _confirm_rule(delta, se, mde, z_crit):
    """Step 3's rule, vectorised: (significant, at least mde / 2)."""
    return delta - z_crit * se > 0, delta >= CONFIRM_MDE_FRACTION * mde


def _result(delta: float, se: float, drift_gap: float, sigma_L: float,
            mde: float, alpha: float, design: str) -> dict:
    if sigma_L <= 0:
        raise ValueError("sigma_L must be positive (run calibrate first)")
    z_crit = _z(alpha)
    void, sig, big, passed = (bool(v) for v in _decide(
        delta, se, drift_gap, sigma_L, mde, z_crit))
    reasons = []
    if void:
        reasons.append(f"void_drift: incumbent moved {drift_gap:+.4f} between "
                       f"its launches (> {DRIFT_SIGMAS:g}*sqrt(2)*sigma_L = "
                       f"{DRIFT_SIGMAS * math.sqrt(2) * sigma_L:.4f}); re-run, "
                       f"do not decide")
    if not sig:
        reasons.append(f"not significant: delta {delta:+.4f} - "
                       f"{z_crit:.2f}*se {se:.4f} <= 0")
    if not big:
        reasons.append(f"below mde: delta {delta:+.4f} < mde {mde:.4f}")
    return {"design": design, "delta": delta, "se": se, "z": delta / se,
            "alpha": alpha, "mde": mde, "drift_gap": drift_gap,
            "passed": passed, "void_drift": void, "reasons": reasons}


def abba(i1: Launch, c1: Launch, c2: Launch, i2: Launch, sigma_L: float,
         mde: float, alpha: float = 0.01) -> dict:
    """Step 2 on launches run in the order I, C, C, I.

    delta = 1/2 [(C1 - I1) + (C2 - I2)], SE = sigma_L (known-variance z-test:
    two launches per arm cannot estimate their own variance). Passes when not
    void, delta - z_alpha * SE > 0 and delta >= mde.
    Returns {design, delta, se, z, alpha, mde, drift_gap, passed, void_drift,
    reasons}.
    """
    vi1, vc1, vc2, vi2 = (_value(x) for x in (i1, c1, c2, i2))
    delta = 0.5 * ((vc1 - vi1) + (vc2 - vi2))
    return _result(delta, sigma_L, vi2 - vi1, sigma_L, mde, alpha, "abba")


def ici(i1: Launch, c: Launch, i2: Launch, sigma_L: float, mde: float,
        alpha: float = 0.01) -> dict:
    """Step 2, cheaper: I, C, I. delta = C - (I1 + I2)/2, SE = sigma_L*sqrt(1.5).
    Linear drift cancels when C sits midway in time. Same output as abba()."""
    vi1, vc, vi2 = (_value(x) for x in (i1, c, i2))
    delta = vc - 0.5 * (vi1 + vi2)
    return _result(delta, sigma_L * math.sqrt(1.5), vi2 - vi1, sigma_L, mde,
                   alpha, "ici")


def confirm(delta_prime: float, sigma_L: float, mde: float,
            alpha: float = 0.05, *, se: float | None = None) -> dict:
    """Step 3: a second ABBA (or ICI, pass its se) on a disjoint sealed prompt
    pool, >= 6 h later. Promote when delta' - z_alpha * se > 0 and
    delta' >= mde / 2. The caller checks that session's void_drift itself.
    The leaderboard records delta', which removes the winner's curse."""
    se = sigma_L if se is None else se
    z_crit = _z(alpha)
    sig, big = (bool(v) for v in _confirm_rule(delta_prime, se, mde, z_crit))
    reasons = []
    if not sig:
        reasons.append(f"confirmation not significant: delta' "
                       f"{delta_prime:+.4f} - {z_crit:.2f}*se {se:.4f} <= 0")
    if not big:
        reasons.append(f"confirmation below mde/2: {delta_prime:+.4f} < "
                       f"{CONFIRM_MDE_FRACTION * mde:.4f}")
    return {"passed": bool(sig and big), "delta": delta_prime, "se": se,
            "alpha": alpha, "reasons": reasons}


# --------------------------------------------------------------------------
# Simulation
# --------------------------------------------------------------------------
def aa_false_promotion_rate(n_sims: int,
                            sigma_L: float = DEFAULT_SIGMA_L,
                            sigma_B: float = DEFAULT_SIGMA_B,
                            drift_per_hour: float = DEFAULT_DRIFT_PER_HOUR,
                            protocol: str = "v2_abba_confirm", *,
                            true_delta: float = 0.0,
                            n_blocks: int = DEFAULT_N_BLOCKS,
                            hours_since_incumbent: float = DEFAULT_HOURS_SINCE_INCUMBENT,
                            launch_spacing_h: float = DEFAULT_LAUNCH_SPACING_H,
                            confirm_gap_h: float = DEFAULT_CONFIRM_GAP_H,
                            calib_launches: int | None = None,
                            seed: int = 0) -> float:
    """Fraction of simulated candidates that get promoted.

    With true_delta = 0 (the default) the candidate is the incumbent, so this
    is the A/A false-promotion rate; with true_delta > 0 it is the power.

    The model, in log units, for one launch started at hour t:
        value = drift_per_hour * t + [candidate] * true_delta + launch_effect
                + mean of n_blocks block effects
    block effects ~ N(0, sigma_B); launch_effect ~ N(0, sqrt(sigma_L^2 -
    sigma_B^2 / n_blocks)) so a launch's SD is sigma_L (floored at
    sigma_B / sqrt(n_blocks) when sigma_L is smaller than block noise allows).
    drift_per_hour is signed: positive means the box gets faster with time,
    which favours whatever runs later (the candidate, under v1).

    Protocols:
      v1_ci_low        naive: one candidate launch now against the incumbent's
                       stored launch from hours_since_incumbent ago; promote
                       iff est - 1.96 * sigma_W > 0, where sigma_W =
                       sigma_B * sqrt(2 / n_blocks) is the within-run sampling
                       SE the bootstrap saw. Only the headline rule (ci_low at
                       32 sessions) is modelled; v1's other gates are not.
      v2_abba_confirm  abba() at alpha 0.01, then a second ABBA confirm_gap_h
                       later through confirm() at alpha 0.05. Both sessions
                       must be non-void. A void session counts as "not
                       promoted": in practice it is re-run, and each re-run is
                       a fresh draw with the same per-attempt rate.
      v2_ici_confirm   the same with ici() and its larger SE.
    The screen (Step 1) is not modelled; it can only lower the rate.

    sigma_L used by the protocol is the true one, or, with calib_launches=k,
    estimated per simulated campaign from k simulated calibration launches
    (calibration drift is ignored).
    """
    if protocol not in PROTOCOLS:
        raise ValueError(f"unknown protocol {protocol!r}; one of {PROTOCOLS}")
    if n_sims <= 0:
        raise ValueError("n_sims must be positive")
    rng = np.random.default_rng(seed)
    block_var = sigma_B ** 2 / n_blocks
    launch_sd = math.sqrt(max(sigma_L ** 2 - block_var, 0.0))

    def launch(t_h: float, cand: bool) -> np.ndarray:
        return (drift_per_hour * t_h + (true_delta if cand else 0.0)
                + rng.normal(0.0, launch_sd, n_sims)
                + rng.normal(0.0, sigma_B, (n_sims, n_blocks)).mean(axis=1))

    if protocol == "v1_ci_low":
        incumbent = launch(0.0, False)
        candidate = launch(hours_since_incumbent, True)
        sigma_w = sigma_B * math.sqrt(2.0 / n_blocks)
        promoted = (candidate - incumbent) - _z(0.025) * sigma_w > 0
        return float(promoted.mean())

    if calib_launches:
        cal = np.stack([launch(0.0, False) for _ in range(calib_launches)])
        sig = cal.std(axis=0, ddof=1)
    else:
        sig = np.full(n_sims, max(sigma_L, math.sqrt(block_var)))
    mde = _mde(sig)

    def session(t0: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """One I..I session from hour t0: (delta, se, I2 - I1)."""
        s = launch_spacing_h
        if protocol == "v2_abba_confirm":
            i1, c1 = launch(t0, False), launch(t0 + s, True)
            c2, i2 = launch(t0 + 2 * s, True), launch(t0 + 3 * s, False)
            return 0.5 * ((c1 - i1) + (c2 - i2)), sig, i2 - i1
        i1, c = launch(t0, False), launch(t0 + s, True)
        i2 = launch(t0 + 2 * s, False)
        return c - 0.5 * (i1 + i2), sig * math.sqrt(1.5), i2 - i1

    z2, z3 = _z(0.01), _z(0.05)
    d1, se1, gap1 = session(hours_since_incumbent)
    step2 = _decide(d1, se1, gap1, sig, mde, z2)[3]
    d2, se2, gap2 = session(hours_since_incumbent + confirm_gap_h)
    void2 = _decide(d2, se2, gap2, sig, mde, z3)[0]
    sig3, big3 = _confirm_rule(d2, se2, mde, z3)
    return float((step2 & sig3 & big3 & ~void2).mean())
