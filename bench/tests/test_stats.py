"""stats.py: noise calibration, the ABBA/ICI decision, confirmation, and the
A/A false-promotion simulation."""
from __future__ import annotations

import math

import numpy as np
import pytest

import stats


# --------------------------------------------------------------- calibrate --
def test_calibrate_recovers_launch_and_block_noise():
    rng = np.random.default_rng(1)
    launch_sd, block_sd = 0.02, 0.03
    launches = []
    for _ in range(400):
        level = rng.normal(0, launch_sd)
        launches.append(list(level + rng.normal(0, block_sd, 3)))
    cal = stats.calibrate(launches)
    assert cal["n_launches"] == 400
    assert abs(cal["sigma_B"] - block_sd) < 0.003
    # sigma_L is the SD of a launch's headline (its block mean): total, not
    # the launch component with block noise subtracted.
    want = math.sqrt(launch_sd ** 2 + block_sd ** 2 / 3)
    assert abs(cal["sigma_L"] - want) < 0.003
    assert cal["mde"] == pytest.approx(max(0.02, 3.2 * cal["sigma_L"]))


def test_mde_has_a_two_percent_floor():
    quiet = [[0.0, 0.0001], [0.0002, 0.0001], [0.0001, 0.0]]
    assert stats.calibrate(quiet)["mde"] == pytest.approx(0.02)


def test_calibrate_needs_two_launches():
    with pytest.raises(ValueError):
        stats.calibrate([[1.0, 1.1, 1.2]])


# -------------------------------------------------------------------- abba --
def test_abba_cancels_linear_drift():
    drift, effect = 0.01, 0.05
    out = stats.abba(0.0, 1 * drift + effect, 2 * drift + effect, 3 * drift,
                     sigma_L=0.01, mde=0.02)
    assert out["delta"] == pytest.approx(effect)
    assert out["se"] == pytest.approx(0.01)
    assert out["passed"] is True
    assert out["void_drift"] is False


def test_abba_takes_per_block_values():
    out = stats.abba([0.0, 0.0, 0.0], [0.1, 0.1], [0.1], [0.0],
                     sigma_L=0.01, mde=0.02)
    assert out["delta"] == pytest.approx(0.1)


def test_abba_requires_the_minimum_effect():
    # z-test passes (0.019 > 2.33 * 0.005) but the effect is under the MDE.
    out = stats.abba(0.0, 0.019, 0.019, 0.0, sigma_L=0.005, mde=0.02)
    assert out["passed"] is False
    assert any("mde" in r for r in out["reasons"])


def test_abba_requires_significance():
    out = stats.abba(0.0, 0.03, 0.03, 0.0, sigma_L=0.02, mde=0.02)
    assert out["passed"] is False            # 0.03 - 2.33 * 0.02 < 0
    assert any("significant" in r for r in out["reasons"])


def test_void_drift_triggers_above_three_root_two_sigma():
    sigma = 0.01
    edge = 3 * math.sqrt(2) * sigma
    just_inside = stats.abba(0.0, 0.2, 0.2, edge * 0.99, sigma_L=sigma, mde=0.02)
    just_outside = stats.abba(0.0, 0.2, 0.2, edge * 1.01, sigma_L=sigma, mde=0.02)
    assert just_inside["void_drift"] is False
    assert just_outside["void_drift"] is True
    assert just_outside["passed"] is False
    assert any("drift" in r for r in just_outside["reasons"])
    # the sign of the incumbent's move does not matter
    assert stats.abba(edge * 1.01, 0.2, 0.2, 0.0, sigma_L=sigma, mde=0.02)["void_drift"]


# --------------------------------------------------------------------- ici --
def test_ici_has_a_larger_standard_error_and_cancels_drift():
    out = stats.ici(0.0, 0.01 + 0.08, 0.02, sigma_L=0.01, mde=0.02)
    assert out["delta"] == pytest.approx(0.08)
    assert out["se"] == pytest.approx(0.01 * math.sqrt(1.5))
    assert out["passed"] is True
    assert set(out) >= {"delta", "se", "z", "passed", "void_drift", "reasons"}


def test_ici_void_drift():
    out = stats.ici(0.0, 0.2, 0.05, sigma_L=0.01, mde=0.02)
    assert out["void_drift"] is True and out["passed"] is False


# ----------------------------------------------------------------- confirm --
def test_confirm_needs_significance_and_half_the_mde():
    assert stats.confirm(0.05, sigma_L=0.01, mde=0.06)["passed"] is True
    assert stats.confirm(0.025, sigma_L=0.01, mde=0.06)["passed"] is False  # < mde/2
    assert stats.confirm(0.035, sigma_L=0.03, mde=0.06)["passed"] is False  # not significant
    assert stats.confirm(0.05, sigma_L=0.01, mde=0.06)["delta"] == 0.05


def test_confirm_accepts_the_ici_standard_error():
    se = 0.02 * math.sqrt(1.5)
    assert stats.confirm(0.035, sigma_L=0.02, mde=0.04)["passed"] is True
    assert stats.confirm(0.035, sigma_L=0.02, mde=0.04, se=se)["passed"] is False


# ------------------------------------------------------------------ screen --
def test_screen_works_in_log_space():
    sentinel = math.log(1000.0)
    assert stats.screen(math.log(985.0), sentinel) is True      # -1.5%
    assert stats.screen(math.log(975.0), sentinel) is False     # -2.5%
    assert stats.screen(math.log(975.0), sentinel, tol=0.03) is True


# -------------------------------------------------- A/A false promotions ----
N_SIMS = 200_000


def test_v2_false_promotion_rate_is_below_one_in_a_thousand():
    rate = stats.aa_false_promotion_rate(
        N_SIMS, stats.DEFAULT_SIGMA_L, stats.DEFAULT_SIGMA_B,
        stats.DEFAULT_DRIFT_PER_HOUR, "v2_abba_confirm", seed=11)
    assert rate <= 1e-3, rate


def test_v1_false_promotion_rate_is_materially_higher():
    v1 = stats.aa_false_promotion_rate(
        N_SIMS, stats.DEFAULT_SIGMA_L, stats.DEFAULT_SIGMA_B,
        stats.DEFAULT_DRIFT_PER_HOUR, "v1_ci_low", seed=11)
    v2 = stats.aa_false_promotion_rate(
        N_SIMS, stats.DEFAULT_SIGMA_L, stats.DEFAULT_SIGMA_B,
        stats.DEFAULT_DRIFT_PER_HOUR, "v2_abba_confirm", seed=11)
    assert v1 >= 0.025, v1          # at least the nominal one-sided 2.5%
    assert v1 >= 20 * max(v2, 1 / N_SIMS)


def test_v1_gets_worse_with_drift_and_v2_does_not():
    args = (N_SIMS, stats.DEFAULT_SIGMA_L, stats.DEFAULT_SIGMA_B)
    v1_flat = stats.aa_false_promotion_rate(*args, 0.0, "v1_ci_low", seed=3)
    v1_drift = stats.aa_false_promotion_rate(*args, 0.01, "v1_ci_low", seed=3)
    v2_drift = stats.aa_false_promotion_rate(*args, 0.01, "v2_abba_confirm", seed=3)
    assert v1_drift > 2 * v1_flat
    assert v2_drift <= 1e-3


def test_estimated_sigma_costs_false_promotions_and_ten_launches_buy_it_back():
    """sigma_L estimated from calibration is noisy; an underestimate shrinks the
    MDE. Measured (1e6 sims): ABBA+confirm 2.3e-5 with sigma_L known, 1.5e-3
    from 6 calibration launches, 6.7e-4 from 10. So 6 launches
    miss the 1e-3 budget; 10 meet it."""
    def rate(k):
        return stats.aa_false_promotion_rate(
            N_SIMS, stats.DEFAULT_SIGMA_L, stats.DEFAULT_SIGMA_B,
            stats.DEFAULT_DRIFT_PER_HOUR, "v2_abba_confirm",
            calib_launches=k, seed=5)
    six, ten = rate(6), rate(10)
    assert 1e-3 < six <= 3e-3, six
    assert ten <= 1e-3, ten


def test_power_at_twice_the_mde():
    mde = max(0.02, 3.2 * stats.DEFAULT_SIGMA_L)
    for protocol in ("v2_abba_confirm", "v2_ici_confirm"):
        power = stats.aa_false_promotion_rate(
            20_000, stats.DEFAULT_SIGMA_L, stats.DEFAULT_SIGMA_B,
            stats.DEFAULT_DRIFT_PER_HOUR, protocol, true_delta=2 * mde, seed=2)
        assert power >= 0.8, (protocol, power)


def test_power_at_the_floor_mde_on_a_quiet_box():
    # sigma_L 0.005 -> MDE is the 2% floor; a 4% win must be found.
    power = stats.aa_false_promotion_rate(
        20_000, 0.005, 0.006, stats.DEFAULT_DRIFT_PER_HOUR, "v2_abba_confirm",
        true_delta=0.04, seed=2)
    assert power >= 0.8, power


def test_simulation_is_seeded():
    a = stats.aa_false_promotion_rate(5000, 0.02, 0.03, 0.002, "v1_ci_low", seed=9)
    b = stats.aa_false_promotion_rate(5000, 0.02, 0.03, 0.002, "v1_ci_low", seed=9)
    assert a == b


def test_unknown_protocol_is_refused():
    with pytest.raises(ValueError):
        stats.aa_false_promotion_rate(10, 0.02, 0.03, 0.0, "v3_vibes")
