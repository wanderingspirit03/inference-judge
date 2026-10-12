"""The TPS metric definitions (DESIGN 2.2): validity, level summary, headline,
n-gram sanity. Pure functions over tps_records rows."""
from __future__ import annotations

import pytest

import metrics

OSL = 16
SLO = {"p99_tpot_ms": 50, "p95_ttft_ms": 2000}


def row(t_send=1.0, ttft=0.1, tpot_ms=10.0, tokens=OSL, usage=OSL,
        finish="length", error=None, prompt_id="p", **extra) -> dict:
    t_first = t_send + ttft
    t_last = t_first + tpot_ms * (tokens - 1) / 1000.0
    r = {"req_id": "r", "prompt_id": prompt_id, "load": "closed", "level": 4,
         "t_send": t_send, "t_first": t_first, "t_last": t_last,
         "tokens_client": tokens, "tokens_usage": usage,
         "finish_reason": finish, "itl_ms": [tpot_ms] * (tokens - 1),
         "error_kind": error, "distinct4": 1.0, "compression": 1.5}
    r.update(extra)
    reason = metrics.tps_invalid_reason(r, OSL)
    r["ok"], r["invalid_reason"] = reason is None, reason
    return r


# --------------------------------------------------------------------------
# validity: the client decides, the server's own account is never trusted
# --------------------------------------------------------------------------
def test_an_honest_request_is_valid():
    assert metrics.tps_invalid_reason(row(), OSL) is None


@pytest.mark.parametrize("kw, reason", [
    ({"error": "http_500"}, "error"),
    ({"finish": "stop"}, "finish_reason"),
    ({"tokens": OSL // 2, "usage": OSL // 2}, "length"),
    ({"tokens": OSL // 2, "usage": OSL}, "length"),          # lie_usage
    ({"tokens": OSL + 1, "usage": OSL + 1}, "length"),
])
def test_invalid_requests_are_named(kw, reason):
    assert metrics.tps_invalid_reason(row(**kw), OSL).startswith(reason)


def test_usage_disagreeing_with_the_client_count_is_invalid():
    # the text is the right length, but the server claims something else
    r = row(tokens=OSL, usage=OSL + 2)
    assert metrics.tps_invalid_reason(r, OSL).startswith("usage_mismatch")


def test_usage_within_one_percent_is_fine_and_missing_usage_is_not_trusted():
    big = 512
    r = row(tokens=big, usage=big + 5)
    assert metrics.tps_invalid_reason(r, big) is None
    r = row(tokens=big, usage=big + 6)
    assert metrics.tps_invalid_reason(r, big).startswith("usage_mismatch")
    # no usage at all: the client count stands on its own
    assert metrics.tps_invalid_reason(row(usage=None), OSL) is None


def test_usage_never_rescues_a_short_reply():
    """Server usage must never be the count: a reply the client counts short
    is invalid however many tokens the server says it sent."""
    r = row(tokens=OSL - 1, usage=OSL)
    assert metrics.tps_invalid_reason(r, OSL) is not None


def test_a_length_tolerance_is_only_used_when_asked():
    r = row(tokens=510, usage=510)
    assert metrics.tps_invalid_reason(r, 512).startswith("length")
    assert metrics.tps_invalid_reason(r, 512, length_tol=0.01) is None


def test_a_reused_prompt_is_invalid():
    assert metrics.tps_invalid_reason(row(prompt_reused=True), OSL) \
        == "prompt_reused"


# --------------------------------------------------------------------------
# level summary
# --------------------------------------------------------------------------
def test_throughput_counts_completions_inside_the_window_only():
    window = (10.0, 20.0)
    rows = ([row(t_send=11.0 + i * 0.5) for i in range(10)]       # in
            + [row(t_send=1.0), row(t_send=25.0)])                # ramp, drain
    s = metrics.tps_level_summary(rows, window, chips=2, slo=SLO)
    assert s["n"] == 12 and s["n_invalid"] == 0
    assert s["n_window"] == 10
    assert s["tps_per_chip"] == pytest.approx(OSL * 10 / 10.0 / 2)
    assert s["goodput_tps_per_chip"] == s["tps_per_chip"]
    assert s["p50_ttft_ms"] == pytest.approx(100.0)
    assert s["p99_tpot_ms"] == pytest.approx(10.0)
    assert s["slo_met"] is True


def test_invalid_requests_count_against_goodput_and_the_level():
    window = (0.0, 10.0)
    rows = [row(t_send=1.0 + i * 0.5) for i in range(8)]
    rows += [row(t_send=2.0, tokens=OSL // 2, usage=OSL)]           # lie
    s = metrics.tps_level_summary(rows, window, chips=1, slo=SLO)
    assert s["n_invalid"] == 1
    assert s["invalid_reasons"] == {"length": 1}
    assert s["goodput_tps_per_chip"] == pytest.approx(OSL * 8 / 10.0)
    # raw tokens/s is what the client counted, never usage
    assert s["tps_per_chip"] == pytest.approx((OSL * 8 + OSL // 2) / 10.0)
    assert metrics.tps_headline([s], SLO) == 0.0


def test_an_invalid_request_in_the_ramp_still_fails_the_level():
    rows = [row(t_send=5.0 + i * 0.1) for i in range(8)]
    rows.append(row(t_send=0.1, finish="stop"))
    s = metrics.tps_level_summary(rows, (4.0, 10.0), chips=1, slo=SLO)
    assert s["n_invalid"] == 1


def test_slo_is_judged_on_p99_tpot_and_p95_ttft():
    window = (0.0, 100.0)
    slow_decode = [row(t_send=1.0 + i, tpot_ms=60.0) for i in range(20)]
    slow_prefill = [row(t_send=1.0 + i, ttft=2.5) for i in range(20)]
    a = metrics.tps_level_summary(slow_decode, window, chips=1, slo=SLO)
    b = metrics.tps_level_summary(slow_prefill, window, chips=1, slo=SLO)
    assert a["slo_met"] is False and b["slo_met"] is False


def test_headline_is_the_best_level_that_meets_the_slo():
    window = (0.0, 10.0)
    lo = metrics.tps_level_summary([row(t_send=1.0 + i) for i in range(4)],
                                   window, 1, SLO)
    hi = metrics.tps_level_summary([row(t_send=1.0 + i * 0.1)
                                    for i in range(40)], window, 1, SLO)
    over = metrics.tps_level_summary([row(t_send=1.0 + i * 0.05, tpot_ms=80)
                                      for i in range(80)], window, 1, SLO)
    assert over["goodput_tps_per_chip"] > hi["goodput_tps_per_chip"]
    assert metrics.tps_headline([lo, hi, over], SLO) == \
        hi["goodput_tps_per_chip"]
    assert metrics.tps_headline([], SLO) == 0.0


def test_headline_ignores_levels_outside_the_sanity_band():
    window = (0.0, 10.0)
    junk = [row(t_send=1.0 + i * 0.1, distinct4=0.02, compression=9.0)
            for i in range(40)]
    s = metrics.tps_level_summary(junk, window, 1, SLO)
    assert metrics.tps_headline([s], SLO) > 0                   # no band given
    band = dict(SLO, min_distinct4=0.5, max_compression=4.0)
    assert metrics.tps_headline([s], band) == 0.0


# --------------------------------------------------------------------------
# n-gram sanity
# --------------------------------------------------------------------------
def test_ngram_sanity_separates_text_from_junk():
    text = ["the quick brown fox jumps over the lazy dog while the cat "
            "watches from a warm and sunny window sill"] * 3
    junk = ["a " * 64]
    good = metrics.ngram_sanity(text)
    bad = metrics.ngram_sanity(junk)
    assert good["n"] == 3
    assert good["distinct4"] == pytest.approx(1.0)
    assert bad["distinct4"] < 0.05
    assert bad["compression"] > 2 * good["compression"]
    assert metrics.ngram_sanity([])["n"] == 0


# --------------------------------------------------------------------------
# one stream's speed, and a mix of input sizes
# --------------------------------------------------------------------------
def test_a_level_reports_one_streams_speed_with_and_without_the_prefill():
    """decode_tps is 1000 / TPOT; stream_tps also pays for the first token.
    Neither is divided by chips and neither moves the goodput headline."""
    rows = [row(t_send=1.0 + i, ttft=0.2, tpot_ms=10.0) for i in range(4)]
    s = metrics.tps_level_summary(rows, (0.0, 10.0), chips=4, slo=SLO)
    assert s["decode_tps"] == pytest.approx(100.0)
    assert s["stream_tps"] == pytest.approx(OSL / (0.2 + 0.15))
    assert "by_isl" not in s and "stream_tps_per_chip" not in s
    assert metrics.tps_headline([s], SLO) == s["goodput_tps_per_chip"]


def mix_rows(tpot_by_isl: dict, per_size: int = 3, ttft=0.1, level=1) -> list[dict]:
    rows, t = [], 1.0
    for _ in range(per_size):
        for isl, tpot in tpot_by_isl.items():
            rows.append(row(t_send=t, ttft=ttft, tpot_ms=tpot, isl=isl, level=level))
            t += 1.0
    return rows


def test_a_sized_mix_is_scored_on_whole_answer_speed_balanced_over_the_sizes():
    """With a list of sizes the level's value is the streams' whole-answer
    speed per chip (request sent to last token), each size weighing the same
    however many requests the window cut gave it, and every size is reported
    on its own with its decode speed and TTFT beside it."""
    rows = mix_rows({128: 8.0, 8000: 10.0})
    rows.append(row(t_send=20.0, tpot_ms=8.0, isl=128, level=1))   # one extra 128
    s = metrics.tps_level_summary(rows, (0.0, 30.0), chips=4, slo=SLO,
                                  sizes=[128, 8000])
    assert s["by_isl"]["128"]["decode_tps"] == pytest.approx(125.0)
    assert s["by_isl"]["8000"]["decode_tps"] == pytest.approx(100.0)
    assert s["by_isl"]["128"]["n"] == 4 and s["by_isl"]["8000"]["n"] == 3
    assert s["by_isl"]["8000"]["p50_ttft_ms"] == pytest.approx(100.0)
    assert s["decode_tps"] == pytest.approx(1000.0 / 9.0)          # mean TPOT 9 ms
    # 16 tokens in 100 ms + 15 gaps: 0.22 s at 128, 0.25 s at 8000
    assert s["by_isl"]["128"]["stream_tps"] == pytest.approx(OSL / 0.22)
    assert s["by_isl"]["8000"]["stream_tps"] == pytest.approx(OSL / 0.25)
    assert s["stream_tps"] == pytest.approx(2 * OSL / 0.47)
    assert s["stream_tps_per_chip"] == pytest.approx(2 * OSL / 0.47 / 4)
    assert "decode_tps_per_chip" not in s
    assert metrics.tps_level_value(s) == s["stream_tps_per_chip"]
    assert metrics.tps_headline([s], SLO) == pytest.approx(s["stream_tps_per_chip"])
    assert metrics.tps_headline_level([s], SLO) is s


def test_a_mix_that_missed_a_size_or_broke_the_slo_has_no_headline():
    missed = metrics.tps_level_summary(mix_rows({128: 8.0}), (0.0, 30.0), 1, SLO,
                                       sizes=[128, 8000])
    assert missed["by_isl"]["8000"]["n"] == 0
    assert missed["stream_tps_per_chip"] == 0.0
    assert metrics.tps_headline([missed], SLO) == 0.0
    slow = metrics.tps_level_summary(mix_rows({128: 8.0, 8000: 80.0}), (0.0, 30.0),
                                     1, SLO, sizes=[128, 8000])
    assert slow["slo_met"] is False and metrics.tps_headline([slow], SLO) == 0.0
    assert metrics.tps_headline_level([slow], SLO) is None


def test_a_mix_is_held_to_the_fixed_ttft_ceiling():
    """The one rule on the first token of a sized mix: the target's fixed
    ceiling, p95 over every request of the level (so, with equal counts, the
    largest size's). Over it the level has no headline, however fast the
    whole answer was."""
    late = mix_rows({128: 2.0, 8000: 2.0}, per_size=10, ttft=2.5)
    s = metrics.tps_level_summary(late, (0.0, 30.0), 1, SLO, sizes=[128, 8000])
    assert s["stream_tps_per_chip"] > 0 and s["p95_ttft_ms"] == pytest.approx(2500.0)
    assert s["slo_met"] is False and metrics.tps_headline([s], SLO) == 0.0
    # only the largest size is late: its share of the requests is over 5%
    rows = (mix_rows({128: 2.0}, per_size=10, ttft=0.05)
            + mix_rows({8000: 2.0}, per_size=10, ttft=2.5))
    s = metrics.tps_level_summary(rows, (0.0, 30.0), 1, SLO, sizes=[128, 8000])
    assert s["by_isl"]["128"]["p95_ttft_ms"] == pytest.approx(50.0)
    assert s["slo_met"] is False and metrics.tps_headline([s], SLO) == 0.0
    ok = metrics.tps_level_summary(mix_rows({128: 2.0, 8000: 2.0}, ttft=1.9),
                                   (0.0, 30.0), 1, SLO, sizes=[128, 8000])
    assert ok["slo_met"] is True and metrics.tps_headline([ok], SLO) > 0


def test_tokens_in_bursts_are_judged_on_the_whole_answer():
    """A server that emits its tokens eight at a time: the first token comes
    later, the request's mean TPOT (what the SLO reads) is no worse, and the
    score moves only with when the last token arrived."""
    honest = metrics.tps_level_summary(mix_rows({128: 10.0}), (0.0, 30.0), 1, SLO,
                                       sizes=[128])
    # same answer, 20% sooner, in two bursts of eight: one long gap, first
    # token 56 ms later
    t_total = 0.8 * (0.1 + 0.15)
    bursty = [row(t_send=1.0 + i, ttft=0.156, tpot_ms=1000.0 * (t_total - 0.156) / 15,
                  isl=128, level=1, itl_ms=[0.0] * 7 + [44.0] + [0.0] * 7)
              for i in range(3)]
    s = metrics.tps_level_summary(bursty, (0.0, 30.0), 1, SLO, sizes=[128])
    assert s["slo_met"] is True
    assert s["by_isl"]["128"]["p50_ttft_ms"] == pytest.approx(156.0)
    assert metrics.tps_headline([s], SLO) == pytest.approx(
        1.25 * metrics.tps_headline([honest], SLO))


def test_holding_the_first_tokens_back_buys_nothing():
    """The gaming route a decode-only number has: emit the first K tokens late
    and in one piece. Decode speed goes up and the size's TTFT carries the
    delay; the whole-answer speed, which is the score, does not move."""
    honest = metrics.tps_level_summary(mix_rows({128: 10.0}), (0.0, 30.0), 1, SLO,
                                       sizes=[128])
    held = [row(t_send=1.0 + i, ttft=0.1 + 0.05, tpot_ms=10.0 * 10 / 15, isl=128,
                level=1) for i in range(3)]     # same end time, first token 50 ms late
    cheat = metrics.tps_level_summary(held, (0.0, 30.0), 1, SLO, sizes=[128])
    assert cheat["decode_tps"] > 1.4 * honest["decode_tps"]
    assert cheat["stream_tps"] == pytest.approx(honest["stream_tps"], rel=1e-6)
    assert cheat["by_isl"]["128"]["p50_ttft_ms"] == pytest.approx(150.0)
    assert metrics.tps_headline([cheat], SLO) == pytest.approx(
        metrics.tps_headline([honest], SLO), rel=1e-6)
