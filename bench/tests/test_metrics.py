import metrics


def test_percentile_of_empty_sample_is_none():
    assert metrics.percentile([], 95) is None


def test_jsonl_round_trip(tmp_path):
    rows = [{"req_id": "a", "tokens_client": 64, "itl_ms": [1.5, 2.0]},
            {"req_id": "b", "tokens_client": 63, "itl_ms": []}]
    p = tmp_path / "t.jsonl"
    metrics.write_jsonl(str(p), rows)
    assert metrics.read_jsonl(str(p)) == rows
