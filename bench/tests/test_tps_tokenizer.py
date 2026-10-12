"""bench/tokenizer.py: the frozen tokenizer the TPS client counts with."""
from __future__ import annotations

import pytest

from tokenizer import WhitespaceTokenizer, load_tokenizer


def test_ws_counts_whitespace_tokens():
    tok = load_tokenizer("ws")
    assert tok.name == "ws"
    assert tok.count("one two\nthree\t four ") == 4
    assert tok.count("") == 0
    assert tok.tokens(" one two\nthree ") == ["one", "two", "three"]


def test_ws_fit_cuts_and_pads_to_exactly_n():
    tok = WhitespaceTokenizer()
    assert tok.fit("a b c d e", 3) == "a b c"
    assert tok.count(tok.fit("a b", 7)) == 7
    assert tok.count(tok.fit("", 5)) == 5
    assert tok.fit("a b", 0) == ""


def test_fingerprint_is_stable():
    assert load_tokenizer("ws").fingerprint() == \
        WhitespaceTokenizer().fingerprint()


def test_an_unknown_tokenizer_spec_is_refused(tmp_path):
    with pytest.raises(SystemExit):
        load_tokenizer(str(tmp_path / "missing.json"))


def test_hf_tokenizer_json(tmp_path):
    tokenizers = pytest.importorskip("tokenizers")
    from tokenizers import models, pre_tokenizers
    vocab = {"[UNK]": 0, "hello": 1, "world": 2, "again": 3}
    hf = tokenizers.Tokenizer(models.WordLevel(vocab, unk_token="[UNK]"))
    hf.pre_tokenizer = pre_tokenizers.Whitespace()
    path = tmp_path / "tokenizer.json"
    hf.save(str(path))
    tok = load_tokenizer(str(path))
    assert tok.count("hello world again") == 3
    assert tok.tokens("hello world again") == ["hello", "world", "again"]
    assert tok.count(tok.fit("hello world", 5)) == 5
    assert len(tok.fingerprint()) == 64
