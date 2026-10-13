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


def test_tokens_are_text_as_a_server_echoes_it_not_vocabulary_pieces(tmp_path):
    tokenizers = pytest.importorskip("tokenizers")
    from tokenizers import decoders, models, pre_tokenizers
    vocab = {"[UNK]": 0, "\u2581hello": 1, "\u2581world": 2}
    hf = tokenizers.Tokenizer(models.WordLevel(vocab, unk_token="[UNK]"))
    hf.pre_tokenizer, hf.decoder = pre_tokenizers.Metaspace(), decoders.Metaspace()
    path = tmp_path / "tokenizer.json"
    hf.save(str(path))
    tok = load_tokenizer(str(path))
    assert [t.strip() for t in tok.tokens(" hello world")] == ["hello", "world"]
    assert tok.count(" hello world") == 2



def test_plain_is_the_text_without_special_tokens(tmp_path):
    assert load_tokenizer("ws").plain("one <pad> two") == "one <pad> two"   # ws has none
    tokenizers = pytest.importorskip("tokenizers")
    from tokenizers import models, pre_tokenizers
    vocab = {"[UNK]": 0, "hello": 1, "world": 2}
    hf = tokenizers.Tokenizer(models.WordLevel(vocab, unk_token="[UNK]"))
    hf.pre_tokenizer = pre_tokenizers.Whitespace()
    hf.add_special_tokens(["<pad>"])
    path = tmp_path / "tokenizer.json"
    hf.save(str(path))
    tok = load_tokenizer(str(path))
    assert tok.count("hello world<pad><pad>") == 4       # they are tokens
    assert tok.plain("hello world<pad><pad>") == "hello world"
    assert tok.plain("hello   world") == "hello   world"  # untouched when there are none
