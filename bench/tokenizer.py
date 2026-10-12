#!/usr/bin/env python3
"""The frozen tokenizer the ruler counts output tokens with.

A tokens/s number is only as honest as its token count, and the server under
test does not get to supply it: its `usage` block is recorded and cross-checked
but never trusted. The harness re-tokenises the text it
actually received with one of these and judges the request on that count.

Two adapters behind one small interface:

  ws          whitespace tokens. The mock server emits exactly one whitespace
              token per output token, so on the mock this count is exact.
  PATH        a Hugging Face tokenizer.json, loaded with the `tokenizers`
              package. Optional: the package is only imported when a path is
              given, so the bench runs without it on a laptop.

Interface (all a caller needs):
  load_tokenizer(spec) -> Tokenizer
  tok.tokens(text) -> list[str]   the tokens of `text`, no special tokens
  tok.count(text) -> int          len(tok.tokens(text))
  tok.fit(text, n) -> str         `text` cut or padded to exactly n tokens
  tok.name, tok.fingerprint()     what to stamp into a summary (sha256)
"""
from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Protocol


class Tokenizer(Protocol):
    name: str

    def tokens(self, text: str) -> list[str]: ...

    def count(self, text: str) -> int: ...

    def fit(self, text: str, n: int) -> str: ...

    def fingerprint(self) -> str: ...


class WhitespaceTokenizer:
    """One token per whitespace-separated word."""

    name = "ws"

    def tokens(self, text: str) -> list[str]:
        return text.split()

    def count(self, text: str) -> int:
        return len(text.split())

    def fit(self, text: str, n: int) -> str:
        words = text.split()
        if not words and n > 0:
            words = ["word"]
        while len(words) < n:              # pad by repeating the text itself
            words = words + words
        return " ".join(words[:max(0, n)])

    def fingerprint(self) -> str:
        return hashlib.sha256(b"ws-tokenizer-v1").hexdigest()


class HFTokenizer:
    """A Hugging Face tokenizer.json, counted without special tokens.

    Re-tokenising decoded text does not always give back the token count the
    server generated (merges across chunk boundaries, stripped think tags), so
    callers that need an exact length with a real tokenizer should allow a
    small tolerance; the ws tokenizer needs none.
    """

    def __init__(self, path: str) -> None:
        try:
            from tokenizers import Tokenizer as _HF   # optional dependency
        except ImportError as exc:                    # pragma: no cover - env
            raise SystemExit(
                f"--tokenizer {path}: the `tokenizers` package is not "
                f"installed in this environment ({exc})") from exc
        self.path = str(path)
        self.name = self.path
        self._tok = _HF.from_file(self.path)

    def _ids(self, text: str) -> list[int]:
        return self._tok.encode(text, add_special_tokens=False).ids

    def tokens(self, text: str) -> list[str]:
        return list(self._tok.encode(text, add_special_tokens=False).tokens)

    def count(self, text: str) -> int:
        return len(self._ids(text))

    def fit(self, text: str, n: int) -> str:
        ids = self._ids(text)
        if not ids and n > 0:
            ids = self._ids("word")
        while len(ids) < n:
            ids = ids + ids
        return self._tok.decode(ids[:max(0, n)])

    def fingerprint(self) -> str:
        return hashlib.sha256(Path(self.path).read_bytes()).hexdigest()


def load_tokenizer(spec: str) -> Tokenizer:
    """`ws`, or the path to a tokenizer.json."""
    if spec in ("", "ws"):
        return WhitespaceTokenizer()
    if not Path(spec).is_file():
        raise SystemExit(f"--tokenizer {spec}: not 'ws' and not a file")
    return HFTokenizer(spec)
