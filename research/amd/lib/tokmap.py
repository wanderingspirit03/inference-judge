"""Token bytes -> token id, from the model's tokenizer.json.  Stdlib only.

ATOM's chat endpoint returns each generated token as text and bytes, not as an
id.  In a byte-level BPE vocabulary every entry spells a distinct byte string,
so the bytes name the id.  Added (special) tokens are matched by their text.
"""
from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path


def _bytes_to_unicode() -> dict[int, str]:
    bs = list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1)) + list(range(ord("®"), ord("ÿ") + 1))
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return dict(zip(bs, map(chr, cs)))


class TokMap:
    def __init__(self, tokenizer_json: Path):
        doc = json.loads(Path(tokenizer_json).read_text())
        inv = {u: b for b, u in _bytes_to_unicode().items()}
        self.by_bytes: dict[bytes, int] = {}
        self.collisions = 0
        for s, i in doc["model"]["vocab"].items():
            try:
                b = bytes(inv[c] for c in s)
            except KeyError:
                b = s.encode("utf-8")
            if b in self.by_bytes:
                self.collisions += 1
                continue
            self.by_bytes[b] = int(i)
        self.special: dict[str, int] = {}
        self.special_ids: set[int] = set()  # added tokens marked special (stop and template tokens)
        for t in doc.get("added_tokens") or []:
            self.special[t["content"]] = int(t["id"])
            if t.get("special"):
                self.special_ids.add(int(t["id"]))
        self.by_id: dict[int, bytes] = {i: b for b, i in self.by_bytes.items()}
        for s, i in self.special.items():
            self.by_id[i] = s.encode("utf-8")

    def text(self, i: int) -> str:
        b = self.by_id.get(int(i))
        return f"<id {i}>" if b is None else b.decode("utf-8", "replace")

    def id_of(self, text: str | None, raw: list[int] | None) -> int:
        if text is not None and text in self.special:
            return self.special[text]
        if raw is not None:
            b = bytes(raw)
            if b in self.by_bytes:
                return self.by_bytes[b]
        if text is not None:
            b = text.encode("utf-8")
            if b in self.by_bytes:
                return self.by_bytes[b]
        raise KeyError(f"no token id for text {text!r} bytes {raw!r}")


@lru_cache(maxsize=4)
def load(model_dir: str) -> TokMap:
    return TokMap(Path(model_dir) / "tokenizer.json")
