#!/usr/bin/env python3
"""Patch ATOM's chat answer so it carries one logprob entry per generated token.

Why: ATOM attaches logprobs to visible text only. Its reasoning filter removes
the marker `<think>` (id 248068, an added token with special=false, 7 ASCII
bytes) from the text when the model generates it, and the tokens of special
ids are skipped, so those positions get no entry and the answer cannot be
placed token by token (shown on 2026-11-17 with research/amd/measurements/
2026-11-17/lpdiag.py: q-q-0001-translate, position 3, `<think>`).

What it changes: in the non-streamed chat answer, when logprobs were asked
for, each choice gets one more field, `token_logprobs`: one entry per
generated token id, in order, straight from the engine's own records (the
same records ATOM's `logprobs` field is cut from), each with `id`, `token`,
`bytes`, `logprob` and `top_logprobs` (each with `id`). Nothing else changes:
the text, `logprobs`, the sampling and the kernels are ATOM's. The field is
built by ATOM's own `candidate` helper, so its values are the ones `logprobs`
shows for the same token.

Run inside the reference image (see build-ref-image.sh). It fails when the
code it patches is not found exactly once, so a different ATOM is not patched
by guess.
"""
import sys
from pathlib import Path

import atom.entrypoints.openai.serving_chat as sc

p = Path(sc.__file__)
src = p.read_text()
MARK = "# ij: token_logprobs"
if MARK in src:
    sys.exit("already patched")
anchor = """        choice["logprobs"] = tracker.take_fields(fields)
"""
if src.count(anchor) != 1:
    sys.exit(f"patch anchor found {src.count(anchor)} times in {p}; not patching")
add = anchor + f"""        {MARK}: one entry per generated token id, straight from the engine's records
        if token_logprobs is not None:
            if len(token_logprobs) != len(token_ids):
                raise LogprobError("Missing or misaligned engine token logprobs")
            choice["token_logprobs"] = [
                _ij_token_entry(logprob_codec, t, rec, top_logprobs)
                for t, rec in zip(token_ids, token_logprobs)
            ]
"""
src = src.replace(anchor, add)
helper = f'''

{MARK}
def _ij_token_entry(codec, token, record, top):
    logprob, rank, *alts = tuple(record)[: 2 + 2 * top]
    entry = codec.candidate(token, logprob if rank <= 20 else -9999.0)
    entry["id"] = int(token)
    entry["top_logprobs"] = [
        dict(codec.candidate(t, v), id=int(t))
        for t, v in zip(alts[::2], alts[1::2])
        if t >= 0
    ]
    return entry
'''
anchor2 = "\ndef _build_chat_choice("
if src.count(anchor2) != 1:
    sys.exit(f"second anchor found {src.count(anchor2)} times; not patching")
src = src.replace(anchor2, helper + "\n" + anchor2)
p.write_text(src)
print(f"patched {p}")
