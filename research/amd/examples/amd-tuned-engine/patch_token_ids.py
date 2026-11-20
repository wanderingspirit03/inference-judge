#!/usr/bin/env python3
"""patch_token_ids.py FILE: make ATOM's non-streaming chat answer carry the
generated token ids in choices[0]["token_ids"] when the request set
`return_token_ids` (ATOM at 352229d6 echoes only the prompt's ids, though
final_output["token_ids"] holds the answer's).  The megakernel path gives no
logprobs, so these ids are what the judge's contract reads.  Fails unless
the expected code is there exactly once (atom/entrypoints/openai/serving_chat.py,
build_chat_response)."""
import sys
from pathlib import Path

OLD = ('    if "prompt_token_ids" in final_output:\n'
       '        update["prompt_token_ids"] = final_output["prompt_token_ids"]\n')
NEW = OLD + ('        response.choices[0]["token_ids"] = [int(t) for t in final_output.get("token_ids") or []]'
             '  # ij: the answer\'s ids\n')

p = Path(sys.argv[1])
src = p.read_text()
if src.count(OLD) != 1:
    sys.exit(f"patch_token_ids: expected the code once in {p}, found it {src.count(OLD)} times")
p.write_text(src.replace(OLD, NEW))
print(f"patch_token_ids: {p} patched")
