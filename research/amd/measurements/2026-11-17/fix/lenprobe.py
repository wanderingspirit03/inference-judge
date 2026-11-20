"""One non-streamed request of the pool row that failed the length check twice, on the patched reference.
Counts which generated token ids leave no text, to explain tokens_client < completion_tokens."""
import json
import sys
import urllib.request

from tokenizers import Tokenizer

URL, POOL, TOK = sys.argv[1], "/opt/bench/pools/qwen38-main.jsonl", "/srv/models/Qwen3.8-27B/tokenizer.json"
row = next(json.loads(l) for l in open(POOL) if json.loads(l)["id"] == "real-main-4000-40-python")
tok = Tokenizer.from_file(TOK)
import uuid
for n in range(int(sys.argv[2]) if len(sys.argv) > 2 else 1):
  head = f"Request reference: {uuid.uuid4()}.\n\n" if len(sys.argv) > 2 else ""
  body = {"model": "qwen3.8-27b", "messages": [{"role": "user", "content": head + row["prompt"]}],
          "max_tokens": 512, "temperature": 0, "ignore_eos": True, "logprobs": True, "top_logprobs": 1,
          "chat_template_kwargs": {"enable_thinking": False}}
  req = urllib.request.Request(URL + "/v1/chat/completions", json.dumps(body).encode(),
                               {"Content-Type": "application/json"})
  ans = json.load(urllib.request.urlopen(req, timeout=600))
  ch = ans["choices"][0]
  text = ch["message"].get("content") or ""
  reasoning = ch["message"].get("reasoning_content") or ""
  ids = [e["id"] for e in ch["token_logprobs"]]
  added = tok.get_added_tokens_decoder()
  specials = [(i, p, added[i].content) for p, i in enumerate(ids) if i in added and added[i].special]
  nonspecial_added = [(i, p, added[i].content) for p, i in enumerate(ids) if i in added and not added[i].special]
  recount = len(tok.encode(text, add_special_tokens=False).ids) + len(tok.encode(reasoning, add_special_tokens=False).ids)
  print(json.dumps({"completion_tokens": ans["usage"]["completion_tokens"], "token_logprobs": len(ids),
                    "client_recount_of_text": recount, "finish": ch["finish_reason"],
                    "special_ids_generated": specials, "added_nonspecial_generated": nonspecial_added,
                    "text_chars": len(text), "reasoning_chars": len(reasoning)}, indent=1))
