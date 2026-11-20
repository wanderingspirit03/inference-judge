#!/usr/bin/env python3
"""A fake reference engine for the stand-in GPU machine: the parts of ATOM's
HTTP interface that research/amd/lib/ref.py uses, answered by toylm.  Stdlib.

  GET  /v1/models
  POST /v1/completions       {"prompt": text, "return_token_ids"} -> prompt_token_ids, text
  POST /v1/chat/completions  {"prompt_token_ids", "logprobs", "top_logprobs"} -> logprobs.content and
                             token_logprobs (the reference image's patch); stops at the stop token
                             {"messages", "return_token_ids"} -> prompt_token_ids
  fake_ref.py serve <port> <model name>
  fake_ref.py tokenizer <dir>     write <dir>/tokenizer.json for tokmap
"""
import json
import os
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

sys.path.insert(0, os.path.dirname(os.path.realpath(__file__)))
import toylm  # noqa: E402


def _byte_chars() -> dict[int, str]:
    bs = list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1)) + list(range(ord("®"), ord("ÿ") + 1))
    cs, n = bs[:], 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return dict(zip(bs, map(chr, cs)))


def write_tokenizer(d: str) -> None:
    os.makedirs(d, exist_ok=True)
    bc = _byte_chars()
    doc = {"model": {"type": "BPE", "vocab": {bc[b]: b for b in range(256)}, "merges": []},
           "added_tokens": [{"id": toylm.EOS, "content": "<|endoftext|>", "special": True}]}
    with open(os.path.join(d, "tokenizer.json"), "w") as f:
        json.dump(doc, f)


def chat_ids(messages: list) -> list[int]:
    text = "".join(f"<{m.get('role')}>{m.get('content')}\n" for m in messages) + "<assistant>"
    return toylm.encode(text)


def entry(i: int, lp: list[float], k: int) -> dict:
    def one(j):
        return {"token": "<|endoftext|>" if j == toylm.EOS else bytes([j]).decode("utf-8", "replace"),
                "logprob": lp[j], "bytes": [] if j == toylm.EOS else [j]}
    e = one(i)
    e["top_logprobs"] = [one(j) for j in toylm.top(lp, k)]
    return e


class H(BaseHTTPRequestHandler):
    name = "fake"

    def log_message(self, *a):
        pass

    def _json(self, code, obj):
        raw = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)

    def do_GET(self):
        if self.path.rstrip("/") == "/v1/models":
            return self._json(200, {"data": [{"id": self.name}]})
        self._json(404, {})

    def do_POST(self):
        b = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        n = int(b.get("max_tokens") or 16)
        if self.path.rstrip("/") == "/v1/completions":
            ids = toylm.encode(b["prompt"])
            gen = toylm.greedy(ids, n, ignore_eos=bool(b.get("ignore_eos")))
            doc = {"choices": [{"text": toylm.decode([g for g, _ in gen]), "finish_reason": "length"}],
                   "usage": {"prompt_tokens": len(ids), "completion_tokens": len(gen)}}
            if b.get("return_token_ids"):
                doc["prompt_token_ids"] = ids
            return self._json(200, doc)
        if self.path.rstrip("/") == "/v1/chat/completions":
            if b.get("prompt_token_ids") is None:
                ids = chat_ids(b["messages"])
                gen = toylm.greedy(ids, n)
                doc = {"choices": [{"message": {"role": "assistant", "content": toylm.decode([g for g, _ in gen])},
                                    "finish_reason": "length"}],
                       "usage": {"prompt_tokens": len(ids), "completion_tokens": len(gen)}}
                if b.get("return_token_ids"):
                    doc["prompt_token_ids"] = ids
                return self._json(200, doc)
            k = int(b.get("top_logprobs") or 0)
            if k > 20:
                return self._json(400, {"error": "top_logprobs must be <= 20"})
            gen = toylm.greedy(list(b["prompt_token_ids"]), n, ignore_eos=bool(b.get("ignore_eos")))
            stopped = bool(gen) and gen[-1][0] == toylm.EOS and not b.get("ignore_eos")
            ch = {"message": {"role": "assistant", "content": toylm.decode([g for g, _ in gen])},
                  "finish_reason": "stop" if stopped else "length",
                  # as ATOM: the stop token has no entry of its own in logprobs.content
                  "logprobs": {"content": [entry(g, lp, k) for g, lp in gen if g != toylm.EOS]
                               if b.get("logprobs") else None}}
            if b.get("logprobs"):
                # as the reference image's patch (reference/atom_token_logprobs.py): every id, with ids
                ch["token_logprobs"] = [dict(entry(g, lp, k), id=g, top_logprobs=[dict(entry(j, lp, 0), id=j)
                                                                                  for j in toylm.top(lp, k)])
                                        for g, lp in gen]
            doc = {"choices": [ch], "usage": {"prompt_tokens": len(b["prompt_token_ids"]), "completion_tokens": len(gen)}}
            return self._json(200, doc)
        self._json(404, {})


def main(argv):
    if argv[:1] == ["tokenizer"]:
        write_tokenizer(argv[1])
        return 0
    if argv[:1] == ["serve"]:
        H.name = argv[2] if len(argv) > 2 else "fake"
        ThreadingHTTPServer(("127.0.0.1", int(argv[1])), H).serve_forever()
    print(__doc__)
    return 2


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
