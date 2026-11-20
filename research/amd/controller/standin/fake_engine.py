#!/usr/bin/env python3
"""A test engine for the stand-in GPU machine: the engine contract of
research/amd/lib/engine.py answered by toylm (the same model as the fake
reference, so it passes the answer check).  Stdlib.

  fake_engine.py <port> [--damage SIGMA] [--quantised] [--ids-only] [--slow S] [--weights FILE]
--damage adds noise to the model so the engine fails the check (3 is a
wrong engine); --quantised is a little noise (sigma 0.3): an honest engine
with smaller weights, whose greedy tokens often differ from the reference's
(agreement about 0.85) but stay in its top few with small gaps (the quality
bar's measures); --ids-only answers with choices[0].token_ids and no
logprobs; --slow sleeps S seconds per request."""
import json
import os
import sys
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

sys.path.insert(0, os.path.dirname(os.path.realpath(__file__)))
import toylm  # noqa: E402

DAMAGE, SLOW, IDS_ONLY = 0.0, 0.0, False
QUANTISED = 0.3             # --quantised: an honest engine with smaller weights, as the toy model sees it


class H(BaseHTTPRequestHandler):
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
            return self._json(200, {"data": [{"id": "toy"}]})
        self._json(404, {})

    def do_POST(self):
        b = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        if self.path.rstrip("/") != "/v1/completions" or not isinstance(b.get("prompt"), list):
            return self._json(400, {"error": "this engine takes token-id prompts on /v1/completions"})
        if SLOW:
            time.sleep(SLOW)
        gen = toylm.greedy([int(x) for x in b["prompt"]], int(b.get("max_tokens") or 16), DAMAGE,
                           ignore_eos=bool(b.get("ignore_eos")))
        k = b.get("logprobs")
        stopped = bool(gen) and gen[-1][0] == toylm.EOS and not b.get("ignore_eos")
        ch = {"index": 0, "text": toylm.decode([g for g, _ in gen]), "finish_reason": "stop" if stopped else "length"}
        if IDS_ONLY:
            ch["token_ids"] = [g for g, _ in gen]
        elif k:
            ch["logprobs"] = {"tokens": [f"token_id:{g}" for g, _ in gen],
                              "token_logprobs": [lp[g] for g, lp in gen],
                              "top_logprobs": [{f"token_id:{j}": lp[j] for j in toylm.top(lp, int(k))} for _, lp in gen]}
        self._json(200, {"choices": [ch], "usage": {"prompt_tokens": len(b["prompt"]), "completion_tokens": len(gen)}})


def main(argv):
    global DAMAGE, SLOW, IDS_ONLY
    port = int(argv[0])
    a = argv[1:]
    while a:
        k = a.pop(0)
        if k == "--damage":
            DAMAGE = float(a.pop(0))
        elif k == "--quantised":
            DAMAGE = QUANTISED
        elif k == "--ids-only":
            IDS_ONLY = True
        elif k == "--slow":
            SLOW = float(a.pop(0))
        elif k == "--weights":          # convert_and_serve.py's converted file: the model's noise comes from it
            w = json.load(open(a.pop(0)))
            if not w.get("source_sha256"):
                raise SystemExit("fake_engine: the weights file has no source")
            DAMAGE = float(w["damage"])
    ThreadingHTTPServer(("127.0.0.1", port), H).serve_forever()


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
