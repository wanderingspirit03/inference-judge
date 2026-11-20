#!/usr/bin/env python3
"""shim.py <port> <atom url>: the engine contract of lib/engine.py on top of
ATOM.  /v1/completions with a token-id prompt and `logprobs` is turned into
ATOM's /v1/chat/completions with `prompt_token_ids` (no chat template), whose
token bytes are mapped back to ids.  For testing ./check and ./submit with a
real engine; not part of the commands."""
import http.client
import json
import os
import sys
import time
import urllib.parse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

sys.path.insert(0, os.path.join(os.path.dirname(os.path.realpath(__file__)), "..", "..", "lib"))
sys.path.append("/opt/ij-amd/amd/lib")  # copied into an agent's tree: the box's copy
import tokmap  # noqa: E402

PORT = int(sys.argv[1])
ATOM = urllib.parse.urlsplit(sys.argv[2])
MODEL_DIR = os.environ.get("IJ_REF_MODEL_DIR", "/srv/models/Qwen3.8-27B")
SERVED = os.environ.get("IJ_REF_NAME", "qwen3.8-27b")
TM = tokmap.load(MODEL_DIR)
# SHIM_DAMAGE=k: at every absolute position divisible by k, answer with the
# second-ranked token instead of the first (a fabricated wrong engine).
DAMAGE = int(os.environ.get("SHIM_DAMAGE") or 0)
# SHIM_IDS_ONLY=1: ask for and return the answer's token ids, no logprobs (the contract allows it).
IDS_ONLY = os.environ.get("SHIM_IDS_ONLY") == "1"


def atom(method, path, body=None, timeout=900):
    c = http.client.HTTPConnection(ATOM.hostname, ATOM.port, timeout=timeout)
    try:
        c.request(method, path, body=None if body is None else json.dumps(body).encode(),
                  headers={"Content-Type": "application/json"})
        r = c.getresponse()
        return r.status, r.read()
    finally:
        c.close()


class H(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def _send(self, code, doc):
        b = json.dumps(doc).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(b)))
        self.end_headers()
        self.wfile.write(b)

    def do_GET(self):
        if self.path != "/v1/models":
            return self._send(404, {"error": "not found"})
        try:
            st, _ = atom("GET", "/v1/models", timeout=5)
        except OSError:
            st = 503
        if st != 200:
            return self._send(503, {"error": "not ready"})
        return self._send(200, {"object": "list", "data": [{"id": SERVED, "object": "model"}]})

    def do_POST(self):
        if self.path != "/v1/completions":
            return self._send(404, {"error": "not found"})
        req = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))))
        ids = req.get("prompt")
        if not (isinstance(ids, list) and ids and all(isinstance(x, int) for x in ids)):
            return self._send(400, {"error": "prompt must be a list of token ids"})
        k = req.get("logprobs")
        body = {"model": SERVED, "messages": [{"role": "user", "content": "-"}], "prompt_token_ids": ids,
                "max_tokens": int(req.get("max_tokens") or 16), "temperature": 0}
        if k and IDS_ONLY:
            body["return_token_ids"] = True
        elif k:
            body.update({"logprobs": True, "top_logprobs": int(k)})
        st, raw = atom("POST", "/v1/chat/completions", body)
        if st != 200:
            return self._send(st, {"error": raw[:300].decode("utf-8", "replace")})
        doc = json.loads(raw)
        ch = doc["choices"][0]
        if not k:
            n = (doc.get("usage") or {}).get("completion_tokens", 0)
            return self._send(200, {"id": doc.get("id"), "object": "text_completion", "model": SERVED,
                                    "choices": [{"index": 0, "text": (ch.get("message") or {}).get("content") or "",
                                                 "finish_reason": ch.get("finish_reason")}],
                                    "usage": {"prompt_tokens": len(ids), "completion_tokens": n,
                                              "total_tokens": len(ids) + n}})
        if IDS_ONLY:
            # an engine that gives token ids and no logprobs (AMD's megakernel path): the contract's
            # choices[0].token_ids; where the server puts them is a card item (NOTES, C4)
            got = next((v for v in (ch.get("token_ids"), (ch.get("message") or {}).get("token_ids"),
                                    ch.get("output_token_ids")) if isinstance(v, list)), None)
            if got is None:
                return self._send(502, {"error": "the engine's answer carried no token ids"})
            got = [int(x) for x in got]
            if ch.get("finish_reason") == "stop" and got and got[-1] in TM.special_ids:
                got = got[:-1]
            return self._send(200, {"id": doc.get("id"), "object": "text_completion", "created": int(time.time()),
                                    "model": SERVED, "choices": [{"index": 0, "text": "", "token_ids": got,
                                                                  "finish_reason": ch.get("finish_reason")}],
                                    "usage": {"prompt_tokens": len(ids), "completion_tokens": len(got),
                                              "total_tokens": len(ids) + len(got)}})
        toks, lps, tops = [], [], []
        if "token_logprobs" in ch:
            # the reference image's patch: one entry per generated token id
            ents = ch["token_logprobs"]
            if ch.get("finish_reason") == "stop" and ents and int(ents[-1]["id"]) in TM.special_ids:
                ents = ents[:-1]
            tid = lambda e: f"token_id:{int(e['id'])}"  # noqa: E731
        else:
            lp = ch.get("logprobs") or {}
            ents = list(lp.get("reasoning_content") or []) + list(lp.get("content") or [])
            tid = lambda e: f"token_id:{TM.id_of(e.get('token'), e.get('bytes'))}"  # noqa: E731
        for e in ents:
            tok, tlp = tid(e), e.get("logprob")
            alts = e.get("top_logprobs") or []
            if DAMAGE and (len(ids) + len(toks)) % DAMAGE == 0 and len(alts) > 1:
                # a deliberately damaged engine: the second-ranked token here
                tok, tlp = tid(alts[1]), alts[1]["logprob"]
            toks.append(tok)
            lps.append(tlp)
            tops.append({tid(t): t["logprob"] for t in alts})
        n = (doc.get("usage") or {}).get("completion_tokens")
        if n is None or not (len(toks) == n or (len(toks) == n - 1 and ch.get("finish_reason") == "stop")):
            # fail closed: a token without an entry cannot be placed
            return self._send(502, {"error": f"ATOM's logprobs cover {len(toks)} of {n} tokens"})
        out = {"id": doc.get("id"), "object": "text_completion", "created": int(time.time()), "model": SERVED,
               "choices": [{"index": 0, "text": "", "finish_reason": ch.get("finish_reason"),
                            "logprobs": {"tokens": toks, "token_logprobs": lps, "top_logprobs": tops}}],
               "usage": {"prompt_tokens": len(ids), "completion_tokens": len(toks),
                         "total_tokens": len(ids) + len(toks)}}
        return self._send(200, out)


ThreadingHTTPServer(("127.0.0.1", PORT), H).serve_forever()
