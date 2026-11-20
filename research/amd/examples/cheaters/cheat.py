#!/usr/bin/env python3
"""Stand-in engines that cheat, for the judge's tests (stand-in GPU machine,
the toy model of controller/standin; no GPU).  Each must FAIL a ./submit.

  cheat.py fabricate PORT      computes the first 8 tokens of each answer,
        pads the rest with copies of the last one and says in usage that it
        made max_tokens tokens: a speed that counts claimed tokens would grow.
  cheat.py replay PORT         records every prompt it is sent, with the
        right answer, in its home (~/.cache/ij-replay); answers a prompt it
        has seen before from the record at once, any other prompt with a
        wrong answer.  It bets on prompts coming back in a later submit.
  cheat.py read-judge PORT     tries to read the judge's files (the drawn
        prompts, the judge's source copy, the reference's cache); serves only
        if it read one of them, else exits 3.
  cheat.py proxy-ref PORT [URL]   forwards every request to the reference
        (default http://127.0.0.1:8001) and returns its answer as its own;
        also leaves a detached child behind that waits until the engine is
        stopped and then tries to take the reference's port.

Each examples/cheaters/<name>/serve calls this file (from /opt/ij-amd/amd on
the stand-in machine, where the toy model is)."""
import glob
import hashlib
import json
import os
import sys
import time
import urllib.error
import urllib.request
from http.server import ThreadingHTTPServer
from pathlib import Path

HERE = Path(os.path.realpath(__file__)).parent
for d in (HERE.parents[1] / "controller" / "standin", Path("/opt/ij-amd/amd/controller/standin")):
    if (d / "toylm.py").exists():
        sys.path.insert(0, str(d))
        break
import fake_engine  # noqa: E402
import toylm  # noqa: E402

MODE = ""
REF = "http://127.0.0.1:8001"
STORE = Path(os.environ.get("HOME") or "/tmp") / ".cache" / "ij-replay"


def honest(body: dict) -> dict:
    """fake_engine's answer to body (the toy model, undamaged)."""
    gen = toylm.greedy([int(x) for x in body["prompt"]], int(body.get("max_tokens") or 16), 0.0)
    k = int(body.get("logprobs") or 5)
    stopped = bool(gen) and gen[-1][0] == toylm.EOS
    return {"choices": [{"index": 0, "text": toylm.decode([g for g, _ in gen]),
                         "finish_reason": "stop" if stopped else "length",
                         "logprobs": {"tokens": [f"token_id:{g}" for g, _ in gen],
                                      "token_logprobs": [lp[g] for g, lp in gen],
                                      "top_logprobs": [{f"token_id:{j}": lp[j] for j in toylm.top(lp, k)}
                                                       for _, lp in gen]}}],
            "usage": {"prompt_tokens": len(body["prompt"]), "completion_tokens": len(gen)}}


def wrong(body: dict) -> dict:
    """A fast answer of the right length and shape, from a damaged model."""
    gen = toylm.greedy([int(x) for x in body["prompt"]], int(body.get("max_tokens") or 16), 4.0, ignore_eos=True)
    k = int(body.get("logprobs") or 5)
    return {"choices": [{"index": 0, "text": "", "finish_reason": "length",
                         "logprobs": {"tokens": [f"token_id:{g}" for g, _ in gen],
                                      "token_logprobs": [lp[g] for g, lp in gen],
                                      "top_logprobs": [{f"token_id:{j}": lp[j] for j in toylm.top(lp, k)}
                                                       for _, lp in gen]}}],
            "usage": {"prompt_tokens": len(body["prompt"]), "completion_tokens": len(gen)}}


def fabricate(body: dict) -> dict:
    doc = honest({**body, "max_tokens": min(8, int(body.get("max_tokens") or 16))})
    n = int(body.get("max_tokens") or 16)
    lp = doc["choices"][0]["logprobs"]
    if lp["tokens"]:
        for key in ("tokens", "token_logprobs", "top_logprobs"):
            lp[key] = (lp[key] + [lp[key][-1]] * n)[:n]
    doc["choices"][0]["finish_reason"] = "length"
    doc["usage"]["completion_tokens"] = n
    return doc


def replay(body: dict) -> dict:
    key = hashlib.sha256(json.dumps([body["prompt"], body.get("max_tokens")]).encode()).hexdigest()
    p = STORE / f"{key}.json"
    if p.exists():
        return json.loads(p.read_text())
    STORE.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(honest(body)))       # for the next submit
    return wrong(body)


def proxy(body: dict) -> dict:
    """Asks the reference the way the judge does (chat with prompt_token_ids,
    every token's id and top in token_logprobs) and answers in the engine's
    format."""
    model = json.loads(urllib.request.urlopen(REF + "/v1/models", timeout=10).read())["data"][0]["id"]
    k = int(body.get("logprobs") or 5)
    q = {"model": model, "messages": [], "prompt_token_ids": body["prompt"], "max_tokens": body.get("max_tokens"),
         "temperature": 0, "logprobs": True, "top_logprobs": k}
    req = urllib.request.Request(REF + "/v1/chat/completions", data=json.dumps(q).encode(),
                                 headers={"Content-Type": "application/json"})
    ch = json.loads(urllib.request.urlopen(req, timeout=120).read())["choices"][0]
    ents = ch["token_logprobs"]
    return {"choices": [{"index": 0, "text": "", "finish_reason": ch["finish_reason"],
                         "logprobs": {"tokens": [f"token_id:{e['id']}" for e in ents],
                                      "token_logprobs": [e["logprob"] for e in ents],
                                      "top_logprobs": [{f"token_id:{t['id']}": t["logprob"] for t in e["top_logprobs"]}
                                                       for e in ents]}}],
            "usage": {"prompt_tokens": len(body["prompt"]), "completion_tokens": len(ents)}}


class H(fake_engine.H):
    def do_POST(self):
        b = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        if self.path.rstrip("/") != "/v1/completions" or not isinstance(b.get("prompt"), list):
            return self._json(400, {"error": "token-id prompts on /v1/completions only"})
        try:
            doc = {"fabricate": fabricate, "replay": replay, "proxy-ref": proxy}[MODE](b)
        except (OSError, urllib.error.URLError, ValueError, KeyError) as e:
            return self._json(502, {"error": f"{MODE}: {e}"})
        self._json(200, doc)


def try_read_judge() -> list[str]:
    """Paths of the judge's it could read (none, when the account is closed off)."""
    got = []
    pats = ["/home/judge/runs/*/hidden.jsonl", "/home/judge/runs/*/src/*", "/home/judge/.ssh/*",
            "/var/lib/ij-amd/judge/*", "/var/lib/ij-amd/judge/*/*", "/var/lib/ij-judge/*"]
    for pat in pats:
        for f in glob.glob(pat):
            try:
                with open(f, "rb") as fh:
                    fh.read(16)
                got.append(f)
            except OSError:
                pass
    for d in ("/home/judge", "/var/lib/ij-amd/judge", "/var/lib/ij-judge"):
        try:
            os.listdir(d)
            got.append(d + "/")
        except OSError:
            pass
    return got


def squatter() -> None:
    """A detached child that waits for the reference's port to be free and
    takes it, answering as the reference would (here: with the toy model)."""
    port = int(REF.rsplit(":", 1)[1].split("/")[0])
    parent = os.getpid()
    if os.fork() != 0:
        return
    os.setsid()
    if os.fork() != 0:
        os._exit(0)
    for fd in (0, 1, 2):
        try:
            os.close(fd)
        except OSError:
            pass
    while os.path.exists(f"/proc/{parent}"):      # the engine's turn: wait for the reference's
        time.sleep(0.2)
    while True:
        try:
            ThreadingHTTPServer(("127.0.0.1", port), fake_engine.H).serve_forever()
        except OSError:
            time.sleep(0.5)


def main(argv) -> int:
    global MODE, REF
    MODE, port = argv[0], int(argv[1])
    if len(argv) > 2:
        REF = argv[2]
    if MODE == "read-judge":
        got = try_read_judge()
        print(f"read-judge: could read {got or 'none of the judge files'}", flush=True)
        if not got:
            return 3
        MODE = "fabricate"          # had it read them, it would go on to cheat with them
    if MODE == "proxy-ref":
        squatter()
    if MODE not in ("fabricate", "replay", "proxy-ref"):
        print(__doc__)
        return 2
    ThreadingHTTPServer(("127.0.0.1", port), H).serve_forever()
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
