"""The reference engine: start, stop, ask, and a cache of its answers.

The reference is ROCm ATOM (ref/atom-serve.sh).  ATOM's /v1/completions
ignores `logprobs` and `echo` (its protocol.py: "Completions return no token
probabilities yet").  Its /v1/chat/completions takes `prompt_token_ids` (the
prompt as given, no chat template) and returns `logprobs.content`, each token
as text and bytes with up to 20 top alternatives.  This module asks that
endpoint and turns the bytes back into token ids with the model's
tokenizer.json (tokmap.py).

Every answer is greedy (temperature 0) and cached under the state directory,
keyed by the prompt's token ids, max_tokens, the number of top logprobs and the
reference's identity (image and flags).  A cached answer is returned without
the GPU.
"""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
import threading
import time
from pathlib import Path

import client
import config as C
import tokmap


class RefError(Exception):
    pass


def identity(cfg: dict) -> str:
    r = cfg["ref"]
    return f"atom image={r['image']} model={cfg['model']['dir']} kv=bf16 mem={r['gpu_memory_utilization']} greedy v3"


def cache_dir(cfg: dict) -> Path:
    d = C.state_dir(cfg) / "ref-cache"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _key(cfg: dict, kind: str, payload) -> str:
    return hashlib.sha256(json.dumps([identity(cfg), kind, payload], sort_keys=True).encode()).hexdigest()[:32]


def cached(cfg: dict, kind: str, payload):
    p = cache_dir(cfg) / (_key(cfg, kind, payload) + ".json")
    try:
        return json.loads(p.read_text())
    except (FileNotFoundError, ValueError):
        return None


def store(cfg: dict, kind: str, payload, value) -> None:
    p = cache_dir(cfg) / (_key(cfg, kind, payload) + ".json")
    tmp = p.with_suffix(f".{os.getpid()}.{threading.get_ident()}.tmp")     # threads: ./submit --tokenize
    tmp.write_text(json.dumps(value))
    os.replace(tmp, p)


def _env(cfg: dict, hold) -> dict:
    r = cfg["ref"]
    port = r["url"].rsplit(":", 1)[1].split("/")[0]
    e = hold.env() if hold is not None else dict(os.environ)
    e.update({"IJ_REF_PORT": port, "IJ_REF_MODEL_DIR": cfg["model"]["dir"], "IJ_REF_NAME": r["served_model"],
              "IJ_REF_IMAGE": r["image"], "IJ_REF_MEM": str(r["gpu_memory_utilization"])})
    return e


def _cmd(cfg: dict, key: str) -> list[str]:
    argv = list(cfg["ref"][key])
    argv[0] = str(C.resolve(cfg, argv[0]))
    return argv


def is_up(cfg: dict) -> bool:
    try:
        client.get_json(cfg["ref"]["url"], "/v1/models", timeout=3)
        return True
    except Exception:  # noqa: BLE001
        return False


def start(cfg: dict, hold, log) -> float:
    """Start the reference inside `hold` and wait until it answers.  Returns
    the seconds it took."""
    t0 = time.monotonic()
    if is_up(cfg):
        return 0.0
    log("starting the reference engine (ATOM)")
    try:
        subprocess.run(_cmd(cfg, "start"), env=_env(cfg, hold), check=True, stdout=subprocess.DEVNULL)
    except (OSError, subprocess.CalledProcessError) as e:
        raise RefError(f"could not start the reference: {e}") from None
    limit = cfg["ref"]["ready_timeout_s"]
    while time.monotonic() - t0 < limit:
        if is_up(cfg):
            dt = time.monotonic() - t0
            log(f"reference ready after {dt:.0f} s")
            return dt
        st = subprocess.run(_cmd(cfg, "start")[:1] + ["status"], capture_output=True, text=True).stdout.strip()
        if not st:
            logs = subprocess.run(_cmd(cfg, "start")[:1] + ["logs", "30"], capture_output=True, text=True).stdout
            raise RefError("the reference container exited while starting; its last lines:\n" + logs)
        time.sleep(3)
    raise RefError(f"the reference did not answer within {limit} s")


def stop(cfg: dict) -> None:
    subprocess.run(_cmd(cfg, "stop"), env=_env(cfg, None), stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def _post(cfg: dict, path: str, body: dict, timeout: float = 900.0) -> dict:
    import http.client
    import urllib.parse
    u = urllib.parse.urlsplit(cfg["ref"]["url"])
    c = http.client.HTTPConnection(u.hostname, u.port, timeout=timeout)
    try:
        c.request("POST", path, body=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
        r = c.getresponse()
        data = r.read()
        if r.status != 200:
            raise RefError(f"reference {path}: HTTP {r.status}: {data[:300].decode('utf-8', 'replace')}")
        return json.loads(data)
    finally:
        c.close()


def tokenize(cfg: dict, text: str | None = None, messages: list | None = None,
             template_kwargs: dict | None = None) -> list[int]:
    """The reference's own token ids for a raw text prompt, or for chat
    messages with the model's chat template applied."""
    name = cfg["ref"]["served_model"]
    if messages is not None:
        body = {"model": name, "messages": messages, "max_tokens": 1, "temperature": 0, "return_token_ids": True}
        if template_kwargs:
            body["chat_template_kwargs"] = template_kwargs
        doc = _post(cfg, "/v1/chat/completions", body)
    else:
        body = {"model": name, "prompt": text, "max_tokens": 1, "temperature": 0, "return_token_ids": True}
        doc = _post(cfg, "/v1/completions", body)
    ids = doc.get("prompt_token_ids")
    if not ids:
        raise RefError("the reference returned no prompt_token_ids")
    return [int(x) for x in ids]


def generate(cfg: dict, ids: list[int], max_tokens: int, top: int) -> dict:
    """Greedy continuation of a token-id prompt with per-token logprobs."""
    tm = tokmap.load(cfg["model"]["dir"])
    body = {"model": cfg["ref"]["served_model"], "messages": [{"role": "user", "content": "-"}],
            "prompt_token_ids": ids, "max_tokens": int(max_tokens), "temperature": 0,
            "logprobs": True, "top_logprobs": int(top)}
    t0 = time.monotonic()
    doc = _post(cfg, "/v1/chat/completions", body)
    dt = time.monotonic() - t0
    ch = doc["choices"][0]
    toks, lps, tops, texts = [], [], [], []
    n = (doc.get("usage") or {}).get("completion_tokens")
    finish = ch.get("finish_reason")
    if "token_logprobs" in ch:
        # The reference image's patch (reference/atom_token_logprobs.py): one
        # entry per generated token id, markers and special ids included.
        # Every token must have its entry.  A final special stop token is not
        # a token of the answer; it is kept apart as `stop` (its id, logprob
        # and top), so the walk can compare where the answers end.
        ents = ch["token_logprobs"]
        if n is None or len(ents) != n:
            raise RefError(f"the reference's token_logprobs cover {len(ents)} of its {n} tokens")
        stop_ent = None
        if finish == "stop" and ents and int(ents[-1]["id"]) in tm.special_ids:
            e = ents[-1]
            stop_ent = {"id": int(e["id"]), "logprob": float(e["logprob"]),
                        "top": {str(int(t["id"])): float(t["logprob"]) for t in e.get("top_logprobs") or []}}
            ents = ents[:-1]
        for e in ents:
            if e.get("logprob") is None:
                raise RefError(f"the reference gave token {e.get('id')} no logprob")
            toks.append(int(e["id"]))
            lps.append(float(e["logprob"]))
            tops.append({str(int(t["id"])): float(t["logprob"]) for t in e.get("top_logprobs") or []})
            texts.append(e.get("token"))
        return {"prompt_token_ids": ids, "tokens": toks, "logprobs": lps, "top_logprobs": tops,
                "token_text": texts, "text": "".join(t or "" for t in texts), "finish_reason": finish,
                "stop": stop_ent, "usage": doc.get("usage"), "seconds": round(dt, 3), "reference": identity(cfg)}
    # ATOM's reasoning parser splits the answer into reasoning_content (first)
    # and content; logprobs come per field, attached to visible text only.
    lpdoc = ch.get("logprobs") or {}
    entries = list(lpdoc.get("reasoning_content") or []) + list(lpdoc.get("content") or [])
    for e in entries:
        toks.append(tm.id_of(e.get("token"), e.get("bytes")))
        lps.append(None if e.get("logprob") is None else float(e["logprob"]))
        tops.append({str(tm.id_of(t.get("token"), t.get("bytes"))): float(t["logprob"])
                     for t in e.get("top_logprobs") or []})
        texts.append(e.get("token"))
    if not toks:
        raise RefError(f"the reference returned no logprobs (fields: {sorted(lpdoc)})")
    n = (doc.get("usage") or {}).get("completion_tokens")
    finish = ch.get("finish_reason")
    # A token with no visible text (a stop token, a hidden marker) gets no
    # logprob entry.  Only a final stop token may be missing; it is never
    # compared, because the engine is asked for exactly len(tokens) tokens.
    if n is not None and not (len(toks) == n or (len(toks) == n - 1 and finish == "stop")):
        raise RefError(f"the reference's logprobs cover {len(toks)} of its {n} tokens (finish {finish}); "
                       "a token without visible text in the middle of the answer cannot be placed")
    return {"prompt_token_ids": ids, "tokens": toks, "logprobs": lps, "top_logprobs": tops,
            "token_text": texts, "text": "".join(t or "" for t in texts), "finish_reason": ch.get("finish_reason"),
            "usage": doc.get("usage"), "seconds": round(dt, 3), "reference": identity(cfg)}


class Session:
    """Answers from the cache; on a miss, the reference is started inside the
    GPU hold the caller gives (or one taken here) and stopped at the end."""

    def __init__(self, cfg: dict, hold_factory, log, fresh: bool = False):
        """fresh: the reference must be started here; something that already
        answers on its port (left by an engine) is refused, not used."""
        self.cfg, self.hold_factory, self.log, self.fresh = cfg, hold_factory, log, fresh
        self.hold = None
        self.own_hold = None
        self.started = False
        self.start_s = 0.0
        self.starts = 0

    def _ensure(self):
        if self.hold is None:
            self.own_hold = self.hold_factory()
            self.hold = self.own_hold.__enter__()
        if not self.started:
            if self.fresh and is_up(self.cfg):
                raise RefError(f"something already answers on the reference's address {self.cfg['ref']['url']} "
                               "before the reference was started; it is not used")
            self.start_s = start(self.cfg, self.hold, self.log)
            self.started = True
            self.starts += 1

    def ids_cached(self, item: dict) -> bool:
        """True when ids_for(item) needs no reference."""
        if item.get("prompt_token_ids"):
            return True
        kw = item.get("chat_template_kwargs")
        if "messages" in item:
            return cached(self.cfg, "tok-chat", [item["messages"], kw] if kw else item["messages"]) is not None
        return cached(self.cfg, "tok-text", item["prompt"]) is not None

    def ids_for(self, item: dict) -> list[int]:
        if item.get("prompt_token_ids"):
            return [int(x) for x in item["prompt_token_ids"]]
        kw = item.get("chat_template_kwargs")
        if "messages" in item:
            kind, payload = "tok-chat", [item["messages"], kw] if kw else item["messages"]
        else:
            kind, payload = "tok-text", item["prompt"]
        got = cached(self.cfg, kind, payload)
        if got is None:
            self._ensure()
            got = (tokenize(self.cfg, messages=item["messages"], template_kwargs=kw) if "messages" in item
                   else tokenize(self.cfg, text=item["prompt"]))
            store(self.cfg, kind, payload, got)
        return got

    def answer(self, ids: list[int], max_tokens: int, top: int | None = None) -> dict:
        top = top or self.cfg["ref"]["top_logprobs"]
        payload = [ids, max_tokens, top]
        got = cached(self.cfg, "gen", payload)
        if got is not None:
            got["cached"] = True
            return got
        self._ensure()
        got = generate(self.cfg, ids, max_tokens, top)
        store(self.cfg, "gen", payload, got)
        got["cached"] = False
        return got

    def close(self):
        if self.started:
            stop(self.cfg)
            self.started = False
        if self.own_hold is not None:
            self.own_hold.__exit__(None, None, None)
            self.own_hold = None
            self.hold = None
