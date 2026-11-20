"""Which generated token has no logprob entry in ATOM's chat answer, and what is it?
Usage: lpdiag.py URL PROMPT_ID [PROMPT_ID...]  (prompts from /root/var-prompts.jsonl; ATOM on URL)
Steps the greedy answer one token at a time through /v1/chat/completions with prompt_token_ids and logprobs (as
ref.generate asks); at the first step with no entry, asks /v1/completions (no reasoning channel) for the same token."""
import json, sys, urllib.request
from tokenizers import Tokenizer

URL = sys.argv[1]
tk = Tokenizer.from_file("/srv/models/Qwen3.8-27B/tokenizer.json")
added = {t["id"]: t for t in json.load(open("/srv/models/Qwen3.8-27B/tokenizer.json"))["added_tokens"]}


def _inv():
    bs = list(range(33, 127)) + list(range(161, 173)) + list(range(174, 256))
    cs, n = bs[:], 0
    for b in range(256):
        if b not in bs:
            bs.append(b); cs.append(256 + n); n += 1
    return {chr(c): b for b, c in zip(bs, cs)}


INV = _inv()
FWD = {v: k for k, v in INV.items()}


def raw(i):
    return added[i]["content"].encode() if i in added else bytes(INV[c] for c in tk.id_to_token(i))


def valid(b):
    try:
        b.decode("utf-8")
        return True
    except UnicodeDecodeError:
        return False


def post(path, body):
    req = urllib.request.Request(URL + path, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    return json.load(urllib.request.urlopen(req, timeout=600))


def chat(ids, n, top):
    c = post("/v1/chat/completions", {"model": "qwen3.8-27b", "messages": [{"role": "user", "content": "-"}],
                                      "prompt_token_ids": ids, "max_tokens": n, "temperature": 0,
                                      "logprobs": True, "top_logprobs": top})
    ch = c["choices"][0]
    lp = ch.get("logprobs") or {}
    return c, ch, list(lp.get("reasoning_content") or []) + list(lp.get("content") or [])


rows = {json.loads(l)["id"]: json.loads(l) for l in open("/root/var-prompts.jsonl")}
for pid in sys.argv[2:]:
    r = rows[pid]
    if "messages" in r:
        d = post("/v1/chat/completions", {"model": "qwen3.8-27b", "messages": r["messages"], "max_tokens": 1,
                                          "return_token_ids": True,
                                          "chat_template_kwargs": r.get("chat_template_kwargs") or {}})
    else:
        d = post("/v1/completions", {"model": "qwen3.8-27b", "prompt": r["prompt"], "max_tokens": 1,
                                     "return_token_ids": True})
    ids = d["prompt_token_ids"]
    c, ch, ents = chat(ids, 128, 1)
    n = c["usage"]["completion_tokens"]
    pre, out, found = list(ids), [], None
    for k in range(n):
        _, sc, se = chat(pre, 1, 5)
        if se:
            tid = tk.token_to_id("".join(FWD[x] for x in se[0]["bytes"]))
            if tid is None:
                found = {"position": k, "error": "entry bytes are not one vocab token", "bytes": se[0]["bytes"]}
                break
            out.append(tid); pre.append(tid)
            continue
        q = post("/v1/completions", {"model": "qwen3.8-27b", "prompt": pre, "max_tokens": 1, "temperature": 0})
        txt = q["choices"][0]["text"]
        cid = tk.encode(txt, add_special_tokens=False).ids
        found = {"position": k, "chat_message_fields": {f: sc["message"].get(f) for f in ("reasoning_content", "content")},
                 "completions_text": txt, "token_ids": cid, "token_bytes": [list(raw(i)) for i in cid],
                 "bytes_are_valid_utf8": [valid(raw(i)) for i in cid],
                 "added_token": [added.get(i) for i in cid],
                 "context_before": tk.decode(out[-15:], skip_special_tokens=False)}
        break
    print(json.dumps({"prompt": pid, "entries_in_full_answer": len(ents), "completion_tokens": n,
                      "finish": ch.get("finish_reason"), "gap": found}, ensure_ascii=False), flush=True)
