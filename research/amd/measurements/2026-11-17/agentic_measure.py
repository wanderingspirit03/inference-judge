"""agentic_measure.py URL OUT POOL[:N] ...: one request at a time, streamed chat completions, greedy, ignore_eos,
220 answer tokens (the DeepSeek target's agentic method), the server's own thinking default. One untimed warm-up
request per pool, then N timed. Per pool: whole-request tokens/s (sum of completion tokens / sum of seconds),
decode tokens/s (tokens after the first / time after the first), median TTFT. Same client for every engine."""
import json, statistics, sys, time, urllib.request

URL, OUT = sys.argv[1].rstrip("/"), sys.argv[2]
OSL = 220


def one(prompt):
    msgs = prompt if isinstance(prompt, list) else [{"role": "user", "content": prompt}]
    body = {"model": "qwen3.8-27b", "messages": msgs, "max_tokens": OSL, "temperature": 0, "ignore_eos": True,
            "stream": True, "stream_options": {"include_usage": True}}
    req = urllib.request.Request(URL + "/v1/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    t0 = time.monotonic(); ttft = None; usage = None
    with urllib.request.urlopen(req, timeout=1800) as r:
        for raw in r:
            line = raw.decode().strip()
            if not line.startswith("data:") or line == "data: [DONE]":
                continue
            d = json.loads(line[5:])
            if d.get("usage"):
                usage = d["usage"]
            for c in d.get("choices") or []:
                dl = c.get("delta") or {}
                if ttft is None and (dl.get("content") or dl.get("reasoning_content")):
                    ttft = time.monotonic() - t0
    total = time.monotonic() - t0
    return {"seconds": total, "ttft": ttft or total, "completion_tokens": (usage or {}).get("completion_tokens"),
            "prompt_tokens": (usage or {}).get("prompt_tokens")}


res = {}
for spec in sys.argv[3:]:
    path, _, n = spec.partition(":")
    rows = [json.loads(l) for l in open(path)]
    n = int(n or len(rows) - 1)
    rows = rows[:n + 1]
    get = lambda r: r.get("messages") or r["prompt"]  # noqa: E731
    one(get(rows[0]))  # warm-up, untimed
    rs = [dict(one(get(r)), id=r.get("id")) for r in rows[1:]]
    tok = sum(x["completion_tokens"] or 0 for x in rs)
    sec = sum(x["seconds"] for x in rs)
    dec = sum(max(0, (x["completion_tokens"] or 0) - 1) for x in rs) / sum(x["seconds"] - x["ttft"] for x in rs)
    res[path] = {"requests": len(rs), "completion_tokens": tok, "seconds": round(sec, 2),
                 "whole_request_tps": round(tok / sec, 2), "decode_tps": round(dec, 2),
                 "median_ttft_s": round(statistics.median(x["ttft"] for x in rs), 3),
                 "median_prompt_tokens": statistics.median(x["prompt_tokens"] or 0 for x in rs), "per_request": rs}
    print(f"{path}: {len(rs)} requests, whole-request {tok / sec:.1f} tok/s, decode {dec:.1f} tok/s, "
          f"median ttft {res[path]['median_ttft_s']} s, median prompt {res[path]['median_prompt_tokens']} tokens",
          flush=True)
json.dump(res, open(OUT, "w"), indent=1)
