"""The tuned engine returns no logprobs, so ./check cannot judge it. This compares its greedy answers with the
reference's cached answers instead: same 43 prompts, same prompt token ids, 128 tokens.
The tuned answer comes back as text, so the reference's tokens are decoded to text and the two texts compared;
the first difference is reported as the index of the reference token that contains the first differing character
(re-encoding the tuned text instead would differ at token boundaries without any real difference)."""
import json, os, sys, urllib.request
os.environ["IJ_AMD_CONFIG"] = "/opt/ij-amd/amd/fix-config.json"
os.environ["IJ_AMD_STATE"] = "/var/lib/ij-amd/fix"
sys.path.insert(0, "/opt/ij-amd/amd/lib")
import common, ref  # noqa: E402
from tokenizers import Tokenizer  # noqa: E402

cfg, _ = common.setup()
tk = Tokenizer.from_file("/srv/models/Qwen3.8-27B/tokenizer.json")
URL = sys.argv[1]
rows = [json.loads(l) for l in open("/root/var-prompts.jsonl")]
out, same, firsts, missing = [], 0, [], []
for r in rows:
    if r.get("prompt_token_ids"):
        ids = [int(x) for x in r["prompt_token_ids"]]
    elif "messages" in r:
        kw = r.get("chat_template_kwargs")
        ids = ref.cached(cfg, "tok-chat", [r["messages"], kw] if kw else r["messages"])
    else:
        ids = ref.cached(cfg, "tok-text", r["prompt"])
    want = ref.cached(cfg, "gen", [ids, cfg["check"]["max_tokens"], cfg["ref"]["top_logprobs"]]) if ids else None
    if want is None:
        missing.append(r["id"])
        print(r["id"], "no cached reference answer", flush=True)
        continue
    w = want["tokens"]
    body = {"model": "qwen3.8-27b", "messages": [{"role": "user", "content": "-"}], "prompt_token_ids": ids,
            "max_tokens": len(w), "temperature": 0, "ignore_eos": True}
    req = urllib.request.Request(URL + "/v1/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    m = json.load(urllib.request.urlopen(req, timeout=600))["choices"][0]["message"]
    got = (m.get("reasoning_content") or "") + (m.get("content") or "")
    pieces = [tk.decode([t], skip_special_tokens=True) for t in w]
    exp = "".join(pieces)
    c = next((i for i, (a, b) in enumerate(zip(exp, got)) if a != b), min(len(exp), len(got)))
    ok = c >= len(exp)
    k, n = len(w), 0
    for i, p in enumerate(pieces):
        n += len(p)
        if n > c:
            k = i
            break
    same += ok
    firsts.append(k)
    out.append({"id": r["id"], "identical": ok, "first_difference_token": k, "ref_tokens": len(w),
                "ref_text_chars": len(exp), "tuned_text_chars": len(got),
                "ref_at_difference": exp[max(0, c - 20):c + 40], "tuned_at_difference": got[max(0, c - 20):c + 40]})
    print(r["id"], "identical" if ok else f"differs at token {k} of {len(w)}", flush=True)
json.dump({"rows": out, "missing": missing}, open("/root/tuned/textcmp.json", "w"), indent=1)
print(f"identical {same} of {len(out)} compared ({len(missing)} without a cached answer); "
      f"median first difference {sorted(firsts)[len(firsts) // 2]} tokens")
