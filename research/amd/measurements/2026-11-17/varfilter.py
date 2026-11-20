"""Keep the variation prompts whose reference answer the reference can place (ATOM drops logprob entries for
special tokens mid-answer; such prompts make ./check refuse the whole run). Uses research/amd's own Session."""
import contextlib, json, os, sys
os.environ["IJ_AMD_CONFIG"] = "/opt/ij-amd/amd/var-config.json"
os.environ["IJ_AMD_STATE"] = "/var/lib/ij-amd/root"
sys.path.insert(0, "/opt/ij-amd/amd/lib")
import common, ref  # noqa: E402

cfg, state = common.setup()
rows = [json.loads(l) for l in open("/root/var-prompts.jsonl")]
s = ref.Session(cfg, lambda: contextlib.nullcontext(None), lambda m: print(m, flush=True))
keep, drop = [], []
try:
    for r in rows:
        ids = s.ids_for(r)
        try:
            s.answer(ids, cfg["check"]["max_tokens"])
            keep.append(r)
        except ref.RefError as e:
            drop.append(r["id"])
            print("drop", r["id"], str(e)[:90], flush=True)
finally:
    s.close()
with open("/root/var-prompts-ok.jsonl", "w") as f:
    for r in keep:
        f.write(json.dumps(r) + "\n")
print(f"kept {len(keep)}, dropped {len(drop)}: {drop}")
