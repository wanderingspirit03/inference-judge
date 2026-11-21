#!/usr/bin/env python3
"""rerun.py: the paper's rerun on one fresh MI350X, laid out to use few GPU hours.  RULES.md (next to this
file) is the frozen plan; this file is its implementation.  Runs on the GPU machine as root, from
$RERUN_ROOT (default /root/rerun):

  amd/    a copy of research/amd (the judge's libraries, the reference scripts, the shim)
  data/   pool-set-0.jsonl pool-set-1.jsonl pool-set-2.jsonl warm.jsonl tasks_512.jsonl
  out/    everything this script writes (one JSON file per step, written whole or not at all)

  rerun.py            run every step that has no output file yet (safe to start again after a stop)
  rerun.py summary    only rebuild out/summary.json from the files there

RERUN_SMOKE=1 is a short trial on the real machine before the run: 2 prompts of each set, 16 answer
tokens, 8 tasks, round 1 with the arms, the reference block; it writes to out-smoke/ and is not a result.

The steps, in order:
  1. round 1: one launch of each configuration; each launch times all three prompt sets, then the rotated
     one-token and whole requests (prompt time and decode rate), then answers the 512 tasks
  2. reference block 1: the reference's own speed, a check that concurrent requests give the same answers
     as serial ones, the walk of every round-1 answer, the 512 tasks
  3. rounds 2 and 3: speed only; round 3's launch also runs the workload arms after its core part
  4. AMD tuned with --max-num-seqs 1: one launch, speed only
  5. reference block 2: the walk of every answer of steps 3 and 4 (answers equal to round 1's cost nothing)
  6. summary

RERUN_STANDIN=1 runs the same code against research/amd/controller/standin (a toy model, no GPU).
"""
from __future__ import annotations

import concurrent.futures as cf
import hashlib
import json
import math
import os
import random
import subprocess
import sys
import time
import traceback
from pathlib import Path

ROOT = Path(os.environ.get("RERUN_ROOT", "/root/rerun"))
AMD = ROOT / "amd"
DATA = ROOT / "data"
OUT = ROOT / "out"
STANDIN = os.environ.get("RERUN_STANDIN") == "1"
SMOKE = os.environ.get("RERUN_SMOKE") == "1"       # a short trial of every code path; its results are discarded
if SMOKE:
    OUT = ROOT / "out-smoke"
OUT.mkdir(parents=True, exist_ok=True)
MACHINE = (ROOT / "MACHINE").read_text().strip() if (ROOT / "MACHINE").exists() else None
os.environ["IJ_AMD_CONFIG"] = str(ROOT / "config.standin.json" if STANDIN else AMD / "config.json")
os.environ["IJ_AMD_STATE"] = str(OUT / "ref-state")

sys.path.insert(0, str(AMD / "lib"))
import answers  # noqa: E402
import client  # noqa: E402
import compare  # noqa: E402
import config as C  # noqa: E402
import quality  # noqa: E402
import ref as R  # noqa: E402
import tokmap  # noqa: E402

# ---- the frozen parameters (RULES.md) -------------------------------------------------------------------
SEED = 20261121
CONFIGS = ["amd_tuned", "agent_tau0", "agent_tau030"]
ROUNDS = [1, 2, 3]
SETS = [0, 1, 2]
MAX_TOKENS = int(os.environ.get("RERUN_MAX_TOKENS", "16" if SMOKE else "128"))
TASK_MAX_TOKENS = 512
REF_WORKERS = 4                     # concurrent reference requests, if the probe finds no token difference
ARM_LENGTHS = [(1, 512), (2, 2048)] if not (STANDIN or SMOKE) else [(1, 48), (2, 64)]      # (set, max_tokens)
ARM_STREAMS = [2, 4]
PORT = 8100
ENGINE_TOP = 5
BOOT = 10000


def log(msg: str) -> None:
    line = f"[{time.strftime('%H:%M:%S', time.gmtime())}Z] {msg}"
    print(line, flush=True)
    with open(OUT / "rerun.log", "a") as f:
        f.write(line + "\n")


def save(name: str, doc) -> None:
    tmp = OUT / (name + ".tmp")
    tmp.write_text(json.dumps(doc, indent=1))
    os.replace(tmp, OUT / name)


def load(name: str):
    p = OUT / name
    return json.loads(p.read_text()) if p.exists() else None


def rows_of(p: Path) -> list[dict]:
    return [json.loads(x) for x in p.read_text().splitlines() if x.strip()]


def sha_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def tok_hash(toks) -> str:
    return hashlib.sha256(json.dumps(toks).encode()).hexdigest()[:16]


def set_items(k: int, max_tokens: int = MAX_TOKENS) -> list[dict]:
    rows = sorted(rows_of(DATA / f"pool-set-{k}.jsonl"), key=lambda r: r["id"])
    if SMOKE:
        rows = rows[:2]
    return [{"id": r["id"], "ids": [int(x) for x in r["prompt_token_ids"]], "max_tokens": max_tokens} for r in rows]


def rotated(items: list[dict], by: int, max_tokens: int | None = None) -> list[dict]:
    """The same prompts with the first `by` tokens moved to the end: the same length, and no request sent
    before is a prefix of them (answers.breakdown does the same for its extra requests)."""
    out = []
    for it in items:
        k = by % max(len(it["ids"]), 1)
        out.append({"id": it["id"], "ids": it["ids"][k:] + it["ids"][:k], "max_tokens": max_tokens or it["max_tokens"]})
    return out


def round_order(r: int) -> list[str]:
    o = list(CONFIGS)
    random.Random(SEED + r).shuffle(o)
    return o


def set_order(r: int) -> list[int]:
    return [SETS[(i + r - 1) % len(SETS)] for i in range(len(SETS))]


# ---- the GPU's state ------------------------------------------------------------------------------------
def gpu_state() -> dict:
    """Temperature, power, clock and memory in use, read from sysfs (empty where a file is missing)."""
    out = {}
    for dev in sorted(Path("/sys/class/drm").glob("card*/device")):
        def rd(p):
            try:
                return int(Path(p).read_text().split()[0])
            except (OSError, ValueError, IndexError):
                return None
        used = rd(dev / "mem_info_vram_used")
        if used is None:
            continue
        out["vram_used_gib"] = round(used / 2**30, 2)
        for hw in sorted(dev.glob("hwmon/hwmon*")):
            for key, f, div in (("temp_c", "temp1_input", 1000), ("power_w", "power1_average", 1e6),
                                ("power_w", "power1_input", 1e6), ("clock_mhz", "freq1_input", 1e6)):
                v = rd(hw / f)
                if v is not None and key not in out:
                    out[key] = round(v / div, 1)
        break
    return out


def wait_gpu_free(limit_s: int = 120) -> None:
    if STANDIN:
        return
    t0 = time.monotonic()
    while time.monotonic() - t0 < limit_s:
        if gpu_state().get("vram_used_gib", 0.0) < 4.0:
            return
        time.sleep(2)
    raise RuntimeError(f"the GPU still holds {gpu_state().get('vram_used_gib')} GiB after {limit_s} s; nothing is timed on it")


def stop_everything(cfg=None) -> None:
    """Stop every engine; with cfg also the reference (one left by a run that was stopped must not stay
    loaded while an engine is timed)."""
    if cfg is not None:
        R.stop(cfg)
    if STANDIN:
        subprocess.run(["pkill", "-f", "fake_engine.py"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    else:
        for name in ("ij-rerun-engine", "ij-engine"):
            subprocess.run(["docker", "rm", "-f", name], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        subprocess.run(["pkill", "-f", "atom-engine/shim.py"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    time.sleep(1)
    wait_gpu_free()


# ---- an engine under test -------------------------------------------------------------------------------
class Engine:
    """One launch of a configuration, answering in the judge's one request shape (lib/engine.py Server.ask)."""

    def __init__(self, cfg: dict, name: str, tag: str):
        self.cfg, self.name, self.tag = cfg, name, tag
        self.url = f"http://127.0.0.1:{PORT}"
        self.inner = f"http://127.0.0.1:{PORT + 1}"
        self.model, self.procs, self.logf, self.env_seen = None, [], None, []
        self.p = None                                   # answers.engine_answers looks at .p for an exit

    def _commands(self) -> list[tuple[list[str], dict | None]]:
        if STANDIN:
            fe = str(AMD / "controller/standin/fake_engine.py")
            extra = {"amd_tuned": ["--ids-only"], "amd_seq1": ["--ids-only"], "agent_tau0": [],
                     "agent_tau030": ["--quantised"]}[self.name]
            return [(["python3", fe, str(PORT), *extra], None)]
        groups = ["--group-add", "video"]
        if subprocess.run(["getent", "group", "render"], stdout=subprocess.DEVNULL).returncode == 0:
            groups += ["--group-add", "render"]
        base = ["docker", "run", "--rm", "--name", "ij-rerun-engine", "--network", "host", "--ipc", "host",
                "--shm-size", "16g", "--device", "/dev/kfd", "--device", "/dev/dri", *groups,
                "--security-opt", "seccomp=unconfined", "-v", "/srv/models:/models:ro",
                "-e", "GPU_MAX_HW_QUEUES=1"]
        if self.name in ("agent_tau0", "agent_tau030"):
            tau = "0" if self.name == "agent_tau0" else "0.3"
            return [(base + ["-e", "PYTHONUNBUFFERED=1", "-e", f"ENG_LOSSY_TAU={tau}", "ij-agent-657edee:paper",
                             "python3", "-m", "eng.main", str(PORT)], None)]
        seqs = "1" if self.name == "amd_seq1" else "4"
        atom = base + ["-e", "ATOM_MONO_ENABLE=1", "-e", "ATOM_DFLASH2_AOT=/opt/qwen38/drafter-objects",
                       "ij-amd-tuned:paper", "env", "-u", "TEMP", "python3", "-m", "atom.entrypoints.openai_server",
                       "--model", "/models/Qwen3.8-27B", "--served-model-name", "qwen3.8-27b",
                       "--kv_cache_dtype", "bf16", "-tp", "1", "--max-model-len", "40000",
                       "--max-num-seqs", seqs, "--server-port", str(PORT + 1), "--method", "dflash",
                       "--draft-model", "/models/Qwen3.8-27B-DFlash2", "--num-speculative-tokens", "7"]
        shim = ["python3", str(AMD / "examples/atom-engine/shim.py"), str(PORT), self.inner]
        return [(atom, None), (shim, dict(os.environ, SHIM_IDS_ONLY="1"))]

    def start(self, limit_s: int = 1800) -> float:
        stop_everything(self.cfg)
        if R.is_up(self.cfg):
            raise RuntimeError("the reference still answers; an engine is not started next to it")
        self.logf = open(OUT / f"serve_{self.tag}.log", "ab")
        cmds = self._commands()
        self.commands = [c for c, _ in cmds]
        for cmd, env in cmds:
            self.procs.append(subprocess.Popen(cmd, env=env, stdout=self.logf, stderr=subprocess.STDOUT))
        self.p = self.procs[0]
        t0 = time.monotonic()
        while time.monotonic() - t0 < limit_s:
            if self.p.poll() is not None:
                raise RuntimeError(f"{self.name} exited with {self.p.returncode} while starting (serve_{self.tag}.log)")
            try:
                data = client.get_json(self.url, "/v1/models", timeout=3).get("data") or []
                if data:
                    self.model = data[0]["id"]
                    dt = time.monotonic() - t0
                    if not STANDIN:
                        r = subprocess.run(["docker", "inspect", "-f", "{{json .Config.Env}}", "ij-rerun-engine"],
                                           capture_output=True, text=True)
                        if r.returncode == 0:
                            self.env_seen = [e for e in json.loads(r.stdout) if e.startswith(("ENG_", "ATOM_", "GPU_"))]
                    log(f"  {self.name} ready after {dt:.0f} s")
                    return dt
            except Exception:  # noqa: BLE001 - not up yet
                pass
            time.sleep(2)
        raise TimeoutError(f"{self.name} did not answer within {limit_s} s")

    def ask(self, ids: list[int], max_tokens: int) -> dict:
        body = client.build_body(self.model, "completions", ids, max_tokens, ENGINE_TOP, stream=False)
        r = client.request(self.url, "completions", body, timeout=self.cfg["engine"]["request_timeout_s"])
        return {"tokens": r["tokens"], "lps": r["lps"], "finish": r["finish"],
                "seconds": r["t_end"] - r["t0"], "prompt_tokens": len(ids)}

    def stop(self) -> None:
        for p in self.procs:
            if p.poll() is None:
                p.terminate()
        stop_everything()
        for p in self.procs:
            try:
                p.wait(timeout=20)
            except subprocess.TimeoutExpired:
                p.kill()
        self.procs = []
        if self.logf:
            self.logf.close()
            self.logf = None


def raw(items: list[dict], outs: list[dict]) -> list[dict]:
    return [{"id": it["id"], "prompt_tokens": len(it["ids"]), "tokens": o.get("tokens"), "lps": o.get("lps") or [],
             "finish": o.get("finish"),
             "seconds": round(o["seconds"], 4) if "seconds" in o else None, "error": o.get("error")}
            for it, o in zip(items, outs)]


def timed_sets(cfg, eng: Engine, r: int, stops, warm_rows) -> dict:
    """Each set in this round's order: its warm-up, then its prompts once each (the timed answers)."""
    res = {}
    for k in set_order(r):
        items = set_items(k)
        before = gpu_state()
        n_warm = answers.warm_up(eng, items, warm_rows, log)
        outs = answers.engine_answers(cfg, eng, items, log)
        sp = answers.speed(items, outs, stops)
        if sp is None:
            raise RuntimeError(f"{eng.name} set {k}: a request failed: "
                               + next(o["error"] for o in outs if "error" in o))
        log(f"  {eng.name} round {r} set {k}: {sp['tokens_per_s']} tokens/s ({sp['tokens']} tokens / {sp['seconds']} s)")
        res[str(k)] = {"warm_requests": n_warm, "speed": sp, "raw": raw(items, outs),
                       "gpu_before": before, "gpu_after": gpu_state()}
    return res


def parts(eng: Engine, sets: dict, stops) -> dict:
    """Prompt time and decode rate from extra requests on rotated prompts, after every timed pass of the launch."""
    out = {}
    for k in SETS:
        items = set_items(k)
        outs = [{"tokens": x["tokens"], "finish": x["finish"], "seconds": x["seconds"],
                 "prompt_tokens": x["prompt_tokens"]} for x in sets[str(k)]["raw"]]
        out[str(k)] = answers.breakdown(eng, items, outs, stops, 1, log)
    return out


def arms(cfg, eng: Engine, stops, warm_rows) -> dict:
    """The workload arms, after the launch's core part (so they cannot change it).  Speed only: these answers
    are not walked.  Every prompt is a rotation not sent before in this launch."""
    res = {"lengths": [], "streams": [], "errors": []}

    def guarded(what, fn):
        try:
            fn()
        except Exception as e:  # noqa: BLE001 - an arm that fails is recorded; the others still run
            log(f"  arm failed, {what}: {e}")
            res["errors"].append({"arm": what, "error": str(e)[:400]})

    def length_arm(i, k, mt):
        items = rotated(set_items(k), 5 + 2 * i, mt)
        eng.ask(answers.warm_shapes(items[:1], warm_rows)[0], mt)      # one untimed answer of this length first
        outs = answers.engine_answers(cfg, eng, items, log)
        sp = answers.speed(items, outs, stops)
        log(f"  arm, {mt} tokens on set {k}: {sp and sp['tokens_per_s']} tokens/s")
        res["lengths"].append({"set": k, "max_tokens": mt, "speed": sp, "errors": [o["error"] for o in outs if "error" in o],
                               "finishes": [o.get("finish") for o in outs]})

    def stream_arm(i, c):
        items = rotated(set_items(0), 11 + 2 * i)
        t0 = time.monotonic()
        got = client.closed_loop(items, c, lambda it: eng.ask(it["ids"], it["max_tokens"]))
        wall = time.monotonic() - t0
        ok = [g for g in got if isinstance(g, dict)]
        toks = sum(answers._counted(g, stops) for g in ok)
        log(f"  arm, {c} streams on set 0: {toks / wall:.1f} tokens/s in total ({len(ok)} of {len(items)} answered)")
        res["streams"].append({"streams": c, "answered": len(ok), "requests": len(items), "tokens": toks,
                               "wall_s": round(wall, 3), "total_tokens_per_s": round(toks / wall, 2),
                               "request_seconds": [round(g["seconds"], 4) for g in ok],
                               "errors": [str(g)[:200] for g in got if not isinstance(g, dict)]})

    for i, (k, mt) in enumerate(ARM_LENGTHS):
        guarded(f"{mt} tokens on set {k}", lambda i=i, k=k, mt=mt: length_arm(i, k, mt))
    for i, c in enumerate(ARM_STREAMS):
        guarded(f"{c} streams", lambda i=i, c=c: stream_arm(i, c))
    if eng.name == "amd_tuned" and not STANDIN:
        guarded("shim cost", lambda: res.update(shim=shim_cost(eng)))
    return res


def shim_cost(eng: Engine) -> dict:
    """What the shim adds to a request of AMD's engine: one-token requests through the shim and straight to
    ATOM's chat endpoint (the request the shim sends), on rotated prompts, alternating."""
    via, direct = [], []
    its = set_items(0)
    for j, it in enumerate(its):
        a, b = rotated([it], 21 + 2 * j)[0], rotated([it], 22 + 2 * j)[0]
        via.append(eng.ask(a["ids"], 1)["seconds"])
        body = {"model": "qwen3.8-27b", "messages": [{"role": "user", "content": "-"}], "prompt_token_ids": b["ids"],
                "max_tokens": 1, "temperature": 0, "return_token_ids": True}
        t0 = time.monotonic()
        client_post(eng.inner, "/v1/chat/completions", body)
        direct.append(time.monotonic() - t0)
    return {"through_shim_s": [round(x, 4) for x in via], "direct_s": [round(x, 4) for x in direct],
            "mean_difference_s": round((sum(via) - sum(direct)) / len(via), 4)}


def client_post(base: str, path: str, body: dict, timeout: float = 900.0) -> dict:
    import http.client
    import urllib.parse
    u = urllib.parse.urlsplit(base)
    c = http.client.HTTPConnection(u.hostname, u.port, timeout=timeout)
    try:
        c.request("POST", path, body=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
        r = c.getresponse()
        data = r.read()
        if r.status != 200:
            raise RuntimeError(f"{path}: HTTP {r.status}: {data[:200]!r}")
        return json.loads(data)
    finally:
        c.close()


def launch(cfg, name: str, r: int, tm, warm_rows, titems, with_tasks: bool, with_arms: bool) -> None:
    """One launch.  Writes launch_<name>_r<r>.json (core), then tasks_<name>.json and arms_<name>.json."""
    tag = f"{name}_r{r}"
    core_done = load(f"launch_{tag}.json") is not None
    need_tasks = with_tasks and load(f"tasks_{name}.json") is None
    need_arms = with_arms and load(f"arms_{name}.json") is None
    if core_done and not need_tasks and not need_arms:
        return
    log(f"=== launch {tag}: sets in order {set_order(r)}" + ("" if not core_done else " (core is done; the rest)"))
    stops = tm.special_ids
    eng = Engine(cfg, name, tag)
    try:
        t0 = time.monotonic()
        ready = eng.start()
        if not core_done:
            sets = timed_sets(cfg, eng, r, stops, warm_rows)
            pr = parts(eng, sets, stops)
            save(f"launch_{tag}.json", {"config": name, "round": r, "machine": MACHINE, "set_order": set_order(r), "ready_s": round(ready, 1),
                                         "commands": eng.commands, "env": eng.env_seen, "sets": sets, "parts": pr,
                                         "core_s": round(time.monotonic() - t0, 1),
                                         "finished_utc": time.strftime("%FT%TZ", time.gmtime())})
        elif need_arms:
            for k in SETS:                      # a launch only for the arms: the same warm-up first
                answers.warm_up(eng, set_items(k), warm_rows, log)
        if need_tasks:
            t1 = time.monotonic()
            texts = answers.engine_task_texts(cfg, eng, titems, tm, log)
            save(f"tasks_{name}.json", {"config": name, "seconds": round(time.monotonic() - t1, 1), "texts": texts})
        if need_arms:
            save(f"arms_{name}.json", {"config": name, "round": r, **arms(cfg, eng, stops, warm_rows)})
    finally:
        eng.stop()


def launches(cfg, plan: list[tuple], tm, warm_rows, titems) -> None:
    """Run the launches of `plan`; one that fails is logged, recorded, and tried once more after the others."""
    for attempt in (1, 2):
        failed = []
        for name, r, with_tasks, with_arms in plan:
            try:
                launch(cfg, name, r, tm, warm_rows, titems, with_tasks, with_arms)
            except Exception as e:  # noqa: BLE001 - recorded, retried once
                log(f"FAILED launch {name} round {r} (attempt {attempt}): {e}")
                fl = load("failures.json") or []
                fl.append({"config": name, "round": r, "attempt": attempt, "error": str(e)[:500],
                           "trace": traceback.format_exc()[-1500:], "utc": time.strftime("%FT%TZ", time.gmtime())})
                save("failures.json", fl)
                failed.append((name, r, with_tasks, with_arms))
        if not failed:
            return
        plan = failed
    log(f"launches that failed twice and are left out: {[(n, r) for n, r, _, _ in plan]}")


# ---- the reference --------------------------------------------------------------------------------------
class RefTimed:
    """The reference in the judge's request shape with logprobs, exactly as submit's time_reference does it
    (check.engine_top alternatives per token).  These answers are not reused by the walk."""

    def __init__(self, cfg):
        self.cfg, self.p = cfg, None
        self.top = cfg["check"]["engine_top"]

    def ask(self, ids, max_tokens):
        d = R.generate(self.cfg, ids, max_tokens, self.top)
        toks = d["tokens"] + ([d["stop"]["id"]] if d.get("stop") else [])
        return {"tokens": toks, "lps": [], "finish": d["finish_reason"], "seconds": d["seconds"],
                "prompt_tokens": len(ids), "doc": d}


class RefPlain:
    """The reference asked for token ids only, no logprobs (the request AMD's shim sends an engine)."""

    def __init__(self, cfg):
        self.cfg, self.p = cfg, None

    def ask(self, ids, max_tokens):
        body = {"model": self.cfg["ref"]["served_model"], "messages": [{"role": "user", "content": "-"}],
                "prompt_token_ids": ids, "max_tokens": int(max_tokens), "temperature": 0, "return_token_ids": True}
        t0 = time.monotonic()
        doc = client_post(self.cfg["ref"]["url"], "/v1/chat/completions", body)
        dt = time.monotonic() - t0
        ch = doc["choices"][0]
        n = int((doc.get("usage") or {}).get("completion_tokens") or 0)
        toks = ch.get("token_ids") or [0] * n          # only the count is used
        return {"tokens": toks, "lps": [], "finish": ch.get("finish_reason"), "seconds": dt, "prompt_tokens": len(ids)}


def ref_speed(cfg, stops, warm_rows) -> None:
    """The reference's speed on the three sets, timed like an engine, with logprobs (the paper's way); then
    without logprobs on rotated prompts."""
    if load("ref_speed.json") is not None:
        return
    res = {"with_logprobs": {}, "without_logprobs_rotated": {}}
    rt = RefTimed(cfg)
    for k in SETS:
        items = set_items(k)
        answers.warm_up(rt, items, warm_rows, log)
        outs = answers.engine_answers(cfg, rt, items, log)
        sp = answers.speed(items, outs, stops)
        if sp is None:
            raise RuntimeError("the reference failed a request: " + next(o["error"] for o in outs if "error" in o))
        log(f"  reference, with logprobs, set {k}: {sp['tokens_per_s']} tokens/s")
        res["with_logprobs"][str(k)] = {"speed": sp, "tokens": [o["tokens"] for o in outs],
                                        "top": [o["doc"]["top_logprobs"] for o in outs]}
    rp = RefPlain(cfg)
    for k in SETS:
        items = rotated(set_items(k), 3)
        outs = answers.engine_answers(cfg, rp, items, log)
        sp = answers.speed(items, outs, set())
        log(f"  reference, no logprobs, rotated set {k}: {sp and sp['tokens_per_s']} tokens/s")
        res["without_logprobs_rotated"][str(k)] = {"speed": sp, "errors": [o["error"] for o in outs if "error" in o]}
    save("ref_speed.json", res)


def ref_probe(cfg) -> int:
    """Do concurrent requests give the reference's serial answers?  Set 0 again, REF_WORKERS at a time,
    against the serial answers of ref_speed.  Any token difference: every later reference request is serial."""
    got = load("ref_probe.json")
    if got is None:
        serial = load("ref_speed.json")["with_logprobs"]["0"]
        items = set_items(0)
        top = cfg["check"]["engine_top"]
        docs = client.closed_loop(items, REF_WORKERS, lambda it: R.generate(cfg, it["ids"], it["max_tokens"], top))
        diff_tokens, max_dlp = 0, 0.0
        for d, st, stp in zip(docs, serial["tokens"], serial["top"]):
            if not isinstance(d, dict):
                diff_tokens += 1
                continue
            toks = d["tokens"] + ([d["stop"]["id"]] if d.get("stop") else [])
            if toks != st:
                diff_tokens += 1
                continue
            for a, b in zip(d["top_logprobs"], stp):
                for t, v in a.items():
                    if t in b:
                        max_dlp = max(max_dlp, abs(v - b[t]))
        got = {"workers_tried": REF_WORKERS, "prompts": len(items), "prompts_with_other_tokens": diff_tokens,
               "max_logprob_difference": round(max_dlp, 6), "workers": REF_WORKERS if diff_tokens == 0 else 1}
        save("ref_probe.json", got)
        log(f"  probe: {diff_tokens} of {len(items)} prompts differ under {REF_WORKERS} concurrent requests; "
            f"largest logprob difference {max_dlp:.5f}; the reference work runs {got['workers']} at a time")
    return int(got["workers"])


def walk_launch(cfg, sess, lim, name: str, r: int, workers: int) -> None:
    tag = f"{name}_r{r}"
    doc = load(f"launch_{tag}.json")
    if doc is None or load(f"walk_{tag}.json") is not None:
        return
    log(f"=== walk {tag}")
    out, pooled = {"config": name, "round": r, "sets": {}}, []
    for k in SETS:
        items = set_items(k)
        outs = [{"tokens": x["tokens"], "lps": x.get("lps") or [], "finish": x["finish"], "seconds": x["seconds"]}
                for x in doc["sets"][str(k)]["raw"]]

        def one(i, items=items, outs=outs):
            return answers.walk(cfg, sess, [items[i]], [outs[i]], lim, lambda m: None)[0]
        with cf.ThreadPoolExecutor(max_workers=workers) as ex:
            rows = list(ex.map(one, range(len(items))))
        objs = [x["item"] for x in rows]
        pooled += objs
        ok, reasons, agg, q = answers.decide(cfg, objs, lim, qb=None)
        ma = q.get("a") or {}
        log(f"  {tag} set {k}: mean gap {ma.get('mean_gap')}, top-k {ma.get('topk_share')}, agreement {agg.get('agreement')}")
        out["sets"][str(k)] = {"aggregate": agg, "quality_a": q, "reasons": reasons,
                               "per_item": [{"id": x["id"], "ok": x["item"].ok, "cut": x["item"].cut,
                                             "positions": x["item"].positions, "agreement": x["agreement"],
                                             "partings": len(x["item"].partings), "first_partings": x["partings"],
                                             "reason": x["item"].reason} for x in rows]}
    ok, reasons, agg, q = answers.decide(cfg, pooled, lim, qb=None)
    out["all_sets"] = {"aggregate": agg, "quality_a": q, "reasons": reasons}
    save(f"walk_{tag}.json", out)


def ref_tasks(cfg, sess, titems, tm, workers: int) -> None:
    if load("tasks_reference.json") is not None:
        return
    log(f"=== the reference answers the {len(titems)} tasks, {workers} at a time")
    t0 = time.monotonic()
    with cf.ThreadPoolExecutor(max_workers=workers) as ex:
        texts = list(ex.map(lambda it: answers.text_of(tm, sess.answer(it["ids"], it["max_tokens"])["tokens"]), titems))
    save("tasks_reference.json", {"seconds": round(time.monotonic() - t0, 1), "texts": texts})


def reference_block(cfg, tm, warm_rows, titems, todo: list[tuple], first: bool) -> None:
    """Start the reference once; do everything of this block that has no output yet; stop it."""
    missing = [(n, r) for n, r in todo if load(f"launch_{n}_r{r}.json") is not None and load(f"walk_{n}_r{r}.json") is None]
    if first:
        need = load("ref_speed.json") is None or load("ref_probe.json") is None or load("tasks_reference.json") is None
    else:
        need = False
    if not missing and not need:
        return
    stop_everything(cfg)
    log("=== reference block" + (" 1" if first else " 2"))
    dt = R.start(cfg, None, log)
    try:
        if first:
            ref_speed(cfg, tm.special_ids, warm_rows)
        workers = ref_probe(cfg) if load("ref_speed.json") is not None else 1
        sess = R.Session(cfg, __import__("contextlib").nullcontext, log)
        lim = answers.limits(cfg)
        for n, r in missing:
            walk_launch(cfg, sess, lim, n, r, workers)
        if first:
            ref_tasks(cfg, sess, titems, tm, workers)
        st = load("ref_starts.json") or []
        st.append({"block": 1 if first else 2, "ready_s": round(dt, 1), "utc": time.strftime("%FT%TZ", time.gmtime())})
        save("ref_starts.json", st)
    finally:
        R.stop(cfg)
        time.sleep(2)
        wait_gpu_free()


# ---- the summary (RULES.md, "Analysis") -----------------------------------------------------------------
def pct(xs: list[float], q: float) -> float:
    return xs[min(len(xs) - 1, max(0, int(q * (len(xs) - 1))))]


def ratio_ci(a: dict, b: dict, label: str) -> dict:
    """a, b: {prompt id: (tokens, seconds)} summed over rounds.  The ratio of pooled speeds a over b, with a
    95% percentile bootstrap over prompts (the same resampled prompts for both)."""
    ids = sorted(set(a) & set(b))

    def rat(sel):
        ta = sum(a[i][0] for i in sel); sa = sum(a[i][1] for i in sel)
        tb = sum(b[i][0] for i in sel); sb = sum(b[i][1] for i in sel)
        return (ta / sa) / (tb / sb)
    rng = random.Random(int.from_bytes(hashlib.sha256(f"{SEED}|{label}".encode()).digest()[:8], "little"))
    boots = sorted(rat([ids[rng.randrange(len(ids))] for _ in ids]) for _ in range(BOOT))
    return {"prompts": len(ids), "ratio": round(rat(ids), 4), "ci95": [round(pct(boots, 0.025), 4), round(pct(boots, 0.975), 4)]}


def mcnemar(b: int, c: int) -> float:
    n = b + c
    return 1.0 if n == 0 else min(1.0, 2.0 * sum(math.comb(n, i) for i in range(min(b, c) + 1)) * 0.5 ** n)


def paired_tasks(tasks, ta, tb, label: str) -> dict:
    oa = [t is not None and quality.correct(task, t) for task, t in zip(tasks, ta)]
    ob = [t is not None and quality.correct(task, t) for task, t in zip(tasks, tb)]
    d = [int(x) - int(y) for x, y in zip(oa, ob)]
    n = len(d)
    rng = random.Random(int.from_bytes(hashlib.sha256(f"{SEED}|{label}".encode()).digest()[:8], "little"))
    boots = sorted(sum(d[rng.randrange(n)] for _ in range(n)) / n for _ in range(BOOT))
    ao, bo = sum(x and not y for x, y in zip(oa, ob)), sum(y and not x for x, y in zip(oa, ob))
    return {"tasks": n, "a_correct": sum(oa), "b_correct": sum(ob), "delta": round(sum(d) / n, 4),
            "delta_ci95": [round(pct(boots, 0.025), 4), round(pct(boots, 0.975), 4)],
            "delta_lb95_one_sided": round(pct(boots, 0.05), 4), "a_only": ao, "b_only": bo,
            "mcnemar_exact_p": round(mcnemar(ao, bo), 6)}


def summary(cfg, tasks) -> None:
    names = CONFIGS + ["amd_seq1"]
    speed, per_prompt, walks, same = {}, {}, {}, {}
    for n in names:
        for r in ROUNDS:
            doc = load(f"launch_{n}_r{r}.json")
            if doc is None:
                continue
            tok = sum(doc["sets"][str(k)]["speed"]["tokens"] for k in SETS)
            sec = sum(doc["sets"][str(k)]["speed"]["seconds"] for k in SETS)
            speed.setdefault(n, {})[str(r)] = {
                "all_sets_tokens_per_s": round(tok / sec, 2), "ready_s": doc["ready_s"],
                "by_set": {str(k): doc["sets"][str(k)]["speed"]["tokens_per_s"] for k in SETS},
                "decode_tps": {str(k): (doc["parts"][str(k)].get("decode_tps") or {}).get("max") for k in SETS},
                "prompt_s": {str(k): (doc["parts"][str(k)].get("prompt_s") or {}).get("max") for k in SETS}}
            for k in SETS:
                for p in doc["sets"][str(k)]["speed"]["per_request"]:
                    t, s = per_prompt.setdefault(n, {}).get(p["id"], (0, 0.0))
                    per_prompt[n][p["id"]] = (t + p["tokens"], s + p["seconds"])
            first = load(f"launch_{n}_r1.json")
            if first is not None and r > 1:
                diff = sum(tok_hash(x["tokens"]) != tok_hash(y["tokens"]) for k in SETS
                           for x, y in zip(doc["sets"][str(k)]["raw"], first["sets"][str(k)]["raw"]))
                same.setdefault(n, {})[str(r)] = {"answers_differing_from_round_1": diff}
            w = load(f"walk_{n}_r{r}.json")
            if w is not None:
                q = w["all_sets"]["quality_a"]
                walks.setdefault(n, {})[str(r)] = {"measure_a": q.get("a"), 
                                                   "agreement": w["all_sets"]["aggregate"].get("agreement"),
                                                   "reasons": w["all_sets"]["reasons"],
                                                   "by_set": {k: v["quality_a"].get("a") for k, v in w["sets"].items()}}
    t0, t3 = load("launch_agent_tau0_r1.json"), load("launch_agent_tau030_r1.json")
    if t0 and t3:
        same["agent_tau030_vs_agent_tau0_round_1"] = {"answers_differing": sum(
            x["tokens"] != y["tokens"] for k in SETS for x, y in zip(t0["sets"][str(k)]["raw"], t3["sets"][str(k)]["raw"]))}
    rs = load("ref_speed.json")
    ref = None
    if rs:
        def pooled(part):
            t = sum(part[str(k)]["speed"]["tokens"] for k in SETS if part[str(k)]["speed"])
            s = sum(part[str(k)]["speed"]["seconds"] for k in SETS if part[str(k)]["speed"])
            return round(t / s, 2) if s else None
        ref = {"with_logprobs_tokens_per_s": pooled(rs["with_logprobs"]),
               "without_logprobs_rotated_tokens_per_s": pooled(rs["without_logprobs_rotated"]),
               "by_set_with_logprobs": {str(k): rs["with_logprobs"][str(k)]["speed"]["tokens_per_s"] for k in SETS}}
        per_prompt["reference"] = {p["id"]: (p["tokens"], p["seconds"]) for k in SETS
                                   for p in rs["with_logprobs"][str(k)]["speed"]["per_request"]}
    ratios = {}
    for a, b in (("agent_tau030", "amd_tuned"), ("agent_tau0", "amd_tuned"), ("agent_tau030", "agent_tau0"),
                 ("amd_seq1", "amd_tuned")):
        if a in per_prompt and b in per_prompt:
            ratios[f"{a}_over_{b}"] = ratio_ci(per_prompt[a], per_prompt[b], f"{a}/{b}")
    for a in names:
        # against the reference: round 1 only on both sides (the reference is timed once)
        d = load(f"launch_{a}_r1.json")
        if d is not None and "reference" in per_prompt:
            one = {p["id"]: (p["tokens"], p["seconds"]) for k in SETS for p in d["sets"][str(k)]["speed"]["per_request"]}
            ratios[f"{a}_round1_over_reference"] = ratio_ci(one, per_prompt["reference"], f"{a}/ref")
    tq, bar = {}, {}
    rt = load("tasks_reference.json")
    texts = {n: (load(f"tasks_{n}.json") or {}).get("texts") for n in CONFIGS}
    if rt:
        for n in CONFIGS:
            if texts[n]:
                tq[f"{n}_vs_reference"] = paired_tasks(tasks, texts[n], rt["texts"], f"{n}/ref")
                # the paper's fixed bar, as the judge applies it: measure A of round 1's walk, measure B of the tasks
                qa = (walks.get(n, {}).get("1") or {}).get("measure_a")
                qb = quality.measure_b(tasks, texts[n], rt["texts"], cfg["quality"]["tasks"]["bootstrap_resamples"])
                if qa is not None:
                    v = quality.verdict(cfg, qa, qb)
                    bar[n] = {"verdict": v["verdict"], "reasons": v["reasons"],
                              "measure_b": {k: qb[k] for k in qb if k not in ("per_task", "method")}}
                for kind in ("math", "mc"):
                    ix = [i for i, t in enumerate(tasks) if t["kind"] == kind]
                    if ix:
                        tq[f"{n}_vs_reference_{kind}"] = paired_tasks([tasks[i] for i in ix], [texts[n][i] for i in ix],
                                                                      [rt["texts"][i] for i in ix], f"{n}/ref/{kind}")
    for a, b in (("agent_tau030", "amd_tuned"), ("agent_tau0", "amd_tuned"), ("agent_tau030", "agent_tau0")):
        if texts.get(a) and texts.get(b):
            tq[f"{a}_vs_{b}"] = paired_tasks(tasks, texts[a], texts[b], f"{a}/{b}/tasks")
    save("summary.json", {"written_utc": time.strftime("%FT%TZ", time.gmtime()), "speed": speed, "reference": ref,
                          "speed_ratios": ratios, "bar": bar, "walks": walks, "repeat_answers": same, "tasks": tq,
                          "arms": {n: load(f"arms_{n}.json") for n in CONFIGS if load(f"arms_{n}.json")},
                          "probe": load("ref_probe.json"), "failures": load("failures.json") or []})
    log("summary.json written")


def manifest(tasks) -> None:
    if load("manifest.json") is not None:
        return
    def run(cmd):
        try:
            r = subprocess.run(cmd, capture_output=True, text=True, timeout=180)
            return r.stdout.strip()[:4000] if r.returncode == 0 else f"ERR {r.stderr.strip()[:200]}"
        except Exception as e:  # noqa: BLE001
            return f"ERR {e}"
    m = {"created_utc": time.strftime("%FT%TZ", time.gmtime()), "standin": STANDIN, "seed": SEED,
         "round_orders": {str(r): round_order(r) for r in ROUNDS}, "set_orders": {str(r): set_order(r) for r in ROUNDS},
         "files": {p.name: sha_file(p) for p in sorted(DATA.glob("*")) if p.is_file()},
         "this_script": sha_file(Path(__file__).resolve()),
         "rules": sha_file(Path(__file__).resolve().parent / "RULES.md") if (Path(__file__).resolve().parent / "RULES.md").exists() else None,
         "tasks": {"count": len(tasks), "kinds": {k: sum(t["kind"] == k for t in tasks) for k in sorted({t["kind"] for t in tasks})}},
         "gpu": gpu_state()}
    if not STANDIN:
        m["tau_in_image"] = {t: run(["docker", "run", "--rm", "--device", "/dev/kfd", "--device", "/dev/dri", "--group-add", "video", "-e", f"ENG_LOSSY_TAU={t}", "ij-agent-657edee:paper", "python3", "-c",
                                     "import eng.fast as f; print(f.LOSSY_TAU, f.lossy_params(f.LOSSY_TAU, f.LOSSY_PRICE))"])
                             for t in ("0", "0.3")}
        m.update({"uname": run(["uname", "-a"]), "amd_smi": run(["amd-smi", "static"]),
                  "rocm": run(["cat", "/opt/rocm/.info/version"]),
                  "images": {i: run(["docker", "image", "inspect", "-f", "{{.Id}}", i]) for i in
                             ("ij-atom-ref:ad256a94-lp1", "ij-atom-tuned:dev973", "ij-amd-tuned:paper", "ij-agent-657edee:paper")}})
    save("manifest.json", m)


def main() -> int:
    cfg = C.load()
    C.state_dir(cfg)
    tm = tokmap.load(cfg["model"]["dir"])
    warm_rows = [r["prompt_token_ids"] for r in rows_of(DATA / "warm.jsonl")]
    tasks = rows_of(DATA / "tasks_512.jsonl")[:8 if SMOKE else None]
    titems = [{"id": t["id"], "ids": [int(x) for x in t["prompt_token_ids"]],
               "max_tokens": int(t.get("max_tokens") or TASK_MAX_TOKENS), "task": t} for t in tasks]
    if sys.argv[1:2] == ["summary"]:
        summary(cfg, tasks)
        return 0
    manifest(tasks)
    if SMOKE:
        launches(cfg, [(n, 1, True, True) for n in round_order(1)], tm, warm_rows, titems)
        reference_block(cfg, tm, warm_rows, titems, [(n, 1) for n in CONFIGS], first=True)
        summary(cfg, tasks)
        log("=== smoke run done; nothing in out-smoke/ is a result")
        return 0
    # 1. round 1, with the tasks
    launches(cfg, [(n, 1, True, False) for n in round_order(1)], tm, warm_rows, titems)
    # 2. reference block 1
    reference_block(cfg, tm, warm_rows, titems, [(n, 1) for n in CONFIGS], first=True)
    summary(cfg, tasks)
    # 3. rounds 2 and 3; the arms ride on round 3
    launches(cfg, [(n, r, False, r == 3) for r in (2, 3) for n in round_order(r)], tm, warm_rows, titems)
    # 4. AMD with one sequence
    launches(cfg, [("amd_seq1", 1, False, False)], tm, warm_rows, titems)
    # 5. reference block 2
    reference_block(cfg, tm, warm_rows, titems, [(n, r) for r in (2, 3) for n in CONFIGS] + [("amd_seq1", 1)], first=False)
    summary(cfg, tasks)
    log("=== all steps done")
    return 0


if __name__ == "__main__":
    sys.exit(main())
