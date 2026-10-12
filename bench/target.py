#!/usr/bin/env python3
"""The target file: the one place that names a chip, an engine or a model.

A target is `targets/<id>.yaml` (DESIGN 2.1). run.sh, quality2 and the shell
tools (bootstrap, check, rehearse) read it, and nothing else in the ruler may
hard-code an accelerator, an engine flag or a model path.

Interface:

  load_target(path) -> dict        validated, with defaults filled in;
                                   raises TargetError naming the bad field
  target_hash(target) -> str       sha256 of the canonical JSON of the dict
  ruler_hash(bench_root, target_file=None) -> str
                                   DESIGN 2.6: sha256 over the ruler files
                                   and the target file, sorted by file name,
                                   each fed as  name NUL bytes NUL
  seal_notes(target) -> [str]      what the target's docker flags give up from
                                   the sealed launch ("privileged",
                                   "network=host", "device <host path>");
                                   [] for a fully sealed one

CLI (shell callers read the target only through this):

  python -I target.py get FILE dotted.key     a scalar, or JSON for list/dict
  python -I target.py validate FILE
  python -I target.py hash FILE               target_hash
  python -I target.py ruler-hash BENCH_ROOT [TARGET_FILE]

Validation is strict about the fields the ruler reads and lenient about whole
sections it does not (provision, probes, ...: bootstrap's). Docker flags
that would undo the seal are refused outright: published ports, writable
mounts, a bridge network, and --gpus (that comes from accelerator.kind).
--privileged, --net=host and --device are allowed, because a TPU VM may need
them, and are reported by seal_notes so that results.json can say the seal is
weaker: a container holding a host device is not fully sealed.
"""
from __future__ import annotations

import copy
import hashlib
import json
import os
import re
import shlex
import sys
from pathlib import Path
from typing import Any

import yaml

RULER_FILES = ("harness.py", "metrics.py", "compare.py", "quality2.py",
               "stats.py", "tokenizer.py", "target.py", "run.sh")

ID_RE = re.compile(r"^[a-z0-9][a-z0-9-]*$")
METRIC_KINDS = ("tps",)
DESIGNS = ("abba", "ici")


class TargetError(ValueError):
    """A target file the ruler must not run with."""


# --------------------------------------------------------------- the schema --
# section -> {key: (checker, required, default)}. A section's unknown keys are
# refused (a typo in `window_s` must not silently fall back to a default);
# unknown top-level sections are kept for the tools that own them.
def _str(v):
    return isinstance(v, str)


def _nonempty(v):
    return isinstance(v, str) and v != ""


def _int_pos(v):
    return isinstance(v, int) and not isinstance(v, bool) and v >= 1


def _int_nonneg(v):
    return isinstance(v, int) and not isinstance(v, bool) and v >= 0


def _num_pos(v):
    return isinstance(v, (int, float)) and not isinstance(v, bool) and v > 0


def _num_nonneg(v):
    return isinstance(v, (int, float)) and not isinstance(v, bool) and v >= 0


def _bool(v):
    return isinstance(v, bool)


def _str_list(v):
    return isinstance(v, list) and all(isinstance(x, str) for x in v)


def _levels(v):
    return isinstance(v, list) and len(v) > 0 and all(_int_pos(x) for x in v)


def _any(v):
    return True


SOURCE_RE = re.compile(r"^[a-z][a-z0-9_]*$")
# where run.sh itself mounts or copies things: a cache may not shadow them
RESERVED_MOUNTS = ("/ro", "/exp", "/src", "/out", "/opt/models", "/opt/inference-judge")


def _sources(v):
    """Engine source trees by name: <name>-src on the host, /src/<name> in
    the launch. [] when the researcher edits no engine tree."""
    return (isinstance(v, list) and len(set(map(str, v))) == len(v)
            and all(isinstance(x, str) and SOURCE_RE.match(x) for x in v))


def _sizes(v):
    """metric.isl: one input size, or a list of distinct sizes served as a
    balanced mix."""
    if isinstance(v, list):
        return len(v) > 0 and all(_int_pos(x) for x in v) and len(set(v)) == len(v)
    return _int_pos(v)


def _plain_abs(v):
    return (isinstance(v, str) and v.startswith("/") and v == v.rstrip("/")
            and not re.search(r"[\s:,]", v))


def _compile_cache(v):
    """{seed: host dir the ruler owns, mount: where a launch sees its private
    copy}."""
    if not (isinstance(v, dict) and set(v) == {"seed", "mount"}
            and _plain_abs(v["seed"]) and _plain_abs(v["mount"])):
        return False
    return not any(v["mount"] == r or v["mount"].startswith(r + "/")
                   for r in RESERVED_MOUNTS)


REQ, OPT = True, False
SCHEMA: dict[str, dict[str, tuple]] = {
    "accelerator": {
        # the target's own word for its hardware; the ruler acts on "gpu" only
        "kind": (lambda v: isinstance(v, str) and bool(ID_RE.match(v)), REQ, None),
        "chips": (_int_pos, REQ, None),
        "hosts": (_int_pos, REQ, None),
        "type": (_str, OPT, None),
        "topology": (_str, OPT, None),
        "hbm_gb_per_chip": (_num_pos, OPT, None),
    },
    "model": {
        "served_name": (_nonempty, REQ, None),
        "revision": (_str, REQ, None),
        "local_dir": (_str, REQ, None),
        "tokenizer": (_nonempty, REQ, None),
        "chat_template": (_str, OPT, ""),
        "hf_id": (_str, OPT, None),
        "gcs": (_str, OPT, None),
    },
    "engine": {
        "name": (_nonempty, REQ, None),
        "image": (_str, REQ, None),
        "docker_flags": (_str_list, REQ, None),
        "env": (lambda v: isinstance(v, dict) and all(
            isinstance(k, str) and isinstance(x, str) for k, x in v.items()), OPT, {}),
        "launch": (_any, OPT, None),
        "port": (_int_pos, OPT, 8000),
        "shm_size": (_str, OPT, ""),
        "sources": (_sources, OPT, []),
        "compile_cache": (_compile_cache, OPT, None),
    },
    "metric": {
        "kind": (lambda v: v in METRIC_KINDS, REQ, None),
        "isl": (_sizes, REQ, None),
        "osl": (_int_pos, REQ, None),
        "concurrency": (_levels, REQ, None),
        "slo": (lambda v: isinstance(v, dict), REQ, None),
        "ramp_s": (_num_nonneg, REQ, None),
        "window_s": (_num_pos, REQ, None),
        # set: the candidate is first launched alone for one window this long
        # and screened before any incumbent launch
        "screen_window_s": (_num_pos, OPT, None),
        "design": (lambda v: v in DESIGNS, OPT, "abba"),
        "blocks": (_int_pos, OPT, 2),
        "prompt_pool": (_str, OPT, ""),
        "confirm_pool": (_str, OPT, ""),
        "pool_size": (_int_pos, OPT, 2048),
        "length_tol": (_num_nonneg, OPT, 0.0),
        "guard_tol": (_num_nonneg, OPT, 0.03),
        "screen_tol": (_num_nonneg, OPT, 0.02),
        "stall_ms": (_num_pos, OPT, 1000),
    },
    "budgets": {
        "build_s": (_int_nonneg, REQ, None),
        "compile_s": (_int_nonneg, REQ, None),
        "health_s": (_int_pos, REQ, None),
        "run_s": (_int_pos, REQ, None),
    },
    # ready_timeout_s: how long a launch may take to answer /v1/models. Left
    # out, it is compile_s + health_s (a server that compiles before it
    # serves). A server that answers first and compiles on its first requests
    # names its own, so a hung start does not cost the compile budget too.
    "health": {
        "url": (_str, OPT, None),
        "ready_timeout_s": (_int_pos, OPT, None),
    },
    "reference": {
        "dir": (_nonempty, REQ, None),
        "prompts": (_str, OPT, ""),
        "tasks": (_str, OPT, ""),
        # R as the baseline writes it: tokens per continuation, and how many
        # cold launches its floors come from. Unset: 128 and 3 for a real
        # engine; 64 and one server measured three times for the mock.
        "max_tokens": (_int_pos, OPT, None),
        "launches": (_int_pos, OPT, None),
    },
    "quality": {
        "allow_missing_logprobs": (_bool, OPT, False),
    },
}
OPTIONAL_SECTIONS = ("quality", "health")
SLO_KEYS = ("p99_tpot_ms", "p95_ttft_ms")


def _fail(field: str, why: str) -> None:
    raise TargetError(f"{field}: {why}")


def _check_section(name: str, sec: Any) -> dict:
    if not isinstance(sec, dict):
        _fail(name, "must be a mapping")
    spec = SCHEMA[name]
    for key in sec:
        if key not in spec:
            _fail(f"{name}.{key}", f"unknown key (known: {', '.join(spec)})")
    out = dict(sec)
    for key, (ok, required, default) in spec.items():
        if key not in sec:
            if required:
                _fail(f"{name}.{key}", "missing")
            if default is not None:
                out[key] = copy.deepcopy(default)
            continue
        if not ok(sec[key]):
            _fail(f"{name}.{key}", f"bad value {sec[key]!r}")
    return out


def _check_metric(m: dict) -> None:
    for key in SLO_KEYS:
        if not _num_pos(m["slo"].get(key)):
            _fail(f"metric.slo.{key}", f"missing or not positive: {m['slo'].get(key)!r}")


def _mount_is_ro(spec: str) -> bool:
    parts = spec.split(":")
    return len(parts) >= 3 and "ro" in parts[2].split(",")


def _check_docker_flags(flags: list[str]) -> None:
    """Refuse every flag that would quietly undo the seal run.sh builds."""
    i = 0
    while i < len(flags):
        f = flags[i]
        key, _, inline = f.partition("=")
        val = inline if "=" in f else (flags[i + 1] if i + 1 < len(flags) else "")
        takes = "=" not in f and key in ("-p", "--publish", "-v", "--volume",
                                         "--mount", "--network", "--net", "--gpus",
                                         "--device")
        if key == "--device" and not val.startswith("/dev/"):
            _fail("engine.docker_flags", f"--device {val}: not a /dev path")
        if key in ("-p", "--publish", "-P", "--publish-all"):
            _fail("engine.docker_flags", f"{f}: no published ports, the ruler "
                  "reaches the server from inside its network namespace")
        if key in ("-v", "--volume") and not _mount_is_ro(val):
            _fail("engine.docker_flags", f"{f} {val}: writable mount (add :ro)")
        if key == "--mount":
            kv = [p.split("=", 1)[0] for p in val.split(",")]
            if "readonly" not in kv and "ro" not in kv and "type=tmpfs" not in val:
                _fail("engine.docker_flags", f"--mount {val}: writable mount "
                      "(add readonly)")
        if key in ("--network", "--net") and val not in ("host", "none"):
            _fail("engine.docker_flags", f"{key} {val}: only none (the default) "
                  "or host (recorded as a weaker seal)")
        if key == "--gpus":
            _fail("engine.docker_flags", "--gpus comes from accelerator.kind gpu")
        i += 2 if takes else 1


def seal_notes(t: dict) -> list[str]:
    """What this target's docker flags give up from the sealed launch."""
    flags = t["engine"]["docker_flags"]
    notes = []
    if "--privileged" in flags:
        notes.append("privileged")
    for i, f in enumerate(flags):
        key, _, inline = f.partition("=")
        val = inline if "=" in f else (flags[i + 1] if i + 1 < len(flags) else "")
        if key in ("--network", "--net") and val == "host":
            notes.append("network=host")
        if key == "--device":           # host[:container[:permissions]]
            notes.append("device " + val.split(":", 1)[0])
    return notes


def validate(raw: Any) -> dict:
    if not isinstance(raw, dict):
        raise TargetError("target: must be a mapping")
    t = dict(raw)
    if not (isinstance(t.get("id"), str) and ID_RE.match(t["id"])):
        _fail("id", f"missing or not [a-z0-9-]+: {t.get('id')!r}")
    for name in SCHEMA:
        if name not in t:
            if name in OPTIONAL_SECTIONS:
                t[name] = {}
            else:
                _fail(name, "missing section")
        t[name] = _check_section(name, t[name])
    _check_metric(t["metric"])
    _check_docker_flags(t["engine"]["docker_flags"])
    if "runnable" in t and not _bool(t["runnable"]):
        _fail("runnable", f"bad value {t['runnable']!r}")
    t.setdefault("runnable", True)
    return t


def load_target(path: str | Path) -> dict:
    p = Path(path)
    try:
        raw = yaml.safe_load(p.read_text())
    except OSError as exc:
        raise TargetError(f"{p}: {exc}") from exc
    except yaml.YAMLError as exc:
        raise TargetError(f"{p}: not YAML: {exc}") from exc
    return validate(raw)


def target_hash(t: dict) -> str:
    return hashlib.sha256(json.dumps(t, sort_keys=True, separators=(",", ":"))
                          .encode()).hexdigest()


def ruler_hash(bench_root: str | Path, target_file: str | Path | None = None) -> str:
    root = Path(bench_root)
    files = [root / n for n in RULER_FILES if (root / n).exists()]
    if target_file is not None and Path(target_file).exists():
        files.append(Path(target_file))
    h = hashlib.sha256()
    for f in sorted(files, key=lambda p: p.name):
        h.update(f.name.encode() + b"\0" + f.read_bytes() + b"\0")
    return h.hexdigest()


# ---------------------------------------------------------------------- CLI --
def _get(t: dict, dotted: str) -> Any:
    value: Any = t
    for key in dotted.split("."):
        if not isinstance(value, dict) or key not in value:
            raise KeyError(dotted)
        value = value[key]
    return value


def _show(value: Any) -> str:
    if isinstance(value, (dict, list)):
        return json.dumps(value)
    if isinstance(value, bool):
        return "true" if value else "false"
    return "" if value is None else str(value)


# --------------------------------------------------------------------------
# The ruler lock (DESIGN 2.6): written once at clock start, checked by every
# run.sh invocation after it.
# --------------------------------------------------------------------------
def write_lock(bench_root: str | Path, target_file: str | Path,
               lock_file: str | Path) -> str:
    """Write the lock for the live ruler and target. Refuses to overwrite: a
    lock that could be rewritten by whoever can run run.sh would lock nothing.
    The operator deletes the old lock by hand to re-lock."""
    lock = Path(lock_file)
    h = ruler_hash(bench_root, target_file)
    text = f"{h}\ntarget {Path(target_file).name}\n"
    try:
        fd = os.open(lock, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    except FileExistsError:
        raise TargetError(f"ruler lock {lock} already exists; remove it by hand "
                          "to re-lock") from None
    with os.fdopen(fd, "w") as fh:
        fh.write(text)
    return h


def check_lock(bench_root: str | Path, target_file: str | Path,
               lock_file: str | Path) -> tuple[bool, str]:
    """(ok, message). The lock's first token is the hash. A `target NAME` line
    says the target file is covered too; a lock without one (a bare hash, the
    way the scorecard writes it) covers the bench files only."""
    lock = Path(lock_file)
    try:
        words = lock.read_text().split()
    except OSError as exc:
        return False, f"ruler lock {lock} unreadable: {exc}"
    if not words:
        return False, f"ruler lock {lock} is empty"
    want, covered = words[0], None
    if "target" in words[1:]:
        covered = words[words.index("target", 1) + 1] \
            if words.index("target", 1) + 1 < len(words) else ""
        if covered != Path(target_file).name:
            return False, (f"ruler lock {lock} was taken for target {covered!r}, "
                           f"this run's target is {Path(target_file).name!r}")
    live = ruler_hash(bench_root, target_file if covered else None)
    if live != want:
        return False, (f"ruler changed since the lock: live ruler_hash {live[:12]} "
                       f"!= locked {want[:12]} ({lock}); a ruler file was edited "
                       "after clock start")
    return True, f"ruler_hash {live[:12]} matches {lock}"


def shell_vars(t: dict, bench_root: str | Path) -> str:
    """The target as bash assignments for run.sh to eval, every value quoted.
    Lists come out as bash arrays; the reference dir is made absolute."""
    m, e, b = t["metric"], t["engine"], t["budgets"]
    ref = Path(t["reference"]["dir"])
    if not ref.is_absolute():
        ref = Path(bench_root) / ref
    isl = m["isl"] if isinstance(m["isl"], list) else [m["isl"]]
    cache = e.get("compile_cache") or {}
    scalars = {
        "T_ID": t["id"], "T_RUNNABLE": "1" if t["runnable"] else "0",
        "T_HASH": target_hash(t),
        "T_ACCEL": t["accelerator"]["kind"], "T_CHIPS": t["accelerator"]["chips"],
        "T_SERVED_NAME": t["model"]["served_name"],
        "T_MODEL_DIR": t["model"]["local_dir"],
        "T_REVISION": t["model"]["revision"], "T_TOKENIZER": t["model"]["tokenizer"],
        "T_ENGINE": e["name"], "T_IMAGE": e["image"], "T_PORT": e["port"],
        "T_SHM": e.get("shm_size") or "",
        "T_CACHE_SEED": cache.get("seed", ""), "T_CACHE_MOUNT": cache.get("mount", ""),
        "T_METRIC": m["kind"],
        "T_ISL": ",".join(str(x) for x in isl), "T_OSL": m["osl"],
        "T_CONCURRENCY": ",".join(str(c) for c in m["concurrency"]),
        "T_SLO": json.dumps(m["slo"], separators=(",", ":")),
        "T_RAMP_S": m["ramp_s"], "T_WINDOW_S": m["window_s"],
        "T_SCREEN_WINDOW_S": m.get("screen_window_s") or "",
        "T_LENGTH_TOL": m.get("length_tol", ""),
        "T_DESIGN": m.get("design", ""),
        "T_BLOCKS": m.get("blocks", ""),
        "T_POOL_SIZE": m.get("pool_size", ""),
        "T_PROMPT_POOL": m.get("prompt_pool", "") or "",
        "T_CONFIRM_POOL": m.get("confirm_pool", "") or "",
        "T_GUARD_TOL": m.get("guard_tol", ""),
        "T_SCREEN_TOL": m.get("screen_tol", ""),
        "T_STALL_MS": m.get("stall_ms", ""),
        "T_BUILD_S": b["build_s"], "T_COMPILE_S": b["compile_s"],
        "T_HEALTH_S": b["health_s"], "T_RUN_S": b["run_s"],
        "T_READY_S": (t["health"].get("ready_timeout_s")
                      or b["compile_s"] + b["health_s"]),
        "T_REF_DIR": str(ref),
        "T_REF_PROMPTS": t["reference"].get("prompts") or "",
        "T_REF_TASKS": t["reference"].get("tasks") or "",
        "T_REF_MAX_TOKENS": (t["reference"].get("max_tokens")
                             or (64 if e["name"] == "mock" else 128)),
        "T_REF_LAUNCHES": (t["reference"].get("launches")
                           or ("" if e["name"] == "mock" else 3)),
        "T_SEAL_NOTES": ",".join(seal_notes(t)),
    }
    lines = [f"{k}={shlex.quote(str(v))}" for k, v in scalars.items()]
    env = [f"{k}={v}" for k, v in sorted((e.get("env") or {}).items())]
    for name, items in (("T_DOCKER_FLAGS", e["docker_flags"]), ("T_ENV", env),
                        ("T_SOURCES", e["sources"])):
        lines.append(f"{name}=(" + " ".join(shlex.quote(str(x)) for x in items) + ")")
    return "\n".join(lines) + "\n"


def main(argv: list[str]) -> int:
    usage = ("usage: target.py get FILE dotted.key | validate FILE | hash FILE | "
             "shell FILE BENCH_ROOT | ruler-hash BENCH_ROOT [TARGET_FILE] | "
             "lock BENCH_ROOT TARGET_FILE LOCK_FILE | "
             "check-lock BENCH_ROOT TARGET_FILE LOCK_FILE")
    if len(argv) < 2:
        print(usage, file=sys.stderr)
        return 2
    cmd = argv[0]
    try:
        if cmd == "ruler-hash":
            print(ruler_hash(argv[1], argv[2] if len(argv) > 2 else None))
            return 0
        if cmd == "lock" and len(argv) == 4:
            load_target(argv[2])
            print(write_lock(argv[1], argv[2], argv[3]))
            return 0
        if cmd == "check-lock" and len(argv) == 4:
            ok, msg = check_lock(argv[1], argv[2], argv[3])
            print(msg, file=sys.stdout if ok else sys.stderr)
            return 0 if ok else 1
        t = load_target(argv[1])
        if cmd == "validate":
            print(f"{argv[1]}: ok ({t['id']}, {t['metric']['kind']}, "
                  f"{'runnable' if t['runnable'] else 'NOT runnable'})")
            return 0
        if cmd == "hash":
            print(target_hash(t))
            return 0
        if cmd == "shell" and len(argv) == 3:
            print(shell_vars(t, argv[2]), end="")
            return 0
        if cmd == "get" and len(argv) == 3:
            print(_show(_get(t, argv[2])))
            return 0
    except TargetError as exc:
        print(f"target.py: {exc}", file=sys.stderr)
        return 1
    except KeyError as exc:
        print(f"target.py: no key {exc.args[0]} in {argv[1]}", file=sys.stderr)
        return 1
    print(usage, file=sys.stderr)
    return 2


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
