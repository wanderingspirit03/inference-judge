"""Bash-command tripwire for the accelerators and the frozen files.

Called by pretooluse-bash.sh for every command:

    IJ_HOOK_CMD=<command> IJ_HOOK_CWD=<cwd> python3 -I _guard.py

Prints one line, "OK" or "<KIND>\t<detail>", KIND being
    ACCEL    the command reaches an accelerator: nvidia-smi or tpu-info under
             any spelling, a /dev/nvidia*, /dev/accel* or /dev/vfio path, or
             host python importing torch/jax/libtpu/vllm/sglang
    FROZEN   a write-shaped command touching a frozen path: loop/, guard/,
             adapters/, the agent's settings folder or the driver's state dir
             (IJ_PROTECTED_ROOTS)

This is a TRIPWIRE.  It sees the command line, not what a script does once it
runs.  The boundary is setup/harden_research.sh: the frozen files are
root-owned and read-only, the state dir belongs to the driver user, the
devices belong to group bench, and the researcher's venv has no CUDA/TPU
stack.  The tripwire's job is to fail loudly and early, with a message that
says what to do instead.
"""
from __future__ import annotations

import fnmatch
import os
import re
import shlex
import sys

ACCEL_BINS = {"nvidia-smi", "tpu-info", "nvidia-debugdump", "nvidia-persistenced", "dcgmi",
              "nvtop", "gpustat", "nvidia-cuda-mps-control", "rocm-smi", "nvitop"}
DEV_RE = re.compile(r"/dev/(nvidia|accel|vfio)")
ACCEL_MODS = ("torch", "jax", "jaxlib", "libtpu", "vllm", "sglang", "sgl_jax", "tensorflow",
              "triton", "cupy", "pycuda", "pynvml", "flashinfer", "torch_xla", "tpu_info")
_MODS = "|".join(ACCEL_MODS)
IMPORT_RE = re.compile(r"(?:^|[\s;:(\"'])(?:import\s+(?:[\w.]+\s*(?:as\s+\w+\s*)?,\s*)*|from\s+)"
                       r"(?P<mod>" + _MODS + r")\b"
                       r"|(?:__import__|import_module)\(\s*['\"](?P<dyn>" + _MODS + r")\b", re.M)
PY_RE = re.compile(r"^(python|python[23](\.\d+)?|pypy3?|ipython3?)$")

WRITE_CMDS = {"tee", "rm", "rmdir", "mv", "cp", "chmod", "chown", "chgrp", "ln", "dd",
              "truncate", "install", "touch", "mkdir", "shred", "unlink", "rsync", "patch",
              "vi", "vim", "nvim", "nano", "ed", "ex", "emacs", "pico", "setfacl", "chattr",
              "tar", "unzip", "cpio", "mkfifo", "mknod", "sponge"}
# commands that, aimed at a directory ABOVE a frozen root, destroy it too
ANCESTOR_CMDS = {"rm", "rmdir", "mv", "chmod", "chown", "chgrp", "rsync", "find", "git", "setfacl"}
GIT_WRITES = {"checkout", "restore", "reset", "apply", "am", "stash", "mv", "rm", "clean",
              "pull", "merge", "rebase", "cherry-pick", "revert", "switch", "worktree"}
PY_WRITE_RE = re.compile(r"""(['"][rwxa]?[wxa]\+?b?['"]|mode\s*=|write_text|write_bytes|\.write\(|"""
                         r"""shutil\.|os\.(remove|unlink|rename|replace|chmod|symlink|truncate|"""
                         r"""rmdir|removedirs|makedirs|mkdir)|rmtree|\.unlink\(|\.rename\(|\.touch\()""")
WRAPPERS = {"sudo", "env", "timeout", "nice", "ionice", "nohup", "stdbuf", "command", "builtin",
            "exec", "time", "xargs", "setsid", "chrt", "taskset", "unbuffer", "doas"}
WRAPPER_ARG_OPTS = {  # options that take a separate argument
    "sudo": {"-u", "-g", "-C", "-D", "-h", "-p", "-r", "-t", "-U"},
    "xargs": {"-n", "-I", "-P", "-L", "-d", "-s", "-E", "-a", "--max-args", "--max-procs"},
    "timeout": {"-s", "-k", "--signal", "--kill-after"},
    "nice": {"-n"}, "ionice": {"-c", "-n"}, "stdbuf": {"-i", "-o", "-e"},
    "chrt": set(), "taskset": set(), "env": {"-u", "-C", "-S"},
}
SHELLS = {"bash", "sh", "dash", "zsh", "ksh"}
KEYWORDS = {"do", "then", "else", "elif", "if", "!", "{", "}", "while", "until", "for", "case"}


def scan(s: str):
    """Split on unquoted operators.  Returns (segments, redirect_targets)."""
    segs, cur, targets = [], [], []
    q = None
    i, n = 0, len(s)
    while i < n:
        c = s[i]
        if q == "'":
            cur.append(c)
            q = None if c == "'" else q
            i += 1
            continue
        if q == '"':
            if c == "\\" and i + 1 < n:
                cur.append(s[i:i + 2]); i += 2; continue
            cur.append(c)
            q = None if c == '"' else q
            i += 1
            continue
        if c == "\\" and i + 1 < n:
            cur.append(s[i:i + 2]); i += 2; continue
        if c in "'\"":
            q = c; cur.append(c); i += 1; continue
        if c == ">" or (c == "<" and not s.startswith("<<", i)):
            write = c == ">"
            i += 1
            while i < n and s[i] in ">|&":
                i += 1
            while i < n and s[i] in " \t":
                i += 1
            j, tq = i, None
            while j < n:
                d = s[j]
                if tq:
                    tq = None if d == tq else tq
                elif d in "'\"":
                    tq = d
                elif d in " \t\n|;&<>()`":
                    break
                j += 1
            word = s[i:j]
            if write and word and not re.match(r"^&?\d*-?$", word):
                try:
                    targets.append(shlex.split(word)[0])
                except (ValueError, IndexError):
                    targets.append(word)
            i = j
            continue
        if c == "<" and s.startswith("<<", i):
            cur.append("<<"); i += 2; continue
        if c in "|;&\n`()":
            segs.append("".join(cur)); cur = []; i += 1; continue
        if c == "$" and i + 1 < n and s[i + 1] == "(":
            segs.append("".join(cur)); cur = []; i += 2; continue
        cur.append(c); i += 1
    segs.append("".join(cur))
    return segs, targets


def tokens(seg: str):
    try:
        return shlex.split(seg, comments=False, posix=True)
    except ValueError:
        return None


def unwrap(toks: list[str]) -> list[str]:
    """Strip keywords, VAR=value prefixes and wrapper commands (sudo, env,
    timeout, xargs ...) with their options, to reach the real command."""
    toks = list(toks)
    while toks:
        t = toks[0]
        if t in KEYWORDS or re.match(r"^[A-Za-z_][A-Za-z0-9_]*=", t):
            toks.pop(0); continue
        b = os.path.basename(t)
        if b in WRAPPERS:
            toks.pop(0)
            takes = WRAPPER_ARG_OPTS.get(b, set())
            while toks and (toks[0].startswith("-") or re.match(r"^[A-Za-z_][A-Za-z0-9_]*=", toks[0])):
                o = toks.pop(0)
                if o in takes and toks:
                    toks.pop(0)
            if b == "timeout" and toks:
                toks.pop(0)           # the duration
            continue
        break
    return toks


DEST_ONLY = {"cp", "install", "ln", "rsync", "scp"}   # only the destination is written


def dest_args(args: list[str]) -> list[str]:
    """The destination of cp/install/ln/rsync: -t DIR, or the last operand."""
    out, pos, k = [], [], 0
    while k < len(args):
        a = args[k]
        if a in ("-t", "--target-directory") and k + 1 < len(args):
            out.append(args[k + 1]); k += 2; continue
        if a.startswith("--target-directory="):
            out.append(a.split("=", 1)[1])
        elif not a.startswith("-"):
            pos.append(a)
        k += 1
    return out + pos[-1:]


def git_subcommand(args: list[str]) -> str:
    k = 0
    while k < len(args):
        a = args[k]
        if a in ("-C", "-c", "--git-dir", "--work-tree", "--namespace"):
            k += 2; continue
        if a.startswith("-"):
            k += 1; continue
        return a
    return ""


class Guard:
    def __init__(self, cwd: str, home: str, roots: list[str]):
        self.cwd = cwd or "/"
        self.home = home
        self.roots = [os.path.normpath(r) for r in roots if r]

    def norm(self, tok: str) -> str:
        if tok.startswith("~/") or tok == "~":
            tok = self.home + tok[1:]
        elif tok.startswith("$HOME/"):
            tok = self.home + tok[5:]
        if not tok.startswith("/"):
            tok = os.path.join(self.cwd, tok)
        return os.path.normpath(tok)

    def hits(self, tok: str, ancestors: bool) -> str | None:
        """The frozen root this token names: a path inside it, a glob that
        matches it, or (ancestors=True) a directory above it."""
        cands = [tok]
        if "=" in tok and not tok.startswith("="):
            cands.append(tok.split("=", 1)[1])
        for c in cands:
            if not c or (c.startswith("-") and "/" not in c):
                continue
            p = self.norm(c)
            glob = any(ch in c for ch in "*?[")
            for r in self.roots:
                if p == r or p.startswith(r + "/"):
                    return r
                if glob and fnmatch.fnmatch(r, p):
                    return r
                if ancestors and r.startswith(p.rstrip("/") + "/"):
                    return r
        return None

    def check(self, cmd: str, depth: int = 0) -> tuple[str, str] | None:
        if DEV_RE.search(cmd):
            return "ACCEL", DEV_RE.search(cmd).group(0) + "*"
        segs, targets = scan(cmd)
        for t in targets:
            r = self.hits(t, ancestors=False)
            if r:
                return "FROZEN", f"a redirect into {r}"
        runs_python = False
        for seg in segs:
            raw = tokens(seg)
            if raw is None:
                continue                      # unbalanced quoting: other checks still apply
            for t in raw:
                if os.path.basename(t) in ACCEL_BINS:
                    return "ACCEL", os.path.basename(t)
            toks = unwrap(raw)
            if not toks:
                continue
            prog = os.path.basename(toks[0])
            args = toks[1:]
            if prog in ("cd", "pushd") and args:
                self.cwd = self.norm(args[-1])      # follow cd within one command line
                continue
            if prog in SHELLS or prog == "eval":
                inner = None
                if prog == "eval":
                    inner = " ".join(args)
                else:
                    for k, a in enumerate(args):
                        if re.match(r"^-[A-Za-z]*c[A-Za-z]*$", a) and k + 1 < len(args):
                            inner = args[k + 1]
                            break
                if inner is not None and depth < 3:
                    v = self.check(inner, depth + 1)
                    if v:
                        return v
                continue
            if PY_RE.match(prog):
                runs_python = True
                v = self.python(args)
                if v:
                    return v
                code = " ".join(a for a in args if not a.startswith("-"))
                if PY_WRITE_RE.search(code):
                    for a in args:
                        for piece in re.findall(r"[~$A-Za-z0-9_./-]*(?:\.claude|loop|research-state)[A-Za-z0-9_./-]*", a):
                            r = self.hits(piece, ancestors=False)
                            if r:
                                return "FROZEN", f"python writing under {r}"
                continue
            write = prog in WRITE_CMDS
            if prog == "sed" or prog == "perl":
                write = any(a == "--in-place" or re.match(r"^-[A-Za-z]*i", a) for a in args)
            if prog == "awk" or prog == "gawk":
                write = "inplace" in args
            if prog == "find":
                write = any(a in ("-delete", "-exec", "-execdir", "-ok", "-fprint", "-fls") for a in args)
            if prog == "git":
                write = git_subcommand(args) in GIT_WRITES
            if not write:
                continue
            anc = prog in ANCESTOR_CMDS and prog != "git"
            if prog in DEST_ONLY:
                args = dest_args(args)
            for a in args:
                r = self.hits(a, ancestors=anc)
                if r:
                    return "FROZEN", f"{prog} on {r}"
        if runs_python and IMPORT_RE.search(cmd):
            m = IMPORT_RE.search(cmd)
            return "ACCEL", "python importing " + (m.group("mod") or m.group("dyn"))
        return None

    def python(self, args: list[str]):
        k = 0
        while k < len(args):
            a = args[k]
            if a == "-m" and k + 1 < len(args):
                mod = args[k + 1].split(".")[0]
                if mod in ACCEL_MODS:
                    return "ACCEL", f"python -m {args[k + 1]}"
                return None
            if a.startswith("-m") and len(a) > 2:
                return ("ACCEL", f"python {a}") if a[2:].split(".")[0] in ACCEL_MODS else None
            if a == "-c":
                return None                   # the code is in the command string; checked by IMPORT_RE
            if a in ("-W", "-X", "--check-hash-based-pycs"):
                k += 2
                continue
            if a.startswith("-"):
                k += 1
                continue
            if a == "-":
                return None
            path = self.norm(a)
            try:
                with open(path, "rb") as f:
                    src = f.read(1 << 20).decode("utf-8", "replace")
            except OSError:
                return None
            m = IMPORT_RE.search(src)
            if m:
                return "ACCEL", f"{a} imports {m.group('mod') or m.group('dyn')}"
            return None
        return None


def main() -> None:
    cmd = os.environ.get("IJ_HOOK_CMD", "")
    cwd = os.environ.get("IJ_HOOK_CWD") or os.getcwd()
    roots = os.environ.get("IJ_PROTECTED_ROOTS", "").split(":")
    g = Guard(cwd, os.environ.get("HOME", "/"), roots)
    v = g.check(cmd)
    print("OK" if not v else f"{v[0]}\t{v[1]}")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:  # noqa: BLE001 - the hook turns this into a deny
        print(f"ERROR\t{type(e).__name__}: {e}")
        sys.exit(0)
