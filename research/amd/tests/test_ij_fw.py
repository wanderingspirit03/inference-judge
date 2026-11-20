"""controller/box/ij-fw, cand's firewall rule, against a fake iptables and
ip6tables (a JSON file of chains) in its test mode (IJ_FW_TEST=1, never as
root).  The e2e test (controller/tests/e2e-standin.sh) runs the real one in
the stand-in machine.  Stdlib only."""
import json
import os
import subprocess
import tempfile
import unittest
from pathlib import Path

AMD = Path(__file__).resolve().parent.parent
FW = AMD / "controller" / "box" / "ij-fw"

# A small iptables: chains in $FAKE_DIR/<tool>.json; -S prints -A lines
# with an extra "-m udp/tcp" word, as the real one does.
FAKE = r'''#!/usr/bin/env python3
import json, os, sys
t = os.path.basename(sys.argv[0])
p = os.path.join(os.environ["FAKE_DIR"], t + ".json")
c = json.load(open(p)) if os.path.exists(p) else {"OUTPUT": []}
a = sys.argv[1:]
if a[:1] == ["-w"]:
    a = a[2:]
op, ch, rest = a[0], (a[1] if len(a) > 1 else None), a[2:]
def show(r):
    out = []
    for i, w in enumerate(r):
        out.append(w)
        if w == "-p" and i + 1 < len(r):
            out += [r[i + 1], "-m", r[i + 1]]
    s, skip = [], False
    for i, w in enumerate(out):
        if skip:
            skip = False; continue
        s.append(w)
        if w == "-p":
            s.append(out[i + 1]); skip = True
    return " ".join(s)
rc = 0
if op == "-N":
    if ch in c: rc = 1
    else: c[ch] = []
elif ch not in c:
    sys.stderr.write(f"No chain/target/match by that name: {ch}\n"); sys.exit(1)
elif op == "-S":
    print(f"-N {ch}" if ch != "OUTPUT" else "-P OUTPUT ACCEPT")
    for r in c[ch]:
        print(f"-A {ch} " + show(r))
elif op == "-F": c[ch] = []
elif op == "-A": c[ch].append(rest)
elif op == "-I": c[ch].insert(int(rest[0]) - 1, rest[1:])
elif op == "-C": rc = 0 if rest in c[ch] else 1
elif op == "-D":
    if rest in c[ch]: c[ch].remove(rest)
    else: rc = 1
json.dump(c, open(p, "w"))
sys.exit(rc)
'''

UID = 1003


class FwTest(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        for t in ("iptables", "ip6tables"):
            (self.tmp / t).write_text(FAKE)
            (self.tmp / t).chmod(0o755)

    def fw(self, *a, uid=UID):
        env = {**os.environ, "IJ_FW_TEST": "1", "IJ_FW_TEST_BIN": str(self.tmp), "IJ_FW_TEST_UID": str(uid),
               "FAKE_DIR": str(self.tmp)}
        env.pop("SUDO_USER", None)
        return subprocess.run([str(FW), *a], capture_output=True, text=True, env=env, timeout=60)

    def chains(self, t="iptables"):
        return json.loads((self.tmp / f"{t}.json").read_text())

    def edit(self, fn, t="iptables"):
        c = self.chains(t)
        fn(c)
        (self.tmp / f"{t}.json").write_text(json.dumps(c))

    def test_check_fails_closed_before_apply(self):
        r = self.fw("check")
        self.assertEqual(r.returncode, 1, r.stdout + r.stderr)
        self.assertIn("not in place", r.stdout)

    def test_apply_then_check_both_tables(self):
        self.assertEqual(self.fw("apply").returncode, 0)
        r = self.fw("check")
        self.assertEqual((r.returncode, r.stdout.strip()), (0, "ok"), r.stderr)
        for t in ("iptables", "ip6tables"):
            c = self.chains(t)
            self.assertEqual(c["OUTPUT"][0], ["-m", "owner", "--uid-owner", str(UID), "-j", "IJ-CAND"])
            self.assertEqual(c["IJ-CAND"][-1], ["-j", "REJECT"])
            self.assertEqual([r[-1] if r[-1] != "tcp-reset" else "REJECT" for r in c["IJ-CAND"]][:3],
                             ["REJECT", "REJECT", "RETURN"])

    def test_apply_is_idempotent_and_puts_the_hook_first(self):
        self.fw("apply")
        self.edit(lambda c: c["OUTPUT"].insert(0, ["-j", "ACCEPT"]))
        self.assertEqual(self.fw("check").returncode, 1)          # an ACCEPT before the hook
        self.assertEqual(self.fw("apply").returncode, 0)
        self.assertEqual(self.fw("apply").returncode, 0)
        c = self.chains()
        hooks = [r for r in c["OUTPUT"] if r[-1] == "IJ-CAND"]
        self.assertEqual(len(hooks), 1)
        self.assertEqual(c["OUTPUT"][0][-1], "IJ-CAND")
        self.assertEqual(len(c["IJ-CAND"]), 4)
        self.assertEqual(self.fw("check").returncode, 0)

    def test_check_finds_each_kind_of_damage(self):
        damage = {
            "no hook": (lambda c: c["OUTPUT"].clear(), "iptables"),
            "no IPv6 hook": (lambda c: c["OUTPUT"].clear(), "ip6tables"),
            "hook for another uid": (lambda c: c["OUTPUT"].__setitem__(0, ["-m", "owner", "--uid-owner", "999",
                                                                           "-j", "IJ-CAND"]), "iptables"),
            "chain flushed": (lambda c: c["IJ-CAND"].clear(), "iptables"),
            "an accept added": (lambda c: c["IJ-CAND"].insert(0, ["-j", "ACCEPT"]), "iptables"),
            "final reject removed": (lambda c: c["IJ-CAND"].pop(), "ip6tables"),
            "loopback widened": (lambda c: c["IJ-CAND"].__setitem__(2, ["-j", "RETURN"]), "iptables"),
            "dns open": (lambda c: c["IJ-CAND"].pop(0), "iptables"),
            "chain gone": (lambda c: c.pop("IJ-CAND"), "iptables"),
        }
        for name, (fn, t) in damage.items():
            with self.subTest(name):
                for f in self.tmp.glob("*.json"):
                    f.unlink()
                self.fw("apply")
                self.edit(fn, t)
                r = self.fw("check")
                self.assertEqual(r.returncode, 1, f"{name}: {r.stdout}")

    def test_check_for_the_wrong_uid_fails(self):
        self.fw("apply", uid=UID)
        self.assertEqual(self.fw("check", uid=UID + 1).returncode, 1)

    def test_other_verbs_are_refused(self):
        self.assertEqual(self.fw("flush").returncode, 2)
        self.assertEqual(self.fw().returncode, 2)


if __name__ == "__main__":
    unittest.main()
