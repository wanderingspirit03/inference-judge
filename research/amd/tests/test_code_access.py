"""controller/code-access.sh: what it opens to the agents, builder and cand
exists in the tree, and none of it is the judge's code, its tests, the cheater
examples or the notes (the e2e checks the modes on the stand-in)."""
import re
import unittest
from pathlib import Path

AMD = Path(__file__).resolve().parent.parent
SRC = (AMD / "controller" / "code-access.sh").read_text()


def opened(role: str, listable: bool = True) -> list[str]:
    """The paths code-access.sh opens for ROLE (with listable: also the directories only listed)."""
    body = SRC.split('if [ "$ROLE" = controller ]; then', 1)[1]
    ctl, box = body.split("\nelse\n", 1)
    part = ctl if role == "controller" else box.split("\nfi\n", 1)[0]
    words = re.findall(r"(?:open|listable)\s+(\S+)" if listable else r"\bopen\s+(\S+)", part)
    for m in re.finditer(r"for f in ((?:[^;\\]|\\\n)+); do\s*\n\s*open \"\$f\"", part):
        words += m.group(1).replace("\\\n", " ").split()
    lib = [f"lib/{p.name}" for p in (AMD / "lib").glob("*.py") if p.name != "ledger.py"] if role == "box" else []
    return [w for w in words if not w.startswith('"')] + lib


CLOSED = ["submit", "score", "lib/ledger.py", "controller/judge", "controller/driver/ij-driver", "examples/cheaters",
          "measurements", "tests", "controller/tests", "README.md", "controller/install-vm.sh"]


class CodeAccess(unittest.TestCase):
    def test_every_opened_path_exists(self):
        for role in ("controller", "box"):
            names = opened(role)
            self.assertTrue(names, role)
            for n in names:
                self.assertTrue((AMD / n).exists(), f"{role}: {n}")

    def test_nothing_of_the_judge_is_opened(self):
        for role in ("controller", "box"):
            for n in opened(role, listable=False) + ["controller/standin"]:
                for c in CLOSED:
                    self.assertFalse(n == c or n.startswith(c + "/") or c.startswith(n + "/"), f"{role}: {n} opens {c}")

    def test_the_box_opens_what_the_agents_md_names(self):
        names = opened("box")
        for n in ("examples/amd-tuned-engine", "reference", "controller/box/ij-box", "check", "try", "config.json"):
            self.assertIn(n, names)

    def test_every_module_check_imports_is_opened_on_the_box(self):
        names = set(opened("box"))
        for script in ("try", "check", "profile", "ref", "gpu"):
            for mod in re.findall(r"^\s*(?:from|import)\s+(\w+)", (AMD / script).read_text(), re.M):
                if (AMD / "lib" / f"{mod}.py").exists():
                    self.assertIn(f"lib/{mod}.py", names, f"{script} imports {mod}")


if __name__ == "__main__":
    unittest.main()
