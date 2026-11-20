"""controller/prime_settings.py: the controller's Prime Agent settings go into
settings.json at every install, keys Prime Agent wrote itself stay, and a
wrong allowedModels never reaches the file."""
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

AMD = Path(__file__).resolve().parent.parent
SCRIPT = AMD / "controller" / "prime_settings.py"
OWNER = __import__("pwd").getpwuid(os.getuid()).pw_name


class PrimeSettings(unittest.TestCase):
    def run_it(self, cfg: dict, cur: dict | None):
        d = Path(tempfile.mkdtemp())
        c, t = d / "c.json", d / "agent" / "settings.json"
        c.write_text(json.dumps(cfg))
        if cur is not None:
            t.parent.mkdir()
            t.write_text(json.dumps(cur))
        r = subprocess.run([sys.executable, str(SCRIPT), str(c), str(t), OWNER], capture_output=True, text=True)
        return r, (json.loads(t.read_text()) if t.exists() else None), t

    def installed(self) -> dict:
        return json.loads((AMD / "controller" / "controller.installed.json").read_text())

    def test_the_installed_settings_set_the_limits_and_keep_the_rest(self):
        live = {"rlmMaxDepth": 4, "harnessRefinements": True, "telemetry": {"enabled": False, "noticeShown": True},
                "defaultModel": "claude-opus-5-5", "defaultProvider": "anthropic",
                "recentModels": ["prime-inference/z-ai/glm-5.3"], "defaultThinkingLevel": "high",
                "agentTraces": {"enabled": False}, "onboardingShown": True}
        r, out, t = self.run_it(self.installed(), live)
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertEqual(out["allowedModels"], ["anthropic/claude-opus-5-5", "anthropic/claude-sonnet-5-5"])
        self.assertEqual(out["recentModels"], ["prime-inference/z-ai/glm-5.3"])          # Prime Agent's own, kept
        self.assertEqual(out["telemetry"], {"enabled": False, "noticeShown": True})       # merged key by key
        self.assertEqual((out["defaultProvider"], out["defaultModel"], out["defaultThinkingLevel"]),
                         ("anthropic", "claude-opus-5-5", "high"))
        self.assertEqual(oct(t.stat().st_mode & 0o777), "0o600")
        self.assertEqual(list(t.parent.glob(".settings.*")), [])                           # no temp file left
        r2, out2, _ = self.run_it(self.installed(), out)
        self.assertIn("in place", r2.stdout)
        self.assertEqual(out2, out)

    def test_a_value_changed_by_hand_is_set_back_to_the_config(self):
        r, out, _ = self.run_it(self.installed(), {"allowedModels": ["anthropic/claude-haiku-4-5"],
                                                   "defaultModel": "x", "agentTraces": {"enabled": True}})
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertEqual(out["allowedModels"], self.installed()["agent"]["settings"]["allowedModels"])
        self.assertEqual(out["defaultModel"], "claude-opus-5-5")
        self.assertEqual(out["agentTraces"], {"enabled": False})

    def test_a_fresh_machine_gets_the_file(self):
        r, out, _ = self.run_it(self.installed(), None)
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertTrue(out["onboardingShown"])

    def test_a_wrong_allowlist_is_refused_and_the_file_left_alone(self):
        for bad in ("anthropic/claude-opus-5-5", [], ["claude-opus-5-5"], [1], ["anthropic/claude-sonnet-5-5"]):
            cfg = self.installed()
            cfg["agent"]["settings"]["allowedModels"] = bad
            cur = {"allowedModels": ["anthropic/claude-opus-5-5"], "keep": 1}
            r, out, _ = self.run_it(cfg, cur)
            self.assertNotEqual(r.returncode, 0, bad)
            self.assertEqual(out, cur, bad)

    def test_a_file_that_is_not_json_is_not_overwritten(self):
        d = Path(tempfile.mkdtemp())
        (d / "c.json").write_text(json.dumps(self.installed()))
        (d / "s.json").write_text("{broken")
        r = subprocess.run([sys.executable, str(SCRIPT), str(d / "c.json"), str(d / "s.json"), OWNER],
                           capture_output=True, text=True)
        self.assertNotEqual(r.returncode, 0)
        self.assertEqual((d / "s.json").read_text(), "{broken")


if __name__ == "__main__":
    unittest.main()
