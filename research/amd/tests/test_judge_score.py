"""The judge's public score list: `ij-judge score` writes it from a ledger and
a machines record (no GPU, no controller needed).  Run:
  python3 -m unittest discover -s research/amd/tests -p 'test_*.py'"""
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

AMD = Path(__file__).resolve().parents[1]
JUDGE = AMD / "controller" / "judge" / "ij-judge"


def judge_env(tmp: Path) -> dict:
    cfg = {"paths": {"ctl_dir": str(tmp / "ctl"), "judge_dir": str(tmp / "judge"),
                     "score_file": str(tmp / "score" / "score.json")},
           "box": {"code_dir": str(AMD)},
           "judge": {"user": "judge", "hidden_prompts": str(tmp / "judge" / "hidden.jsonl"),
                     "amd_config": str(AMD / "config.json")}}
    for d in ("ctl", "judge", "score"):
        (tmp / d).mkdir()
    (tmp / "ctl.json").write_text(json.dumps(cfg))
    return {**os.environ, "IJ_CTL_CONFIG": str(tmp / "ctl.json")}


class ScoreTest(unittest.TestCase):
    def run_score(self, tmp: Path, env: dict) -> dict:
        r = subprocess.run([sys.executable, str(JUDGE), "score"], env=env, capture_output=True, text=True)
        self.assertEqual(r.returncode, 0, r.stderr)
        return json.loads((tmp / "score" / "score.json").read_text())

    def test_empty(self):
        with tempfile.TemporaryDirectory() as t:
            tmp = Path(t)
            doc = self.run_score(tmp, judge_env(tmp))
            self.assertEqual(doc["submissions"], [])
            self.assertIn("BF16 weights", doc["memory_bound_tps"])
            self.assertTrue(any("not measured yet" in x for x in doc["text"]))

    def test_with_submissions(self):
        with tempfile.TemporaryDirectory() as t:
            tmp = Path(t)
            env = judge_env(tmp)
            rows = [{"id": 1, "commit": "a" * 40, "note": "good", "submitted": "2026-11-17 15:08:20 UTC",
                     "status": "passed", "tokens_per_s": 80.0, "machine": "m1", "machine_reference_tps": 78.0,
                     "agreement": 0.99, "protocol": 2},
                    {"id": 2, "commit": "b" * 40, "note": "bad", "submitted": "2026-11-17 15:09:00 UTC",
                     "status": "failed", "reasons": ["agreement 0.5 < 0.95"], "machine": "m1"}]
            (tmp / "judge" / "ledger.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
            (tmp / "judge" / "machines.json").write_text(json.dumps(
                {"m1": {"reference": {"tokens_per_s": 78.0, "engine": "atom", "protocol": 2}, "measured": "x"}}))
            doc = self.run_score(tmp, env)
            self.assertEqual([s["id"] for s in doc["submissions"]], [1, 2])
            self.assertEqual(doc["machines"]["m1"]["reference_tps"], 78.0)
            text = "\n".join(doc["text"])
            self.assertIn("best passing submission: #1", text)
            self.assertIn("agreement 0.5 < 0.95", text)
            self.assertEqual(oct(os.stat(tmp / "score" / "score.json").st_mode & 0o777), "0o644")

    def test_old_protocol_is_marked_and_not_best(self):
        with tempfile.TemporaryDirectory() as t:
            tmp = Path(t)
            env = judge_env(tmp)
            rows = [{"id": 1, "commit": "a" * 40, "note": "old", "submitted": "2026-11-17 15:08:20 UTC",
                     "status": "passed", "tokens_per_s": 78.3, "machine": "m1", "machine_reference_tps": 78.2,
                     "agreement": 0.99}]
            (tmp / "judge" / "ledger.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
            (tmp / "judge" / "machines.json").write_text(json.dumps(
                {"m1": {"reference": {"tokens_per_s": 78.2, "engine": "atom"}, "measured": "x",
                        "tuned": {"tokens_per_s": 570.6, "measured": "y"}}}))
            text = "\n".join(self.run_score(tmp, env)["text"])
            self.assertIn("best passing submission: none", text)
            self.assertIn("78.3 p1", text)
            self.assertIn("old protocol 1", text)
            self.assertIn("570.6 tokens/s (measured y; by an older method, to be measured again)", text)

    def test_measurement_runs_are_marked_and_never_best(self):
        with tempfile.TemporaryDirectory() as t:
            tmp = Path(t)
            env = judge_env(tmp)
            base = {"submitted": "2026-11-17 15:08:20 UTC", "status": "passed", "machine": "m1",
                    "machine_reference_tps": 78.0, "agreement": 0.99, "protocol": 2}
            rows = [{**base, "id": 1, "commit": "measure-tuned", "note": "tuned 1", "tokens_per_s": 600.0,
                     "measure": "tuned"},
                    {**base, "id": 2, "commit": "measure-atom", "note": "atom 1", "tokens_per_s": 79.0, "measure": "atom"},
                    {**base, "id": 3, "commit": "c" * 40, "note": "agent", "tokens_per_s": 90.0}]
            (tmp / "judge" / "ledger.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
            (tmp / "judge" / "machines.json").write_text(json.dumps(
                {"m1": {"reference": {"tokens_per_s": 78.0, "engine": "atom", "protocol": 2}, "measured": "x",
                        "tuned": {"tokens_per_s": 600.0, "protocol": 2, "measured": "y"}}}))
            doc = self.run_score(tmp, env)
            text = "\n".join(doc["text"])
            self.assertIn("best passing submission: #3", text)
            self.assertIn("[operator's measurement: AMD's tuned engine", text)
            self.assertIn("600.0 m", text)
            self.assertIn("600.0 tokens/s (measured y)", text)
            self.assertEqual([s.get("measure") for s in doc["submissions"]], ["tuned", "atom", None])
            rows[2]["status"] = "failed"
            (tmp / "judge" / "ledger.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
            self.assertIn("best passing submission: none", "\n".join(self.run_score(tmp, env)["text"]))

    def test_agents_md_states_the_formula_and_no_unproven_claims(self):
        r = subprocess.run([sys.executable, str(AMD / "controller" / "agents_text.py"), "agents-md",
                            str(AMD / "config.json")], capture_output=True, text=True)
        self.assertEqual(r.returncode, 0, r.stderr)
        t = " ".join(r.stdout.split())
        self.assertIn("the sum of the output tokens over the sum of the whole-request seconds, prompt time included", t)
        self.assertIn("3 min 12 s", t)
        self.assertIn("4,700 GB/s", t)
        self.assertIn("not the GPU's limit", t)
        self.assertIn("700 to 1,000 tokens/s", t)
        self.assertNotIn("will be measured again", t)
        self.assertNotIn("78 tokens/s", t)
        self.assertIn("`./score` shows both speeds as the judge measured them on each GPU machine", t)
        self.assertIn("about 65 tokens/s for the reference engine and about 260 for the tuned engine", t)
        self.assertIn("`/srv/models` also holds `Qwen3.8-27B-DFlash2`", t)
        self.assertIn("sees `/srv/models` read-only at `/models` and no other folder of the machine", t)
        self.assertIn("within 30 minutes of its start", t)
        self.assertIn("9 min 03 s measured on 2026-11-18 for plain ATOM", t)
        self.assertNotIn("Its files cannot be read or changed", t)
        self.assertNotIn("returns GPU counters", t)
        self.assertIn("Smaller weights (4-bit and the like) are allowed", t)
        # the bar the user set on 2026-11-18 (config.json), and that it replaces the agreement limit
        self.assertIn("top-5 share at least 0.98; mean gap at most 0.08 nats per token", t)
        self.assertIn("reference's) at least -0.15", t)
        self.assertIn("The bar replaces the earlier rule", t)
        self.assertNotIn("not set yet", t)

    def test_agents_md_names_the_two_models_of_the_allowlist_and_every_level(self):
        r = subprocess.run([sys.executable, str(AMD / "controller" / "agents_text.py"), "agents-md",
                            str(AMD / "config.json")], capture_output=True, text=True)
        t = " ".join(r.stdout.split())
        cfg = json.loads((AMD / "controller" / "controller.installed.json").read_text())
        allowed = cfg["agent"]["settings"]["allowedModels"]
        self.assertEqual(len(allowed), 2)
        self.assertIn(f"Two models are available: `{allowed[0]}` and `{allowed[1]}`.", t)
        self.assertIn("`rlm.spawn` takes `model` and `thinking`; choose them per task.", t)
        for level in ("off", "minimal", "low", "medium", "high", "xhigh", "max"):
            self.assertRegex(r.stdout, rf"\n- `{level}`: ")
        self.assertNotRegex(t.split("## Models")[1], r"\d+ ?(%|tokens|s\b|x\b)")   # no figures
        # the main agent's command runs the first model at the level the note names
        a = cfg["agent"]
        cmd = a["command"]
        self.assertEqual(f"{cmd[cmd.index('--provider') + 1]}/{cmd[cmd.index('--model') + 1]}", allowed[0])
        self.assertEqual(cmd[cmd.index("--thinking") + 1], "high")
        self.assertEqual((a["settings"]["defaultProvider"], a["settings"]["defaultModel"],
                          a["settings"]["defaultThinkingLevel"]), ("anthropic", "claude-opus-5-5", "high"))
        self.assertEqual(a["goal_token_budget"], 4294967295)
        self.assertEqual(cmd[-1], "Work on the goal.")
        self.assertNotIn("--continue", cmd)
        self.assertNotIn("--resume", cmd)

    def test_agents_md_says_not_set_while_the_bar_is_null(self):
        with tempfile.TemporaryDirectory() as d:
            c = json.loads((AMD / "config.json").read_text())
            c["quality"]["bar"] = {"min_topk_share": {"1": None, "2": None, "5": None}, "max_mean_gap": None,
                                   "min_task_delta_lb": None}
            p = Path(d) / "c.json"
            p.write_text(json.dumps(c))
            r = subprocess.run([sys.executable, str(AMD / "controller" / "agents_text.py"), "agents-md", str(p)],
                               capture_output=True, text=True)
        t = " ".join(r.stdout.split())
        self.assertIn("The limits of the bar are not set yet", t)
        self.assertNotIn("mean gap at most", t)
        self.assertNotIn("The bar replaces", t)

    def test_agents_md_names_the_limits_only_once_set(self):
        with tempfile.TemporaryDirectory() as d:
            c = json.loads((AMD / "config.json").read_text())
            c["quality"]["bar"] = {"min_topk_share": {"5": 0.9}, "max_mean_gap": 0.25, "min_task_delta_lb": -0.05}
            p = Path(d) / "c.json"
            p.write_text(json.dumps(c))
            r = subprocess.run([sys.executable, str(AMD / "controller" / "agents_text.py"), "agents-md", str(p)],
                               capture_output=True, text=True)
        t = " ".join(r.stdout.split())
        self.assertIn("top-5 share at least 0.9; mean gap at most 0.25 nats per token", t)
        self.assertIn("at least -0.05", t)
        self.assertNotIn("not set yet", t)

    def test_speed_parts_and_requests_are_published(self):
        with tempfile.TemporaryDirectory() as t:
            tmp = Path(t)
            env = judge_env(tmp)
            per = [{"id": "hidden-1", "prompt_tokens": 100, "tokens": 128, "seconds": 1.6, "finish": "length"},
                   {"id": "hidden-2", "prompt_tokens": 2000, "tokens": 64, "seconds": 1.0, "finish": "stop"}]
            sp = {"tokens_per_s": 73.85, "tokens": 192, "seconds": 2.6, "requests": 2, "per_request_min": 64.0,
                  "per_request_max": 80.0, "per_request": per,
                  "parts": {"whole_tps": {"min": 72.0, "max": 75.0}, "decode_tps": {"min": 90.0, "max": 91.5},
                            "prompt_s": {"min": 0.4, "max": 0.45}, "samples": [{}, {}, {}], "repeats": 2}}
            rows = [{"id": 1, "commit": "a" * 40, "note": "good", "submitted": "2026-11-17 15:08:20 UTC",
                     "status": "passed", "tokens_per_s": 73.85, "machine": "m1", "machine_reference_tps": 70.0,
                     "agreement": 0.99, "protocol": 2, "speed": sp}]
            (tmp / "judge" / "ledger.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
            (tmp / "judge" / "machines.json").write_text(json.dumps(
                {"m1": {"reference": {"tokens_per_s": 70.0, "engine": "atom", "protocol": 2}, "measured": "x"}}))
            doc = self.run_score(tmp, env)
            text = "\n".join(doc["text"])
            self.assertIn("prompt time included", text)
            self.assertIn("192 answer tokens / 2.6 s", text)
            self.assertIn("decode 90.0-91.5 tokens/s", text)
            self.assertIn("whole request 72.0-75.0 tokens/s", text)
            d = "\n".join(doc["details"]["1"])
            self.assertIn("hidden-2: 2000 prompt tokens, 64 answer tokens in 1.000 s = 64.0 tokens/s (stop)", d)
            ijc = AMD / "controller" / "bin" / "ijc"
            cenv = {**env, "IJ_CTL_CONFIG": env["IJ_CTL_CONFIG"]}
            r = subprocess.run([sys.executable, str(ijc), "score", "1"], env=cenv, capture_output=True, text=True)
            self.assertEqual(r.returncode, 0, r.stderr)
            self.assertIn("hidden-1: 100 prompt tokens", r.stdout)
            r = subprocess.run([sys.executable, str(ijc), "score", "9"], env=cenv, capture_output=True, text=True)
            self.assertEqual(r.returncode, 1)


if __name__ == "__main__":
    unittest.main()
