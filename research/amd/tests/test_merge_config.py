"""controller/merge_config.py: an install adds missing keys and keeps every value
already set, judge.draw as a whole."""
import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "controller"))
import merge_config  # noqa: E402

DEFAULTS = {"judge": {"draw": {"100": 4, "1000": 4, "2000": 4, "4000": 4}, "cap": 10},
            "box": {"limits": {"mem": 1, "cpu": 2}}}


class MergeConfig(unittest.TestCase):
    def run_merge(self, target):
        with tempfile.TemporaryDirectory() as d:
            s, t = Path(d) / "s.json", Path(d) / "t.json"
            s.write_text(json.dumps(DEFAULTS))
            if target is not None:
                t.write_text(json.dumps(target))
            merge_config.main([str(s), str(t)])
            return json.loads(t.read_text())

    def test_a_hand_set_draw_is_kept_whole(self):
        draw = {"2000": 4, "4000": 4, "8000": 4}
        out = self.run_merge({"judge": {"draw": draw}})
        self.assertEqual(out["judge"]["draw"], draw)
        self.assertEqual(out["judge"]["cap"], 10)          # missing keys still added

    def test_other_dicts_still_get_missing_keys(self):
        out = self.run_merge({"box": {"limits": {"mem": 5}}})
        self.assertEqual(out["box"]["limits"], {"mem": 5, "cpu": 2})

    def test_the_installed_defaults_keep_the_hand_set_values(self):
        # the values set by hand on the controller (2026-11-18) survive an install of the real defaults
        defaults = json.loads((Path(merge_config.__file__).parent / "controller.installed.json").read_text())
        mine = json.loads(json.dumps(defaults))
        mine["driver"].update({"price_usd_per_hour": 3.0, "do_ssh_key_ids": [123]})
        mine["agent"]["enabled"] = False
        mine["judge"]["draw"] = {"2000": 4, "4000": 4, "8000": 4}
        with tempfile.TemporaryDirectory() as d:
            s, t = Path(d) / "s.json", Path(d) / "t.json"
            s.write_text(json.dumps(defaults))
            t.write_text(json.dumps(mine))
            merge_config.main([str(s), str(t)])
            out = json.loads(t.read_text())
        self.assertEqual(out["driver"]["price_usd_per_hour"], 3.0)
        self.assertEqual(out["driver"]["do_ssh_key_ids"], [123])
        self.assertIs(out["agent"]["enabled"], False)
        self.assertEqual(out["judge"]["draw"], {"2000": 4, "4000": 4, "8000": 4})

    def test_no_target_gets_the_defaults(self):
        self.assertEqual(self.run_merge(None), DEFAULTS)


if __name__ == "__main__":
    unittest.main()
