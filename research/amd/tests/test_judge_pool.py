"""The judge's hidden pool (ij-judge draw): every run draws fresh sessions,
a drawn session is never drawn again, and a run is refused when a bucket is
used up.  No GPU, no controller.  Run:
  python3 -m unittest discover -s research/amd/tests -p 'test_*.py'"""
import importlib.machinery
import importlib.util
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

from test_judge_score import JUDGE, judge_env

N = 0


def load_judge(tmp: Path, draw: dict, pool: list[dict], warm: list[dict] = ()):
    env = judge_env(tmp)
    cfg = json.loads(Path(env["IJ_CTL_CONFIG"]).read_text())
    cfg["judge"]["draw"] = draw
    Path(env["IJ_CTL_CONFIG"]).write_text(json.dumps(cfg))
    (tmp / "judge" / "pool.jsonl").write_text("".join(json.dumps(r) + "\n" for r in pool))
    (tmp / "judge" / "warm.jsonl").write_text("".join(json.dumps(r) + "\n" for r in warm))
    return reload_judge(tmp)


def reload_judge(tmp: Path):
    """A new copy of the module on the same files (as a new process would)."""
    global N
    old = os.environ.get("IJ_CTL_CONFIG")
    os.environ["IJ_CTL_CONFIG"] = str(tmp / "ctl.json")
    sys.modules.pop("ijctl", None)          # it reads the variable when imported
    try:
        N += 1
        loader = importlib.machinery.SourceFileLoader(f"ij_judge_{N}", str(JUDGE))
        spec = importlib.util.spec_from_loader(loader.name, loader)
        mod = importlib.util.module_from_spec(spec)
        loader.exec_module(mod)
    finally:
        if old is None:
            os.environ.pop("IJ_CTL_CONFIG", None)
        else:
            os.environ["IJ_CTL_CONFIG"] = old
    return mod


def pool_rows(buckets: dict, per_session: int = 1) -> list[dict]:
    rows = []
    for b, n in buckets.items():
        for s in range(n):
            for k in range(per_session):
                rows.append({"id": f"{b}-{s}-{k}", "session": f"{b}-{s}", "bucket": b,
                             "prompt_token_ids": [int(b) % 250, s, k, 7]})
    return rows


class PoolTest(unittest.TestCase):
    def test_draws_never_repeat_a_session(self):
        with tempfile.TemporaryDirectory() as t:
            J = load_judge(Path(t), {"100": 2, "1000": 2}, pool_rows({"100": 10, "1000": 10}, per_session=3))
            seen = set()
            for i in range(5):
                rows = J.draw(f"run-{i}")
                self.assertEqual(len(rows), 4)
                self.assertEqual(sorted(r["bucket"] for r in rows), ["100", "100", "1000", "1000"])
                sess = {r["session"] for r in rows}
                self.assertEqual(len(sess), 4)
                self.assertFalse(sess & seen, "a session was drawn twice")
                seen |= sess
            self.assertEqual(J.pool_left(), {"100": 0, "1000": 0})

    def test_exhausted_bucket_refuses_and_marks_nothing(self):
        with tempfile.TemporaryDirectory() as t:
            J = load_judge(Path(t), {"100": 2, "1000": 2}, pool_rows({"100": 4, "1000": 3}))
            J.draw("a")
            before = J.pool_left()
            with self.assertRaises(J.PoolExhausted) as e:
                J.draw("b")
            self.assertIn("bucket 1000: 1 unused", str(e.exception))
            self.assertEqual(J.pool_left(), before)

    def test_used_survives_a_new_process(self):
        with tempfile.TemporaryDirectory() as t:
            tmp = Path(t)
            pool = pool_rows({"100": 6})
            J = load_judge(tmp, {"*": 3}, pool)
            first = {r["session"] for r in J.draw("a")}
            J2 = reload_judge(tmp)
            used = [json.loads(x)["session"] for x in (tmp / "judge" / "pool-used.jsonl").read_text().splitlines()]
            self.assertEqual(set(used), first)
            second = {r["session"] for r in J2.draw("b")}
            self.assertFalse(first & second)

    def test_run_files_send_ids_only(self):
        with tempfile.TemporaryDirectory() as t:
            J = load_judge(Path(t), {"*": 2}, pool_rows({"100": 3}), warm=[{"id": "w", "prompt_token_ids": [1, 2, 3]}])
            files = J.run_files("x")
            hidden = [json.loads(x) for x in files["hidden.jsonl"].decode().splitlines()]
            self.assertEqual(len(hidden), 2)
            self.assertTrue(all(set(h) == {"prompt_token_ids"} for h in hidden), hidden)
            self.assertEqual(json.loads(files["warm.jsonl"].decode()), {"prompt_token_ids": [1, 2, 3]})

    def test_released_run_gives_its_sessions_back(self):
        with tempfile.TemporaryDirectory() as t:
            J = load_judge(Path(t), {"*": 2}, pool_rows({"100": 3}))
            J.draw("a")
            self.assertEqual(J.pool_left(), {"100": 1})
            J.release("a", "no judged prompt was sent")
            self.assertEqual(J.pool_left(), {"100": 3})
            J.draw("b")
            self.assertEqual(J.pool_left(), {"100": 1})

    def test_run_names_never_repeat_after_the_numbers_start_again(self):
        # the submission numbers started again at #1 on 2026-11-18; pool-used.jsonl holds the old sub-1, sub-2
        with tempfile.TemporaryDirectory() as t:
            J = load_judge(Path(t), {"*": 2}, pool_rows({"100": 8}))
            J.draw("sub-1")
            J.release("sub-1", "no judged prompt was sent")       # the old #1 gave its draw back
            J.draw("sub-2")                                      # the old #2 kept its draw
            self.assertEqual(J.pool_left(), {"100": 6})
            r1, r2 = J.run_name(1), J.run_name(2)
            self.assertTrue(r1.startswith("sub-1-") and r1 != J.run_name(1))
            J.draw(r1)                                           # the new #1, kept
            J.draw(r2)
            J.release(r2, "no judged prompt was sent")            # the new #2 gave its draw back
            self.assertEqual(J.pool_left(), {"100": 4})          # neither old release nor new touched the others

    def test_no_warm_rows_refuses_and_draws_nothing(self):
        with tempfile.TemporaryDirectory() as t:
            J = load_judge(Path(t), {"*": 2}, pool_rows({"100": 3}))
            with self.assertRaises(J.PoolExhausted):
                J.run_files("x")
            self.assertEqual(J.pool_left(), {"100": 3})

    def test_add_rows_dedupes_by_id(self):
        with tempfile.TemporaryDirectory() as t:
            J = load_judge(Path(t), {"*": 1}, pool_rows({"100": 2}))
            k = J.add_rows(J.POOL, [{"id": "100-0-0", "bucket": "100", "prompt_token_ids": [1]},
                                    {"id": "new", "bucket": "2000", "prompt_token_ids": [2]}])
            self.assertEqual(k, 1)
            self.assertEqual(J.pool_left(), {"100": 2, "2000": 1})


WARM = [{"id": "w", "prompt_token_ids": [1, 2, 3]}]


def task_rows(n: int) -> list[dict]:
    return [{"id": f"t{i}", "prompt_token_ids": [9, i], "answer": "B", "kind": "mc", "max_tokens": 1, "lang": "en"}
            for i in range(n)]


def with_tasks(tmp: Path, n_tasks: int, draw_n: int, bar_set: bool):
    """A judge with a task pool of n_tasks, judge.task_draw draw_n, and an AMD
    config whose quality bar is set or not."""
    J = load_judge(tmp, {"*": 2}, pool_rows({"100": 6}), warm=WARM)
    amd = json.loads((Path(J.CFG["judge"]["amd_config"])).read_text())
    if bar_set:
        amd["quality"]["bar"] = {"min_topk_share": {"5": 0.9}, "max_mean_gap": 0.5, "min_task_delta_lb": -0.1}
    else:              # the real config has a bar since 2026-11-18; "not set" is this test's own
        amd["quality"]["bar"] = {"min_topk_share": {"1": None, "2": None, "5": None}, "max_mean_gap": None,
                                 "min_task_delta_lb": None}
    (tmp / "amd.json").write_text(json.dumps(amd))
    cfg = json.loads((tmp / "ctl.json").read_text())
    cfg["judge"].update(amd_config=str(tmp / "amd.json"), task_draw=draw_n)
    (tmp / "ctl.json").write_text(json.dumps(cfg))
    J = reload_judge(tmp)
    J.add_rows(J.TASKS, task_rows(n_tasks), tasks=True)
    return J


class TaskPoolTest(unittest.TestCase):
    def test_each_run_draws_fresh_tasks_with_their_answers(self):
        with tempfile.TemporaryDirectory() as t:
            J = with_tasks(Path(t), 10, 4, bar_set=False)
            self.assertEqual(J.tasks_left(), 10)
            a = J.run_files("a", tasks=True)
            b = J.run_files("b", tasks=True)
            ta = [json.loads(x) for x in a["tasks.jsonl"].decode().splitlines()]
            tb = [json.loads(x) for x in b["tasks.jsonl"].decode().splitlines()]
            self.assertEqual((len(ta), len(tb)), (4, 4))
            self.assertFalse({r["id"] for r in ta} & {r["id"] for r in tb})
            self.assertEqual(set(ta[0]), {"id", "prompt_token_ids", "answer", "kind", "max_tokens"})
            self.assertEqual(J.tasks_left(), 2)
            self.assertIn("--tasks", J.judge_args(a))
            self.assertNotIn("tasks.jsonl", J.run_files("plain"))   # tasks=False: no draw
            self.assertEqual(J.tasks_left(), 2)

    def test_release_gives_back_prompts_and_tasks(self):
        with tempfile.TemporaryDirectory() as t:
            J = with_tasks(Path(t), 6, 3, bar_set=False)
            J.run_files("a", tasks=True)
            self.assertEqual((J.pool_left(), J.tasks_left()), ({"100": 4}, 3))
            J.release("a", "no judged prompt was sent")
            self.assertEqual((J.pool_left(), J.tasks_left()), ({"100": 6}, 6))

    def test_bar_not_set_goes_on_without_tasks(self):
        with tempfile.TemporaryDirectory() as t:
            J = with_tasks(Path(t), 2, 4, bar_set=False)
            files = J.run_files("a", tasks=True)
            self.assertNotIn("tasks.jsonl", files)
            self.assertNotIn("--tasks", J.judge_args(files))
            self.assertEqual(J.pool_left(), {"100": 4})

    def test_bar_set_refuses_without_enough_tasks_and_draws_nothing(self):
        with tempfile.TemporaryDirectory() as t:
            J = with_tasks(Path(t), 2, 4, bar_set=True)
            self.assertTrue(J.bar_set())
            with self.assertRaises(J.PoolExhausted) as e:
                J.run_files("a", tasks=True)
            self.assertIn("task pool has 2 unused tasks", str(e.exception))
            self.assertEqual((J.pool_left(), J.tasks_left()), ({"100": 6}, 2))

    def test_task_rows_need_an_answer_and_a_kind(self):
        with tempfile.TemporaryDirectory() as t:
            J = with_tasks(Path(t), 0, 4, bar_set=False)
            with self.assertRaises(SystemExit):
                J.add_rows(J.TASKS, [{"id": "x", "prompt_token_ids": [1], "kind": "essay"}], tasks=True)
            self.assertEqual(J.tasks_left(), 0)

    def test_quality_table_lists_runs_with_measures(self):
        with tempfile.TemporaryDirectory() as t:
            J = with_tasks(Path(t), 0, 4, bar_set=False)
            q = {"a": {"agreement_strict": 0.8, "topk_share": {"1": 0.8, "2": 0.9, "5": 1.0}, "mean_gap": 0.05,
                       "answers_cut": 0, "answers_failed": 0},
                 "b": {"tasks": 64, "engine_correct": 60, "reference_correct": 61, "delta": -0.016, "delta_lb95": -0.06}}
            lines = J.quality_table([{"id": 3, "status": "failed", "note": "q4", "quality": q}, {"id": 4, "status": "error"}])
            self.assertEqual(len(lines), 2)
            self.assertIn("0.8 0.8 0.9 1.0 0.05 0 0 | 64 60 61 -0.016 -0.06", lines[1])

    def test_the_same_question_is_added_once(self):
        with tempfile.TemporaryDirectory() as t:
            J = with_tasks(Path(t), 3, 4, bar_set=False)
            again = [dict(r, id=f"other-seed-{i}") for i, r in enumerate(task_rows(5))]
            self.assertEqual(J.add_rows(J.TASKS, again + [dict(again[4], id="dup-in-file")], tasks=True), 2)
            self.assertEqual(J.tasks_left(), 5)


if __name__ == "__main__":
    unittest.main()
