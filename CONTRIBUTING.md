# Contributing

This page covers setup, the tests, the rules every change follows, and how to
add a target, a cheat or an agent adapter.

## Set up

You need [uv](https://docs.astral.sh/uv/), `bash`, `git` and `jq`. Docker is
optional, but some tests and one cheat-suite row skip without it.

```bash
make -C bench venv      # creates bench/.venv (Python 3.12) with the test dependencies
```

Every suite below runs with `bench/.venv/bin/python`. No suite needs an
accelerator, a model, a network or an account. If `shellcheck` is installed, the
tests that lint shell scripts use it.

## Run the tests

Run each suite on its own: `bench/`, `setup/` and `research/loop/` each have a
`conftest.py`.

| Suite | Command | Time, where the sources say |
|---|---|---|
| The judge (`bench/`) | `make -C bench test` | about 25 minutes, because the end-to-end runs are slow |
| One judge test file | `cd bench && .venv/bin/python -m pytest tests/test_stats.py` | seconds to a few minutes |
| The cheat suite | `bench/.venv/bin/python scorecard/run_scorecard.py` | about 3 minutes; `--quick` runs fewer simulations, `--only s1,s4` runs some groups |
| The cheat suite's own tests | `bench/.venv/bin/python -m pytest scorecard/tests -q` | |
| Setup scripts | `bench/.venv/bin/python -m pytest setup/tests` | needs no root, network or Docker |
| The laptop rehearsal | `IJ_LOCALBOX=1 bench/.venv/bin/python -m pytest -m slow setup/tests/test_local_box.py` | needs Docker; `bash setup/local_box/run_local_box.sh` alone takes about 75 seconds warm |
| The loop (`research/loop/`) | `bench/.venv/bin/python -m pytest research/loop/tests` | |
| The guard (`research/guard/`) | `bash research/guard/test-hooks.sh` | |

`LOCALBOX_TIER=2 bash setup/local_box/run_local_box.sh` runs the rehearsal with
the real judge and takes about 40 minutes. `./demo.sh --quick` runs the cheat
suite and the rehearsal together and exits 0 only if everything passed.

The judge's timing tests assume a quiet machine. If a test in
`bench/tests/test_tps_harness.py`, or a throughput test in
`bench/tests/test_run_sh.py`, fails while the machine is busy, run it again
alone (`-k <name>`) before you look for a bug.

## Rules for changes

1. **Write the test first.** Show it failing, then make it pass.
2. **A change to the judge needs a cheat-suite row or a test that fails without
   it.** The judge is `run.sh`, `harness.py`, `metrics.py`, `compare.py`,
   `quality2.py`, `stats.py`, `tokenizer.py` and `target.py` in `bench/`
   (`RULER_FILES` in `bench/target.py`). The cheat suite tests each scenario
   with two fake judges (`scorecard/tests/test_scenarios.py`): one with the
   defence must pass the row, one without it must fail. A row that a judge
   without the defence also passes proves nothing.
3. **Never weaken a check to make a run pass.** Fix the cause. If the check
   was wrong, say why in the change and add the test that shows it.
4. **Keep the judge free of any one engine or accelerator.** Everything that
   names a chip, an engine flag or a model path belongs in a target file
   (`bench/target.py` says so at the top). The judge acts on `accelerator.kind:
   gpu` only.
5. **Shell scripts run on bash 3.2 as well as bash 5.** The tests run on macOS
   laptops, where `/bin/bash` is 3.2. Avoid `mapfile`, associative arrays and
   `${var,,}`.
6. **Do not put numbers in the docs that no run produced.**

A box refuses to run when a judge file has changed since it was locked, and
results made under different ruler hashes are never compared. Expect to
re-measure the baseline and the calibration after a change to a judge file.

## Add a target

A target is one file, `targets/<id>.yaml`. Start from `targets/cpu-mock.yaml`
(runs on a laptop) or `targets/sketch-tpu-8chip.yaml` (an example of a file that
is not ready to run), and follow [docs/your-first-target.md](docs/your-first-target.md).

1. Check the file: `bench/.venv/bin/python bench/target.py validate targets/<id>.yaml`.
   It names the field that is wrong.
2. Preview the install: `PATH="$PWD/bench/.venv/bin:$PATH" bash setup/bootstrap.sh
   --target targets/<id>.yaml --plan`.
3. Add the file name to the list in `test_shipped_targets_validate` in
   `bench/tests/test_target.py`.
4. Leave `runnable: false` until a launch of the baseline has answered on the
   real hardware. Say in a comment at the top what has and has not been run.
   Edit the file before the baseline is measured: the ruler fingerprints it.

## Add a mock cheat

The mock server (`bench/mock_server.py`) misbehaves on request. A new cheat
touches four places.

1. A `Behaviour` subclass in `bench/mock_server.py`, added to the `BEHAVIOURS`
   tuple and to the list in the module docstring.
2. A test in `bench/tests/test_mock_modes.py` that shows what the server does.
3. A case in `CASES` in `scorecard/scenarios/s1_tps_gaming.py`: the behaviour
   name, whether the judge should credit it, and a plain-words description.
4. The fake judges in `scorecard/tests/test_scenarios.py`: teach `Strict` how
   the cheat looks (its `measure_tps`) and `Naive` that it is fooled, then fix
   the counts that tests such as `test_tps_naive_judge_credits_the_cheats`
   assert. Add the row to the table in `scorecard/README.md`.

Run the cheat suite. If the real judge credits the new cheat, you have found a
hole: fix the judge under rule 2.

## Add an agent adapter

The loop does not know which agent does the research. It runs one script,
`research/adapters/<name>/session.sh`, once per session. Three adapters ship
(`claude`, `codex`, `pi`), and `research/adapters/TEMPLATE/` is a starting
point. [research/adapters/README.md](research/adapters/README.md) is the
contract; read it first.

1. Copy `research/adapters/TEMPLATE/` to `research/adapters/<name>/` and change
   the lines marked `EDIT`.
2. Implement the three verbs of `session.sh`:
   - `run --model M --effort E` reads the prompt on stdin, starts one session
     that needs no person, and prints exactly one JSON object on stdout. The
     driver reads `is_error` and `result` from it to tell a finished session,
     a quota limit, an expired login and a crash apart.
   - `version` prints the agent's version.
   - `auth` exits 0 when the agent is logged in, 1 when it is not and 2 when it
     cannot tell.
3. Make the agent send each shell command and file edit through the guard
   (`research/guard/`) before it runs. The guard is a tripwire. The Unix
   permissions that `setup/harden_research.sh` sets are the enforcement.
4. Optional: add `setup.sh`, so that `setup/bootstrap.sh --agent <name>` can
   install the agent (`research/adapters/codex/setup.sh` is an example).
5. Optional: put the files your agent needs in
   `research/adapters/<name>/project/`. Setup copies them to the top of the
   research repo.

Try the adapter without the loop, then ask the driver for the command it would
run. It prints one only when a session is due, so start a clock in a throwaway
state directory first:

```bash
(cd research && echo "Say hello and stop." | adapters/<name>/session.sh run --model primary --effort high)
export IJ_STATE_DIR=$(mktemp -d)
research/loop/run.sh --start-clock
IJ_AGENT=<name> research/loop/run.sh --dry-run
```

`research/loop/tests/test_adapters.py` shows how the shipped adapters are
tested. `claude`, `codex` and `pi` stand-in programs sit first on the `PATH`
(`research/loop/tests/conftest.py`), so no test starts a real agent or needs a
subscription. The tests check what each adapter sends its agent (the prompt,
the model and the effort), that it prints one result object, that a failure
never exits 0, that SIGTERM reaches the agent, and what `version` and `auth`
return. They also run the TEMPLATE around a small stand-in agent of its own,
and check that Claude Code's and Codex's `project/` files and Pi's `guard.ts`
send tool calls through the guard. A passing run says the adapters follow the
tools' documentation, not that the tools behave that way. Add your adapter to
the `AGENTS` list in that file and give `conftest.py` a stand-in for your
agent's command.

## Pull requests

Say what changed, which suites you ran, and what you could not run (anything
that needs Docker or an accelerator, for example).
