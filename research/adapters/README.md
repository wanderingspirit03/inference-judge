# Agent adapters

The loop does not know which coding agent does the research. It runs one
script, `adapters/<name>/session.sh`, once per session, and reads one JSON
object back. An adapter is that script plus whatever files the agent needs.

| Folder | Agent | What is in it | Proven by |
|---|---|---|---|
| `claude/` | Claude Code | `session.sh`, three helper agents (scout, hypothesizer, sceptic), hook settings | a real loop, before the loop called it through `session.sh`; tests with a stand-in program since (a real session through the adapter stopped at an expired login) |
| `codex/` | Codex CLI | `session.sh`, hook settings | tests with a stand-in program, and one short real session on a laptop; no real loop |
| `pi/` | Pi | `session.sh`, a guard extension | tests with a stand-in program, and one short real session on a laptop; no real loop |
| `TEMPLATE/` | yours | a `session.sh` to copy | |

Pick one with `IJ_AGENT` (default `claude`), or with `--agent <name>` when you
run `setup/bootstrap.sh`. If the agent's command is not on the `PATH` that
`sudo` allows, name it with `IJ_AGENT_BIN`; each `session.sh` reads that
variable. `loop/run.sh` stops at once, `--dry-run` included, when
`adapters/$IJ_AGENT/session.sh` is missing or not executable.

The Codex and Pi adapters were written from those tools' documentation and
tested against stand-in programs that print the documented output. Then each
agent ran one short real session through its adapter, on a laptop, in a
throwaway copy of `research/`. The prompt was a five-step wiring check, not
research: quote the standing instructions, run `echo`, create a file, try to
edit `loop/run.sh`, try to run `nvidia-smi`. There was no driver, no judge, no
hardened machine, no separate Unix users and no managed settings file, and
each agent ran once.

Seen in that session, for Codex (0.162.0) and for Pi (1.0.3, run with a named
model, `--model <provider>/<model>`):

- The adapter read the agent's real output: the result text, the token counts
  and the session id (for Pi also the cost), with exit status 0.
- The guard worked. The edit of `loop/run.sh` and the `nvidia-smi` call were
  refused, and the two allowed calls ran. In Codex the hooks in
  `.codex/hooks.json` did this; in Pi, `guard.ts` loaded and did it.
- `audit.log` got one line for each call that ran. A call the guard refused
  leaves no line.
- The agent had the repo's `AGENTS.md` in its context. Pi had it with
  `--no-approve` set.

Pi also ran two failing sessions: its configured default model could not be
reached ("Connection error."), and a model name the provider does not have
gave a 404 "model_not_found". Each came back as `is_error: true` with Pi's
message and a non-zero exit status. The driver counts both as crashes, not
login failures.

Still unseen for Codex and Pi:

- A real loop: the driver starting many sessions on a hardened Linux machine,
  as the `researcher` user, with the managed settings file in place.
- The words they use for a quota limit or an expired login, so whether the
  driver sorts those into "limit" and "auth" and not "crash".
- For Codex, that the hooks are still enforced from
  `/etc/codex/requirements.toml` on a hardened machine.

Claude Code has not been seen through the adapter either. Its one real
session on a laptop did not get past an expired login: the adapter returned
Claude Code's error object with a non-zero exit status.

Read `codex/session.sh` or `pi/session.sh`, and run step 5 of "Write your
own" below, before you trust one with a long run. The laptop rehearsal
(`setup/local_box/`) runs the Claude adapter only, with a stand-in agent.

## What the loop asks of `session.sh`

```
session.sh run --model M --effort E     the prompt arrives on stdin
session.sh version                      print the agent's version
session.sh auth                         exit 0: logged in   1: not logged in   2: cannot tell
```

`run` starts in the research repo, as the researcher's Unix user. It runs one
session that needs no person, then prints exactly one JSON object on stdout:

```json
{"type": "result", "is_error": false, "result": "the agent's last message"}
```

- `is_error` is `true` when the session failed, and `result` then holds the
  agent's own error text. The driver reads that text to tell a quota limit and
  an expired login from a crash, so pass it on unchanged.
- Print the object even when the agent printed nothing you can read. The
  helpers in `common.sh` then write `"the agent printed no result"`, followed
  by the end of whatever the agent did print.
- The exit status is the agent's, and never 0 after a failure. The driver
  takes a session that exits 0 for a success unless `result` names a limit or
  a login. `adapter_finish` in `common.sh` turns a 0 into 1 when `is_error`
  is `true`.
- When a session fails and the agent's own login check also fails, the Codex
  and Pi adapters put `not logged in: ` in front of `result`. The driver reads
  that as a login failure, whatever else the message says. Pi checks a login
  per model, so its adapter does this only when `IJ_PRIMARY_MODEL` or
  `IJ_FALLBACK_MODEL` names the model; for the same reason its `auth` exits 2
  unless `IJ_PRIMARY_MODEL` is set.
- Optional numbers are recorded when present: `num_turns`, `total_cost_usd`,
  `duration_ms`, and any number under `usage`.
- Nothing else goes to stdout. Diagnostics go to stderr.

`M` is `primary` or `fallback`. The driver moves to `fallback` after repeated
crashes. Each adapter decides what the two words mean: Claude maps them to
`opus` and `sonnet` at the top of its `session.sh`; Codex and Pi use the model
their own configuration names. Set `IJ_PRIMARY_MODEL` or `IJ_FALLBACK_MODEL`
to send a model name instead, and the adapter passes it to the agent as it is.

With Codex and Pi the two words mean the same model. The driver's move to
`fallback` then runs that model again (for `IJ_FALLBACK_SESSIONS` sessions,
default 3), and it pauses if the crashes go on. Set `IJ_FALLBACK_MODEL` to a
model name to get a real second model, or `IJ_FALLBACK_SESSIONS=0` to skip the
step.

`E` is the reasoning effort (`IJ_EFFORT`, default `high`).

The driver ends a session that runs too long with SIGTERM. Pass it on to the
agent. `common.sh` does that for you (`adapter_capture`).

What the session must leave behind is files in the research repo, and
[`../AGENTS.md`](../AGENTS.md) tells the agent which: an experiment folder, a
prediction, and a file in `queue/`.

## The guard

`../guard/` holds three scripts that look at a tool call before or after it
runs. Each reads one JSON object on stdin:

```json
{"tool_name": "Bash", "tool_input": {"command": "ls"}, "cwd": "/home/researcher/research"}
```

| Script | Called | Reads |
|---|---|---|
| `pretooluse-bash.sh` | before a shell command | `tool_input.command` |
| `pretooluse-edit.sh` | before a file edit | `tool_input.file_path`, `tool_input.notebook_path` or `tool_input.path`, or the files named in a Codex `apply_patch` |
| `posttooluse-audit.sh` | after a tool call runs | writes one line to `audit.log`; a call that was blocked leaves no line |

Exit status 0 allows the call. Exit status 2 blocks it, and stderr says why in
words meant for the agent. The scripts exit only with 0 or 2, because Claude
Code and Codex treat any other status as a hook error and may still run the
call. `pi/guard.ts` blocks on any status except 0. The Bash script assumes the
judge is installed at `/opt/bench`.

The guard needs to know where the research repo is and which paths are frozen.
It reads them from `/etc/inference-judge/guard.env`, which
`setup/harden_research.sh` writes. A variable already in the environment
(`IJ_RESEARCH_DIR`, `IJ_PROTECTED_ROOTS`) wins over the file. With neither, the
research repo is `CLAUDE_PROJECT_DIR` when Claude Code sets it and otherwise
the folder that contains `guard/`. In it, `loop/`, `guard/`, `adapters/`,
`.claude/` and `.codex/` are frozen, and so is the driver's state directory.

The guard is a tripwire. It catches an agent that drifts, and it explains the
rule. It does not stop a process the agent starts. The boundary is the Unix
permissions that `setup/harden_research.sh` sets: the judge, the loop, the
guard and the adapters are owned by other users, and the researcher cannot
write to them.

How each agent reaches the guard:

- **Claude Code** runs the scripts as hooks
  (`claude/project/.claude/settings.json`; on a hardened machine, root-owned
  managed settings).
- **Codex** runs them as hooks (`codex/project/.codex/hooks.json`; on a hardened
  machine, `/etc/codex/requirements.toml`; the file has not been seen enforcing
  them). According to Codex's documentation, its hosted tools, such as web search, do not pass through hooks; that has not
  been seen in a session.
- **Pi** has no hook file, so `pi/session.sh` loads `pi/guard.ts` with `-e`.
  The extension calls the same scripts and blocks a call when a script is
  missing (Pi's own `read`, `grep`, `find` and `ls` are not checked). The
  session runs with `--no-approve`. According to Pi's documentation, Pi then
  skips extensions and settings in the research repo's own `.pi/` folder, and
  extensions in the researcher's home folder still load; neither has been seen
  in a session. That lets you add a web search tool there, and it means an
  extension there can register a tool whose calls the guard checks only by the
  path they name.

## Files the agent needs in the research repo

Anything under `adapters/<name>/project/` is copied to the top of the research
repo at setup. Claude Code's `CLAUDE.md`, `.claude/settings.json` and
`.claude/agents/` live there, and so does Codex's `.codex/hooks.json`. On a
hardened machine the folders among them (`.claude/`, `.codex/`) are frozen;
`AGENTS.md` and `CLAUDE.md` are not.

The standing instructions for every agent are in `../AGENTS.md`. According to
their documentation, Codex and Pi read that file themselves (Pi with
`--no-approve` too); in their laptop sessions both had it in their context.
Claude Code reads `CLAUDE.md`, which imports it with `@AGENTS.md`; that import
has not been checked in a session without a person. The driver's prompt names
`AGENTS.md` too, for an agent that reads neither.

## `setup.sh`

An adapter may have a `setup.sh`, which `setup/bootstrap.sh` and
`setup/harden_research.sh` call with one verb:

```
setup.sh install          install the agent's command
setup.sh login            print the line that tells the operator how to log in
setup.sh managed-path     print where the agent's root-owned settings file goes, or nothing
setup.sh managed-config   print that file's contents
```

Bootstrap runs `install` only when no command with the adapter's name is on
the `PATH`. `setup/check.sh` and bootstrap show the `login` line; the operator
runs it as the researcher and logs in by hand; no token is copied.
`setup/harden_research.sh` writes the `managed-config` output to the
`managed-path` file, root-owned; it passes `INSTALL_DIR`, `RESEARCH_DIR`,
`STATE_DIR`, `RESULTS_DIR` and `FROZEN` in the environment. An adapter with no
`setup.sh` is not an error: the operator installs the command, and no managed
settings file is written. `claude/setup.sh`, `codex/setup.sh` and `pi/setup.sh`
show each case.

## Write your own

1. Copy `TEMPLATE/` to `adapters/<name>/` and change the lines marked `EDIT`.
2. Make the agent call the guard before each shell command and file edit
   (`common.sh` puts the guard's folder in `GUARD_DIR`).
3. Put files the agent needs in `adapters/<name>/project/`.
4. Optional: add `setup.sh` so that `setup/bootstrap.sh --agent <name>` can
   install the agent (see `codex/setup.sh`).
5. Try it without the loop:

   ```bash
   cd research && echo "Say hello and stop." | adapters/<name>/session.sh run --model primary --effort high
   ```

6. Then `IJ_AGENT=<name> research/loop/run.sh --dry-run` shows the command the
   driver would run, once the clock is started (`--start-clock`; set
   `IJ_STATE_DIR` to a throwaway folder for both).

`session.sh` is also where a design with several agents goes. Start an
orchestrator, a planner with workers, or two agents that argue. The loop reads
one result object and the files the session leaves, and nothing else. The
`claude/` folder is one worked example: one session that can call three helper
agents.
