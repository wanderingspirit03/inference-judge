#!/usr/bin/env bash
# Synthetic tests for the three guard scripts.  No box, no GPU, no agent needed:
# every case is a JSON payload on stdin and an expected exit code.
#
#     bash research/guard/test-hooks.sh
#
# Exit code 0 = every case behaved. 1 = at least one case did not.
#
# Contract: exit 0 allows the tool call, exit 2 blocks it with the stderr text
# shown to the agent.  Any other exit code is a hook error, so a script that
# crashes is itself a failure here.
set -uo pipefail

HOOKS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASH_HOOK="$HOOKS_DIR/pretooluse-bash.sh"
EDIT_HOOK="$HOOKS_DIR/pretooluse-edit.sh"
AUDIT_HOOK="$HOOKS_DIR/posttooluse-audit.sh"

PASS=0; FAIL=0

# The cases name a research repo at /home/researcher/research.  Nothing from the
# machine these tests run on may leak in: not its guard.env, not its variables.
unset IJ_PROTECTED_ROOTS IJ_ALLOWED_WRITE_ROOTS IJ_STATE_DIR IJ_AUDIT_LOG CLAUDE_PROJECT_DIR
export IJ_GUARD_ENV=/nonexistent/guard.env
export IJ_RESEARCH_DIR=/home/researcher/research

# The edit cases below use two engine trees (vllm, sglang) whatever target this box
# was bootstrapped for; the section on engine.sources points this at its own files.
NO_SOURCES_FILE=/nonexistent/engine-sources
TWO_TREES=$(mktemp)
printf 'vllm\nsglang\n' > "$TWO_TREES"
trap 'rm -f "$TWO_TREES"' EXIT
export IJ_ENGINE_SOURCES_FILE="$TWO_TREES"

# run <expected-code> <hook> <json> <description>
run() {
  local want="$1" hook="$2" json="$3" desc="$4" got out
  out=$(printf '%s' "$json" | "$hook" 2>&1); got=$?
  if [ "$got" -eq "$want" ]; then
    printf '\033[32mok  \033[0m %s\n' "$desc"; PASS=$((PASS+1))
  else
    printf '\033[31mFAIL\033[0m %s  (wanted exit %s, got %s)\n' "$desc" "$want" "$got"
    printf '       %s\n' "$(printf '%s' "$out" | head -2 | tr '\n' ' ')"
    FAIL=$((FAIL+1))
  fi
}

bash_json()  { printf '{"tool_name":"Bash","tool_input":{"command":%s}}' "$(jq -Rn --arg c "$1" '$c')"; }
# the same, run from the research repo (relative paths resolve against cwd)
rbash_json() { printf '{"tool_name":"Bash","cwd":"/home/researcher/research","tool_input":{"command":%s}}' "$(jq -Rn --arg c "$1" '$c')"; }
edit_json()  { printf '{"tool_name":"%s","cwd":"/home/researcher/research","tool_input":{"file_path":%s}}' "${2:-Write}" "$(jq -Rn --arg c "$1" '$c')"; }
# Pi's write and edit tools name the file in tool_input.path
path_json()  { printf '{"tool_name":"%s","cwd":"/home/researcher/research","tool_input":{"path":%s}}' "${2:-write}" "$(jq -Rn --arg c "$1" '$c')"; }
# Codex's apply_patch: the patch text is tool_input.command; $2 is the cwd
patch_json() { printf '{"tool_name":"apply_patch","cwd":"%s","tool_input":{"command":%s}}' "${2:-/home/researcher/research}" "$(jq -Rn --arg c "$1" '$c')"; }

echo "=== Bash hook: the one allowed privileged command ==="
run 0 "$BASH_HOOK" "$(bash_json 'sudo -u bench /opt/bench/run.sh prefill-chunk-1024')" "the exact run.sh line"
run 0 "$BASH_HOOK" "$(bash_json '  sudo -u bench /opt/bench/run.sh baseline-vllm  ')" "same, padded with spaces"
run 2 "$BASH_HOOK" "$(bash_json 'sudo -u bench /opt/bench/run.sh id1 && cat /etc/shadow')" "run.sh line with a && chain"
run 2 "$BASH_HOOK" "$(bash_json 'sudo -u bench /opt/bench/run.sh Bad_ID')" "run.sh with an invalid id"
run 2 "$BASH_HOOK" "$(bash_json 'sudo -u bench /opt/bench/run.sh x --mock http://127.0.0.1:9')" "run.sh with extra arguments"

echo
echo "=== Bash hook: the GPU and the container runtime ==="
run 2 "$BASH_HOOK" "$(bash_json 'nvidia-smi')"                       "bare nvidia-smi"
run 2 "$BASH_HOOK" "$(bash_json 'watch -n1 nvidia-smi --query-gpu=utilization.gpu --format=csv')" "nvidia-smi inside a pipeline"
run 2 "$BASH_HOOK" "$(bash_json 'docker run --gpus all inference-judge-base bash')" "docker run"
run 0 "$BASH_HOOK" "$(bash_json 'ls experiments/my-exp/dockerfiles/')"  "the word dockerfiles is not docker"
run 0 "$BASH_HOOK" "$(bash_json 'vim experiments/my-exp/Dockerfile')"   "editing a Dockerfile is fine"

echo
echo "=== Bash hook: reading the ruler's results ==="
run 0 "$BASH_HOOK" "$(bash_json 'cat /opt/bench/results/baseline-vllm/results.json')"          "cat a results.json"
run 0 "$BASH_HOOK" "$(bash_json 'jq -r .verdict /opt/bench/results/baseline-vllm/results.json')" "jq a results.json"
run 0 "$BASH_HOOK" "$(bash_json 'ls -la /opt/bench/results/')"                                  "ls the results dir"
run 0 "$BASH_HOOK" "$(bash_json 'tail -50 /opt/bench/results/my-exp/server.log')"               "tail a server log"
run 0 "$BASH_HOOK" "$(bash_json 'grep -c error /opt/bench/results/my-exp/turns.jsonl')"         "grep the per-turn records"
run 0 "$BASH_HOOK" "$(bash_json 'for f in /opt/bench/results/*/results.json; do jq -r .verdict "$f"; done')" "the RUNBOOK verdict loop"
run 0 "$BASH_HOOK" "$(bash_json 'cat /opt/bench/results/e1/results.json | jq .screen')"         "cat piped into jq"
run 0 "$BASH_HOOK" "$(bash_json 'python3 analyse.py /opt/bench/results/e1/turns.jsonl')"        "a python script reading results"

echo
echo "=== Bash hook: quoting is respected ==="
# The old splitter chopped on every | ; & and > regardless of quotes, so the
# ordinary way to look at a result was denied and the researcher learned to
# work around the hook.
run 0 "$BASH_HOOK" "$(bash_json 'jq ".long | keys" /opt/bench/results/exp-one/results.json')" "a quoted jq pipe"
run 0 "$BASH_HOOK" "$(bash_json "jq -r '.stats | .delta' /opt/bench/results/e1/results.json")" "a single-quoted jq pipe"
run 0 "$BASH_HOOK" "$(bash_json 'jq "select(.ttft_ms > 500) | .conv_id" /opt/bench/results/e1/turns.jsonl')" "a quoted > comparison in a jq filter"
run 0 "$BASH_HOOK" "$(bash_json 'awk -F, "\$3 > 100 {print}" /opt/bench/results/e1/turns.jsonl')" "a quoted > comparison in awk"
run 0 "$BASH_HOOK" "$(bash_json "python3 -c 'import json,sys; d=json.load(open(sys.argv[1])); print([t for t in d if t>1])' /opt/bench/results/e1/results.json")" "a python one-liner with > and ; inside quotes"
run 0 "$BASH_HOOK" "$(bash_json 'grep -c "rm -rf" /opt/bench/results/e1/server.log')"  "a quoted word that only looks write-shaped"
run 0 "$BASH_HOOK" "$(bash_json 'jq -r ".verdict" /opt/bench/results/e1/results.json | tr "a-z" "A-Z"')" "a real pipeline of two readers"
# ...and the denies still hold when the same characters are NOT quoted.
run 2 "$BASH_HOOK" "$(bash_json 'jq ".long | keys" /opt/bench/results/e1/results.json > /tmp/keys')" "quoted jq pipe, real redirect"
run 2 "$BASH_HOOK" "$(bash_json 'jq ".long" /opt/bench/results/e1/results.json; rm -rf /opt/bench/results/e1')" "quoted jq pipe, real rm after a ;"
run 2 "$BASH_HOOK" "$(bash_json 'echo "$(cat /opt/bench/pools/main.jsonl)"')" "command substitution reaching the pools"
run 2 "$BASH_HOOK" "$(bash_json 'cat "/opt/bench/harness.py"')"                    "a quoted path outside results/"
run 2 "$BASH_HOOK" "$(bash_json 'jq ".a" "/opt/bench/results/e1/x.json')"          "unbalanced quoting"

echo
echo "=== Bash hook: everything else under /opt/bench stays blocked ==="
run 2 "$BASH_HOOK" "$(bash_json 'cat /opt/bench/pools/main.jsonl')"        "reading a prompt pool"
run 2 "$BASH_HOOK" "$(bash_json 'cat /opt/bench/harness.py')"                  "reading the harness"
run 2 "$BASH_HOOK" "$(bash_json 'cat /opt/bench/leaderboard/cpu-mock.json')"            "reading the leaderboard state"
run 2 "$BASH_HOOK" "$(bash_json 'ls /opt/bench')"                             "listing the ruler"
run 2 "$BASH_HOOK" "$(bash_json 'cat /opt/bench/results/../canary/baseline.json')" "traversal out of results/"

echo
echo "=== Bash hook: write-shaped commands mentioning the ruler ==="
run 2 "$BASH_HOOK" "$(bash_json 'echo x > /opt/bench/results/e1/results.json')"    "redirect into results"
run 2 "$BASH_HOOK" "$(bash_json 'cat x | tee /opt/bench/results/e1/results.json')" "tee into results"
run 2 "$BASH_HOOK" "$(bash_json 'rm -rf /opt/bench/results/e1')"                   "rm a result"
run 2 "$BASH_HOOK" "$(bash_json 'cp /opt/bench/results/e1/results.json /tmp/r.json')" "cp out of results"
run 2 "$BASH_HOOK" "$(bash_json 'sed -i s/REJECTED/PROMOTED/ /opt/bench/results/e1/results.json')" "sed -i on a result"
run 2 "$BASH_HOOK" "$(bash_json 'chmod 777 /opt/bench/results')"                   "chmod the results dir"
run 2 "$BASH_HOOK" "$(bash_json 'ln -sf /tmp/fake /opt/bench/results/e1')"         "symlink over a result"
run 2 "$BASH_HOOK" "$(bash_json 'jq . /opt/bench/results/e1/results.json > /tmp/r.json')" "reading but redirecting to a file"
run 2 "$BASH_HOOK" "$(bash_json 'bash -c "cat /opt/bench/results/e1/results.json"')" "wrapping the read in another shell"
run 2 "$BASH_HOOK" "$(bash_json 'sudo -u bench cat /opt/bench/results/e1/results.json')" "sudo to read results"

echo
echo "=== Bash hook: ordinary work is untouched ==="
run 0 "$BASH_HOOK" "$(bash_json 'git -C /home/researcher/vllm-src diff')"        "git in the engine checkout"
run 0 "$BASH_HOOK" "$(bash_json 'python3 -m pytest tests/')"                 "running tests"
run 0 "$BASH_HOOK" "$(bash_json 'rm -rf /tmp/scratch')"                      "rm outside /opt/bench"

echo
echo "=== Edit/Write hook ==="
run 2 "$EDIT_HOOK" "$(edit_json /opt/bench/run.sh)"                       "Write to the ruler"
run 2 "$EDIT_HOOK" "$(edit_json /opt/bench/results/e1/results.json Edit)" "Edit a results.json (reads are Bash's job)"
run 2 "$EDIT_HOOK" "$(edit_json /etc/sudoers.d/inference-judge-researcher)"             "Write to sudoers"
run 2 "$EDIT_HOOK" "$(edit_json /home/researcher/.bashrc)"                    "Write to the researcher's home root"
run 2 "$EDIT_HOOK" "$(edit_json /tmp/scratch.py)"                         "Write to /tmp"
run 2 "$EDIT_HOOK" "$(edit_json /home/researcher/research/../../etc/passwd)"   "traversal out of the research repo"
run 0 "$EDIT_HOOK" "$(edit_json /home/researcher/research/LAB_NOTEBOOK.md)"    "Write the notebook"
run 0 "$EDIT_HOOK" "$(edit_json /home/researcher/research/experiments/e1/launch.sh)" "Write an experiment"
run 0 "$EDIT_HOOK" "$(edit_json /home/researcher/vllm-src/vllm/engine/core.py Edit)" "Edit the vLLM checkout"
run 0 "$EDIT_HOOK" "$(edit_json /home/researcher/sglang-src/python/sglang/srt/server.py)" "Write the SGLang checkout"

echo
echo "=== Edit/Write hook: the engine trees follow the target (engine.sources) ==="
# bootstrap writes /etc/inference-judge/engine-sources from the target; no file opens no tree.
ENG_TMP=$(mktemp -d)
printf 'tpu_inference\nvllm\n' > "$ENG_TMP/tpu-sources"
printf 'sglang-jax\n../etc\nevil dir\n' > "$ENG_TMP/odd-sources"
export IJ_ENGINE_SOURCES_FILE="$ENG_TMP/tpu-sources"
run 0 "$EDIT_HOOK" "$(edit_json /home/researcher/tpu_inference-src/tpu_inference/layers/attn.py Edit)" "target lists tpu_inference: edit it"
run 0 "$EDIT_HOOK" "$(edit_json /home/researcher/vllm-src/vllm/engine/core.py Edit)"                   "target lists vllm: edit it"
run 2 "$EDIT_HOOK" "$(edit_json /home/researcher/sglang-src/python/sglang/srt/server.py)"              "target does not list sglang: denied"
run 2 "$EDIT_HOOK" "$(edit_json /home/researcher/other-src/x.py)"                                      "a tree no one listed: denied"
run 2 "$EDIT_HOOK" "$(edit_json /home/researcher/tpu_inference-src/../.bashrc)"                        "traversal out of a listed tree"
run 2 "$EDIT_HOOK" "$(edit_json /home/researcher/tpu_inference/x.py)"                                  "the name without -src: denied"
export IJ_ENGINE_SOURCES_FILE="$ENG_TMP/odd-sources"
run 0 "$EDIT_HOOK" "$(edit_json /home/researcher/sglang-jax-src/python/x.py)"                          "a well-formed name is accepted"
run 2 "$EDIT_HOOK" "$(edit_json /home/researcher/etc-src/x.py)"                                        "malformed names in the file are ignored"
run 2 "$EDIT_HOOK" "$(edit_json /home/researcher/vllm-src/x.py)"                                       "a file that omits vllm closes vllm"
export IJ_ENGINE_SOURCES_FILE="$NO_SOURCES_FILE"
run 2 "$EDIT_HOOK" "$(edit_json /home/researcher/vllm-src/vllm/engine/core.py Edit)"                   "no file: no engine tree is open"
run 2 "$EDIT_HOOK" "$(edit_json /home/researcher/sglang-src/python/sglang/srt/server.py)"              "no file: nor this one"
run 0 "$EDIT_HOOK" "$(edit_json /home/researcher/research/LAB_NOTEBOOK.md)"                            "no file: the research repo stays open"
: > "$ENG_TMP/empty-sources"
export IJ_ENGINE_SOURCES_FILE="$ENG_TMP/empty-sources"
run 2 "$EDIT_HOOK" "$(edit_json /home/researcher/vllm-src/vllm/engine/core.py Edit)"                   "a target that lists no tree: none is open"
run 0 "$EDIT_HOOK" "$(edit_json /home/researcher/research/experiments/e1/launch.sh)"                   "a target that lists no tree: experiments stay open"
export IJ_ENGINE_SOURCES_FILE="$TWO_TREES"
rm -rf "$ENG_TMP"

echo
echo "=== Bash hook: accelerators, any spelling ==="
# The real boundary is setup/harden_research.sh: the devices belong to group
# bench and the researcher's venv is CPU-only.  These are the tripwire.
run 2 "$BASH_HOOK" "$(bash_json '/usr/bin/nvidia-smi -L')"                     "nvidia-smi by absolute path"
run 2 "$BASH_HOOK" "$(bash_json 'n""vidia-smi -L')"                            "nvidia-smi split by empty quotes"
run 2 "$BASH_HOOK" "$(bash_json 'env FOO=1 /usr/local/nvidia/bin/nvidia-smi')"  "nvidia-smi behind env"
run 2 "$BASH_HOOK" "$(bash_json 'tpu-info')"                                   "tpu-info"
run 2 "$BASH_HOOK" "$(bash_json '/usr/local/bin/tpu-info --help')"            "tpu-info by path"
run 2 "$BASH_HOOK" "$(bash_json 'ls -la /dev/nvidia0')"                        "a /dev/nvidia device"
run 2 "$BASH_HOOK" "$(bash_json 'cat /dev/nvidiactl')"                         "/dev/nvidiactl"
run 2 "$BASH_HOOK" "$(bash_json 'ls /dev/accel*')"                             "the TPU /dev/accel devices"
run 2 "$BASH_HOOK" "$(bash_json 'ls /dev/vfio/')"                              "the TPU vfio devices"
run 2 "$BASH_HOOK" "$(bash_json "python3 -c 'import torch; print(torch.cuda.is_available())'")" "python -c importing torch"
run 2 "$BASH_HOOK" "$(bash_json 'python -c "import jax; print(jax.devices())"')" "python -c importing jax"
run 2 "$BASH_HOOK" "$(bash_json "python3 -c 'from vllm import LLM'")"           "python -c importing vllm"
run 2 "$BASH_HOOK" "$(bash_json 'python3 -m vllm.entrypoints.openai.api_server --model m')" "python -m vllm"
run 2 "$BASH_HOOK" "$(bash_json 'python3 -m sglang.launch_server --model-path m')" "python -m sglang"
run 2 "$BASH_HOOK" "$(bash_json 'cd /tmp && python3 - <<EOF
import libtpu
EOF')"                                                                          "python heredoc importing libtpu"
GUARD_TMP=$(mktemp -d)
printf 'import os\nimport torch\nprint(torch.cuda.device_count())\n' > "$GUARD_TMP/probe.py"
printf 'import json, sys\nprint(json.load(open(sys.argv[1]))["verdict"])\n' > "$GUARD_TMP/clean.py"
run 2 "$BASH_HOOK" "$(bash_json "python3 $GUARD_TMP/probe.py")"                "a host script importing torch"
run 2 "$BASH_HOOK" "$(bash_json "timeout 60 python3 -u $GUARD_TMP/probe.py")"  "the same behind timeout"
run 0 "$BASH_HOOK" "$(bash_json "python3 $GUARD_TMP/clean.py /opt/bench/results/e1/results.json")" "a host script that imports no accelerator stack"
run 0 "$BASH_HOOK" "$(rbash_json 'grep -rn "import torch" experiments/')"      "grepping for an import is fine"
run 0 "$BASH_HOOK" "$(rbash_json 'printf "import torch\n" > experiments/e1/probe.py')" "writing container code that imports torch is fine"

echo
echo "=== Bash hook: the frozen files and the driver's state ==="
run 2 "$BASH_HOOK" "$(rbash_json 'sed -i s/exit\ 2/exit\ 0/ guard/pretooluse-bash.sh')" "sed -i on a guard script"
run 2 "$BASH_HOOK" "$(rbash_json "echo 'exit 0' > guard/pretooluse-edit.sh")" "redirect over a guard script"
run 2 "$BASH_HOOK" "$(rbash_json 'echo "{}" >| ./.claude/settings.json')"     "clobber-redirect over settings"
run 2 "$BASH_HOOK" "$(bash_json 'echo x >> /home/researcher/research/loop/run.sh')" "append to the driver"
run 2 "$BASH_HOOK" "$(rbash_json 'cat /tmp/x | tee loop/runner.py')"            "tee into loop/"
run 2 "$BASH_HOOK" "$(rbash_json 'cp /tmp/settings.json .claude/settings.json')" "cp over settings"
run 2 "$BASH_HOOK" "$(rbash_json 'mv guard guard.off')"                         "mv the guard away"
run 2 "$BASH_HOOK" "$(rbash_json 'rm -rf .claude')"                             "rm the .claude dir"
run 2 "$BASH_HOOK" "$(rbash_json 'chmod -x guard/pretooluse-bash.sh')"          "chmod a guard script"
run 2 "$BASH_HOOK" "$(rbash_json 'echo "{}" > .codex/hooks.json')"              "redirect over Codex's hook file"
run 2 "$BASH_HOOK" "$(rbash_json 'rm -rf .codex')"                              "rm the .codex dir"
run 2 "$BASH_HOOK" "$(rbash_json 'cp /tmp/session.sh adapters/claude/session.sh')" "cp over an adapter"
run 2 "$BASH_HOOK" "$(rbash_json "sed -i 's/--effort/--x/' adapters/codex/session.sh")" "sed -i on an adapter"
run 2 "$BASH_HOOK" "$(rbash_json 'cd experiments && rm -rf ../adapters')"       "rm adapters/ by a relative path"
run 2 "$BASH_HOOK" "$(rbash_json 'ln -sf /dev/null loop/preflight.sh')"         "symlink over preflight"
run 2 "$BASH_HOOK" "$(rbash_json 'cd experiments && rm -rf ../loop')"           "rm loop/ by a relative path"
run 2 "$BASH_HOOK" "$(rbash_json 'rm -rf *')"                                   "rm * in the research repo"
run 2 "$BASH_HOOK" "$(rbash_json 'git checkout -- .claude/settings.json')"      "git checkout over settings"
run 2 "$BASH_HOOK" "$(rbash_json "python3 -c \"open('.claude/settings.json','w').write('{}')\"")" "python writing settings"
run 2 "$BASH_HOOK" "$(rbash_json "bash -c 'echo x > loop/run.sh'")"             "a write wrapped in bash -c"
run 2 "$BASH_HOOK" "$(bash_json 'echo "{}" > /home/driver/research-state/events.jsonl')" "overwrite the driver's event log"
run 2 "$BASH_HOOK" "$(bash_json 'truncate -s0 /home/researcher/research-state/runs.jsonl')" "truncate a ledger"
run 2 "$BASH_HOOK" "$(rbash_json 'vim guard/_lib.sh')"                          "an editor on a guard script"
run 0 "$BASH_HOOK" "$(rbash_json 'cat guard/pretooluse-bash.sh')"               "reading a guard script is fine"
run 0 "$BASH_HOOK" "$(rbash_json 'grep -n deny guard/*.sh | head')"             "grepping the guard is fine"
run 0 "$BASH_HOOK" "$(rbash_json 'cat adapters/README.md adapters/claude/session.sh')" "reading an adapter is fine"
run 0 "$BASH_HOOK" "$(bash_json 'tail -20 /home/driver/research-state/events.jsonl')" "reading the event log"
run 0 "$BASH_HOOK" "$(bash_json 'cat /home/driver/research-state/BRIEF.md')"     "reading the BRIEF"
run 0 "$BASH_HOOK" "$(rbash_json 'cp loop/preflight.sh /tmp/pf-copy.sh')"       "copying out of loop/ is fine"
run 0 "$BASH_HOOK" "$(rbash_json 'rm -f experiments/e1/*.log')"                 "rm inside experiments/"
run 0 "$BASH_HOOK" "$(rbash_json 'git add -A && git commit -qm notes')"         "committing the notebook"

echo
echo "=== Bash hook: the researcher's ordinary tools stay usable ==="
run 0 "$BASH_HOOK" "$(bash_json 'date; tail /opt/bench/results/e1/run.log')"    "date, then a results read"
run 0 "$BASH_HOOK" "$(bash_json 'date -u +%FT%TZ')"                             "date"
run 0 "$BASH_HOOK" "$(bash_json 'sleep 600; cat /opt/bench/results/e1/results.json')" "sleep, then a results read"
run 0 "$BASH_HOOK" "$(bash_json 'timeout 60 cat /opt/bench/results/e1/results.json')" "timeout around a reader"
run 0 "$BASH_HOOK" "$(bash_json 'ls /opt/bench/results | xargs -n1 echo')"     "xargs around a reader"
run 0 "$BASH_HOOK" "$(bash_json 'ls -d /opt/bench/results/*/ | xargs -I{} cat {}results.json')" "xargs -I around cat"
run 2 "$BASH_HOOK" "$(bash_json 'ls /opt/bench/results | xargs rm -rf')"       "xargs around rm"
run 2 "$BASH_HOOK" "$(bash_json 'timeout 5 bash -c "cat /opt/bench/results/e1/results.json"')" "timeout around a shell"
run 0 "$BASH_HOOK" "$(bash_json 'sudo -u bench /opt/bench/run.sh exp-1 2>&1 | tail -20')" "run.sh output piped to tail"
run 0 "$BASH_HOOK" "$(bash_json 'sudo -u bench /opt/bench/run.sh exp-1 | grep -i verdict')" "run.sh output piped to grep"
run 0 "$BASH_HOOK" "$(bash_json 'timeout 2700 sudo -u bench /opt/bench/run.sh exp-1')" "run.sh under timeout"
run 2 "$BASH_HOOK" "$(bash_json 'sudo -u bench /opt/bench/run.sh exp-1 | grep -f /opt/bench/pools/main.jsonl')" "run.sh piped to a grep that reads a prompt pool"
run 2 "$BASH_HOOK" "$(bash_json 'sudo -u bench /opt/bench/run.sh exp-1 | sh')"   "run.sh piped to a shell"
run 0 "$BASH_HOOK" "$(rbash_json 'bash loop/preflight.sh exp-1')"               "pre-flight"
run 0 "$BASH_HOOK" "$(rbash_json 'loop/preflight.sh exp-1 | tail -1')"          "pre-flight, last line"
run 0 "$BASH_HOOK" "$(rbash_json 'ls queue | head')"                            "list the queue"
run 0 "$BASH_HOOK" "$(rbash_json "printf '%s' '{\"id\":\"exp-1\",\"kind\":\"candidate\"}' > queue/exp-1.json")" "queue an experiment"
run 0 "$BASH_HOOK" "$(rbash_json 'loop/bg.sh sweep python3 tools/sweep.py')"     "start a detached CPU job"

echo
echo "=== Edit/Write hook: the frozen files and the driver's state ==="
run 2 "$EDIT_HOOK" "$(edit_json /home/researcher/research/.claude/settings.json)"          "Write settings.json"
run 2 "$EDIT_HOOK" "$(edit_json /home/researcher/research/.claude/settings.local.json)"    "Write settings.local.json"
run 2 "$EDIT_HOOK" "$(edit_json guard/pretooluse-bash.sh Edit)"                        "Edit a guard script, relative path"
run 2 "$EDIT_HOOK" "$(edit_json /home/researcher/research/guard/_guard.py)"                "Write the command checker"
run 2 "$EDIT_HOOK" "$(edit_json /home/researcher/research/adapters/claude/session.sh Edit)" "Edit an adapter"
run 2 "$EDIT_HOOK" "$(edit_json /home/researcher/research/adapters/mine/session.sh)"       "Write a new adapter mid-run"
run 2 "$EDIT_HOOK" "$(edit_json /home/researcher/research/.codex/hooks.json)"              "Write Codex's hook file"
run 2 "$EDIT_HOOK" "$(edit_json /home/researcher/research/.claude/agents/sceptic.md Edit)" "Edit the sceptic's mandate"
run 2 "$EDIT_HOOK" "$(edit_json /home/researcher/research/loop/run.sh)"                    "Write the driver"
run 2 "$EDIT_HOOK" "$(edit_json loop/../loop/runner.py MultiEdit)"                     "MultiEdit the runner via .."
run 2 "$EDIT_HOOK" "$(edit_json /home/researcher/research-state/BRIEF.md)"                 "Write the BRIEF"
run 2 "$EDIT_HOOK" "$(edit_json /home/driver/research-state/events.jsonl)"              "Write the event log"
run 0 "$EDIT_HOOK" "$(edit_json /home/researcher/research/queue/exp-1.json)"               "Write a queue file"
run 0 "$EDIT_HOOK" "$(edit_json /home/researcher/research/HYPOTHESES.jsonl Edit)"          "Edit HYPOTHESES.jsonl"
run 0 "$EDIT_HOOK" "$(edit_json /home/researcher/research/loopnotes.md)"                   "a name that only starts with loop"
run 0 "$EDIT_HOOK" "$(edit_json /home/researcher/research/AGENTS.md Edit)"                 "the standing instructions stay editable"
run 0 "$EDIT_HOOK" "$(edit_json /home/researcher/research/guardrails.md)"                  "a name that only starts with guard"
rm -rf "$GUARD_TMP"

echo
echo "=== Edit/Write hook: Pi's tools name the file in tool_input.path ==="
run 0 "$EDIT_HOOK" "$(path_json experiments/e1/launch.sh)"                 "pi write, an experiment"
run 0 "$EDIT_HOOK" "$(path_json /home/researcher/research/NEXT.md edit)"       "pi edit, the notes"
run 2 "$EDIT_HOOK" "$(path_json loop/run.sh edit)"                         "pi edit, the driver"
run 2 "$EDIT_HOOK" "$(path_json /opt/bench/run.sh)"                        "pi write, the ruler"
run 2 "$EDIT_HOOK" "$(path_json ../.bashrc)"                               "pi write, out of the research repo"

echo
echo "=== Edit/Write hook: Codex's apply_patch names its files in the patch ==="
# Every file a patch adds, updates, deletes or moves to is checked, each against
# the cwd of the call.  One refused file refuses the patch.
P_ADD='*** Begin Patch
*** Add File: experiments/e1/launch.sh
+#!/bin/bash
+exec serve
*** End Patch'
P_UPDATE='*** Begin Patch
*** Update File: LAB_NOTEBOOK.md
@@
-old
+new
*** End Patch'
P_FROZEN='*** Begin Patch
*** Update File: loop/run.sh
@@
-AUTH_RE
+NOPE
*** End Patch'
P_TRAVERSE='*** Begin Patch
*** Add File: experiments/e1/../../loop/x.py
+print(1)
*** End Patch'
P_UP='*** Begin Patch
*** Update File: ../../guard/_lib.sh
@@
-deny
+allow
*** End Patch'
P_MOVE='*** Begin Patch
*** Update File: experiments/e1/a.py
*** Move to: loop/a.py
@@
-a
+b
*** End Patch'
P_MOVE_OK='*** Begin Patch
*** Update File: experiments/e1/a.py
*** Move to: experiments/e1/b.py
@@
-a
+b
*** End Patch'
P_TWO='*** Begin Patch
*** Add File: experiments/e1/ok.py
+print(1)
*** Update File: adapters/codex/session.sh
@@
-x
+y
*** End Patch'
P_DELETE='*** Begin Patch
*** Delete File: .codex/hooks.json
*** End Patch'
P_DELETE_OK='*** Begin Patch
*** Delete File: experiments/e1/old.log
*** End Patch'
P_TMP='*** Begin Patch
*** Add File: /tmp/x.py
+print(1)
*** End Patch'
P_RULER='*** Begin Patch
*** Update File: /opt/bench/run.sh
@@
-a
+b
*** End Patch'
P_ENGINE='*** Begin Patch
*** Update File: vllm/engine/core.py
@@
-a
+b
*** End Patch'
P_LOOKALIKE='*** Begin Patch
*** Add File: experiments/e1/notes.md
+A patch names its files like this:
+*** Update File: loop/run.sh
*** End Patch'
P_NOFILE='*** Begin Patch
@@
-a
+b
*** End Patch'
P_CRLF_OK=$(printf '*** Begin Patch\r\n*** Add File: experiments/e1/w.py\r\n+x\r\n*** End Patch\r\n')
P_CRLF=$(printf '*** Begin Patch\r\n*** Update File: loop/run.sh\r\n@@\r\n-a\r\n+b\r\n*** End Patch\r\n')
run 0 "$EDIT_HOOK" "$(patch_json "$P_ADD")"                    "a patch that adds an experiment file"
run 0 "$EDIT_HOOK" "$(patch_json "$P_UPDATE")"                 "a patch that updates the notebook"
run 0 "$EDIT_HOOK" "$(patch_json "$P_MOVE_OK")"                "a patch that renames inside experiments/"
run 0 "$EDIT_HOOK" "$(patch_json "$P_DELETE_OK")"              "a patch that deletes an experiment log"
run 0 "$EDIT_HOOK" "$(patch_json "$P_ENGINE" /home/researcher/vllm-src)" "a patch in a listed engine checkout"
run 0 "$EDIT_HOOK" "$(patch_json "$P_LOOKALIKE")"              "a file line inside added text is not a file line"
run 0 "$EDIT_HOOK" "$(patch_json "$P_CRLF_OK")"                "a patch with CRLF line ends"
run 2 "$EDIT_HOOK" "$(patch_json "$P_FROZEN")"                 "a patch that updates the driver"
run 2 "$EDIT_HOOK" "$(patch_json "$P_TRAVERSE")"               "a patch that adds into loop/ through .."
run 2 "$EDIT_HOOK" "$(patch_json "$P_UP" /home/researcher/research/experiments/e1)" "a patch from experiments/e1 reaching ../../guard"
run 2 "$EDIT_HOOK" "$(patch_json "$P_MOVE")"                   "a patch that moves a file into loop/"
run 2 "$EDIT_HOOK" "$(patch_json "$P_TWO")"                    "a patch whose second file is an adapter"
run 2 "$EDIT_HOOK" "$(patch_json "$P_DELETE")"                 "a patch that deletes Codex's hook file"
run 2 "$EDIT_HOOK" "$(patch_json "$P_TMP")"                    "a patch that adds a file in /tmp"
run 2 "$EDIT_HOOK" "$(patch_json "$P_RULER")"                  "a patch that updates the ruler"
run 2 "$EDIT_HOOK" "$(patch_json "$P_ENGINE" /home/researcher/other-src)" "a patch in a tree no one listed"
run 2 "$EDIT_HOOK" "$(patch_json "$P_CRLF")"                   "a CRLF patch that updates the driver"
run 2 "$EDIT_HOOK" "$(patch_json "$P_NOFILE")"                 "a patch that names no file is refused, not waved through"
run 2 "$EDIT_HOOK" '{"tool_name":"apply_patch","cwd":"/home/researcher/research","tool_input":{}}' "apply_patch with no patch text is refused"

echo
echo "=== guard.env: where setup tells the guard what to protect ==="
# setup/harden_research.sh writes /etc/inference-judge/guard.env root-only.  It
# is how an agent that sets no variable of its own (Codex, Pi) gets the same
# rules as one that does.
ENV_TMP=$(mktemp -d)
cat > "$ENV_TMP/guard.env" <<'EOF'
# written by setup
IJ_RESEARCH_DIR=/srv/lab/research
IJ_PROTECTED_ROOTS=/srv/lab/research/loop:/srv/lab/research/guard:/srv/state:/opt/inference-judge
IJ_STATE_DIR=/srv/state
PATH=/nonexistent
export IJ_ALLOWED_WRITE_ROOTS=/
not a setting
EOF
srv_json() { printf '{"tool_name":"Write","cwd":"/srv/lab/research","tool_input":{"file_path":%s}}' "$(jq -Rn --arg c "$1" '$c')"; }
unset IJ_RESEARCH_DIR
export IJ_GUARD_ENV="$ENV_TMP/guard.env" IJ_AUDIT_LOG="$ENV_TMP/elsewhere.log"
run 0 "$EDIT_HOOK" "$(srv_json /srv/lab/research/LAB_NOTEBOOK.md)"       "the file names the research repo"
run 0 "$EDIT_HOOK" "$(srv_json experiments/e1/launch.sh)"                "and a relative path resolves inside it"
run 2 "$EDIT_HOOK" "$(srv_json /srv/lab/research/loop/run.sh)"           "the file names the frozen roots"
run 2 "$EDIT_HOOK" "$(srv_json /srv/state/BRIEF.md)"                     "the state dir it names is frozen"
run 2 "$EDIT_HOOK" "$(srv_json /opt/inference-judge/guard/_lib.sh)"      "the installed copy of the guard is frozen"
run 2 "$EDIT_HOOK" "$(srv_json /home/researcher/research/LAB_NOTEBOOK.md)"   "another research repo is not this one"
run 2 "$EDIT_HOOK" "$(srv_json /tmp/x)"                                  "a line that is not KEY=value sets nothing"
run 2 "$BASH_HOOK" "$(printf '{"tool_name":"Bash","cwd":"/srv/lab/research","tool_input":{"command":"echo x > loop/run.sh"}}')" "the Bash hook reads the file too"
run 2 "$BASH_HOOK" "$(printf '{"tool_name":"Bash","cwd":"/srv/lab/research","tool_input":{"command":"rm -rf /opt/inference-judge/loop"}}')" "the Bash hook protects the installed loop"
run 0 "$BASH_HOOK" "$(printf '{"tool_name":"Bash","cwd":"/srv/lab/research","tool_input":{"command":"rm -f experiments/e1/x.log"}}')" "ordinary work is untouched"
# a variable already in the environment wins over the file
export IJ_RESEARCH_DIR=/home/researcher/research IJ_PROTECTED_ROOTS=/home/researcher/research/loop
run 0 "$EDIT_HOOK" "$(edit_json /home/researcher/research/LAB_NOTEBOOK.md)"  "a variable in the environment wins over the file"
run 2 "$EDIT_HOOK" "$(edit_json /home/researcher/research/loop/run.sh)"      "and so does its list of frozen roots"
run 0 "$EDIT_HOOK" "$(edit_json /home/researcher/research/guard/x.sh)"       "which here does not list guard/"
unset IJ_PROTECTED_ROOTS
export IJ_GUARD_ENV=/nonexistent/guard.env
rm -rf "$ENV_TMP"

echo
echo "=== with nothing set, the research repo is where the guard sits ==="
unset IJ_RESEARCH_DIR IJ_AUDIT_LOG
HERE_REPO=$(dirname "$HOOKS_DIR")
here_json() { printf '{"tool_name":"Write","cwd":"%s","tool_input":{"file_path":%s}}' "$HERE_REPO" "$(jq -Rn --arg c "$1" '$c')"; }
run 0 "$EDIT_HOOK" "$(here_json "$HERE_REPO/experiments/e1/launch.sh")"   "an experiment next to this guard"
run 2 "$EDIT_HOOK" "$(here_json "$HERE_REPO/guard/_lib.sh")"              "this guard itself"
run 2 "$EDIT_HOOK" "$(here_json loop/run.sh)"                             "the loop next to this guard"
# Not /home/researcher/research: on an installed machine that is where this suite sits.
run 2 "$EDIT_HOOK" "$(here_json /srv/elsewhere/research/LAB_NOTEBOOK.md)"     "a research repo somewhere else"
CLAUDE_PROJECT_DIR=/home/researcher/research; export CLAUDE_PROJECT_DIR
run 0 "$EDIT_HOOK" "$(edit_json /home/researcher/research/LAB_NOTEBOOK.md)"   "Claude Code's project dir names it too"
run 2 "$EDIT_HOOK" "$(edit_json /home/researcher/research/.claude/settings.json)" "and its settings folder is frozen"
unset CLAUDE_PROJECT_DIR
export IJ_RESEARCH_DIR=/home/researcher/research

echo
echo "=== PostToolUse audit hook (must never block) ==="
IJ_AUDIT_LOG=$(mktemp)
export IJ_AUDIT_LOG
run 0 "$AUDIT_HOOK" '{"tool_name":"Bash","tool_input":{"command":"ls"}}' "audit a Bash call"
run 0 "$AUDIT_HOOK" '{"tool_name":"Write"}'                              "audit a call with no input"
run 0 "$AUDIT_HOOK" 'not json at all'                                    "audit garbage input"
if [ "$(wc -l < "$IJ_AUDIT_LOG")" -ge 2 ]; then
  printf '\033[32mok  \033[0m audit.log received the lines\n'; PASS=$((PASS+1))
else
  printf '\033[31mFAIL\033[0m audit.log received the lines\n'; FAIL=$((FAIL+1))
fi
rm -f "$IJ_AUDIT_LOG"
# with no IJ_AUDIT_LOG the log is audit.log in the research repo
AUDIT_TMP=$(mktemp -d)
unset IJ_AUDIT_LOG
IJ_RESEARCH_DIR="$AUDIT_TMP" "$AUDIT_HOOK" <<< '{"tool_name":"apply_patch","tool_input":{"command":"*** Begin Patch"}}'
if grep -q apply_patch "$AUDIT_TMP/audit.log" 2>/dev/null; then
  printf '\033[32mok  \033[0m the audit log defaults to the research repo\n'; PASS=$((PASS+1))
else
  printf '\033[31mFAIL\033[0m the audit log defaults to the research repo\n'; FAIL=$((FAIL+1))
fi
rm -rf "$AUDIT_TMP"

echo
printf '%d passed, %d failed\n' "$PASS" "$FAIL"
[ "$FAIL" -eq 0 ]
