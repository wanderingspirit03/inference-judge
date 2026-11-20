#!/bin/bash
# e2e-standin.sh: the controller end to end against stand-ins, on the
# controller VM, as root (sudo bash e2e-standin.sh).  Real: the users, sudo,
# ssh with the driver's own host key, rsync, the thin clients, the judge, the
# driver's code and its limits, tmux, prime-agent's `prompt`.  Stand-in: the
# GPU machine (a Docker container from ij-standin-box, no GPU), the
# reference and the engines (a toy model), DigitalOcean (fake_do.py), and
# Prime Agent's run (a sleeping shell instead of a model session).
#
# It works in a clone of the engine repository (/home/agents/e2e-repo), with
# its own machines ledger, and moves the judge's test records aside at the
# end, so the installed controller is left as it was.
set -u
A=/opt/ij-amd/amd
C=$A/controller
CTL=/var/lib/ij-ctl
T=$CTL/test
JD=/var/lib/ij-judge
PORT=8650
PASS=0; FAIL=0
ok()  { echo "  ok: $*"; PASS=$((PASS+1)); }
bad() { echo "  FAILED: $*"; FAIL=$((FAIL+1)); }
check() { local what="$1"; shift; if "$@"; then ok "$what"; else bad "$what"; fi; }
ag() { local d="$1"; shift; sudo -u agents -H bash -c "cd $d && $*"; }
drv() { sudo -u driver -H env IJ_CTL_CONFIG="$T/cfg.json" "$C/driver/ij-driver" "$@"; }
mkcfg() {  # python expression on c (the config) -> $T/cfg.json
  python3 - "$1" <<'EOF'
import json, sys
c = json.load(open("/etc/ij-amd/controller.json"))
d = c["driver"]
d.update({"api_base": "http://127.0.0.1:8650", "do_token_file": "/var/lib/ij-ctl/test/do-token",
          "ledger": "/var/lib/ij-ctl/test/machines.jsonl", "standin": True, "price_usd_per_hour": None,
          "do_ssh_key_ids": [],
          "poll_s": 2, "max_creations_per_day": 2, "active_timeout_s": 120})
c["agent"]["enabled"] = False
exec(sys.argv[1])
json.dump(c, open("/var/lib/ij-ctl/test/cfg.json", "w"), indent=1)
EOF
  chown driver: "$T/cfg.json"
}
state() { python3 -c "import json; print(json.load(open('$CTL/box.json'))['state'])"; }
box_sh() { docker exec "$(docker ps -q --filter label=ij.fake=1 | head -1)" sh -c "$1"; }
# result NOTE: "status|reasons|reference_starts" of the judge's last submission with that note
result() { python3 - "$1" <<'EOF'
import json, sys
last = {}
for l in open("/var/lib/ij-judge/ledger.jsonl"):
    r = json.loads(l); last[r["id"]] = {**last.get(r["id"], {}), **r}
r = [x for x in last.values() if x.get("note") == sys.argv[1]][-1]
res = json.load(open(f"/var/lib/ij-judge/subs/{r['id']}/result.json"))
print(f"{r.get('status')}|{' / '.join(r.get('reasons') or [])}|{(res.get('times') or {}).get('reference_starts')}")
EOF
}

echo "== setup"
rm -rf "$T"; install -d -m 700 -o driver -g driver "$T"
head -c 24 /dev/urandom | base64 > "$T/do-token"; chown driver: "$T/do-token"; chmod 600 "$T/do-token"
cp "$T/do-token" /root/ij-fake-do-token
docker rm -f $(docker ps -aq --filter label=ij.fake=1) >/dev/null 2>&1
docker build -q -t ij-standin-box "$C/standin" >/dev/null || { echo "docker build of the stand-in failed"; exit 2; }
pkill -f "fake_do.py $PORT" 2>/dev/null; sleep 0.5
nohup python3 "$C/standin/fake_do.py" $PORT /root/ij-fake-do-token 2.0 > /tmp/fake_do.log 2>&1 &
FAKE=$!
sleep 1
# The judge's own records (real submissions) and its real pools are set aside for the test and put back at
# the end, also when the test stops early (trap).  The test never draws from a real pool.
KEEP_JD="ledger.jsonl machines.json seq subs refdraws pool.jsonl warm.jsonl tasks.jsonl pool-used.jsonl"
KEEP_CTL="box.json box_known_hosts"
SAVED=$JD/saved-for-e2e-$(date -u +%Y%m%d-%H%M%S)
SAVED_CTL=$CTL/saved-for-e2e-$(basename "$SAVED" | cut -d- -f4-)
install -d -m 700 -o judge -g judge "$SAVED"
install -d -m 700 -o driver -g driver "$SAVED_CTL"
for f in $KEEP_JD; do [ -e "$JD/$f" ] && mv "$JD/$f" "$SAVED/"; done
for f in $KEEP_CTL; do [ -e "$CTL/$f" ] && cp -p "$CTL/$f" "$SAVED_CTL/"; done
# put_back: the test's judge files to $JD/e2e-$STAMP, the saved ones back; once
PUT_BACK=0
put_back() {
  [ $PUT_BACK = 0 ] || return 0
  PUT_BACK=1
  STAMP=${STAMP:-$(date -u +%Y%m%d-%H%M%S)}
  mkdir -p $JD/e2e-$STAMP
  for f in $KEEP_JD; do [ -e $JD/$f ] && mv $JD/$f $JD/e2e-$STAMP/; done
  for f in $KEEP_JD; do [ -e "$SAVED/$f" ] && mv "$SAVED/$f" $JD/; done
  rmdir "$SAVED"
  [ -e $JD/subs ] || install -d -m 700 -o judge -g judge $JD/subs
  chown -R judge: $JD/e2e-$STAMP
  for f in $KEEP_CTL; do rm -f "$CTL/$f"; [ -e "$SAVED_CTL/$f" ] && mv "$SAVED_CTL/$f" "$CTL/"; done
  rmdir "$SAVED_CTL" 2>/dev/null
}
trap 'put_back' EXIT
install -d -m 700 -o judge -g judge "$JD/subs"
# the stand-in pool: 150 sessions in each bucket of the installed judge.draw, toy-model token ids
python3 - "$C/standin" "$JD" <<'EOF'
import json, sys
sys.path.insert(0, sys.argv[1])
import toylm
topics = ["harbour", "invoice", "glacier", "compiler", "violin", "orchard", "satellite", "bakery", "lantern", "museum"]
verbs = ["describes", "explains", "lists", "compares", "questions", "summarises"]
draw = json.load(open("/etc/ij-amd/controller.json"))["judge"].get("draw") or {"100": 1}
buckets = [b for b in draw if b != "*"] or ["100"]
with open(sys.argv[2] + "/pool.jsonl", "w") as f:
    for n, b in enumerate(buckets, 1):
        for s in range(150):
            t = (f"Note {b}/{s}: the {topics[s % 10]} {verbs[s % 6]} the {topics[(s * 7 + n) % 10]}. " * n).strip()
            f.write(json.dumps({"id": f"sb{b}-{s}", "session": f"sb{b}-{s}", "bucket": b,
                                "prompt_token_ids": toylm.encode(t)}) + "\n")
with open(sys.argv[2] + "/warm.jsonl", "w") as f:
    for i in range(4):
        f.write(json.dumps({"id": f"w{i}", "prompt_token_ids": toylm.encode(f"Warm-up text {i}, never judged. " * 3)}) + "\n")
EOF
# the stand-in task pool of measure B: 750 one-token letter picks, 250 for each of seeds 3, 4 and 5 (the toy
# model has about 259 a seed). The installed bar is set (2026-11-18), so every run needs 64 tasks; the six
# runs that keep their draw take 384.
for s in 3 4 5; do python3 "$C/standin/standin_tasks.py" 250 $s; done > "$JD/tasks.jsonl"
check "the installed quality bar is set (the checks below expect it)" python3 -c "
import sys; sys.path.insert(0, '$A/lib'); import json, quality
sys.exit(not quality.is_set(json.load(open('$A/config.json'))))"
chown judge: "$JD/pool.jsonl" "$JD/warm.jsonl" "$JD/tasks.jsonl"; chmod 600 "$JD/pool.jsonl" "$JD/warm.jsonl" "$JD/tasks.jsonl"
rm -rf /home/agents/e2e-repo /home/agents/e2e-wt2 /home/agents/e2e-wt3 /home/agents/e2e-wt4
sudo -u agents git clone -q /home/agents/engine /home/agents/e2e-repo
ag /home/agents/e2e-repo "printf '#!/bin/bash\nexec python3 $C/standin/fake_engine.py \"\$1\"\n' > serve && chmod +x serve && echo one > marker && printf 'build-out/\n' > .gitignore && git add -A && git commit -qm 'good engine'"
ag /home/agents/e2e-repo "git worktree add -q /home/agents/e2e-wt2 -b damaged && cd /home/agents/e2e-wt2 && printf '#!/bin/bash\nexec python3 $C/standin/fake_engine.py \"\$1\" --damage 3\n' > serve && echo two > marker && git commit -qam 'damaged engine'"
python3 -c "import json; json.dump({'state':'none','since':__import__('time').time(),'why':'e2e test start'}, open('$CTL/box.json','w'))"
chown driver: "$CTL/box.json"

echo "== 1. the driver refuses without a cap, and without the token"
mkcfg 'd["spend_cap_usd"] = None'
out=$(drv run 2>&1); rc=$?
check "cap unset: exit 4 (got $rc)" test $rc = 4
check "cap unset: says why" grep -q "spend_cap_usd is not set" <<<"$out"
mkcfg 'd["do_token_file"] = "/var/lib/ij-ctl/test/absent"'
out=$(drv run 2>&1); rc=$?
check "token absent: exit 4 (got $rc)" test $rc = 4
check "token absent: says where to put it" grep -q "no DigitalOcean token: put it in" <<<"$out"

echo "== 2. no GPU machine"
out=$(ag /home/agents/e2e-repo ./gpu); echo "$out" | sed 's/^/    /'
check "./gpu says there is none and since when" grep -q "there is no GPU machine, since 20" <<<"$out"
out=$(ag /home/agents/e2e-repo "timeout 8 ./try true" 2>&1); rc=$?
echo "$out" | sed 's/^/    /'
check "./try waits (killed by timeout: rc $rc)" test $rc = 124
check "./try says it waits" grep -q "waiting: there is no GPU machine" <<<"$out"

echo "== 3. the driver creates the stand-in machine"
mkcfg 'pass'
t0=$(date +%s); drv tick; echo "    tick took $(( $(date +%s) - t0 )) s"
check "state ready (got $(state))" test "$(state)" = ready
check "known_hosts holds exactly the driver's key for ij-box" test "$(wc -l < $CTL/box_known_hosts)" = 1
check "the driver found the price on page 2 and registered its public key before the droplet" python3 -c "
import json, urllib.request
c = json.load(urllib.request.urlopen('http://127.0.0.1:8650/fake/calls'))
s = [x for x in c if x['path'] == '/v2/sizes']
k = [i for i, x in enumerate(c) if x['method'] == 'POST' and x['path'] == '/v2/account/keys']
d = [x for x in c if x['method'] == 'POST' and x['path'] == '/v2/droplets']
assert len(s) >= 2 and len(k) == 1 and d and d[0]['body']['ssh_keys'], (len(s), k, d[:1])
assert c.index(d[0]) > k[0]
"
IP=$(python3 -c "import json; print(json.load(open('$CTL/box.json'))['ip'])")
out=$(sudo -u driver ssh -i $CTL/ssh/driver -o HostKeyAlias=ij-box -o StrictHostKeyChecking=yes -o UserKnownHostsFile=$CTL/box_known_hosts -o BatchMode=yes root@$IP true 2>&1); rc=$?
check "ssh to the box accepts only the generated host key (rc $rc)" test $rc = 0
out=$(sudo -u driver ssh -i $CTL/ssh/driver -o StrictHostKeyChecking=yes -o UserKnownHostsFile=/dev/null -o BatchMode=yes root@$IP true 2>&1); rc=$?
check "without the pinned key ssh refuses (rc $rc)" test $rc = 255
sleep 3   # the judge's remeasure starts in the background

echo "== 4. thin clients from two working trees at once"
ag /home/agents/e2e-repo "./try bash -c 'echo tree \$(cat marker); sleep 3; echo tree \$(cat marker) done'" > /tmp/e2e-t1 2>&1 &
P1=$!
ag /home/agents/e2e-wt2 "./try bash -c 'echo tree \$(cat marker); sleep 3; echo tree \$(cat marker) done'" > /tmp/e2e-t2 2>&1 &
P2=$!
wait $P1; R1=$?; wait $P2; R2=$?
sed 's/^/    1: /' /tmp/e2e-t1; sed 's/^/    2: /' /tmp/e2e-t2
check "./try in tree 1 sees its own files (rc $R1)" grep -q "tree one done" /tmp/e2e-t1
check "./try in tree 2 sees its own files (rc $R2)" grep -q "tree two done" /tmp/e2e-t2
out=$(ag /home/agents/e2e-repo "./try bash -c 'exit 7'" 2>&1); rc=$?
check "./try returns the command's exit code (7, got $rc)" test $rc = 7
ag /home/agents/e2e-repo "./box bash -c 'mkdir -p build-out && echo kept > build-out/x'" >/dev/null 2>&1
out=$(ag /home/agents/e2e-repo "./box bash -c 'cat build-out/x; whoami; ls -a | tr \"\n\" \" \"'" 2>&1); echo "    $out"
check "./box: a build output made on the box (git ignores it) stays there" grep -q kept <<<"$out"
check "./box runs as builder" grep -q builder <<<"$out"
check "./box: .git is not copied" bash -c '! grep -q " \.git " <<<"$1"' _ "$out"
out=$(ag /home/agents/e2e-repo "mkdir -p sub && cd sub && ../box pwd" 2>&1)
check "./box keeps the subdirectory ($out)" grep -q "/sub$" <<<"$out"
ag /home/agents/e2e-repo "./try sleep 300" > /tmp/e2e-kill 2>&1 &
sleep 6; pkill -TERM -u agents -f "try sleep 300"; sleep 4
out=$(ag /home/agents/e2e-repo "./box bash -c 'pgrep -u builder -f \"[s]leep 300\" || echo none'")
check "stopping the client stops the command on the box ($out)" test "$out" = none
out=$(ag /home/agents/e2e-repo "./ref 'Hello'" 2>&1); rc=$?
check "./ref answers (rc $rc)" grep -q "top" <<<"$out"
ag /home/agents/e2e-repo ./check > /tmp/e2e-c1 2>&1 & P1=$!
ag /home/agents/e2e-wt2 ./check > /tmp/e2e-c2 2>&1 & P2=$!
wait $P1; R1=$?; wait $P2; R2=$?
grep -E "PASS|FAIL" /tmp/e2e-c1 /tmp/e2e-c2 | sed 's/^/    /'
check "./check passes the good engine (rc $R1)" test $R1 = 0
check "./check fails the damaged engine (rc $R2)" test $R2 = 1
grep "^speed" /tmp/e2e-c1 | sed 's/^/    /'
check "./check prints the speed in parts" grep -q "speed (public prompts, not a score): whole request .* decode .* prompt time" /tmp/e2e-c1
grep "^quality" /tmp/e2e-c1 | sed 's/^/    /'
check "./check prints measure A, measure B on 16 public tasks, and PASS against the bar" bash -c \
  'grep -q "^quality A: .*top-5" /tmp/e2e-c1 && grep -q "^quality B: tasks right: engine [0-9]*/16, reference 16/16" /tmp/e2e-c1 && grep -q "^quality: PASS against the fixed bar" /tmp/e2e-c1'
check "./check fails the damaged engine against the bar" grep -q "^quality: FAIL against the fixed bar" /tmp/e2e-c2
out=$(ag /home/agents/e2e-repo ./gpu 2>&1); echo "$out" | head -3 | sed 's/^/    /'
check "./gpu reports the ready machine and the queue" grep -q "held by" <<<"$out"
PROF_BEFORE=" $(ls /home/agents/ij-profiles 2>/dev/null | tr '\n' ' ') "    # the test's own ./profile output is removed at the end
out=$(ag /home/agents/e2e-repo "./profile true" 2>&1); rc=$?
echo "    ./profile on the stand-in (no rocprofv3): rc $rc: $(echo "$out" | tail -1)"

echo "== 5. ./submit: a passing and a failing one"
# Forget the machine's reference speed (the remeasure at creation measured it in a run of its own), so the
# next submission measures it inside its own walk: one reference start for both.
python3 - <<'EOF'
import json, os
p = "/var/lib/ij-judge/machines.json"
st = os.stat(p)
m = json.load(open(p))
for v in m.values():
    v.pop("reference", None)
json.dump(m, open(p, "w"), indent=1)
os.chown(p, st.st_uid, st.st_gid)
EOF
ag /home/agents/e2e-repo "./submit --note 'e2e good'" > /tmp/e2e-s1 2>&1; R1=$?
ag /home/agents/e2e-wt2 "./submit --note 'e2e damaged'" > /tmp/e2e-s2 2>&1; R2=$?
tail -4 /tmp/e2e-s1 | sed 's/^/    1: /'; tail -4 /tmp/e2e-s2 | sed 's/^/    2: /'
check "the good engine passes (rc $R1)" test $R1 = 0
check "the damaged engine fails (rc $R2)" test $R2 = 1
check "what ./submit shows names no pool prompt and no prompt text" python3 -c "
import json, sys
sys.path.insert(0, '$C/standin')
import toylm
seen = open('/tmp/e2e-s1').read() + open('/tmp/e2e-s2').read()
hid = [json.loads(l) for l in open('$JD/pool.jsonl') if l.strip()]
sys.exit(any(h['id'] in seen or toylm.decode(h['prompt_token_ids'])[:30] in seen for h in hid))"
check "each submission drew its own sessions, none twice; the reference's draw went back" python3 -c "
import json, sys
u = [json.loads(l) for l in open('$JD/pool-used.jsonl')]
freed = {x['run'] for x in u if x.get('released')}
kept = [x for x in u if 'session' in x and x['run'] not in freed and not str(x['session']).startswith('standin-task-')]
runs = {x['run'] for x in kept}
k = sum((json.load(open('/etc/ij-amd/controller.json'))['judge'].get('draw') or {'*': 8}).values())
print('    kept runs', sorted(runs), 'released', sorted(freed), 'sessions per run', k)
sys.exit(len({x['session'] for x in kept}) != len(kept) or len(kept) != k * len(runs) or len(runs) < 2
         or not any(r.startswith('ref-') for r in freed))"
check "the good submission recorded both measures, 64 fresh tasks, and passed the bar" python3 -c "
import json, sys
last = {}
for l in open('/var/lib/ij-judge/ledger.jsonl'):
    x = json.loads(l); last[x['id']] = {**last.get(x['id'], {}), **x}
g = [x for x in last.values() if x.get('note') == 'e2e good'][-1]
q = json.load(open(f\"/var/lib/ij-judge/subs/{g['id']}/result.json\"))['quality']
print('    quality:', q['verdict'], '; A top-5', q['a']['topk_share']['5'], 'gap', q['a']['mean_gap'],
      '; B', q['b'].get('engine_correct'), '/', q['b'].get('tasks'), 'vs reference', q['b'].get('reference_correct'))
sys.exit(not (q['verdict'] == 'pass' and q['b']['tasks'] == 64 and q['a']['topk_share']['5'] == 1.0
              and (g.get('quality') or {}).get('verdict') == q['verdict'] and 'per_task' not in g['quality'].get('b', {})))"
check "the runs drew tasks, none twice" python3 -c "
import json, sys
u = [json.loads(l) for l in open('$JD/pool-used.jsonl')]
freed = {x['run'] for x in u if x.get('released')}
t = [x['session'] for x in u if str(x.get('session', '')).startswith('standin-task-') and x['run'] not in freed]
print('    tasks drawn and kept:', len(t))
sys.exit(len(t) < 128 or len(set(t)) != len(t))"
check "the good engine ran as cand on the box" box_sh "test -d /home/cand/run/src && stat -c %U /home/cand/run/src | grep -qx cand"
r=$(result "e2e good")
check "the good submission started the reference once and measured its speed in that start (${r##*|})" \
  test "${r##*|}" = 1
check "the machine's reference speed came from that run, under protocol 2" python3 -c "
import json, sys
m = json.load(open('/var/lib/ij-judge/machines.json'))
last = {}
for l in open('/var/lib/ij-judge/ledger.jsonl'):
    x = json.loads(l); last[x['id']] = {**last.get(x['id'], {}), **x}
g = [x for x in last.values() if x.get('note') == 'e2e good'][-1]
res = json.load(open(f\"/var/lib/ij-judge/subs/{g['id']}/result.json\"))
ref = (m.get(str(g['machine'])) or {}).get('reference') or {}
print('    reference', ref.get('tokens_per_s'), 'protocol', ref.get('protocol'), '; in the run:', (res.get('reference_speed') or {}).get('tokens_per_s'))
sys.exit(not (ref.get('protocol') == 2 and ref.get('tokens_per_s') == res['reference_speed']['tokens_per_s']
              and g.get('machine_reference_tps') == ref['tokens_per_s']))"
check "the good submission's result has the speed in parts" python3 -c "
import json, sys, glob
last = {}
for l in open('/var/lib/ij-judge/ledger.jsonl'):
    x = json.loads(l); last[x['id']] = {**last.get(x['id'], {}), **x}
g = [x for x in last.values() if x.get('note') == 'e2e good'][-1]
sp = json.load(open(f\"/var/lib/ij-judge/subs/{g['id']}/result.json\"))['speed']
p = sp['parts']
print('    parts: whole', p['whole_tps'], 'decode', p['decode_tps'], 'prompt', p['prompt_s'], '; requests', len(sp['per_request']))
sys.exit(not (p['whole_tps'] and p['decode_tps'] and p['prompt_s'] and len(sp['per_request']) == sp['requests']))"
check "what ./submit shows has the score line and the parts" bash -c 'grep -q "score: .* answer tokens / .* s of whole requests" /tmp/e2e-s1 && grep -q "parts, not the score: whole request" /tmp/e2e-s1'
out=$(ag /home/agents/e2e-repo ./score); echo "$out" | sed 's/^/    /'
check "./score shows the speed in parts of the passed submission" bash -c 'grep -q "speed of passed submissions" <<<"$1" && grep -q "decode .* tokens/s" <<<"$1"' _ "$out"
gid=$(python3 -c "
import json
last = {}
for l in open('/var/lib/ij-judge/ledger.jsonl'):
    x = json.loads(l); last[x['id']] = {**last.get(x['id'], {}), **x}
print([x for x in last.values() if x.get('note') == 'e2e good'][-1]['id'])")
out2=$(ag /home/agents/e2e-repo "./score $gid"); echo "$out2" | head -4 | sed 's/^/    /'
check "./score N lists the requests one by one" grep -q "hidden-1: [0-9]* prompt tokens" <<<"$out2"
check "./score lists both" bash -c 'grep -q "e2e good" <<<"$1" && grep -q "e2e damaged" <<<"$1"' _ "$out"
check "./score shows the machine's reference speed" grep -q "reference engine (AMD's ATOM, pinned image plus the logprob patch): [0-9]" <<<"$out"
check "./score shows the tuned engine as not measured" grep -q "not measured yet on this machine" <<<"$out"
out=$(ag /home/agents/e2e-repo "./submit --note 'during a box job'" 2>&1 & ag /home/agents/e2e-repo "./box sleep 1" 2>&1; wait)
check "a ./box run and a submit at once both finish" true

echo "== 5a. an engine that makes its own weights at start-up from the read-only models (NOTES: smaller weights)"
# The stand-in has no Docker, so serve runs as cand on the host (the path without a Dockerfile). builder leaves
# readable files where it can write, to record what such an engine sees of the machine beyond the models.
BX=$(docker ps -q --filter label=ij.fake=1 | head -1)
box_sh "rm -f /home/cand/convert-report.json"
docker exec -u builder "$BX" sh -c 'umask 022; echo b > /var/lib/ij-amd/builder/e2e-probe; echo b > /tmp/ij-e2e-probe; echo b > /home/builder/work/e2e-probe'
ag /home/agents/e2e-repo "git worktree add -q /home/agents/e2e-wt4 -b convert && cd /home/agents/e2e-wt4 && printf '#!/bin/bash\nexec python3 $C/standin/convert_and_serve.py \"\$1\"\n' > serve && git commit -qam 'engine that converts at start-up'"
ag /home/agents/e2e-wt4 "./submit --note 'e2e converts'" > /tmp/e2e-conv 2>&1; rc=$?
r=$(result "e2e converts"); echo "    converts: rc $rc: ${r:0:200}"
check "the engine that converts at start-up passes a judged ./submit (rc $rc)" test "$rc" = 0
rep=$(box_sh "cat /home/cand/convert-report.json 2>/dev/null"); echo "    its report: $rep"
check "it ran as cand, read the models, wrote its weights under its own tmp and served from them" python3 -c "
import json, sys
r = json.loads(sys.argv[1])
sys.exit(not (r['models'] == '/srv/models/standin' and r['models_bytes_read'] > 0 and r['weights'].startswith('/home/cand/tmp/')))" "$rep"
check "the models folder was not writable for the judged engine" python3 -c "
import json, sys; sys.exit(json.loads(sys.argv[1])['models_writable'] is not False)" "$rep"
check "builder's home stays closed to it" python3 -c "
import json, sys; sys.exit(json.loads(sys.argv[1])['probes_readable']['/home/builder/work/e2e-probe'] is not False)" "$rep"
echo "    recorded, not judged (NOTES): files builder made readable in /var/lib/ij-amd/builder and /tmp, seen by the" \
     "engine on the host: $(python3 -c "import json, sys; print(json.loads(sys.argv[1])['probes_readable'])" "$rep" 2>/dev/null)"
box_sh "rm -f /var/lib/ij-amd/builder/e2e-probe /tmp/ij-e2e-probe /home/builder/work/e2e-probe"

echo "== 5b. cheating engines (examples/cheaters) each fail; status failed, not error"
ag /home/agents/e2e-repo "git worktree add -q /home/agents/e2e-wt3 -b cheat" >/dev/null 2>&1
cheat() {  # cheat NAME NOTE: commit examples/cheaters/NAME/serve and cheat.py as the engine and submit it
  # copied by root: the examples are closed to agents (and cand cannot read /opt), so the commit carries them
  install -o agents -g agents -m 755 "$A/examples/cheaters/$1/serve" /home/agents/e2e-wt3/serve
  install -o agents -g agents -m 644 "$A/examples/cheaters/cheat.py" /home/agents/e2e-wt3/cheat.py
  ag /home/agents/e2e-wt3 "git add serve cheat.py && git commit -qm 'cheater $1' --allow-empty && ./submit --note '$2'" > /tmp/e2e-cheat-$2 2>&1
  echo $?
}
rc=$(cheat fabricate cheat-fabricate); r=$(result cheat-fabricate); echo "    fabricate: rc $rc: ${r:0:300}"
check "fabricate fails (rc 1)" test "$rc" = 1
check "fabricate: on the bar's top-5 share, the walk stopped early, no request refused" bash -c '[[ "$1" == failed\|*"top-5 share"* && "$1" == *"the walk stopped after"* && "$1" != *"request failed"* ]]' _ "$r"
rc1=$(cheat replay cheat-replay-1); rc2=$(cheat replay cheat-replay-2); r=$(result cheat-replay-2)
echo "    replay: rc $rc1 then $rc2: ${r:0:300}"
check "replay fails both times (rc $rc1, $rc2)" test "$rc1$rc2" = 11
check "replay: the second on the bar's top-5 share" bash -c '[[ "$1" == failed\|*"top-5 share"* ]]' _ "$r"
n=$(box_sh "ls /home/cand/.cache/ij-replay 2>/dev/null | wc -l")
check "replay did record what it was sent ($n answers in cand's home)" test "$n" -gt 0
rc=$(cheat read-judge cheat-read-judge); r=$(result cheat-read-judge); echo "    read-judge: rc $rc: ${r:0:300}"
check "read-judge fails (rc 1)" test "$rc" = 1
check "read-judge read none of the judge's files" bash -c '[[ "$1" == failed\|*"could read none of the judge files"* ]]' _ "$r"
check "read-judge got no judged prompt, so its draw went back to the pool" python3 -c "
import json, sys
last = {}
for l in open('$JD/ledger.jsonl'):
    r = json.loads(l); last[r['id']] = {**last.get(r['id'], {}), **r}
n = [x for x in last.values() if x.get('note') == 'cheat-read-judge'][-1]['id']
u = [json.loads(l) for l in open('$JD/pool-used.jsonl')]
sys.exit(not any(x.get('released') and x['run'].startswith(f'sub-{n}-') for x in u))"
BOX=$(docker ps -q --filter label=ij.fake=1 | head -1)
out=$(docker exec -u judge -e HOME=/home/judge -w /home/judge "$BOX" python3 -c "import sys; sys.path.insert(0, '$A/examples/cheaters'); import cheat; print(len(cheat.try_read_judge()))" 2>&1 | tail -1)
check "control: as judge the same paths are readable ($out)" test "${out:-0}" -gt 0 2>/dev/null
rc=$(cheat proxy-ref cheat-proxy-ref); r=$(result cheat-proxy-ref); echo "    proxy-ref: rc $rc: ${r:0:300}"
check "proxy-ref fails (rc 1)" test "$rc" = 1
check "proxy-ref: its requests found no reference (connection refused)" bash -c '[[ "$1" == failed\|*"proxy-ref"*"refused"* ]]' _ "$r"
check "proxy-ref: the reference was not started before the engine ended" bash -c '[[ "$1" == *\|0 || "$1" == *\|1 ]]' _ "$r"
left=$(box_sh "pgrep -u cand | wc -l")
check "no process of cand is left on the box ($left)" test "$left" = 0
check "cand cannot become judge or root" bash -c "! docker exec -u cand $BOX sudo -n true 2>/dev/null && ! docker exec -u cand $BOX sudo -n -u judge true 2>/dev/null"
check "cand cannot list the judge's home or state" bash -c "! docker exec -u cand $BOX ls /home/judge /var/lib/ij-amd/judge 2>/dev/null"
full=$(docker exec -u judge -e HOME=/home/judge -e USER=judge -e LOGNAME=judge -w /home/judge "$BOX" bash $C/tests/replay-control.sh 2>&1)
out=$(tail -1 <<<"$full")
echo "    replay control (same prompts twice, on the box as judge): $out"
[ "$out" = "failed passed" ] || sed 's/^/      /' <<<"$full"
check "control: replay fails, then passes on the same prompts" test "$out" = "failed passed"
printf '%s\n' '{"id": "pp1", "prompt": "A new pool row about tides", "bucket": "100"}' \
  '{"id": "pp2", "prompt": "Another pool row about kites", "bucket": "100", "session": "pp-s"}' \
  '{"id": "pp3", "prompt": "Same session as pp2", "bucket": "100", "session": "pp-s"}' > /tmp/e2e-pp.jsonl
chmod 644 /tmp/e2e-pp.jsonl
n0=$(wc -l < $JD/pool.jsonl)
out=$(sudo -u judge $C/judge/ij-judge prepare-pool /tmp/e2e-pp.jsonl 2>&1); echo "$out" | tail -1 | sed 's/^/    /'
n1=$(wc -l < $JD/pool.jsonl)
check "prepare-pool tokenizes on the box and adds 3 rows ($n0 -> $n1)" test $((n1 - n0)) = 3
sudo -u judge $C/judge/ij-judge prepare-pool /tmp/e2e-pp.jsonl >/dev/null 2>&1
check "prepare-pool again adds nothing" test "$(wc -l < $JD/pool.jsonl)" = "$n1"
# the operator's commands of the GPU session (gpu-session/run-session.sh, steps 2 and 3)
ls "$JD/refdraws" 2>/dev/null | sort > /tmp/e2e-rd0
out=$(sudo -u judge $C/judge/ij-judge reference-draws 2 2>&1); echo "$out" | tail -1 | sed 's/^/    /'
new=$(ls "$JD/refdraws" 2>/dev/null | sort | comm -13 /tmp/e2e-rd0 -)
check "reference-draws 2 keeps each draw's requests in refdraws/ ($(echo $new))" python3 -c "
import json, sys
names = sys.argv[1].split()
assert len(names) == 2, names
for n in names:
    rows = json.load(open('$JD/refdraws/' + n)).get('per_request') or []
    assert rows and all('tokens' in r and 'finish' in r for r in rows), n
" "$new"
check "reference-draws gave its draws back to the pool" python3 -c "
import json, sys
lines = [json.loads(l) for l in open('$JD/pool-used.jsonl')]
freed = {u['run'] for u in lines if u.get('released')}
for n in sys.argv[1].split():
    assert n[:-5] in freed, n
" "$new"
install -d -m 755 /tmp/e2e-meas
printf '#!/bin/bash\nexec python3 %s/standin/fake_engine.py "$1"\n' "$C" > /tmp/e2e-meas/serve
chmod 755 /tmp/e2e-meas/serve
out=$(sudo -u judge $C/judge/ij-judge measure atom /tmp/e2e-meas "e2e measurement" 2>&1); rc=$?
echo "$out" | tail -1 | sed 's/^/    /'
check "measure atom judges the engine through a submission's path (rc $rc)" test "$rc" = 0
check "the measurement is in the ledger as kind atom and is never the best" python3 -c "
import json
by = {}
for l in open('$JD/ledger.jsonl'):                 # a run's rows are merged in order, as the judge reads them
    r = json.loads(l); by.setdefault(r['id'], {}).update(r)
m = [r for r in by.values() if r.get('measure') == 'atom']
assert m and m[-1].get('status') in ('passed', 'failed'), m[-1:]
doc = json.load(open('/var/lib/ij-score/score.json'))
best = [t for t in doc['text'] if t.startswith('best passing submission')][0]
assert not best.startswith(f\"best passing submission: #{m[-1]['id']} \"), best
assert any(s.get('measure') == 'atom' for s in doc['submissions'])
assert any('not ranked' in t for t in doc['text'])
"
# the GPU session's preflight over real ssh to the stand-in box: its checks of the firewall, the launcher and
# builder pass; it stops at the reference image (no docker on the stand-in) or at judge.draw (the test's buckets)
GS=/var/lib/ij-ctl/gpu-session/e2e-preflight
rm -rf "$GS"
IJ_CTL_CONFIG="$T/cfg.json" STAMP=e2e-preflight bash $A/measurements/2026-11-17/gpu-session/run-session.sh 1 >/dev/null 2>&1; rc=$?
sed 's/^/      /' "$GS/session.log"
check "run-session.sh 1 stops at preflight on the stand-in (rc $rc)" bash -c '[ "$1" = 5 ] && grep -q "STOP: preflight failed" "$2"' _ "$rc" "$GS/session.log"
check "run-session.sh 1: ssh, the firewall, the launcher and builder checks pass first" bash -c \
  '! grep -qE "no ready GPU machine|not in place|launcher is not installed|builder is in the docker|host key" "$1" && grep -qE "reference image is not on the machine|judge.draw does not match" "$1"' _ "$GS/session.log"
check "run-session.sh 1 copied its log to the box" box_sh "test -s /root/gpu-session/e2e-preflight/session.log"
rm -rf "$GS"; box_sh "rm -rf /root/gpu-session/e2e-preflight"

echo "== 5c. containment: cand's firewall rule, the Docker launcher, builder's leftovers"
# The stand-in box runs with NET_ADMIN, so setup-box.sh applied the real rule in its network namespace.
check "the firewall rule for cand is in place on the box" box_sh "$C/box/ij-fw check | grep -qx ok"
check "judge may run the check through sudo" bash -c "docker exec -u judge $BOX sudo -n $C/box/ij-fw check >/dev/null"
check "builder and cand may not" bash -c "! docker exec -u builder $BOX sudo -n $C/box/ij-fw check 2>/dev/null && ! docker exec -u cand $BOX sudo -n $C/box/ij-fw check 2>/dev/null"
conn='import socket, sys
try:
    socket.create_connection((sys.argv[1], int(sys.argv[2])), 3).close(); print("connected")
except Exception as e:
    print(type(e).__name__)'
out_c=$(docker exec -u cand "$BOX" python3 -c "$conn" 192.0.2.1 9 2>&1)
out_j=$(docker exec -u judge "$BOX" python3 -c "$conn" 192.0.2.1 9 2>&1)
echo "    a connection off the machine (192.0.2.1:9): cand $out_c, judge $out_j"
check "cand's connection off loopback is refused at once; judge's is not" bash -c '[ "$1" = ConnectionRefusedError ] && [ "$2" != ConnectionRefusedError ]' _ "$out_c" "$out_j"
out=$(docker exec -u cand "$BOX" python3 -c 'import socket
s = socket.socket(); s.bind(("127.0.0.1", 0)); s.listen()
socket.create_connection(s.getsockname(), 3).close(); print("connected")' 2>&1)
check "cand reaches its own loopback port ($out)" test "$out" = connected
docker exec -d "$BOX" python3 -c 'import socket
s = socket.socket(); s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1); s.bind(("127.0.0.1", 53)); s.listen()
import time; t = time.time()
while time.time() - t < 60:
    s.settimeout(1)
    try: s.accept()[0].close()
    except OSError: pass'
sleep 1
out_c=$(docker exec -u cand "$BOX" python3 -c "$conn" 127.0.0.1 53 2>&1)
out_j=$(docker exec -u judge "$BOX" python3 -c "$conn" 127.0.0.1 53 2>&1)
check "loopback port 53 (a resolver): judge connects ($out_j), cand is refused ($out_c)" bash -c '[ "$1" = connected ] && [ "$2" = ConnectionRefusedError ]' _ "$out_j" "$out_c"
CU=$(docker exec "$BOX" id -u cand)
box_sh "iptables -D OUTPUT -m owner --uid-owner $CU -j IJ-CAND"
check "with the hook removed the check fails" bash -c "! docker exec $BOX $C/box/ij-fw check >/dev/null"
ag /home/agents/e2e-repo "./submit --note 'e2e no firewall'" > /tmp/e2e-nofw 2>&1; rc=$?
r=$(result "e2e no firewall"); echo "    without the rule: rc $rc: ${r:0:300}"
check "without the rule ./submit stops with an error naming the firewall" bash -c '[[ "$1" == error\|*"firewall rule for the engine account is not in place"* ]]' _ "$r"
check "without the rule no judged prompt was sent and the draw went back to the pool" python3 -c "
import json, sys
last = {}
for l in open('$JD/ledger.jsonl'):
    r = json.loads(l); last[r['id']] = {**last.get(r['id'], {}), **r}
n = [x for x in last.values() if x.get('note') == 'e2e no firewall'][-1]['id']
res = json.load(open(f'$JD/subs/{n}/result.json'))
u = [json.loads(l) for l in open('$JD/pool-used.jsonl')]
sys.exit(not ((res.get('times') or {}).get('judged_sent') == 0
              and any(x.get('released') and x['run'].startswith(f'sub-{n}-') for x in u)))"
box_sh "$C/box/ij-fw apply >/dev/null"
check "ij-fw apply puts it back" box_sh "$C/box/ij-fw check | grep -qx ok"
check "the launcher is in sudoers for builder, judge and cand" box_sh "grep -q 'builder,judge,cand ALL=(root) NOPASSWD: $C/box/ij-docker' /etc/sudoers.d/ij-box"
check "builder is not in the docker group" bash -c "! docker exec $BOX id -nG builder | grep -qw docker"
lr() { docker exec -u "$1" "$BOX" sudo -n $C/box/ij-docker "${@:2}" 2>&1; }
for a in "-v /:/host" "--privileged" "--network host" "--pid host" "--device /dev/mem" "--cap-add SYS_ADMIN" \
         "--security-opt seccomp=unconfined" "--user 0" "--mount type=bind,src=/,dst=/h"; do
  # shellcheck disable=SC2086
  out=$(lr cand run $a ij-cand/engine:latest true); rc=$?
  check "cand: ij-docker run $a is refused through sudo (rc $rc)" bash -c '[ "$1" = 2 ] && grep -q "refused" <<<"$2"' _ "$rc" "$out"
done
out=$(lr builder run -v /:/host ij-builder/engine:latest true); rc=$?
check "builder: ij-docker run -v /:/host is refused (rc $rc)" bash -c '[ "$1" = 2 ] && grep -q "refused" <<<"$2"' _ "$rc" "$out"
out=$(lr cand build /home/cand); rc=$?
check "cand cannot build images (rc $rc)" test "$rc" = 2
out=$(lr builder run ij-cand/engine:latest true); rc=$?
check "builder cannot run cand's image (rc $rc)" test "$rc" = 2
docker exec -d -u builder "$BOX" setsid sleep 7777
sleep 1
n0=$(box_sh "pgrep -u builder -fc 'sleep 7777'")
out=$(docker exec -u judge "$BOX" sudo -n $C/box/ij-reap --machine 2>&1); echo "    ij-reap --machine: $out"
n1=$(box_sh "pgrep -u builder -fc 'sleep 7777' || true")
check "ij-reap --machine kills builder's detached process too ($n0 -> $n1)" test "$n0" = 1 -a "$n1" = 0

echo "== 6. the judge's files are closed to agents"
check "agents cannot read the hidden pool" bash -c "! sudo -u agents cat $JD/pool.jsonl 2>/dev/null"
check "agents cannot list the judge's directory" bash -c "! sudo -u agents ls $JD 2>/dev/null"
check "agents cannot write the ledger" bash -c "! sudo -u agents bash -c 'echo x >> $JD/ledger.jsonl' 2>/dev/null"
check "agents cannot write the public score" bash -c "! sudo -u agents bash -c 'echo x > /var/lib/ij-score/score.json' 2>/dev/null"
check "agents cannot run other judge commands" bash -c "! sudo -u agents sudo -n -u judge $C/judge/ij-judge score 2>/dev/null"
check "agents cannot run a shell as judge" bash -c "! sudo -u agents sudo -n -u judge cat $JD/pool.jsonl 2>/dev/null"
check "agents cannot read the judge's key" bash -c "! sudo -u agents cat $JD/ssh/ij_box 2>/dev/null"
check "agents cannot read the driver's key" bash -c "! sudo -u agents cat $CTL/ssh/driver 2>/dev/null"
check "agents cannot read the machines records" bash -c "! sudo -u agents ls $CTL/records 2>/dev/null"
out=$(sudo -u agents ssh -i /home/agents/.ssh/ij_box -o HostKeyAlias=ij-box -o UserKnownHostsFile=$CTL/box_known_hosts -o BatchMode=yes judge@$IP true 2>&1); rc=$?
check "agents' key does not open judge@box (rc $rc)" test $rc = 255
check "agents can read the public score" sudo -u agents cat /var/lib/ij-score/score.json >/dev/null

echo "== 6a. the judge's code is closed to agents, builder and cand (code-access.sh)"
CLOSED="$C/judge/ij-judge $A/lib/ledger.py $A/submit $A/examples/cheaters/cheat.py $A/tests/test_judge_score.py $A/measurements/2026-11-17/NOTES.md $C/driver/ij-driver"
for f in $CLOSED $A/lib/answers.py $A/config.json; do
  check "controller: agents cannot read ${f#$A/}" bash -c "! sudo -u agents cat '$f' >/dev/null 2>&1"
done
check "controller: agents cannot list the code" bash -c "! sudo -u agents ls $A >/dev/null 2>&1 && ! sudo -u agents ls $A/controller >/dev/null 2>&1"
check "controller: judge and driver read the judge's code" bash -c "sudo -u judge cat $C/judge/ij-judge >/dev/null && sudo -u driver cat $C/judge/ij-judge >/dev/null"
check "controller: agents read their command and its module" bash -c "sudo -u agents cat $C/bin/ijc $C/lib/ijctl.py >/dev/null"
BOX=$(docker ps -q --filter label=ij.fake=1 | head -1)
for u in builder cand; do
  for f in $CLOSED; do
    check "box: $u cannot read ${f#$A/}" bash -c "! docker exec -u $u $BOX cat '$f' >/dev/null 2>&1"
  done
  check "box: $u cannot list the code" bash -c "! docker exec -u $u $BOX ls $A >/dev/null 2>&1"
done
check "box: judge reads the judge's code" docker exec -u judge "$BOX" cat $C/judge/ij-judge $A/lib/ledger.py >/dev/null
check "box: builder reads what AGENTS.md names (amd-tuned-engine, atom-engine, reference)" \
  docker exec -u builder "$BOX" cat $A/examples/amd-tuned-engine/serve $A/examples/amd-tuned-engine/Dockerfile $A/reference/atom-serve.sh $A/reference/setup.sh >/dev/null
out=$(docker exec -u builder "$BOX" sudo -n $C/box/ij-docker --help 2>&1); rc=$?
check "box: builder still runs sudo ij-docker (--help: rc $rc)" test "$rc" = 0
check "box: the launcher passes reference/atom-serve.sh's [ -x ] test for cand but cannot be read" \
  bash -c "docker exec -u cand $BOX test -x $C/box/ij-docker && ! docker exec -u cand $BOX cat $C/box/ij-docker >/dev/null 2>&1"
docker exec "$BOX" python3 -c "import sys; sys.path.insert(0, '$A/lib'); import answers, ledger" 2>/dev/null   # root leaves __pycache__
check "box: what root compiled stays readable to judge (setgid: group ijcode)" \
  bash -c "docker exec -u judge $BOX sh -c 'cat $A/lib/__pycache__/*.pyc' >/dev/null"
python3 -c "import sys; sys.path.insert(0, '$A/lib'); import answers, ledger" 2>/dev/null
out=$(sudo -u driver rsync -an "$A/" /tmp/e2e-rsync-probe/ 2>&1); rc=$?
check "controller: after root imports from the tree, the driver can still copy all of it (rsync rc $rc)" test "$rc" = 0
# ./try and ./check above ran as builder through the closed tree; push-code sets the access again on a ready machine
out=$(drv push-code 2>&1); rc=$?; echo "$out" | tail -2 | sed 's/^/    /'
check "ij-driver push-code: the code to the ready machine and its access again (rc $rc)" test "$rc" = 0
check "after push-code builder still cannot read the judge's code" bash -c "! docker exec -u builder $BOX cat $C/judge/ij-judge >/dev/null 2>&1"
out=$(ag /home/agents/e2e-repo "./try bash -c 'echo after-push'" 2>&1)
check "after push-code ./try still works" grep -q after-push <<<"$out"

if [ "${E2E_DRIVER:-0}" != 1 ]; then
  echo "== 7 and 8 (reclaim, daily limit, spending cap with a stand-in agent) skipped: set E2E_DRIVER=1 to run them"
else
echo "== 7. a spot reclaim, and the daily limit (2 in the test config)"
ID=$(python3 -c "import json; print(json.load(open('$CTL/box.json'))['id'])")
curl -s -X POST http://127.0.0.1:$PORT/fake/reclaim/$ID >/dev/null
drv tick
check "after the reclaim a second machine is made (state $(state))" test "$(state)" = ready
check "the reclaim is in the ledger as gone" grep -q '"event": "gone"' $T/machines.jsonl
ID=$(python3 -c "import json; print(json.load(open('$CTL/box.json'))['id'])")
curl -s -X POST http://127.0.0.1:$PORT/fake/reclaim/$ID >/dev/null
out=$(drv tick 2>&1); echo "$out" | tail -2 | sed 's/^/    /'
check "the third creation in 24 h is refused" grep -q "daily limit: 2 machines created" <<<"$out"
out=$(ag /home/agents/e2e-repo ./gpu); echo "    $out" | head -1
check "./gpu shows why there is no machine" grep -q "daily limit" <<<"$out"
creates=$(grep -c '"event": "create"' $T/machines.jsonl)
check "exactly 2 creations in the ledger ($creates)" test "$creates" = 2
curl -s http://127.0.0.1:$PORT/fake/calls > $T/calls.json
check "every create asked for gpu-mi350x1-288gb-spot in ric1, with no droplet existing" python3 -c "
import json, sys
c = [x for x in json.load(open('$T/calls.json')) if x['method'] == 'POST' and x['path'] == '/v2/droplets']
print('    creates:', [(x['body']['size'], x['body']['region'], x['live']) for x in c])
sys.exit(not c or any(x['body']['size'] != 'gpu-mi350x1-288gb-spot' or x['body']['region'] != 'ric1' or x['live'] for x in c))"

echo "== 8. the spending cap: destroy and stop (price \$7200/h = \$2/s, cap \$250)"
rm -f $T/machines.jsonl
python3 -c "import json; json.dump({'state':'none','since':__import__('time').time()}, open('$CTL/box.json','w'))"
pkill -f "fake_do.py $PORT"; sleep 0.5
nohup python3 "$C/standin/fake_do.py" $PORT /root/ij-fake-do-token 7200 > /tmp/fake_do.log 2>&1 &
FAKE=$!; sleep 1
curl -s -X POST http://127.0.0.1:$PORT/fake/fail-delete/2 >/dev/null   # the first two deletions fail
mkcfg 'd["spend_cap_usd"] = 250; d["min_hours_left_to_create"] = 0.01; c["agent"].update({"enabled": True, "restart_delay_s": 1, "command": ["bash", "-c", "echo stand-in for Prime Agent; sleep 100000"]})'
t0=$(date +%s)
timeout 600 sudo -u driver -H env IJ_CTL_CONFIG="$T/cfg.json" "$C/driver/ij-driver" run > /tmp/e2e-cap.log 2>&1; rc=$?
sed 's/^/    /' /tmp/e2e-cap.log | grep -v "^    \[judge"
echo "    run took $(( $(date +%s) - t0 )) s"
check "the driver exits 3 at the cap (got $rc)" test $rc = 3
check "the run is marked stopped (state $(state))" test "$(state)" = stopped
check "failed deletions are retried, not taken as the end" grep -q "error in this pass (retrying).*HTTP 500" /tmp/e2e-cap.log
check "the droplet was deleted" bash -c "curl -s http://127.0.0.1:$PORT/fake/calls | grep -q '\"DELETE\"'"
check "no droplet is left" test -z "$(docker ps -q --filter label=ij.fake=1)"
spent=$(sudo -u driver env IJ_CTL_CONFIG=$T/cfg.json $C/driver/ij-driver status | grep spent)
echo "    $spent"
check "the agent's tmux session is gone" bash -c "! sudo -u agents tmux -L ij has-session -t prime 2>/dev/null"
RD=$(ls -d $CTL/records/run-* | tail -1)
check "the run's records: prompt.json with AGENTS.md in it" grep -q "This repository starts empty" $RD/prompt.json
check "the run's records: start and end" test -f $RD/started -a -f $RD/ended
out=$(drv run 2>&1); rc=$?
check "a stopped run does not start again (rc $rc)" test $rc = 3

echo "== 9. a driver stopped after the droplet was made: the next pass sets that machine up"
rm -f $T/machines.jsonl
python3 -c "import json; json.dump({'state':'none','since':__import__('time').time()}, open('$CTL/box.json','w'))"
chown driver: $CTL/box.json
pkill -f "fake_do.py $PORT"; sleep 0.5
nohup python3 "$C/standin/fake_do.py" $PORT /root/ij-fake-do-token 2.0 > /tmp/fake_do.log 2>&1 &
FAKE=$!; sleep 1
mkcfg 'pass'
drv tick > /tmp/e2e-resume-1 2>&1 &
# stopped right after the droplet was made (state creating with its id; the stand-in's set-up takes only
# seconds, so a later moment is hard to hit); the next pass runs the whole set-up, as after a restart
for i in $(seq 1 600); do
  python3 -c "import json, sys; s = json.load(open('$CTL/box.json')); sys.exit(not (s['state'] == 'creating' and s.get('id')))" 2>/dev/null && break
  sleep 0.1
done
pkill -KILL -u driver -f "ij-driver tick"; sleep 1
st1=$(state)
echo "    stopped in state $st1"
drv tick > /tmp/e2e-resume-2 2>&1; rc=$?
tail -3 /tmp/e2e-resume-2 | sed 's/^/    /'
check "stopped in creating ($st1), the next pass ends ready (state $(state), rc $rc)" bash -c '[ "$1" = creating ] && [ "$2" = ready ]' _ "$st1" "$(state)"
check "the next pass resumed the set-up, it did not make a second machine" bash -c \
  "grep -q 'resum' /tmp/e2e-resume-2 && [ \"\$(grep -c '\"event\": \"create\"' $T/machines.jsonl)\" = 1 ] && [ \"\$(docker ps -q --filter label=ij.fake=1 | wc -l)\" = 1 ]"
fi

echo "== cleanup"
kill $FAKE 2>/dev/null
docker rm -f $(docker ps -aq --filter label=ij.fake=1) >/dev/null 2>&1
STAMP=$(date -u +%Y%m%d-%H%M%S)
put_back
sudo -u judge $C/judge/ij-judge score
mkdir -p $CTL/records-e2e-$STAMP && mv $CTL/records/run-* $CTL/records-e2e-$STAMP/ 2>/dev/null; chown -R driver: $CTL/records-e2e-$STAMP; chmod 750 $CTL/records-e2e-$STAMP
rm -f $CTL/agent-current $CTL/agent-last-start /root/ij-fake-do-token
rm -rf /tmp/e2e-* /tmp/ij-e2e-*
if [ ! -e $CTL/box.json ]; then         # none was saved: a fresh controller
  python3 -c "import json; json.dump({'state':'none','since':__import__('time').time(),'why':'the driver has not run yet'}, open('$CTL/box.json','w'))"
  chown driver: $CTL/box.json
fi
rm -rf /home/agents/e2e-repo /home/agents/e2e-wt2 /home/agents/e2e-wt3 /home/agents/e2e-wt4 /home/agents/.ij-run
if [ -n "${PROF_BEFORE+set}" ]; then                # the agents' own ./profile output stays; the test's goes
  for d in /home/agents/ij-profiles/*; do
    [ -e "$d" ] && [[ "$PROF_BEFORE" != *" $(basename "$d") "* ]] && rm -rf "$d"
  done
fi
echo "== $PASS passed, $FAIL failed (judge test records moved to $JD/e2e-$STAMP; the judge's own records put back)"
[ $FAIL = 0 ]
