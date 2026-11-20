#!/bin/bash
# On the box: ./submit --reference, a passing submission (the ATOM test
# engine), a failing one (serve exits at once), then ./score.  Test engines
# live in throw-away git repos under /root.
set -u
IJ=/opt/ij-amd
mk() {  # mk <dir> <serve body>
    rm -rf "$1"; mkdir -p "$1"
    printf '#!/bin/bash\n%s\n' "$2" > "$1/serve"; chmod +x "$1/serve"
    git -C "$1" init -q && git -C "$1" -c user.name=t -c user.email=t@t add serve \
        && git -C "$1" -c user.name=t -c user.email=t@t commit -qm "test engine"
}
mk /root/eng-good "exec $IJ/examples/atom-engine/serve \"\$@\""
mk /root/eng-broken 'echo "no engine here" >&2; exit 3'
echo "== submit --reference"; time "$IJ/submit" --reference; echo "rc=$?"
echo "== submit good";   (cd /root/eng-good && time "$IJ/submit" --note "ATOM test engine"); echo "rc=$?"
echo "== submit broken"; (cd /root/eng-broken && time "$IJ/submit" --note "serve exits at once"); echo "rc=$?"
echo "== score"; time "$IJ/score"
