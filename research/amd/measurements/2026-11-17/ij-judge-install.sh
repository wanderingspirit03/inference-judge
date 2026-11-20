set -uo pipefail
cd /root/inference-judge
IJ_BOOTSTRAP_LIB=1 . setup/bootstrap.sh
set +e
parse_args --target targets/mi350x-qwen38-27b-tps.yaml --agent TEMPLATE --no-clock --skip-model
resolve_agent
resolve_target
for f in s_preflight s_docker s_apt s_users s_uv s_engine_image s_ruler_image s_helper_images s_bench s_bench_venv s_target s_research; do
  log "$f"
  ( set -e; "$f" ) || { echo "STEP FAILED: $f" >&2; exit 1; }
done
usermod -aG render,video bench
assert_bench_access && echo ALL-OK
