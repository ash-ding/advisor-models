#!/bin/bash
# Start the SWE-smith agent server (:8081) and eval server (:8082) on this host.
#
# Hosts without a docker CLI but with a podman engine (e.g. lumen): mini-swe-agent
# is pointed at podman, and swesmith's `docker image inspect/pull` shell-outs go
# through a docker -> podman symlink placed first on the servers' PATH. The docker
# Python SDK reaches the engine through DOCKER_HOST.
# Logs, PIDs and the shim live in $RUN_DIR (default swe_servers/, gitignored).
# Stop with: kill $(cat swe_servers/*.pid)
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
RUN_DIR="${RUN_DIR:-$REPO_ROOT/swe_servers}"
HOST="${HOST:-127.0.0.1}"
cd "$REPO_ROOT"
mkdir -p "$RUN_DIR/bin" "$RUN_DIR/agent_logs"

if ! command -v docker >/dev/null && command -v podman >/dev/null; then
  ln -sf "$(command -v podman)" "$RUN_DIR/bin/docker"
  export PATH="$RUN_DIR/bin:$PATH"
  export MSWEA_DOCKER_EXECUTABLE=podman
fi

setsid nohup .venv/bin/python -m advisor_models.swe_smith.agent_server \
  --host "$HOST" --port 8081 --log-dir "$RUN_DIR/agent_logs" \
  </dev/null >"$RUN_DIR/agent_server.log" 2>&1 &
echo $! >"$RUN_DIR/agent_server.pid"
setsid nohup .venv/bin/python -m advisor_models.swe_smith.eval_server \
  --host "$HOST" --port 8082 \
  </dev/null >"$RUN_DIR/eval_server.log" 2>&1 &
echo $! >"$RUN_DIR/eval_server.pid"

for _ in $(seq 1 30); do
  if curl -sf -m 5 "http://$HOST:8081/health" >/dev/null && curl -sf -m 5 "http://$HOST:8082/health" >/dev/null; then
    echo "agent server pid $(cat "$RUN_DIR/agent_server.pid"), eval server pid $(cat "$RUN_DIR/eval_server.pid"): healthy"
    exit 0
  fi
  sleep 2
done
echo "servers did not become healthy; see $RUN_DIR/*.log" >&2
exit 1
