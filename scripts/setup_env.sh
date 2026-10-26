#!/bin/bash
# Build both virtual environments.
#
# The training venv (SkyRL/skyrl-train) does not list the packages the
# advisor_models environments import (litellm, math-verify, ...), so they are
# added on top. The constraints file keeps the versions vLLM/Ray were locked
# with; without it protobuf jumps from 4.x to 7.x.
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

uv sync

cd SkyRL/skyrl-train
uv sync --extra vllm
uv pip install --python .venv/bin/python \
  -c "$REPO_ROOT/scripts/train_venv_constraints.txt" \
  "litellm==1.74.0" "google-cloud-aiplatform>=1.38" \
  math-verify pylcs rank-bm25 sacrebleu evaluate
