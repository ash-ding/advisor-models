#!/bin/bash
# Install the SWE-smith harness into the root venv (needed by the eval server).
#
# Only the `validate` extra plus the docker SDK are installed; `all` pulls in
# sglang/modal, which the harness does not need. Pinned to the commit this setup
# was verified with.
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
SWE_SMITH_DIR="${SWE_SMITH_DIR:-$(dirname "$REPO_ROOT")/SWE-smith}"
SWE_SMITH_COMMIT=9b74ac0

[ -d "$SWE_SMITH_DIR" ] || git clone https://github.com/SWE-bench/SWE-smith "$SWE_SMITH_DIR"
git -C "$SWE_SMITH_DIR" checkout -q "$SWE_SMITH_COMMIT"
uv pip install --python "$REPO_ROOT/.venv/bin/python" -e "$SWE_SMITH_DIR[validate]" docker
