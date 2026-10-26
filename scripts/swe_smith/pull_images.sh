#!/bin/bash
# Pull every image referenced by SWE-smith dataset files and alias it for swesmith.
#
# The HF dataset's image_name uses the old Docker Hub org (jyangballin/), which
# the agent server starts. Current swesmith derives the name with the new org
# (swebench/) and shells out to `docker image inspect`, then `docker pull` with a
# short name that the podman engine refuses. Tagging the pulled image under the
# swesmith name makes the inspect succeed, so eval runs in the same image the
# agent worked in.
# Usage: scripts/swe_smith/pull_images.sh data/swe_smith/train_X.json [more.json ...]
set -euo pipefail
for img in $(python3 -c "
import json, sys
names = set()
for f in sys.argv[1:]:
    for line in open(f):
        names.add(json.loads(json.loads(line)['reward_spec']['ground_truth_json'])['image_name'])
print('\n'.join(sorted(names)))" "$@"); do
  podman pull -q "docker.io/$img" >/dev/null
  alias="docker.io/swebench/${img#*/}"
  podman tag "docker.io/$img" "$alias"
  echo "$img -> $alias"
done
