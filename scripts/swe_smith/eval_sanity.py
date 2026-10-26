"""Gold patch must score 1.0 and an empty patch 0.0 through the eval server.

Usage: EVAL_SERVER_URL=http://127.0.0.1:8082 PYTHONPATH=. .venv/bin/python \
           scripts/swe_smith/eval_sanity.py data/swe_smith/train_X.json 0
"""
import json
import sys
import time

from advisor_models.swe_smith.config import compute_score

rows = [json.loads(line) for line in open(sys.argv[1])]
inst = json.loads(rows[int(sys.argv[2])]["reward_spec"]["ground_truth_json"])
print("instance:", inst["instance_id"])
for name, patch in [("gold", inst["patch"]), ("empty", "")]:
    t = time.time()
    print(name, "->", compute_score(patch, inst), f"{time.time() - t:.0f}s", flush=True)
