"""Run one SWE-smith sample end to end through SWESmithEnv (the class training uses).

Usage (servers running, training venv):
    AGENT_SERVER_URL=http://127.0.0.1:8081 EVAL_SERVER_URL=http://127.0.0.1:8082 \
    PYTHONPATH=.:SkyRL/skyrl-train SkyRL/skyrl-train/.venv/bin/python \
        scripts/swe_smith/smoke_env.py data/swe_smith/train_X.json 0

The advisor is a fixed string here (no trained advisor yet); the agent is the
Claude model AGENT_MODEL maps to, via the agent server.
"""
import json
import sys
import time

from omegaconf import OmegaConf

from advisor_models.swe_smith.env import SWESmithEnv

ADVICE = (
    "Reproduce the failure first with a minimal script or the relevant test, "
    "locate the faulty logic, make the smallest fix, re-run the tests, then submit."
)

row = [json.loads(line) for line in open(sys.argv[1])][int(sys.argv[2])]
extras = {k: v for k, v in row.items() if k != "prompt"}
env = SWESmithEnv(OmegaConf.create({}), extras)
print("instance:", json.loads(row["reward_spec"]["ground_truth_json"])["instance_id"])
print("agent model:", env.agent_model, flush=True)

t0 = time.time()
env.init(row["prompt"])
print("session:", env.session_id, flush=True)
for rl_step in range(1, 21):
    out = env.step(ADVICE)
    print(
        f"rl_step {rl_step}: done={out['done']} reward={out['reward']} "
        f"agent_steps={env.total_steps_taken} t={time.time() - t0:.0f}s",
        flush=True,
    )
    if out["done"]:
        md = out.get("metadata", {})
        for k in ("student_response", "other_info"):
            if md.get(k):
                print(f"--- {k} ---\n{str(md[k])[:1500]}")
        break
