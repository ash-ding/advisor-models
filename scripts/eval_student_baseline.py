"""Student-only baseline (no advisor) on the RuleArena Taxes and MTOB test sets.

The upstream eval scripts only score advisor pipelines. This reproduces the
"student alone" baseline from the paper with upstream's own pieces:

- rule_arena: the 25 test problems `rule_arena/construct_dataset.py` selects
  (seed 42, Level 0, 75/25), answered exactly as its `get_initial_response`
  does (single user turn, no system prompt, default temperature) and scored by
  `rule_arena.config.compute_score`.
- mtob: `data/mtob/validation.parquet` from `mtob/construct_dataset.py`,
  prompted with `BASELINE_PROMPT_TEMPLATE` (the student prompt minus advice),
  temperature 0 as in `eval_mtob.py`, scored with chrF.

Both run the test set `--num_runs` times and report the same multi-run CI as
the eval scripts. Each module's `config` is imported script-style, so run one
domain per process.

Usage:
    PYTHONPATH=. .venv/bin/python scripts/eval_student_baseline.py rule_arena \
        --student_model claude-haiku-4-5
    PYTHONPATH=. .venv/bin/python scripts/eval_student_baseline.py mtob \
        --student_model claude-haiku-4-5 --dataset_path data/mtob/validation.parquet
"""

import argparse
import json
import random
import sys
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from utils.eval_utils import compute_multi_run_statistics, format_ci_string
from utils.llm import chat_completion, resolve_model

REPO_ROOT = Path(__file__).resolve().parents[1]


def rule_arena_examples(args):
    domain_dir = REPO_ROOT / "advisor_models" / "rule_arena"
    sys.path.insert(0, str(domain_dir))
    import construct_dataset as ra  # noqa: E402  (script-style imports inside)
    from config import build_prompt, compute_score  # noqa: E402

    # Same call order as construct_dataset.main(), so the same 25 problems.
    random.seed(42)
    data_path = domain_dir / "RuleArena" / "tax" / "synthesized_problems" / "comp_0.json"
    problems = ra.get_problems_from_data_path(str(data_path), args.train_size + args.val_size)
    random.shuffle(problems)
    test = problems[args.train_size : args.train_size + args.val_size]

    examples = [
        {
            "messages": [{"role": "user", "content": build_prompt(p.get("info_dict", {}))}],
            "ground_truth": p["ground_truth"],
        }
        for p in test
    ]
    return examples, {}, lambda resp, gt: compute_score(resp, gt)[0]


def mtob_examples(args):
    import pandas as pd

    sys.path.insert(0, str(REPO_ROOT / "advisor_models" / "mtob"))
    from config import (  # noqa: E402
        BASELINE_PROMPT_TEMPLATE,
        compute_translation_score,
        extract_translation,
    )

    df = pd.read_parquet(args.dataset_path)
    examples = []
    for _, row in df.iterrows():
        prompt = BASELINE_PROMPT_TEMPLATE.format(
            original_question=row["original_question"],
            reference_materials="\n".join(row["reference_materials"]),
        )
        examples.append(
            {
                "messages": [{"role": "user", "content": prompt}],
                "ground_truth": row["reward_spec"]["ground_truth"],
            }
        )
    # eval_mtob.py: temperature 0, except 1.0 for gpt-5 students.
    temperature = 1.0 if "gpt-5" in args.student_model else 0.0

    def score(resp, gt):
        return compute_translation_score(extract_translation(resp), gt)

    return examples, {"temperature": temperature}, score


def run_once(examples, model, call_kwargs, score_fn, max_workers):
    def one(ex):
        try:
            r = chat_completion(model, ex["messages"], **call_kwargs)
            text = r.choices[0].message.content or ""
            finish = r.choices[0].finish_reason
        except Exception as e:  # counted, scored 0 like the upstream evals
            text, finish = "", f"error: {type(e).__name__}: {e}"[:200]
        return {
            "response": text,
            "finish_reason": finish,
            "ground_truth": ex["ground_truth"],
            "score": float(score_fn(text, ex["ground_truth"])),
        }

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        return list(pool.map(one, examples))


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("domain", choices=["rule_arena", "mtob"])
    parser.add_argument("--student_model", default="claude-haiku-4-5")
    parser.add_argument("--num_runs", type=int, default=5)
    parser.add_argument("--max_workers", type=int, default=12)
    parser.add_argument("--train_size", type=int, default=75, help="rule_arena split")
    parser.add_argument("--val_size", type=int, default=25, help="rule_arena split")
    parser.add_argument("--dataset_path", default="data/mtob/validation.parquet")
    parser.add_argument("--output_dir", default="results/baselines")
    args = parser.parse_args()

    build = rule_arena_examples if args.domain == "rule_arena" else mtob_examples
    examples, call_kwargs, score_fn = build(args)
    print(
        f"{args.domain}: {len(examples)} test examples, student {args.student_model} "
        f"-> {resolve_model(args.student_model)}, kwargs {call_kwargs}, {args.num_runs} runs",
        flush=True,
    )

    runs = []
    for i in range(args.num_runs):
        t = time.time()
        results = run_once(examples, args.student_model, call_kwargs, score_fn, args.max_workers)
        scores = [r["score"] for r in results]
        finishes = Counter(str(r["finish_reason"]).split(":")[0] for r in results)
        print(
            f"run {i + 1}: mean {sum(scores) / len(scores):.4f}  "
            f"finish_reasons {dict(finishes)}  {time.time() - t:.0f}s",
            flush=True,
        )
        runs.append(results)

    stats = compute_multi_run_statistics([[r["score"] for r in run] for run in runs])
    metric = "Accuracy" if args.domain == "rule_arena" else "chrF (0-1)"
    print(format_ci_string(stats, metric))

    out_dir = REPO_ROOT / args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"{args.domain}_{args.student_model}.json"
    out.write_text(
        json.dumps({"args": vars(args), "call_kwargs": call_kwargs, "stats": stats, "runs": runs}, indent=1)
    )
    print(f"saved {out}")


if __name__ == "__main__":
    main()
