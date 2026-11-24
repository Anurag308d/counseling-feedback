"""
Turn the goal-setting eval set (data/goal_eval_v0/eval_set_v0.jsonl, built in the project's
goal-setting-eval folder) into the prompt file that generate_eval_serverless.py reads.

    python make_goal_eval_dataset.py
    modal run generate_eval_serverless.py --model sft_labeledsplit --dataset-name goal_eval_v0

Each prompt keeps its row_id (as example_id) and kind (as source); conv_index is the context
number, so bootstrap resampling in the scorer is by context.
"""
import argparse
import json
import os


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", default="data/goal_eval_v0/eval_set_v0.jsonl")
    parser.add_argument("--out", default="data/goal_eval_v0/test_prompts.json")
    args = parser.parse_args()

    with open(args.src) as f:
        rows = [json.loads(line) for line in f if line.strip()]
    contexts = sorted({r["context_id"] for r in rows})
    ctx_index = {c: i for i, c in enumerate(contexts)}
    prompts = []
    for r in rows:
        assert r["prompt"].endswith("### Response:"), r["row_id"]
        prompts.append({
            "text": r["prompt"],
            "example_id": r["row_id"],
            "source": r["kind"],
            "conv_index": ctx_index[r["context_id"]],
            "helper_index": 0,
        })
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(prompts, f)
    print(f"Wrote {len(prompts)} prompts from {len(contexts)} contexts to {args.out}")


if __name__ == "__main__":
    main()
