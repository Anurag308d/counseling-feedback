"""
Score generations on the goal-setting eval set (part 3 of the brief).

    python make_goal_eval_dataset.py      # once
    modal run generate_eval_serverless.py --model sft_labeledsplit --dataset-name goal_eval_v0
    python evaluate_goal_eval.py exp/sft_labeledsplit_goal_eval_v0_eval.json

Metrics (95% CIs from a bootstrap over the 20 contexts):
  detection_balanced_acc  perfect flag vs expected_perfect, over all 130 rows
  perfect_rate_by_kind    share called perfect for strong / degraded / neg_appropriate / neg_mistimed
                          (strong and neg_appropriate should be high, the others low)
  overcorrection          neg_appropriate rows called NOT perfect (pushing for homework where it does not fit)
  missed_mistimed         neg_mistimed rows called perfect
  goal_setting_named      degraded and mistimed rows whose badareas include "Goal Setting"
                          (only possible for a model that knows the area)
  suggestions_penalty     strong rows with "Suggestions" in badareas: the "homework read as
                          unsolicited advice" hypothesis from the brief
  pairwise_acc            share of preferred-vs-rejected pairs where P(perfect=true) is higher for
                          the preferred reply, overall and per degradation type
  length_baseline_pairwise  same, if you just prefer the longer reply (the set's known length confound)
Diagnosis and alternative quality need an LLM judge and are scored separately.
"""
import argparse
import collections
import json
import os
import random

import numpy as np

from evaluate_baseline import parse_output

KINDS = ["strong", "degraded", "neg_appropriate", "neg_mistimed"]


def compute(rows, pairs):
    out = {"n": len(rows)}
    gold = np.array([r["expected_perfect"] for r in rows])
    pred = np.array([(r["pred"]["perfect"] if r["pred"] is not None else not r["expected_perfect"]) for r in rows])
    out["valid_rate"] = float(np.mean([r["pred"] is not None for r in rows]))
    correct = gold == pred
    out["detection_acc"] = float(correct.mean())
    out["detection_balanced_acc"] = float(np.mean([correct[gold == c].mean() for c in (True, False) if (gold == c).any()]))
    for kind in KINDS:
        sel = [p for r, p in zip(rows, pred) if r["kind"] == kind]
        if sel:
            out[f"perfect_rate_{kind}"] = float(np.mean(sel))
    if "perfect_rate_neg_appropriate" in out:
        out["overcorrection"] = 1 - out["perfect_rate_neg_appropriate"]
    if "perfect_rate_neg_mistimed" in out:
        out["missed_mistimed"] = out["perfect_rate_neg_mistimed"]

    flawed = [r for r in rows if r["kind"] in ("degraded", "neg_mistimed")]
    if flawed:
        out["goal_setting_named"] = float(np.mean([
            r["pred"] is not None and "Goal Setting" in (r["pred"].get("badareas") or []) for r in flawed]))
    strong = [r for r in rows if r["kind"] == "strong"]
    if strong:
        out["suggestions_penalty"] = float(np.mean([
            r["pred"] is not None and "Suggestions" in (r["pred"].get("badareas") or []) for r in strong]))

    by_id = {r["row_id"]: r for r in rows}
    scored, length_wins = [], []
    per_type = collections.defaultdict(list)
    for p in pairs:
        a, b = by_id.get(p["preferred_row"]), by_id.get(p["rejected_row"])
        if a is None or b is None or a["p_true"] is None or b["p_true"] is None:
            continue
        win = 1.0 if a["p_true"] > b["p_true"] else 0.5 if a["p_true"] == b["p_true"] else 0.0
        scored.append(win)
        per_type[p["degradation"]].append(win)
        la, lb = a["response_words"], b["response_words"]
        length_wins.append(1.0 if la > lb else 0.5 if la == lb else 0.0)
    if scored:
        out["pairwise_acc"] = float(np.mean(scored))
        out["length_baseline_pairwise"] = float(np.mean(length_wins))
        out["pairwise_by_degradation"] = {k: float(np.mean(v)) for k, v in sorted(per_type.items())}
    return out


CI_KEYS = ["detection_balanced_acc", "overcorrection", "missed_mistimed", "goal_setting_named",
           "suggestions_penalty", "pairwise_acc"]


def bootstrap(rows, pairs, n_boot, seed):
    by_ctx = collections.defaultdict(list)
    for r in rows:
        by_ctx[r["context_id"]].append(r)
    pairs_by_ctx = collections.defaultdict(list)
    for p in pairs:
        pairs_by_ctx[p["context_id"]].append(p)
    ctxs = list(by_ctx)
    rng = random.Random(seed)
    samples = collections.defaultdict(list)
    for _ in range(n_boot):
        draw = rng.choices(ctxs, k=len(ctxs))
        # Re-key duplicated contexts so pairs still find their own rows.
        rs, ps = [], []
        for i, c in enumerate(draw):
            for r in by_ctx[c]:
                rs.append(dict(r, row_id=f"{i}:{r['row_id']}"))
            for p in pairs_by_ctx[c]:
                ps.append(dict(p, preferred_row=f"{i}:{p['preferred_row']}", rejected_row=f"{i}:{p['rejected_row']}"))
        m = compute(rs, ps)
        for k in CI_KEYS:
            if m.get(k) is not None:
                samples[k].append(m[k])
    return {k: [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))] for k, v in samples.items()}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("generations")
    parser.add_argument("--eval_set", default="data/goal_eval_v0/eval_set_v0.jsonl")
    parser.add_argument("--pairs", default="data/goal_eval_v0/pairs_v0.jsonl")
    parser.add_argument("--n_boot", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    with open(args.eval_set) as f:
        gold = {r["row_id"]: r for r in map(json.loads, filter(str.strip, f))}
    with open(args.pairs) as f:
        pairs = [json.loads(line) for line in f if line.strip()]
    with open(args.generations) as f:
        gen = json.load(f)

    rows = []
    for res in gen["results"]:
        g = gold[res["example_id"]]
        rows.append({
            "row_id": g["row_id"], "context_id": g["context_id"], "kind": g["kind"],
            "degradation": g["degradation"], "expected_perfect": g["expected_perfect"],
            "response_words": g["response_words"], "flaw": g["flaw"],
            "pred": parse_output(res["raw_output"]), "p_true": res.get("p_true"),
            "raw_output": res["raw_output"],
        })

    metrics = compute(rows, pairs)
    if args.n_boot:
        metrics["ci95"] = bootstrap(rows, pairs, args.n_boot, args.seed)
    report = {"generations": args.generations, "meta": gen.get("meta", {}), "metrics": metrics}
    out = os.path.splitext(args.generations)[0] + "_goal_metrics.json"
    with open(out, "w") as f:
        json.dump(report, f, indent=2)
    with open(os.path.splitext(out)[0] + "_per_example.jsonl", "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")

    print(f"n={metrics['n']}")
    for k, v in metrics.items():
        if isinstance(v, float):
            ci = metrics.get("ci95", {}).get(k)
            print(f"  {k:30s} {v:.3f}" + (f"  [{ci[0]:.3f}, {ci[1]:.3f}]" if ci else ""))
    for k, v in metrics.get("pairwise_by_degradation", {}).items():
        print(f"  pairwise {k:21s} {v:.3f}")
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
