"""
Score model generations on the held-out FeedbackESConv test split.

    pip install sacrebleu rouge-score scikit-learn numpy        # bert-score too, for --bertscore
    python evaluate_baseline.py exp/sft_labeledsplit_feedback_qesconv_labeledsplit_eval.json --bertscore

Input: the JSON written by generate_eval_serverless.py. Gold labels come from
data/feedback_qesconv_labeledsplit/test.json, matched on example_id.

Metrics, reported separately for each subset of the test split:
  original          real Helper turns (headline numbers)
  gold_alternative  Helper turns replaced by the gold alternative, all labeled perfect
  all               both

  valid_rate            output parses as JSON with the required keys
  perfect_accuracy      "perfect" flag vs gold (invalid outputs count as wrong)
  perfect_balanced_acc  mean of recall on perfect and on not-perfect
  perfect_auroc         from P("true") on the first generated token
  goodareas_* / badareas_*  set-based micro P/R/F1 and macro F1 over the 8 areas
                        (badareas only on examples whose gold is not perfect)
  feedback_* / alternative_*  BLEU (sacrebleu corpus), ROUGE-L F1, BERTScore F1, on examples
                        where gold and prediction are both not perfect
95% confidence intervals come from a bootstrap over test conversations, since examples from
one conversation are correlated.
"""
import argparse
import collections
import json
import os
import random

import numpy as np

AREAS = ["Reflections", "Questions", "Suggestions", "Validation", "Self-disclosure",
         "Empathy", "Professionalism", "Structure"]
RESPONSE_MARKER = "### Response:"
CI_METRICS = ["valid_rate", "perfect_accuracy", "perfect_balanced_acc", "perfect_auroc",
              "goodareas_micro_f1", "goodareas_macro_f1", "badareas_micro_f1", "badareas_macro_f1",
              "feedback_bleu", "feedback_rougeL", "feedback_bertscore_f1",
              "alternative_bleu", "alternative_rougeL", "alternative_bertscore_f1"]


def parse_output(raw):
    """Return the first JSON object in raw, or None if there is none or keys are missing."""
    start = raw.find("{")
    if start < 0:
        return None
    try:
        obj, _ = json.JSONDecoder().raw_decode(raw[start:])
    except json.JSONDecodeError:
        return None
    if not isinstance(obj, dict) or not isinstance(obj.get("perfect"), bool):
        return None
    if not isinstance(obj.get("goodareas", []), list):
        return None
    if obj["perfect"] is False:
        if not all(k in obj for k in ("feedback", "badareas", "alternative")):
            return None
        if not isinstance(obj["badareas"], list):
            return None
    return obj


def set_prf(pairs):
    """pairs: list of (gold_set, pred_set). Returns micro P/R/F1, macro F1 and per-area F1."""
    tp = fp = fn = 0
    per = {a: [0, 0, 0] for a in AREAS}
    for g, p in pairs:
        tp += len(g & p)
        fp += len(p - g)
        fn += len(g - p)
        for a in AREAS:
            per[a][0] += (a in g and a in p)
            per[a][1] += (a in p and a not in g)
            per[a][2] += (a in g and a not in p)

    def f1(t, f_p, f_n):
        prec = t / (t + f_p) if t + f_p else 0.0
        rec = t / (t + f_n) if t + f_n else 0.0
        return prec, rec, (2 * prec * rec / (prec + rec) if prec + rec else 0.0)

    p, r, f = f1(tp, fp, fn)
    per_f1 = {a: f1(*per[a])[2] for a in AREAS}
    supported = [a for a in AREAS if per[a][0] + per[a][2] > 0]
    macro = float(np.mean([per_f1[a] for a in supported])) if supported else 0.0
    return {"micro_p": p, "micro_r": r, "micro_f1": f, "macro_f1": macro, "per_area_f1": per_f1}


def bleu(hyps, refs):
    import sacrebleu
    if not hyps:
        return None
    return sacrebleu.corpus_bleu(hyps, [refs]).score


def compute(rows, with_detail=False):
    """rows: scored example dicts (see score_rows). Returns a flat metric dict."""
    from sklearn.metrics import roc_auc_score

    n = len(rows)
    out = {"n": n}
    if n == 0:
        return out
    out["valid_rate"] = float(np.mean([r["pred"] is not None for r in rows]))

    gold_perfect = np.array([r["gold"]["perfect"] for r in rows])
    # An invalid output gets the wrong label, so it is never rewarded.
    pred_perfect = np.array([(r["pred"]["perfect"] if r["pred"] is not None else not r["gold"]["perfect"])
                             for r in rows])
    correct = gold_perfect == pred_perfect
    out["perfect_accuracy"] = float(correct.mean())
    recalls = [float(correct[gold_perfect == c].mean()) for c in (True, False) if (gold_perfect == c).any()]
    out["perfect_balanced_acc"] = float(np.mean(recalls))
    out["gold_perfect_rate"] = float(gold_perfect.mean())
    out["pred_perfect_rate"] = float(pred_perfect.mean())
    scored = [(r["gold"]["perfect"], r["p_true"]) for r in rows if r["p_true"] is not None]
    if scored and len({g for g, _ in scored}) == 2:
        out["perfect_auroc"] = float(roc_auc_score([g for g, _ in scored], [p for _, p in scored]))

    good = set_prf([(set(r["gold"].get("goodareas", [])),
                     set(r["pred"].get("goodareas", [])) if r["pred"] else set()) for r in rows])
    for k, v in good.items():
        if k != "per_area_f1" or with_detail:
            out[f"goodareas_{k}"] = v

    flawed = [r for r in rows if not r["gold"]["perfect"]]
    out["n_gold_not_perfect"] = len(flawed)
    if flawed:
        bad = set_prf([(set(r["gold"].get("badareas", [])),
                        set(r["pred"].get("badareas", [])) if r["pred"] else set()) for r in flawed])
        for k, v in bad.items():
            if k != "per_area_f1" or with_detail:
                out[f"badareas_{k}"] = v

    both = [r for r in flawed if r["pred"] is not None and r["pred"]["perfect"] is False]
    out["n_text_pairs"] = len(both)
    for field in ("feedback", "alternative"):
        if not both:
            continue
        out[f"{field}_bleu"] = bleu([str(r["pred"][field]) for r in both], [str(r["gold"][field]) for r in both])
        out[f"{field}_rougeL"] = float(np.mean([r[f"{field}_rougeL"] for r in both]))
        if both[0].get(f"{field}_bertscore_f1") is not None:
            out[f"{field}_bertscore_f1"] = float(np.mean([r[f"{field}_bertscore_f1"] for r in both]))
    return out


def score_rows(rows, use_bertscore, bertscore_model):
    """Attach per-example ROUGE-L (and BERTScore) so the bootstrap only has to average them."""
    from rouge_score import rouge_scorer
    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    pairs = [r for r in rows if not r["gold"]["perfect"] and r["pred"] is not None and r["pred"]["perfect"] is False]
    for field in ("feedback", "alternative"):
        for r in pairs:
            r[f"{field}_rougeL"] = scorer.score(str(r["gold"][field]), str(r["pred"][field]))["rougeL"].fmeasure
        if use_bertscore and pairs:
            from bert_score import score as bert_score
            _, _, f1 = bert_score([str(r["pred"][field]) for r in pairs], [str(r["gold"][field]) for r in pairs],
                                  model_type=bertscore_model, lang="en", verbose=False, batch_size=32)
            for r, f in zip(pairs, f1.tolist()):
                r[f"{field}_bertscore_f1"] = f


def bootstrap(rows, n_boot, seed):
    by_conv = collections.defaultdict(list)
    for r in rows:
        by_conv[r["conv_index"]].append(r)
    convs = list(by_conv)
    rng = random.Random(seed)
    samples = collections.defaultdict(list)
    for _ in range(n_boot):
        resampled = [r for c in rng.choices(convs, k=len(convs)) for r in by_conv[c]]
        m = compute(resampled)
        for k in CI_METRICS:
            if m.get(k) is not None:
                samples[k].append(m[k])
    return {k: [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))] for k, v in samples.items()}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("generations")
    parser.add_argument("--gold", default="data/feedback_qesconv_labeledsplit/test.json")
    parser.add_argument("--bertscore", action="store_true", help="also compute BERTScore (needs bert-score)")
    parser.add_argument("--bertscore_model", default="roberta-large")
    parser.add_argument("--n_boot", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", default=None, help="report path (default: <generations>_metrics.json)")
    args = parser.parse_args()

    with open(args.generations) as f:
        gen = json.load(f)
    with open(args.gold) as f:
        gold = json.load(f)
    gold_by_id = {g["example_id"]: g for g in gold}

    rows = []
    for res in gen["results"]:
        g = gold_by_id[res["example_id"]]
        rows.append({
            "example_id": res["example_id"],
            "conv_index": res["conv_index"],
            "source": g["source"],
            "gold": json.loads(g["text"].split(RESPONSE_MARKER, 1)[1]),
            "pred": parse_output(res["raw_output"]),
            "p_true": res.get("p_true"),
            "raw_output": res["raw_output"],
        })
    if len(rows) < len(gold):
        print(f"Note: {len(rows)} of {len(gold)} test examples have generations")

    score_rows(rows, args.bertscore, args.bertscore_model)

    report = {"generations": args.generations, "meta": gen.get("meta", {}), "subsets": {}}
    for name, subset in (("original", [r for r in rows if r["source"] == "original"]),
                         ("gold_alternative", [r for r in rows if r["source"] == "gold_alternative"]),
                         ("all", rows)):
        if not subset:
            continue
        metrics = compute(subset, with_detail=True)
        if args.n_boot:
            metrics["ci95"] = bootstrap(subset, args.n_boot, args.seed)
        report["subsets"][name] = metrics

    out = args.out or os.path.splitext(args.generations)[0] + "_metrics.json"
    with open(out, "w") as f:
        json.dump(report, f, indent=2)
    with open(os.path.splitext(out)[0] + "_per_example.jsonl", "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")

    for name, m in report["subsets"].items():
        print(f"\n== {name} (n={m['n']}, gold not perfect={m['n_gold_not_perfect']}, text pairs={m['n_text_pairs']})")
        for k in CI_METRICS:
            if m.get(k) is None:
                continue
            ci = m.get("ci95", {}).get(k)
            ci_s = f"  [{ci[0]:.3f}, {ci[1]:.3f}]" if ci else ""
            print(f"  {k:28s} {m[k]:.3f}{ci_s}")
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
