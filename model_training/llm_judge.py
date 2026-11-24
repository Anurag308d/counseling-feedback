"""
LLM judge on Groq (OpenAI-compatible API), for the parts of the brief that need one.

Setup (once, on your laptop; never paste the key anywhere else):
    Windows cmd:   set GROQ_API_KEY=gsk_...
    PowerShell:    $env:GROQ_API_KEY="gsk_..."
    macOS/Linux:   export GROQ_API_KEY=gsk_...

Tasks:
  pairwise   Part 1. Model feedback vs gold feedback on a sample of held-out test examples where both
             say "not perfect". Each pair is judged in both orders to cancel position bias.
      python llm_judge.py pairwise exp/sft_labeledsplit_feedback_qesconv_labeledsplit_eval_metrics_per_example.jsonl --n 60
  diagnosis  Part 3. For each degraded or mistimed eval row the model flagged, does its feedback name
             the flaw that was introduced? Rows the model called perfect count as not diagnosed.
      python llm_judge.py diagnosis exp/sft_labeledsplit_goal_eval_v0_eval_goal_metrics_per_example.jsonl
  models     List the Groq models your key can use (pass one with --model).
  kappa      Agreement between the judge and your own labels (fill the "human" column of the
             *_handlabel.csv file that each task writes, with the same values the judge uses).
      python llm_judge.py kappa exp/..._judge_diagnosis_handlabel.csv

Every judge call is cached in <input>_judge_<task>.jsonl, so a run stopped by Groq's free-tier
rate limits can simply be restarted and picks up where it stopped.

The default judge is llama-3.3-70b-versatile, an open model and not a frontier one; say so in
the write-up, and report kappa against your hand labels before trusting it.
"""
import argparse
import csv
import json
import os
import random
import re
import sys
import time
import urllib.error
import urllib.request

API_URL = "https://api.groq.com/openai/v1/chat/completions"
DEFAULT_MODEL = "llama-3.3-70b-versatile"
RESPONSE_MARKER = "### Response:"

PAIRWISE_SYSTEM = """You are an expert counseling supervisor. You compare two pieces of written feedback on a peer counselor's (Helper's) last response in an emotional-support conversation.
Good feedback correctly identifies what the response does poorly, explains how to improve it in a way that fits the conversation, and suggests a better alternative response.
Answer with JSON only: {"reason": "<one sentence>", "winner": "A" | "B" | "tie"}"""

DIAGNOSIS_SYSTEM = """You are an expert in behavioral activation and counseling supervision. A Helper's reply was deliberately written with one specific flaw. You are shown the flaw and a model's feedback on the reply.
Decide whether the feedback identifies that specific flaw (the same problem, in any wording). Feedback that only criticises other things, or is too generic to point at this flaw, does not count.
Answer with JSON only: {"reason": "<one sentence>", "names_flaw": true | false}"""


MODELS_URL = "https://api.groq.com/openai/v1/models"
# Tried in order when the requested model is not available on the account.
FALLBACK_MODELS = ["llama-3.3-70b-versatile", "openai/gpt-oss-120b", "moonshotai/kimi-k2-instruct",
                   "meta-llama/llama-4-maverick-17b-128e-instruct", "qwen/qwen3-32b"]
_active = {}


def _headers():
    key = os.environ.get("GROQ_API_KEY")
    if not key:
        sys.exit("Set GROQ_API_KEY in this terminal first (see the top of llm_judge.py).")
    return {"Authorization": f"Bearer {key}", "Content-Type": "application/json",
            "User-Agent": "counseling-feedback-judge"}


def available_models():
    req = urllib.request.Request(MODELS_URL, headers=_headers())
    with urllib.request.urlopen(req, timeout=30) as resp:
        return sorted(m["id"] for m in json.load(resp)["data"])


def pick_fallback(requested):
    models = available_models()
    for m in FALLBACK_MODELS:
        if m in models and m != requested:
            print(f"  Model {requested} is not available on this account; switching to {m}.")
            return m
    sys.exit(f"Model {requested} is not available. Models on this account: {', '.join(models)}\n"
             "Rerun with --model <one of these> (a large chat model is best).")


def parse_json_text(text):
    start, end = text.find("{"), text.rfind("}")
    return json.loads(text[start:end + 1])


CALL_DELAY = float(os.environ.get("JUDGE_DELAY_SECONDS", "2"))  # pause between calls, for free-tier limits


def call_groq(messages, model, max_retries=8):
    time.sleep(CALL_DELAY)
    model = _active.get(model, model)
    use_json_mode = True
    attempt = 0
    while attempt < max_retries:
        payload = {"model": model, "messages": messages, "temperature": 0, "max_tokens": 400}
        if use_json_mode:
            payload["response_format"] = {"type": "json_object"}
        req = urllib.request.Request(API_URL, data=json.dumps(payload).encode(), headers=_headers())
        try:
            with urllib.request.urlopen(req, timeout=60) as resp:
                data = json.load(resp)
            return parse_json_text(data["choices"][0]["message"]["content"])
        except urllib.error.HTTPError as e:
            msg = e.read().decode(errors="ignore")[:300]
            if e.code == 404 or "model_not_found" in msg or "decommissioned" in msg:
                new_model = pick_fallback(model)
                for k in list(_active) + [model]:
                    _active[k] = new_model
                model = new_model
                continue
            if e.code == 400 and use_json_mode and "response_format" in msg:
                use_json_mode = False
                continue
            if e.code == 429 or e.code >= 500:
                if "per day" in msg.lower() or "tpd" in msg.lower() or "rpd" in msg.lower():
                    sys.exit("Daily Groq limit reached for this model. Rerun later (finished calls are kept), "
                             "or pass --model with another model from: python llm_judge.py models")
                wait = float(e.headers.get("retry-after") or 0) or min(60, 5 * 2 ** attempt)
                print(f"  Groq {e.code}, waiting {wait:.0f}s: {msg[:150]}")
                time.sleep(wait)
                attempt += 1
                continue
            sys.exit(f"Groq error {e.code}: {msg}")
        except (urllib.error.URLError, ConnectionError, TimeoutError, OSError) as e:
            # Dropped connections (e.g. ConnectionResetError on a flaky network): back off and retry.
            wait = min(60, 5 * 2 ** attempt)
            print(f"  Network error ({e}); retrying in {wait}s")
            time.sleep(wait)
            attempt += 1
        except (json.JSONDecodeError, KeyError, ValueError):
            attempt += 1
            time.sleep(2)
    sys.exit("Groq call kept failing. Rerun the same command later; finished items are kept.")


def judge_model(requested):
    return _active.get(requested, requested)


def load_cache(path):
    cache = {}
    if os.path.exists(path):
        with open(path) as f:
            for line in f:
                if line.strip():
                    rec = json.loads(line)
                    cache[rec["key"]] = rec
    return cache


def conversation_of(prompt):
    return prompt.split("### Input:\n", 1)[1].split(RESPONSE_MARKER, 1)[0].strip()


def run_pairwise(args):
    with open(args.per_example) as f:
        rows = [json.loads(line) for line in f if line.strip()]
    gold_test = {}
    with open(args.gold) as f:
        for g in json.load(f):
            gold_test[g["example_id"]] = g
    pool = [r for r in rows if r.get("source", "original") == "original" and not r["gold"]["perfect"]
            and r["pred"] is not None and r["pred"]["perfect"] is False]
    random.Random(args.seed).shuffle(pool)
    pool = pool[:args.n]
    out_path = os.path.splitext(args.per_example)[0] + "_judge_pairwise.jsonl"
    cache = load_cache(out_path)
    with open(out_path, "a") as out:
        for i, r in enumerate(pool):
            conv = conversation_of(gold_test[r["example_id"]]["text"])
            model_fb = {"feedback": r["pred"]["feedback"], "badareas": r["pred"]["badareas"],
                        "alternative": r["pred"]["alternative"]}
            gold_fb = {"feedback": r["gold"]["feedback"], "badareas": r["gold"]["badareas"],
                       "alternative": r["gold"]["alternative"]}
            for order in ("model_first", "gold_first"):
                key = f"{r['example_id']}:{order}"
                if key in cache:
                    continue
                a, b = (model_fb, gold_fb) if order == "model_first" else (gold_fb, model_fb)
                user = (f"Conversation:\n{conv}\n\nFeedback A:\n{json.dumps(a)}\n\nFeedback B:\n{json.dumps(b)}\n\n"
                        "Which feedback is better?")
                verdict = call_groq([{"role": "system", "content": PAIRWISE_SYSTEM},
                                     {"role": "user", "content": user}], args.model)
                w = str(verdict.get("winner", "tie")).strip().upper()
                model_slot = "A" if order == "model_first" else "B"
                outcome = "tie" if w not in ("A", "B") else ("model" if w == model_slot else "gold")
                rec = {"key": key, "example_id": r["example_id"], "order": order, "outcome": outcome,
                       "reason": verdict.get("reason", ""), "judge": judge_model(args.model)}
                out.write(json.dumps(rec) + "\n")
                out.flush()
                cache[key] = rec
            print(f"  {i + 1}/{len(pool)} judged")
    recs = [cache[k] for k in cache if k.split(":")[0] in {str(r["example_id"]) for r in pool}]
    n = len(recs)
    if n:
        for o in ("model", "gold", "tie"):
            print(f"{o:6s} wins: {sum(x['outcome'] == o for x in recs) / n:.3f}")
        print(f"({n} judgments on {len(pool)} examples, both orders; judge {judge_model(args.model)})")
    write_handlabel(out_path, recs, ["outcome"], args.seed)


def run_diagnosis(args):
    with open(args.per_example) as f:
        rows = [json.loads(line) for line in f if line.strip()]
    with open(args.eval_set) as f:
        evalset = {r["row_id"]: r for r in map(json.loads, filter(str.strip, f))}
    targets = [r for r in rows if r["kind"] in ("degraded", "neg_mistimed")]
    out_path = os.path.splitext(args.per_example)[0] + "_judge_diagnosis.jsonl"
    cache = load_cache(out_path)
    with open(out_path, "a") as out:
        for i, r in enumerate(targets):
            key = r["row_id"]
            if key in cache:
                continue
            if r["pred"] is None or r["pred"]["perfect"] is not False:
                rec = {"key": key, "names_flaw": False, "reason": "model called it perfect or output invalid",
                       "judged": False}
            else:
                e = evalset[key]
                user = (f"Conversation:\n{conversation_of(e['prompt'])}\n\nIntroduced flaw:\n{e['flaw']}\n\n"
                        f"Model feedback:\n{r['pred'].get('feedback', '')}\n"
                        f"Areas the model marked as needing improvement: {r['pred'].get('badareas', [])}\n\n"
                        "Does the feedback identify the introduced flaw?")
                verdict = call_groq([{"role": "system", "content": DIAGNOSIS_SYSTEM},
                                     {"role": "user", "content": user}], args.model)
                rec = {"key": key, "names_flaw": bool(verdict.get("names_flaw")),
                       "reason": verdict.get("reason", ""), "judged": True}
            rec.update({"degradation": r["degradation"], "kind": r["kind"], "judge": judge_model(args.model) if rec.get("judged") else None})
            out.write(json.dumps(rec) + "\n")
            out.flush()
            cache[key] = rec
            print(f"  {i + 1}/{len(targets)} {key}: {rec['names_flaw']}")
    recs = [cache[r["row_id"]] for r in targets if r["row_id"] in cache]
    by_type = {}
    for rec in recs:
        by_type.setdefault(rec["degradation"], []).append(rec["names_flaw"])
    print(f"\nDiagnosis rate overall: {sum(x['names_flaw'] for x in recs) / len(recs):.3f} (n={len(recs)})")
    for k, v in sorted(by_type.items()):
        print(f"  {k:16s} {sum(v) / len(v):.3f} (n={len(v)})")
    write_handlabel(out_path, [x for x in recs if x.get("judged")], ["names_flaw"], args.seed)


def write_handlabel(out_path, recs, fields, seed, n=30):
    sample = list(recs)
    random.Random(seed).shuffle(sample)
    path = os.path.splitext(out_path)[0] + "_handlabel.csv"
    if os.path.exists(path):
        return
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["key", "judge", "human", "judge_reason"])
        for rec in sample[:n]:
            w.writerow([rec["key"], rec[fields[0]], "", rec.get("reason", "")])
    print(f"Wrote {path}: fill the 'human' column, then run: python llm_judge.py kappa {path}")


def run_kappa(args):
    from sklearn.metrics import cohen_kappa_score
    with open(args.csv, newline="", encoding="utf-8") as f:
        rows = [r for r in csv.DictReader(f) if r["human"].strip()]
    norm = lambda v: re.sub(r"\s+", "", str(v)).lower()
    judge = [norm(r["judge"]) for r in rows]
    human = [norm(r["human"]) for r in rows]
    agree = sum(a == b for a, b in zip(judge, human)) / len(rows)
    print(f"n={len(rows)} agreement={agree:.3f} cohen_kappa={cohen_kappa_score(judge, human):.3f}")


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="task", required=True)
    p = sub.add_parser("pairwise")
    p.add_argument("per_example")
    p.add_argument("--gold", default="data/feedback_qesconv_labeledsplit/test.json")
    p.add_argument("--n", type=int, default=60)
    d = sub.add_parser("diagnosis")
    d.add_argument("per_example")
    d.add_argument("--eval_set", default="data/goal_eval_v0/eval_set_v0.jsonl")
    for s in (p, d):
        s.add_argument("--model", default=DEFAULT_MODEL)
        s.add_argument("--seed", type=int, default=42)
    sub.add_parser("models", help="list the Groq models your key can use")
    k = sub.add_parser("kappa")
    k.add_argument("csv")
    args = parser.parse_args()
    if args.task == "models":
        print("\n".join(available_models()))
        return
    {"pairwise": run_pairwise, "diagnosis": run_diagnosis, "kappa": run_kappa}[args.task](args)


if __name__ == "__main__":
    main()
