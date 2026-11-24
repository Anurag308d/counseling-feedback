# Results

Outputs of the runs in the final report. Copy them into `model_training/exp/` to rescore.

| Run | What it is |
| --- | --- |
| `sft_labeledsplit` | Llama 3.1-8B-Instruct, QLoRA SFT on the 320 train conversations (paper settings) |
| `base_instruct` | Llama 3.1-8B-Instruct, prompted, no training |
| `sft_mixed_x1` | as `sft_labeledsplit`, plus 208 goal-setting examples once (3.1% of training) |
| `sft_mixed_x3` | as `sft_labeledsplit`, plus the same 208 examples three times (8.8%) |

Files in `runs/`, per run:

- `<run>_feedback_qesconv_labeledsplit_eval.json`: raw greedy generations and P(perfect) on the 1,697 held-out prompts.
- `<run>_feedback_qesconv_labeledsplit_eval_metrics.json`: old-task metrics by subset (`original` = the 982 real turns, used for the headline numbers) with bootstrap CIs over conversations.
- `<run>_goal_eval_v0_eval.json`, `..._goal_metrics.json`, `..._goal_metrics_per_example.jsonl`: generations and metrics on the 130-row goal-setting eval (CIs over contexts).
- `*_judge_diagnosis.jsonl`, `*_judge_pairwise.jsonl`: LLM judge verdicts, each recording the judge model used.

Judge: openai/gpt-oss-120b on Groq. For `sft_mixed_x1` diagnosis, rows 74 to 110 were judged by openai/gpt-oss-20b after the daily limit. The judge was validated against an AI annotator (Claude), not human labels: `base_instruct_diagnosis_ai_annotator.csv`.

## Headline numbers

| Metric | sft_labeledsplit | base_instruct | sft_mixed_x1 | sft_mixed_x3 |
| --- | --- | --- | --- | --- |
| Old task: perfect-flag balanced acc | 0.770 | 0.500 | 0.777 | 0.745 |
| Old task: perfect-flag AUROC | 0.859 | 0.792 | 0.867 | 0.847 |
| Old task: weakness areas micro F1 | 0.471 | 0.320 | 0.495 | 0.439 |
| Goal eval: detection balanced acc | 0.523 | 0.500 | 0.584 | 0.559 |
| Goal eval: strong replies called perfect | 100% | 0% | 0% | 100% |
| Goal eval: degraded replies called perfect | 96.2% | 0% | 7.6% | 87.6% |
| Goal eval: names Goal Setting on flawed replies | 0% | 15.5% | 71.8% | 7.3% |
| Goal eval: judge says feedback names the flaw | 2.7% | 31.8% | 33.6% | 5.5% |
