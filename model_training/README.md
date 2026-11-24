# Model training & sampling

## Held-out baseline on Modal (Llama 3.1-8B-Instruct)

The paper trained on all labeled data, so it has no clean test set. These steps build one.
Run everything from `model_training/`. You need the Modal CLI (`pip install modal`, `modal setup`),
access to `meta-llama/Meta-Llama-3.1-8B-Instruct` on Hugging Face, and a Modal secret named
`anurag-bridge-huggingface` holding `HF_TOKEN`:
`modal secret create anurag-bridge-huggingface HF_TOKEN=hf_...`
The shared salt-nlp workspace has other people's apps, so all app, volume and secret names start
with `anurag-bridge` (set `MODAL_NAME_PREFIX` to change it).

1. Split by conversation (already committed; rerun only to change the split):
   `python make_labeled_split.py`
   writes `data/feedback_qesconv_labeledsplit/` with 320 train and 80 test conversations
   (6,482 and 1,697 examples). Each test example is tagged `original` (a real Helper turn, 982)
   or `gold_alternative` (the turn replaced by the gold alternative and labeled perfect, 715).
2. Smoke test, about 15 minutes and under $1:
   `modal run finetune_llama_serverless.py --run-name sft_smoke --max-train-samples 200 --max-steps 20`
   `modal run generate_eval_serverless.py --model sft_smoke --max-prompts 20`
3. Full SFT on the train split with the paper's settings (about 1 to 1.5 hours on an H100):
   `modal run --detach finetune_llama_serverless.py --run-name sft_labeledsplit`
4. One greedy generation per test prompt, plus P("perfect" = true):
   `modal run generate_eval_serverless.py --model sft_labeledsplit`
   `modal run generate_eval_serverless.py --model meta-llama/Meta-Llama-3.1-8B-Instruct --prompt-style instruct --tag base_instruct`
5. Score locally (`pip install sacrebleu rouge-score scikit-learn numpy bert-score`):
   `python evaluate_baseline.py exp/sft_labeledsplit_feedback_qesconv_labeledsplit_eval.json --bertscore`
   Headline numbers are the `original` subset.

## Original paper commands

### Fine-tuning the model

```
poetry run accelerate launch finetune_llama.py \
--output_dir="<SFT PATH>" \
--max_steps=766 \
--logging_steps=10 \
--save_steps=100 \
--per_device_train_batch_size=1 \
--gradient_accumulation_steps=32 \
--gradient_checkpointing=False \
--group_by_length=False \
--learning_rate=3e-4 \
--lr_scheduler_type="cosine" \
--warmup_steps=100 \
--weight_decay=0.01 \
--optim="paged_adamw_32bit" \
--bf16=True \
--remove_unused_columns=False \
--run_name="sft_model" \
--report_to="wandb"
```

#### Generating 10 samples from the model and scoring

```
poetry run accelerate launch generate_llama_exp.py \
--output_dir="<SFT PATH>" \
--bf16=True \
--dataset_name="feedback_qesconv_dpo_pre" \
--start_index=0 '

poetry run accelerate launch generate_llama_exp_probs.py \
--output_dir="<SFT PATH>" \
--bf16=True \
--file_name="exp/feedback_qesconv_dpo_pre_sft_model_generations_0.json"

```


### DPO alignment

```
poetry run accelerate launch dpo_llama.py  \
--model_name_or_path="<SFT PATH/final_merged_checkpoint>"
--output_dir="<DPO PATH>" \

```


### Ablations

##### + new data training

```
poetry run accelerate launch generate_llama.py \
--output_dir="<SFT PATH>" \
--bf16=True \
--dataset_name="feedback_qesconv_dpo_pre" 
```

```
poetry run accelerate launch finetune_llama.py \
--model_name="<SFT PATH>" \
--output_dir="<NEW DATA PATH>" \
--dataset_name="feedback_ablation_data" \
--max_steps=200 \
--logging_steps=10 \
--save_steps=100 \
--per_device_train_batch_size=1 \
--gradient_accumulation_steps=32 \
--gradient_checkpointing=False \
--group_by_length=False \
--learning_rate=5e-6 \
--lr_scheduler_type="cosine" \
--warmup_steps=10 \
--weight_decay=0.05 \
--optim="paged_adamw_32bit" \
--bf16=True \
--remove_unused_columns=False \
--run_name="ablation_new_data" \
--report_to="wandb"
```

##### + best scores training

```
poetry run accelerate launch finetune_llama.py \
--model_name="<SFT PATH>" \
--output_dir="<BEST SCORES PATH>" \
--dataset_name="feedback_ablation_preference" \
--max_steps=200 \
--logging_steps=10 \
--save_steps=100 \
--per_device_train_batch_size=1 \
--gradient_accumulation_steps=32 \
--gradient_checkpointing=False \
--group_by_length=False \
--learning_rate=5e-6 \
--lr_scheduler_type="cosine" \
--warmup_steps=10 \
--weight_decay=0.05 \
--optim="paged_adamw_32bit" \
--bf16=True \
--remove_unused_columns=False \
--run_name="ablation_best_scores" \
--report_to="wandb"
```
##### sampling and scoring with SFT
```

poetry run accelerate launch generate_llama_exp.py \
--output_dir="<SFT PATH>" \
--bf16=True \
--dataset_name="feedback_qesconv" 

poetry run accelerate launch generate_llama_exp.py \
--output_dir="<NEW DATA PATH>" \
--bf16=True \
--dataset_name="feedback_qesconv" 

poetry run accelerate launch generate_llama_exp.py \
--output_dir="<BEST SCORES PATH>" \
--bf16=True \
--dataset_name="feedback_qesconv" 

poetry run accelerate launch generate_llama_exp.py \
--output_dir="<DPO PATH>" \
--bf16=True \
--dataset_name="feedback_qesconv" 

```

```

poetry run accelerate launch generate_llama_exp_probs.py \
--output_dir="<SFT PATH>" \
--bf16=True \
--file_name="exp/feedback_qesconv_sft_model_generations_0.json"

poetry run accelerate launch generate_llama_exp_probs.py \
--output_dir="<SFT PATH>" \
--bf16=True \
--file_name="exp/feedback_qesconv_dpo_model_generations_0.json"

poetry run accelerate launch generate_llama_exp_probs.py \
--output_dir="<SFT PATH>" \
--bf16=True \
--file_name="exp/feedback_qesconv_ablation_preference_model_generations_0.json"

poetry run accelerate launch generate_llama_exp_probs.py \
--output_dir="<SFT PATH>" \
--bf16=True \
--file_name="exp/feedback_qesconv_ablation_data_model_generations_0.json"

```

We used a single A100 GPU.


### Acknowledgements

We use TRL - Transformer Reinforcement Learning library [TRL](https://github.com/huggingface/trl) and build upon provided implementation [examples](https://github.com/huggingface/trl/tree/main/examples/research_projects)
