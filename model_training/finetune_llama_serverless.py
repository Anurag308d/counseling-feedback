"""
Fine-tune Llama 3.1-8B-Instruct (QLoRA) on Modal for serverless execution.

Run from the model_training directory.

Smoke test (about 15 minutes, under $1):
    modal run finetune_llama_serverless.py --run-name sft_smoke --max-train-samples 200 --max-steps 20

Full held-out baseline run (paper settings, train split only):
    modal run --detach finetune_llama_serverless.py --run-name sft_labeledsplit

Assumes the labeled split exists at model_training/data/feedback_qesconv_labeledsplit/
(create it with `python make_labeled_split.py`) and a Modal secret named "anurag-bridge-huggingface"
holding HF_TOKEN (see PREFIX below).

Outputs go to the "anurag-bridge-models" volume, under /root/model_training/models/<run-name>/:
    final_checkpoint/          LoRA adapter
    final_merged_checkpoint/   merged bf16 model, used by generate_eval_serverless.py
    train_config.json, train_log.json

Changes from the version we were given (see the part 1 plan):
  * Examples end with the Llama 3 EOS token instead of the literal "<s>"/"</s>", which are
    Llama 2 tokens and plain text to the Llama 3 tokenizer, so the model never learned to stop.
  * The pad token is a dedicated Llama 3.1 pad token, not EOS: the LM collator masks every pad
    id out of the loss, so pad == EOS would also hide the EOS the model needs to learn.
  * The rope_scaling override (linear, 8.0) is removed; Llama 3.1 ships its own "llama3" rope
    scaling and the generation scripts never applied the override.
  * Hyperparameters follow the paper's README recipe (effective batch 32, lr 3e-4, cosine,
    100 warmup steps, weight decay 0.01, no packing, max length 2048). The paper ran 766 steps
    on all 8,179 examples, which is 3 epochs; here we train 3 epochs on the train split.
  * Package versions are pinned, base weights are cached on a volume, and the timeout is 4 h.
"""
import json
import os

import modal

# The salt-nlp Modal workspace is shared, so every app, volume and secret name carries a prefix
# that cannot collide with other people's (override with MODAL_NAME_PREFIX).
PREFIX = os.environ.get("MODAL_NAME_PREFIX", "anurag-bridge")
HF_SECRET_NAME = os.environ.get("MODAL_HF_SECRET", f"{PREFIX}-huggingface")

app = modal.App(f"{PREFIX}-finetune")

MODEL_VOLUME = modal.Volume.from_name(f"{PREFIX}-models", create_if_missing=True)
HF_CACHE_VOLUME = modal.Volume.from_name(f"{PREFIX}-hf-cache", create_if_missing=True)

BASE_MODEL = "meta-llama/Meta-Llama-3.1-8B-Instruct"
MODELS_DIR = "/root/model_training/models"
# Llama 3.1 reserves this token for padding (id 128004).
PAD_TOKEN = "<|finetune_right_pad_id|>"

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch==2.5.1",
        "transformers==4.46.3",
        "accelerate==1.1.1",
        "peft==0.13.2",
        "datasets==3.1.0",
        "trl==0.12.2",
        "bitsandbytes==0.45.0",
        "numpy<2.2",
        "scipy",
    )
    .env({"HF_HOME": "/root/hf-cache", "TOKENIZERS_PARALLELISM": "false"})
    .add_local_dir(".", remote_path="/root/model_training",
                   ignore=["venv", "__pycache__", "*.git*", "models", "exp"])
)


@app.function(
    image=image,
    gpu="H100",
    timeout=4 * 60 * 60,  # hard cap on spend: about $16 at H100 rates
    volumes={MODELS_DIR: MODEL_VOLUME, "/root/hf-cache": HF_CACHE_VOLUME},
    secrets=[modal.Secret.from_name(HF_SECRET_NAME)],
)
def train_llama_model(
    run_name: str,
    model_name: str = BASE_MODEL,
    dataset_name: str = "feedback_qesconv_labeledsplit",
    num_train_epochs: float = 3.0,
    max_steps: int = -1,
    max_train_samples: int = 0,
    learning_rate: float = 3e-4,
    per_device_train_batch_size: int = 1,
    gradient_accumulation_steps: int = 32,
    warmup_steps: int = 100,
    weight_decay: float = 0.01,
    lr_scheduler_type: str = "cosine",
    max_seq_length: int = 2048,
    packing: bool = False,
    gradient_checkpointing: bool = False,
    lora_r: int = 8,
    lora_alpha: float = 16,
    lora_dropout: float = 0.05,
    seed: int = 42,
):
    import torch
    from datasets import load_dataset
    from peft import AutoPeftModelForCausalLM, LoraConfig
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, set_seed
    from trl import SFTConfig, SFTTrainer

    set_seed(seed)
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    access_token = os.environ.get("HF_TOKEN")
    output_dir = os.path.join(MODELS_DIR, run_name)
    os.makedirs(output_dir, exist_ok=True)
    config = dict(locals())
    config.pop("access_token")
    with open(os.path.join(output_dir, "train_config.json"), "w") as f:
        json.dump({k: v for k, v in config.items() if isinstance(v, (str, int, float, bool))}, f, indent=2)

    tokenizer = AutoTokenizer.from_pretrained(model_name, token=access_token)
    tokenizer.pad_token = PAD_TOKEN
    tokenizer.padding_side = "right"
    assert tokenizer.pad_token_id != tokenizer.eos_token_id
    print(f"eos={tokenizer.eos_token!r} ({tokenizer.eos_token_id}) pad={tokenizer.pad_token!r} ({tokenizer.pad_token_id})")

    data_dir = f"/root/model_training/data/{dataset_name}"
    dataset = load_dataset("json", data_files={"train": f"{data_dir}/train.json"},
                           cache_dir="/root/hf-cache/datasets")["train"]
    if max_train_samples:
        dataset = dataset.shuffle(seed=seed).select(range(min(max_train_samples, len(dataset))))
    # The tokenizer adds <|begin_of_text|> itself; we only append EOS so the model learns to stop.
    dataset = dataset.map(lambda x: {"text": x["text"] + tokenizer.eos_token},
                          remove_columns=[c for c in dataset.column_names if c != "text"])
    print(f"train examples: {len(dataset)}")
    print("sample text:\n" + dataset[0]["text"])

    ids = tokenizer(dataset[0]["text"])["input_ids"]
    assert ids[0] == tokenizer.bos_token_id and ids[-1] == tokenizer.eos_token_id, \
        "expected <|begin_of_text|> ... EOS after tokenization"

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    base_model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map={"": 0},
        torch_dtype=torch.bfloat16,
        token=access_token,
    )
    base_model.config.use_cache = False
    base_model.config.pad_token_id = tokenizer.pad_token_id

    peft_config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        target_modules=["q_proj", "v_proj"],
        bias="none",
        task_type="CAUSAL_LM",
    )

    sft_config = SFTConfig(
        output_dir=output_dir,
        num_train_epochs=num_train_epochs,
        max_steps=max_steps,
        per_device_train_batch_size=per_device_train_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        learning_rate=learning_rate,
        lr_scheduler_type=lr_scheduler_type,
        warmup_steps=warmup_steps,
        weight_decay=weight_decay,
        optim="paged_adamw_32bit",
        bf16=True,
        gradient_checkpointing=gradient_checkpointing,
        group_by_length=False,
        logging_steps=10,
        save_steps=100,
        save_total_limit=3,
        report_to="none",
        seed=seed,
        dataset_text_field="text",
        max_seq_length=max_seq_length,
        packing=packing,
        dataset_num_proc=4,
    )

    trainer = SFTTrainer(
        model=base_model,
        args=sft_config,
        train_dataset=dataset,
        peft_config=peft_config,
        processing_class=tokenizer,
    )

    print("Starting training...")
    results = trainer.train()
    print("Training results:", results)
    with open(os.path.join(output_dir, "train_log.json"), "w") as f:
        json.dump(trainer.state.log_history, f, indent=2)

    adapter_dir = os.path.join(output_dir, "final_checkpoint")
    trainer.model.save_pretrained(adapter_dir)
    tokenizer.save_pretrained(adapter_dir)
    MODEL_VOLUME.commit()

    del trainer, base_model
    torch.cuda.empty_cache()

    print("Merging LoRA weights...")
    model = AutoPeftModelForCausalLM.from_pretrained(adapter_dir, device_map={"": 0},
                                                     torch_dtype=torch.bfloat16, token=access_token)
    model = model.merge_and_unload()
    merged_dir = os.path.join(output_dir, "final_merged_checkpoint")
    model.save_pretrained(merged_dir, safe_serialization=True)
    tokenizer.save_pretrained(merged_dir)
    MODEL_VOLUME.commit()
    print(f"TRAINING COMPLETE: {merged_dir}")
    return merged_dir


@app.local_entrypoint()
def main(
    run_name: str = "sft_labeledsplit",
    model_name: str = BASE_MODEL,
    dataset_name: str = "feedback_qesconv_labeledsplit",
    num_train_epochs: float = 3.0,
    max_steps: int = -1,
    max_train_samples: int = 0,
    learning_rate: float = 3e-4,
    gradient_accumulation_steps: int = 32,
    warmup_steps: int = 100,
):
    if max_steps > 0 and max_steps < warmup_steps:
        warmup_steps = max(1, max_steps // 10)
    print(f"run={run_name} model={model_name} dataset={dataset_name} epochs={num_train_epochs} "
          f"max_steps={max_steps} max_train_samples={max_train_samples} lr={learning_rate} "
          f"grad_accum={gradient_accumulation_steps} warmup={warmup_steps}")
    path = train_llama_model.remote(
        run_name=run_name,
        model_name=model_name,
        dataset_name=dataset_name,
        num_train_epochs=num_train_epochs,
        max_steps=max_steps,
        max_train_samples=max_train_samples,
        learning_rate=learning_rate,
        gradient_accumulation_steps=gradient_accumulation_steps,
        warmup_steps=warmup_steps,
    )
    print(f"Model saved to: {path}")
