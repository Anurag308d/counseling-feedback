"""
Generate one greedy feedback output per held-out test prompt with vLLM on Modal, and record
P("perfect" = true) from the first generated token's log-probabilities.

This replaces generate_llama_exp_serverless.py for evaluation. That script draws 10 samples
per prompt, plus a second "improved" generation, one at a time with plain HF generate, which
would cost roughly $160 of H100 time on the test split. This one batches everything and
should take 15 to 20 minutes including model load.

Run from model_training/:
    # retrained baseline (a run name on the trained-llama-models volume)
    modal run generate_eval_serverless.py --model sft_labeledsplit

    # base Llama 3.1-8B-Instruct, prompted with the task and taxonomy
    modal run generate_eval_serverless.py --model meta-llama/Meta-Llama-3.1-8B-Instruct \
        --prompt-style instruct --tag base_instruct

    # quick check on 5 prompts
    modal run generate_eval_serverless.py --model sft_smoke --max-prompts 5

Writes exp/<tag>_<dataset>_eval.json locally (and a copy on the anurag-bridge-models volume under exp/).
Score it with: python evaluate_baseline.py exp/<tag>_<dataset>_eval.json
"""
import json
import os

import modal

# The salt-nlp Modal workspace is shared, so every app, volume and secret name carries a prefix
# that cannot collide with other people's (override with MODAL_NAME_PREFIX).
PREFIX = os.environ.get("MODAL_NAME_PREFIX", "anurag-bridge")
HF_SECRET_NAME = os.environ.get("MODAL_HF_SECRET", f"{PREFIX}-huggingface")

app = modal.App(f"{PREFIX}-generate-eval")

MODEL_VOLUME = modal.Volume.from_name(f"{PREFIX}-models", create_if_missing=True)
HF_CACHE_VOLUME = modal.Volume.from_name(f"{PREFIX}-hf-cache", create_if_missing=True)
MODELS_DIR = "/root/model_training/models"
BASE_MODEL = "meta-llama/Meta-Llama-3.1-8B-Instruct"

# The SFT data reads '### Response:{"perfect": true, ...}'. We put '{"perfect":' into the prompt so
# the next token is exactly " true" or " false", and so the prompt's last tokens match training.
PREFILL = '{"perfect":'

AREAS = ["Reflections", "Questions", "Suggestions", "Validation", "Self-disclosure",
         "Empathy", "Professionalism", "Structure"]

INSTRUCT_SYSTEM = f"""You give feedback to peer counselors (Helpers) on their last response in an emotional-support conversation with a Seeker.
Reply with a single JSON object and nothing else.
- If the Helper's last response is already appropriate: {{"perfect": true, "goodareas": [...]}}
- Otherwise: {{"perfect": false, "goodareas": [...], "feedback": "...", "badareas": [...], "alternative": "..."}}
"goodareas" lists what the response does well and "badareas" what needs improvement, using only these areas: {", ".join(AREAS)}.
"feedback" states the goal of the response and how it could be improved, in one or two sentences. "alternative" is an improved Helper response."""

image = (
    modal.Image.debian_slim(python_version="3.11")
    # vllm 0.6.4 only declares transformers>=4.45, so pip otherwise pulls transformers 5.x, whose
    # tokenizer class vllm cannot use. Pin the same transformers stack as the training image.
    .pip_install("vllm==0.6.4.post1", "transformers==4.46.3", "tokenizers==0.20.3",
                 "huggingface_hub==0.26.5", "datasets==3.1.0", "numpy<2")
    .env({"HF_HOME": "/root/hf-cache", "TOKENIZERS_PARALLELISM": "false"})
    .add_local_dir(".", remote_path="/root/model_training",
                   ignore=["venv", "__pycache__", "*.git*", "models", "exp"])
)


def build_prompt_ids(tokenizer, text, prompt_style):
    """text is a full prompt ending in '### Response:' (labeled-split test_prompts.json format)."""
    if prompt_style == "sft":
        return tokenizer(text + PREFILL, add_special_tokens=True)["input_ids"]
    if prompt_style == "instruct":
        conversation = text.split("### Input:\n", 1)[1].split("### Response:", 1)[0].strip()
        messages = [
            {"role": "system", "content": INSTRUCT_SYSTEM},
            {"role": "user", "content": f"Conversation:\n{conversation}\n\nGive feedback to the Helper's last response."},
        ]
        chat = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        return tokenizer(chat + PREFILL, add_special_tokens=False)["input_ids"]
    raise ValueError(f"unknown prompt_style {prompt_style}")


@app.function(
    image=image,
    gpu="H100",
    timeout=2 * 60 * 60,
    volumes={MODELS_DIR: MODEL_VOLUME, "/root/hf-cache": HF_CACHE_VOLUME},
    secrets=[modal.Secret.from_name(HF_SECRET_NAME)],
)
def generate(model: str, tag: str, dataset_name: str, prompts_file: str, prompt_style: str,
             max_prompts: int, max_new_tokens: int):
    import math

    from vllm import LLM, SamplingParams
    from vllm.inputs import TokensPrompt

    os.environ.setdefault("HUGGING_FACE_HUB_TOKEN", os.environ.get("HF_TOKEN", ""))
    merged = os.path.join(MODELS_DIR, model, "final_merged_checkpoint")
    model_path = merged if os.path.isdir(merged) else model
    print(f"Loading {model_path}")

    with open(f"/root/model_training/data/{dataset_name}/{prompts_file}") as f:
        rows = json.load(f)
    if max_prompts:
        rows = rows[:max_prompts]

    llm = LLM(model=model_path, dtype="bfloat16", max_model_len=4096, seed=42,
              gpu_memory_utilization=0.9)
    tokenizer = llm.get_tokenizer()

    true_ids = tokenizer.encode(" true", add_special_tokens=False)
    false_ids = tokenizer.encode(" false", add_special_tokens=False)
    assert len(true_ids) == 1 and len(false_ids) == 1, (true_ids, false_ids)
    true_id, false_id = true_ids[0], false_ids[0]
    stop_ids = [i for i in {tokenizer.eos_token_id,
                            tokenizer.convert_tokens_to_ids("<|eot_id|>"),
                            tokenizer.convert_tokens_to_ids("<|end_of_text|>")} if i is not None]

    prompts = [TokensPrompt(prompt_token_ids=build_prompt_ids(tokenizer, r["text"], prompt_style)) for r in rows]
    params = SamplingParams(temperature=0.0, max_tokens=max_new_tokens, logprobs=20,
                            stop_token_ids=stop_ids, seed=42)
    outputs = llm.generate(prompts, params)

    results = []
    for row, out in zip(rows, outputs):
        completion = out.outputs[0]
        p_true = None
        if completion.logprobs:
            first = completion.logprobs[0]
            lp_t = first[true_id].logprob if true_id in first else None
            lp_f = first[false_id].logprob if false_id in first else None
            if lp_t is not None or lp_f is not None:
                pt = math.exp(lp_t) if lp_t is not None else 0.0
                pf = math.exp(lp_f) if lp_f is not None else 0.0
                p_true = pt / (pt + pf)
        raw = PREFILL + completion.text
        for junk in ("</s>", "<|eot_id|>", "<|end_of_text|>"):
            raw = raw.replace(junk, "")
        results.append({
            "example_id": row.get("example_id"),
            "source": row.get("source"),
            "conv_index": row["conv_index"],
            "helper_index": row["helper_index"],
            "prompt": row["text"],
            "raw_output": raw.strip(),
            "p_true": p_true,
            "finish_reason": completion.finish_reason,
            "n_output_tokens": len(completion.token_ids),
        })

    meta = {"model": model, "model_path": model_path, "tag": tag, "dataset_name": dataset_name,
            "prompts_file": prompts_file, "prompt_style": prompt_style, "n": len(results),
            "decoding": {"temperature": 0.0, "max_new_tokens": max_new_tokens}}
    payload = {"meta": meta, "results": results}
    exp_dir = os.path.join(MODELS_DIR, "exp")
    os.makedirs(exp_dir, exist_ok=True)
    with open(os.path.join(exp_dir, f"{tag}_{dataset_name}_eval.json"), "w") as f:
        json.dump(payload, f)
    MODEL_VOLUME.commit()
    return payload


@app.local_entrypoint()
def main(
    model: str = "sft_labeledsplit",
    tag: str = "",
    dataset_name: str = "feedback_qesconv_labeledsplit",
    prompts_file: str = "test_prompts.json",
    prompt_style: str = "sft",
    max_prompts: int = 0,
    max_new_tokens: int = 512,
):
    tag = tag or model.replace("/", "_")
    payload = generate.remote(model, tag, dataset_name, prompts_file, prompt_style,
                              max_prompts, max_new_tokens)
    os.makedirs("exp", exist_ok=True)
    path = os.path.join("exp", f"{tag}_{dataset_name}_eval.json")
    with open(path, "w") as f:
        json.dump(payload, f)
    finished = sum(r["finish_reason"] == "stop" for r in payload["results"])
    print(f"Wrote {len(payload['results'])} outputs to {path} ({finished} stopped on EOS)")
    print(f"Score with: python evaluate_baseline.py {path}")
