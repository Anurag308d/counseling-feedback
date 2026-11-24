"""
Build goal-setting training data (part 4) and the mixed SFT datasets.

    python build_goal_train.py

Reads data/goal_train_v0/contexts_train.py (24 homework contexts, 8 negative controls; all
synthetic and disjoint from the eval set) and writes:
    data/goal_train_v0/train.json            208 examples in the FeedbackESConv format
    data/feedback_mixed_x1/train.json        labeled-split train + goal data once   (~3% goal data)
    data/feedback_mixed_x3/train.json        labeled-split train + goal data 3 times (~9% goal data)

Gold labels follow the eval set's expectations: strong replies are perfect with Goal Setting
as a good area; each degradation is not perfect with Goal Setting as a bad area, feedback naming
the specific flaw, and the strong reply as the alternative; negative controls flip that (the
empathic reply is perfect, the mistimed homework is not).
"""
import importlib.util
import json
import os
import random

SEED = 42
ORDER = ["open", "rationale", "plan", "barriers", "followup"]
HEADER = ("Below is an instruction that describes a task, paired with an input that provides further "
          "context. Write a response that appropriately completes the request.\n\n### Instruction:\n"
          "Give feedback to the Helper's last response.\n\n### Input:\n")

# Replacement lines for the generic degradations, kept close to the original segment lengths
# so that reply length does not give the label away.
POOLS = {
    "no_barriers": [
        "I'm sure it will go really well, and you'll feel great about it afterwards.",
        "I really think you'll find it easier than you expect once you get started.",
        "You clearly have what it takes, so I'm confident it will work out fine.",
        "It sounds like a good plan, and I have a feeling it will go smoothly for you.",
    ],
    "no_rationale": [
        "So let's turn that into something concrete that you can try out this week.",
        "Okay, let's go ahead and turn that idea into a clear plan for the coming days.",
        "Right, let's make that into a small plan you can put into practice this week.",
        "Great, so let's shape that into something specific for the next few days.",
    ],
    "no_followup": [
        "Good luck with it, and I really hope you enjoy it when the time comes.",
        "I hope it goes well for you, and that you get something good out of it.",
        "Have a lovely time with it, and take care of yourself this week.",
        "Enjoy it, and remember to be kind to yourself while you're doing it.",
    ],
}

FEEDBACK = {
    "vague": ["The goal is to {goal}. The plan is too vague to act on; it could be improved by agreeing on a specific activity with a day, time and place.",
              "The goal is to {goal}. It could be improved by making the plan concrete: what exactly the seeker will do, when and where."],
    "values_unlinked": ["The goal is to {goal}. The activity suggested isn't connected to what the seeker said matters to them ({values}); it could be improved by building the plan around that instead.",
                        "The goal is to {goal}. It could be improved by choosing an activity linked to the seeker's own values ({values}) rather than a generic one."],
    "too_ambitious": ["The goal is to {goal}. The plan is far too much for someone who is struggling; it could be improved by starting with one small, achievable step.",
                      "The goal is to {goal}. It could be improved by grading the task down to something the seeker can realistically manage this week."],
    "imposed": ["The goal is to {goal}. The plan is dictated rather than agreed; it could be improved by offering it as a suggestion and checking whether the seeker wants to try it.",
                "The goal is to {goal}. It could be improved by choosing the plan together with the seeker instead of telling them what they must do."],
    "no_barriers": ["The goal is to {goal}. It doesn't explore what might get in the way; it could be improved by asking about possible barriers and how to handle them.",
                    "The goal is to {goal}. It could be improved by anticipating obstacles with the seeker rather than assuming the plan will go smoothly."],
    "no_rationale": ["The goal is to {goal}. It doesn't explain why the activity might help; it could be improved by briefly sharing the reasoning behind it.",
                     "The goal is to {goal}. It could be improved by giving the seeker a short rationale for why taking this step can help their mood."],
    "no_followup": ["The goal is to {goal}. It doesn't plan to review how it went; it could be improved by agreeing to check in on the activity next time.",
                    "The goal is to {goal}. It could be improved by planning a follow-up so the seeker knows you will look at how it went together."],
}
MISTIMED_FEEDBACK = ["The goal is to support the seeker in this moment. {why} It could be improved by listening and validating their feelings before suggesting any plan.",
                     "The goal is to respond to what the seeker needs right now. {why} It could be improved by holding off on homework and focusing on understanding and support."]


def load_contexts(path):
    spec = importlib.util.spec_from_file_location("contexts_train", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.POSITIVE, mod.NEGATIVE


def example(context, reply, label, ctx_id, kind):
    conv = "\n".join(f"{who}: {text}" for who, text in context) + f"\nHelper: {reply}"
    return {"text": HEADER + conv + "\n\n### Response:" + json.dumps(label),
            "context_id": ctx_id, "kind": kind}


def build(positive, negative, rng):
    out = []
    for c in positive:
        strong = " ".join(c["seg"][k] for k in ORDER)
        out.append(example(c["context"], strong, {"perfect": True, "goodareas": ["Goal Setting", "Validation"]},
                           c["id"], "strong"))
        overrides = {
            "vague": {"plan": c["vague"]},
            "values_unlinked": {"open": c["values_open"], "plan": c["values_plan"]},
            "too_ambitious": {"plan": c["too_ambitious"]},
            "imposed": {"plan": c["imposed"]},
            "no_barriers": {"barriers": rng.choice(POOLS["no_barriers"])},
            "no_rationale": {"rationale": rng.choice(POOLS["no_rationale"])},
            "no_followup": {"followup": rng.choice(POOLS["no_followup"])},
        }
        for deg, over in overrides.items():
            seg = dict(c["seg"], **over)
            reply = " ".join(seg[k] for k in ORDER)
            label = {
                "perfect": False,
                "goodareas": [] if deg == "values_unlinked" else ["Validation"],
                "feedback": rng.choice(FEEDBACK[deg]).format(goal=c["goal"], values=c["values"]),
                "badareas": ["Goal Setting", "Suggestions"] if deg == "imposed" else ["Goal Setting"],
                "alternative": strong,
            }
            out.append(example(c["context"], reply, label, c["id"], deg))
    for n in negative:
        out.append(example(n["context"], n["appropriate"], {"perfect": True, "goodareas": n["appropriate_good"]},
                           n["id"], "neg_appropriate"))
        out.append(example(n["context"], n["mistimed"], {
            "perfect": False, "goodareas": [],
            "feedback": rng.choice(MISTIMED_FEEDBACK).format(why=n["why"]),
            "badareas": ["Goal Setting", "Empathy"], "alternative": n["appropriate"]}, n["id"], "neg_mistimed"))
    return out


def main():
    rng = random.Random(SEED)
    positive, negative = load_contexts("data/goal_train_v0/contexts_train.py")

    # Guard against leakage into the eval set: no shared context ids or replies.
    with open("data/goal_eval_v0/eval_set_v0.jsonl") as f:
        eval_rows = [json.loads(line) for line in f if line.strip()]
    eval_ids = {r["context_id"] for r in eval_rows}
    eval_replies = {r["response"] for r in eval_rows}
    assert not eval_ids & {c["id"] for c in positive + negative}

    goal = build(positive, negative, rng)
    assert not any(ex["text"].split("\nHelper: ")[-1].split("\n\n### Response:")[0] in eval_replies for ex in goal)
    os.makedirs("data/goal_train_v0", exist_ok=True)
    with open("data/goal_train_v0/train.json", "w") as f:
        json.dump(goal, f)

    with open("data/feedback_qesconv_labeledsplit/train.json") as f:
        base = json.load(f)
    goal_rows = [{"text": g["text"], "helper_index": -1, "conv_index": -1, "source": "goal_train_v0"} for g in goal]
    for name, copies in (("feedback_mixed_x1", 1), ("feedback_mixed_x3", 3)):
        mixed = base + goal_rows * copies
        random.Random(SEED).shuffle(mixed)
        os.makedirs(f"data/{name}", exist_ok=True)
        with open(f"data/{name}/train.json", "w") as f:
            json.dump(mixed, f)
        share = len(goal_rows) * copies / len(mixed)
        print(f"{name}: {len(mixed)} examples, goal-setting share {share:.1%}")
    kinds = {}
    for g in goal:
        kinds[g["kind"]] = kinds.get(g["kind"], 0) + 1
    print(f"goal_train_v0: {len(goal)} examples {kinds}")


if __name__ == "__main__":
    main()
