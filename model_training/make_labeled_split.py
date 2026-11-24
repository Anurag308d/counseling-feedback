"""
Split the labeled FeedbackESConv data into held-out train/test sets, by conversation.

Every example is the conversation up to one Helper turn, so examples from the same
conversation share most of their text. Splitting by turn would leak test context into
training, so whole conversations (conv_index) go to one side only.

Run from model_training/ (no GPU needed):
    python make_labeled_split.py

Writes to data/feedback_qesconv_labeledsplit/:
    train.json         labeled training examples (same format as feedback_qesconv/train.json)
    test.json          labeled held-out examples, gold JSON kept after "### Response:"
    test_prompts.json  the same test examples cut after "### Response:" (generation input)
    split_manifest.json  seed, sizes, test conversation ids, balance stats

About the data: every non-perfect example in feedback_qesconv/train.json has a twin with the same
conv_index and helper_index whose last Helper turn is replaced by the gold "alternative" and labeled
perfect. So of 8,179 examples, 4,721 are original Helper turns (1,263 perfect, 27%) and 3,458 are
gold alternatives (all perfect). Each test example gets "example_id" and "source"
("original" or "gold_alternative") so evaluate_baseline.py can report the two separately; the
headline numbers should use source == "original".
"""
import argparse
import collections
import json
import os
import random

RESPONSE_MARKER = "### Response:"
AREAS = ["Reflections", "Questions", "Suggestions", "Validation", "Self-disclosure",
         "Empathy", "Professionalism", "Structure"]


def gold(example):
    return json.loads(example["text"].split(RESPONSE_MARKER, 1)[1])


def stats(examples):
    good, bad = collections.Counter(), collections.Counter()
    perfect = 0
    for ex in examples:
        g = gold(ex)
        perfect += bool(g["perfect"])
        good.update(g.get("goodareas", []))
        bad.update(g.get("badareas", []))
    n = len(examples)
    return {
        "n": n,
        "perfect_rate": perfect / n,
        "goodarea_rate": {a: good[a] / n for a in AREAS},
        "badarea_rate": {a: bad[a] / n for a in AREAS},
    }


def max_gap(a, b):
    gaps = [abs(a["perfect_rate"] - b["perfect_rate"])]
    for key in ("goodarea_rate", "badarea_rate"):
        gaps += [abs(a[key][x] - b[key][x]) for x in AREAS]
    return max(gaps)


def last_helper_turn(example):
    return example["text"].split(RESPONSE_MARKER, 1)[0].rstrip().split("\n")[-1].strip()


def tag_sources(examples):
    """Mark examples whose last Helper turn is the gold alternative of their non-perfect twin."""
    by_turn = collections.defaultdict(list)
    for ex in examples:
        ex["source"] = "original"
        by_turn[(ex["conv_index"], ex["helper_index"])].append(ex)
    for group in by_turn.values():
        flawed = [ex for ex in group if not gold(ex)["perfect"]]
        for bad in flawed:
            alt_line = ("Helper: " + gold(bad)["alternative"]).strip()
            for ex in group:
                if ex is not bad and gold(ex)["perfect"] and last_helper_turn(ex) == alt_line:
                    ex["source"] = "gold_alternative"
    return collections.Counter(ex["source"] for ex in examples)


def split(examples, test_frac, seed):
    convs = sorted({ex["conv_index"] for ex in examples})
    rng = random.Random(seed)
    rng.shuffle(convs)
    n_test = round(len(convs) * test_frac)
    test_convs = set(convs[:n_test])
    train = [ex for ex in examples if ex["conv_index"] not in test_convs]
    test = [ex for ex in examples if ex["conv_index"] in test_convs]
    return train, test, sorted(test_convs)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", default="data/feedback_qesconv/train.json")
    parser.add_argument("--out_dir", default="data/feedback_qesconv_labeledsplit")
    parser.add_argument("--test_frac", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_gap", type=float, default=0.03,
                        help="re-draw (seed+1, ...) until every label rate differs by at most this")
    parser.add_argument("--max_tries", type=int, default=200)
    args = parser.parse_args()

    with open(args.source) as f:
        examples = json.load(f)
    source_counts = tag_sources(examples)
    print(f"examples by source: {dict(source_counts)}")

    for attempt in range(args.max_tries):
        seed = args.seed + attempt
        train, test, test_convs = split(examples, args.test_frac, seed)
        tr_stats, te_stats = stats(train), stats(test)
        gap = max_gap(tr_stats, te_stats)
        if gap <= args.max_gap:
            break
    else:
        raise SystemExit(f"No seed within {args.max_tries} tries met max_gap={args.max_gap}")

    train_convs = {ex["conv_index"] for ex in train}
    assert not train_convs & set(test_convs), "conversation leak between train and test"
    assert len(train) + len(test) == len(examples)

    for i, ex in enumerate(test):
        ex["example_id"] = i
    prompts = [{"text": ex["text"].split(RESPONSE_MARKER, 1)[0] + RESPONSE_MARKER,
                "helper_index": ex["helper_index"],
                "conv_index": ex["conv_index"],
                "example_id": ex["example_id"],
                "source": ex["source"]} for ex in test]

    os.makedirs(args.out_dir, exist_ok=True)
    for name, data in (("train.json", train), ("test.json", test), ("test_prompts.json", prompts)):
        with open(os.path.join(args.out_dir, name), "w") as f:
            json.dump(data, f)
    manifest = {
        "source": args.source,
        "seed": seed,
        "test_frac": args.test_frac,
        "n_conversations": {"train": len(train_convs), "test": len(test_convs)},
        "n_examples": {"train": len(train), "test": len(test)},
        "max_label_rate_gap": gap,
        "n_test_examples_by_source": dict(collections.Counter(ex["source"] for ex in test)),
        "test_stats_original_only": stats([ex for ex in test if ex["source"] == "original"]),
        "train_stats": tr_stats,
        "test_stats": te_stats,
        "test_conv_indices": test_convs,
    }
    with open(os.path.join(args.out_dir, "split_manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"seed={seed} conversations train/test={len(train_convs)}/{len(test_convs)} "
          f"examples train/test={len(train)}/{len(test)} max label-rate gap={gap:.3f}")
    print(f"perfect rate train={tr_stats['perfect_rate']:.3f} test={te_stats['perfect_rate']:.3f}")


if __name__ == "__main__":
    main()
