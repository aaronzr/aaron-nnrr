"""Builds the two-step arithmetic data in data/: (a op1 b) op2 c, single digits.
Intermediate and final are single digits, differ from each other, and differ
from every operand.

  two_step_50.json, two_step_pool.json (300), fewshot.json, patch_pairs.json
"""

import itertools
import json
import os
import random

SEED = 0
OPS = {"+": lambda x, y: x + y, "-": lambda x, y: x - y, "*": lambda x, y: x * y}
DIGITS = range(1, 10)
OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")


def question_text(a, op1, b, op2, c):
    # Same surface format as the GPT-3 arithmetic set used by Lu et al.
    return f"Question: What is ({a} {op1} {b}) {op2} {c}?\nAnswer:"


def all_valid_problems():
    problems = []
    for a, b, c in itertools.product(DIGITS, DIGITS, DIGITS):
        for op1, op2 in itertools.product(OPS, OPS):
            inter = OPS[op1](a, b)
            final = OPS[op2](inter, c)
            if not (0 <= inter <= 9 and 0 <= final <= 9):
                continue
            if inter == final:
                continue
            if inter in (a, b, c) or final in (a, b, c):
                continue
            problems.append({
                "a": a, "b": b, "c": c, "op1": op1, "op2": op2,
                "pattern": f"{op1}{op2}",
                "intermediate": str(inter),
                "final": str(final),
                "context": question_text(a, op1, b, op2, c),
                "completion": f" {final}",
            })
    return problems


def balanced_sample(problems, n, rng, exclude_ids=()):
    """Round-robin across op patterns so no single pattern dominates."""
    by_pattern = {}
    for p in problems:
        if p["context"] in exclude_ids:
            continue
        by_pattern.setdefault(p["pattern"], []).append(p)
    for group in by_pattern.values():
        rng.shuffle(group)
    patterns = sorted(by_pattern)
    picked = []
    i = 0
    while len(picked) < n and any(by_pattern.values()):
        group = by_pattern[patterns[i % len(patterns)]]
        if group:
            picked.append(group.pop())
        i += 1
    return picked


def make_patch_pairs(pool, rng, max_pairs=200):
    """
    Pairs (source A, target B) with the same op pattern, so the prompts tokenize
    to the same length and positions line up when we swap states.

    hybrid = A's intermediate combined with B's second step: interA op2 cB.
    We only keep pairs where the hybrid is a single digit and differs from every
    other answer we might confuse it with, so "the model said the hybrid" is
    unambiguous.
    """
    by_pattern = {}
    for p in pool:
        by_pattern.setdefault(p["pattern"], []).append(p)

    pairs = []
    for pattern, group in by_pattern.items():
        for src, tgt in itertools.permutations(group, 2):
            if src["intermediate"] == tgt["intermediate"] or src["final"] == tgt["final"]:
                continue
            hybrid = OPS[tgt["op2"]](int(src["intermediate"]), tgt["c"])
            if not 0 <= hybrid <= 9:
                continue
            h = str(hybrid)
            if h in (src["final"], tgt["final"], src["intermediate"], tgt["intermediate"]):
                continue
            pairs.append({
                "pattern": pattern,
                "source": src["context"],
                "target": tgt["context"],
                "source_final": src["final"],
                "source_intermediate": src["intermediate"],
                "target_final": tgt["final"],
                "target_intermediate": tgt["intermediate"],
                "hybrid": h,
            })
    rng.shuffle(pairs)
    return pairs[:max_pairs]


def main():
    rng = random.Random(SEED)
    problems = all_valid_problems()

    # Few-shot examples: fixed, and removed from everything else.
    fewshot = balanced_sample(problems, 4, rng)
    fewshot_ids = {p["context"] for p in fewshot}

    main50 = balanced_sample(problems, 50, rng, exclude_ids=fewshot_ids)
    pool = balanced_sample(problems, 300, random.Random(SEED + 1), exclude_ids=fewshot_ids)

    pairs = make_patch_pairs(pool, rng)

    os.makedirs(OUT_DIR, exist_ok=True)
    for name, obj in [("two_step_50.json", main50), ("two_step_pool.json", pool),
                      ("fewshot.json", fewshot), ("patch_pairs.json", pairs)]:
        with open(os.path.join(OUT_DIR, name), "w") as f:
            json.dump(obj, f, indent=2)

    print(f"valid problems in total: {len(problems)}")
    print(f"main set: {len(main50)} | pool: {len(pool)} | few-shot: {len(fewshot)} | patch pairs: {len(pairs)}")
    patterns = {}
    for p in main50:
        patterns[p["pattern"]] = patterns.get(p["pattern"], 0) + 1
    print("main set op patterns:", dict(sorted(patterns.items())))
    print("example:", main50[0]["context"].replace("\n", " "), "->",
          main50[0]["final"], "(intermediate", main50[0]["intermediate"] + ")")
    print("example pair:", pairs[0])


if __name__ == "__main__":
    main()
