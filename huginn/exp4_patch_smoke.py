"""Hybrid-answer patching smoke test.

Target B is run and at loop k its R4 output is replaced by source A's R4 output.
Outcomes: B's answer, A's answer, or the hybrid (A's intermediate with B's second op).
Controls: self-patch (must be a no-op) and fresh noise.

  python exp4_patch_smoke.py
  python exp4_patch_smoke.py --positions last
"""

import os

import numpy as np

from common import RESULTS, base_parser, get_runner, load_json, plt, progress, results_path, save_json
from huginn_harness import R4, Intervention, build_input_ids

CLASSES = ["target", "source", "hybrid", "other"]


def classify(pred, pair):
    if pred == pair["target_final"]:
        return "target"
    if pred == pair["source_final"]:
        return "source"
    if pred == pair["hybrid"]:
        return "hybrid"
    return "other"


def main():
    p = base_parser(__doc__)
    p.add_argument("--pairs", default="patch_pairs.json")
    p.add_argument("--n", type=int, default=30, help="number of usable pairs to run")
    p.add_argument("--steps", type=int, default=16)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--positions", choices=["all", "last"], default="all")
    p.add_argument("--skip-correct-filter", action="store_true",
                   help="debugging only: use pairs even if Huginn gets them wrong")
    args = p.parse_args()

    runner = get_runner(args)
    fewshot = load_json("fewshot.json")
    pairs = load_json(args.pairs)
    R = args.steps

    def prep(question):
        ids = build_input_ids(runner.tok, fewshot, question, args.device)
        init = runner.make_init_state(ids.shape[1], kind="seed", seed=args.seed)
        return ids, init

    # which prompts does Huginn get right? (reuse exp1 results when available)
    known = {}
    for fname in ("exp1_accuracy.json", "pool_accuracy.json"):
        path = os.path.join(RESULTS, fname)
        if os.path.exists(path):
            for row in load_json(path)["items"]:
                if str(R) in row["by_steps"]:
                    known[row["context"]] = row["by_steps"][str(R)]["correct"]

    def is_correct(question, answer):
        if question not in known:
            ids, init = prep(question)
            known[question] = runner.run(ids, R, init, capture_blocks=()).pred_text == answer
        return known[question]

    usable = []
    for pair in pairs:
        if len(usable) >= args.n:
            break
        if args.skip_correct_filter or (is_correct(pair["source"], pair["source_final"])
                                        and is_correct(pair["target"], pair["target_final"])):
            usable.append(pair)
    if not usable:
        raise SystemExit("no pairs where Huginn gets both problems right; try more pairs or more loops")

    positions = None if args.positions == "all" else [-1]
    counts = {c: {cls: np.zeros(R) for cls in CLASSES} for c in ("patch", "self", "noise")}
    self_unchanged = np.zeros(R)  # self-patch must reproduce the unpatched answer exactly
    logit_diff = np.zeros((len(usable), R))  # hybrid logit minus target logit, main patch
    rows = []

    for i, pair in enumerate(progress(usable, "exp4")):
        src_ids, src_init = prep(pair["source"])
        tgt_ids, tgt_init = prep(pair["target"])
        if src_ids.shape != tgt_ids.shape:
            continue
        j = len(rows)
        src_states = runner.run(src_ids, R, src_init, capture_blocks=(R4,)).block_states[R4]
        tgt_run = runner.run(tgt_ids, R, tgt_init, capture_blocks=(R4,))
        tgt_states = tgt_run.block_states[R4]
        noise = runner.make_init_state(tgt_ids.shape[1], kind="seed", seed=10_000 + i)
        row = {"pair": pair, "patch": [], "self": [], "noise": []}
        for k in range(1, R + 1):
            for cond, state in (("patch", src_states[k - 1]), ("self", tgt_states[k - 1]), ("noise", noise)):
                out = runner.run(tgt_ids, R, tgt_init, capture_blocks=(),
                                 intervention=Intervention(loop=k, state=state, positions=positions))
                cls = classify(out.pred_text, pair)
                counts[cond][cls][k - 1] += 1
                row[cond].append(out.pred_text)
                if cond == "self":
                    self_unchanged[k - 1] += out.pred_text == tgt_run.pred_text
                if cond == "patch":
                    logit_diff[j, k - 1] = (runner.digit_logit(out.logits, pair["hybrid"])
                                            - runner.digit_logit(out.logits, pair["target_final"]))
        rows.append(row)

    n = len(rows)
    if n == 0:
        raise SystemExit("every usable pair had mismatched prompt lengths")
    frac = {c: {cls: (counts[c][cls] / n).round(3).tolist() for cls in CLASSES} for c in counts}
    any_hybrid = sum(any(classify(pred, r["pair"]) == "hybrid" for pred in r["patch"]) for r in rows)

    save_json({"n_pairs": n, "loops": R, "positions": args.positions, "fractions": frac,
               "pairs_with_any_hybrid": any_hybrid,
               "self_patch_unchanged": (self_unchanged / n).round(3).tolist(),
               "hybrid_minus_target_logit_mean": logit_diff[:n].mean(0).round(3).tolist(),
               "rows": rows}, f"exp4_patch_{args.positions}.json")

    pyplot = plt()
    loops = np.arange(1, R + 1)
    fig, axes = pyplot.subplots(1, 2, figsize=(11, 4.2))
    ax = axes[0]
    colors = {"target": "#1f77b4", "source": "#ff7f0e", "hybrid": "#d62728", "other": "#7f7f7f"}
    for cls in CLASSES:
        ax.plot(loops, counts["patch"][cls] / n, color=colors[cls], marker="o", ms=3, label=f"answer = {cls}")
    ax.plot(loops, self_unchanged / n, "k:", label="control: self-patch leaves answer unchanged")
    ax.plot(loops, counts["noise"]["target"] / n, color="#2ca02c", ls="--", label="control: noise-patch gives target")
    ax.set_ylim(-0.02, 1.02)
    ax.set_xlabel("loop k where A's state is swapped into B")
    ax.set_ylabel("fraction of pairs")
    ax.set_title(f"hybrid-answer patching, {args.positions} positions (n={n})")
    ax.legend(fontsize=8)
    ax = axes[1]
    m, s = logit_diff[:n].mean(0), logit_diff[:n].std(0)
    ax.plot(loops, m, color="#d62728")
    ax.fill_between(loops, m - s, m + s, color="#d62728", alpha=0.15)
    ax.axhline(0, color="k", lw=0.8)
    ax.set_xlabel("loop k")
    ax.set_ylabel("logit(hybrid) - logit(target answer)")
    ax.set_title("does the hybrid gain ground?")
    fig.tight_layout()
    fig.savefig(results_path(f"exp4_patch_{args.positions}.png"), dpi=150)

    print(f"\n{n} pairs, {R} loops, patching {args.positions} positions")
    print("  loop | target source hybrid other | self-ctrl unchanged | noise-ctrl gives target")
    for k in range(R):
        f = [counts["patch"][c][k] / n for c in CLASSES]
        print(f"  {k+1:>4} | " + "  ".join(f"{x:5.0%}" for x in f)
              + f" | {self_unchanged[k]/n:17.0%} | {counts['noise']['target'][k]/n:22.0%}")
    print(f"  pairs that produced the hybrid at some loop: {any_hybrid}/{n}")
    if self_unchanged.min() < n:
        print("  WARNING: self-patch changed some answers; the hook is not a no-op, investigate before trusting results")
    print(f"  plot -> results/exp4_patch_{args.positions}.png")


if __name__ == "__main__":
    main()
