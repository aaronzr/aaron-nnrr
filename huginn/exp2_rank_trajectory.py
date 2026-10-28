"""Rank of the final / intermediate / 'the' tokens at every block and loop,
with both lenses, on correctly answered items (Lu et al. Fig 4 on our data).

  python exp2_rank_trajectory.py   (needs results/correct_r16.json)
"""

import os

import numpy as np

from common import RESULTS, base_parser, get_runner, load_json, plt, progress, results_path, save_json
from huginn_harness import build_input_ids

BLOCKS = (0, 1, 2, 3)
NAMES = {0: "R1", 1: "R2", 2: "R3", 3: "R4"}


def main():
    p = base_parser(__doc__)
    p.add_argument("--data", default=os.path.join(RESULTS, "correct_r16.json"))
    p.add_argument("--steps", type=int, default=16)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    runner = get_runner(args)
    fewshot = load_json("fewshot.json")
    items = load_json(args.data)[: args.limit]
    if not items:
        raise SystemExit("no items: run exp1_accuracy.py first (or pass --data)")

    # ranks[lens][block] -> array [n_items, n_loops, 3]  (final, intermediate, "the")
    ranks = {lens: {b: [] for b in BLOCKS} for lens in ("logit", "coda")}
    for item in progress(items, "exp2"):
        ids = build_input_ids(runner.tok, fewshot, item["context"], args.device)
        init = runner.make_init_state(ids.shape[1], kind="seed", seed=args.seed)
        out = runner.run(ids, num_steps=args.steps, init_state=init, capture_blocks=BLOCKS)
        for b in BLOCKS:
            per_lens = {"logit": [], "coda": []}
            for state in out.block_states[b]:
                for lens, fn in (("logit", runner.logit_lens), ("coda", runner.coda_lens)):
                    logits = fn(state)
                    per_lens[lens].append([
                        runner.digit_rank(logits, item["final"]),
                        runner.digit_rank(logits, item["intermediate"]),
                        runner.rank_of(logits, runner.the_ids),
                    ])
            for lens in per_lens:
                ranks[lens][b].append(per_lens[lens])

    arr = {lens: {b: np.array(ranks[lens][b]) for b in BLOCKS} for lens in ranks}
    np.savez(results_path("exp2_ranks.npz"),
             **{f"{lens}_{NAMES[b]}": arr[lens][b] for lens in arr for b in BLOCKS})

    def summary(a):
        logs = np.log10(a)
        return {
            "geo_mean_rank": (10 ** logs.mean(0)).round(2).tolist(),       # [loop, token]
            "inter_ahead_of_final": (a[:, :, 1] < a[:, :, 0]).mean(0).round(3).tolist(),
        }

    out_json = {lens: {NAMES[b]: summary(arr[lens][b]) for b in BLOCKS} for lens in arr}
    out_json["n_items"] = len(items)
    out_json["token_order"] = ["final", "intermediate", "the"]
    save_json(out_json, "exp2_rank_summary.json")

    pyplot = plt()
    fig, axes = pyplot.subplots(1, 3, figsize=(15, 4.2))
    loops = np.arange(1, args.steps + 1)
    for ax, (lens, b) in zip(axes[:2], (("logit", 2), ("coda", 3))):
        a = np.log10(arr[lens][b])
        for j, (label, color) in enumerate((("final", "#1f77b4"), ("intermediate", "#d62728"), ("'the'", "#7f7f7f"))):
            m, s = a[:, :, j].mean(0), a[:, :, j].std(0)
            ax.plot(loops, 10 ** m, label=label, color=color)
            ax.fill_between(loops, 10 ** (m - s), 10 ** (m + s), color=color, alpha=0.15)
        ax.set_yscale("log")
        ax.set_xlabel("loop")
        ax.set_ylabel("rank (geometric mean, 1 = top)")
        ax.set_title(f"{lens} lens at {NAMES[b]}  (n={len(items)})")
        ax.legend()
    ax = axes[2]
    for lens, b, style in (("logit", 2, "-"), ("coda", 3, "--")):
        a = arr[lens][b]
        ax.plot(loops, (a[:, :, 1] < a[:, :, 0]).mean(0), style, label=f"{lens} lens @ {NAMES[b]}")
    ax.set_ylim(0, 1)
    ax.set_xlabel("loop")
    ax.set_ylabel("fraction of problems")
    ax.set_title("intermediate ranked above final")
    ax.legend()
    fig.tight_layout()
    fig.savefig(results_path("exp2_rank_trajectory.png"), dpi=150)

    print(f"\n{len(items)} problems, {args.steps} loops")
    for lens, b in (("logit", 2), ("coda", 3)):
        s = out_json[lens][NAMES[b]]
        first_top = [next((k + 1 for k, v in enumerate(col) if v <= 1.5), None)
                     for col in zip(*s["geo_mean_rank"])][:2]
        flip = s["inter_ahead_of_final"]
        peak = int(np.argmax(flip)) + 1
        print(f"  {lens} lens @ {NAMES[b]}: final reaches ~rank 1 at loop {first_top[0]}, "
              f"intermediate at loop {first_top[1]}; intermediate most often ahead at loop {peak} "
              f"({flip[peak - 1]:.0%} of problems)")
    print("  plot -> results/exp2_rank_trajectory.png")
    if len(items) < 20:
        print("  note: fewer than 20 problems; run exp1 on the pool for more")


if __name__ == "__main__":
    main()
