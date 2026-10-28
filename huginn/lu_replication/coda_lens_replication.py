"""Coda lens replication of Lu et al. (2025), arXiv 2507.02199.

  --part fig23   Fig 2/3 (right) + appendix tables 5-7, 100 questions
  --part fig4    Fig 4 (right) + appendix Fig 6, 67 questions
  --part all     both

Settings follow their repo: revision 2a364bd, 16 loops, set_seed(42) per question.
Results are compared against their saved pickles in data/lu_reference/.
Outputs: lu_replication/results_coda/
"""

import argparse
import json
import os
import pickle
import re
import sys
import time

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

from huginn_harness import Runner, build_input_ids, load_huginn  # noqa: E402

DATA = os.path.join(HERE, "data")
REF = os.path.join(DATA, "lu_reference")
OUT = os.path.join(HERE, "results_coda")
STEPS = 16
SEED = 42
BLOCKS = ("R1", "R2", "R3", "R4")


def is_signed_numeric(s):
    """Their rule (frequency_analysis_combined.py)."""
    return bool(re.fullmatch(r"(-\d+|\d+|-)", s))


def load_ref(name):
    with open(os.path.join(REF, name), "rb") as f:
        return np.array(pickle.load(f), dtype=object if "top5" in name else float)


def rank_of_id(logits, token_id):
    """1-based rank, same as their sort-and-find (ties resolved toward the better rank)."""
    return int((logits > logits[token_id]).sum()) + 1


def unrolled_states(runner, input_ids, init):
    """Run 16 loops; return [prelude1, prelude2, R1..R4 of loop 1, ..., R1..R4 of loop 16] (66 states)
    plus the model's predicted first token."""
    prelude_out = []
    hooks = [blk.register_forward_hook(lambda m, i, o: prelude_out.append(o[0].detach()))
             for blk in runner.model.transformer.prelude]
    try:
        out = runner.run(input_ids, STEPS, init, capture_blocks=(0, 1, 2, 3))
    finally:
        for h in hooks:
            h.remove()
    core = [out.block_states[b][k] for k in range(STEPS) for b in range(4)]
    return prelude_out + core, out.pred_id


def spearman(a, b):
    ra = np.argsort(np.argsort(a)).astype(float)
    rb = np.argsort(np.argsort(b)).astype(float)
    return float(np.corrcoef(ra, rb)[0, 1])


def plt():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as pyplot
    return pyplot




def run_fig23(runner, args, device, report):
    data = json.load(open(os.path.join(DATA, "arith_1dc_first104.json")))
    fewshot, tests = data[:4], data[4:104][: args.limit]
    their_ranks = load_ref("coda_arithmetic_rank_16_with_prelude.pkl")[: len(tests)]
    their_top5 = load_ref("coda_arithmetic_results_top5_16_with_prelude.pkl")[: len(tests)]
    their_preds = json.load(open(os.path.join(REF, "fig2_predicted_token_ids.json")))["token_ids"][: len(tests)]

    ranks = np.zeros((len(tests), 66))
    top5 = []
    preds = []
    t0 = time.time()
    for i, q in enumerate(tests):
        ids = build_input_ids(runner.tok, fewshot, q["context"], device)
        init = runner.make_init_state(ids.shape[1], kind="seed", seed=SEED)
        states, pred = unrolled_states(runner, ids, init)
        preds.append(pred)
        row_top5 = []
        for j, s in enumerate(states):
            logits = runner.coda_lens(s)
            ranks[i, j] = rank_of_id(logits, pred)
            if j >= 2:  # their top-5 file covers the 64 recurrent states only
                row_top5.append([runner.tok.decode([int(t)]) for t in logits.topk(5).indices])
        top5.append(row_top5)
        if (i + 1) % 10 == 0 or i == len(tests) - 1:
            print(f"  fig2/3: {i + 1}/{len(tests)} questions, {time.time() - t0:.0f}s")

    os.makedirs(OUT, exist_ok=True)
    np.save(os.path.join(OUT, "fig2_coda_ranks.npy"), ranks)
    json.dump({"pred_token_ids": preds, "top5": top5}, open(os.path.join(OUT, "fig2_fig3_coda_details.json"), "w"))

    agree = sum(a == b for a, b in zip(preds, their_preds))
    ours_mean, theirs_mean = ranks.mean(0), their_ranks.mean(0)
    rho = spearman(ours_mean, theirs_mean)
    r4_idx = [2 + 4 * k + 3 for k in range(STEPS)]
    other_idx = [2 + 4 * k + b for k in range(STEPS) for b in range(3)]

    def numeric_share(t5):
        share = np.zeros((4, STEPS))
        for row in t5:
            for j, toks in enumerate(row):
                k, b = divmod(j, 4)
                share[b, k] += sum(is_signed_numeric(t.strip()) for t in toks) / 5
        return share / len(t5)

    ours_share = numeric_share(top5)
    theirs_share = numeric_share(their_top5)

    report.append("## Fig 2 (coda lens, all 66 states)\n")
    report.append(f"- Predicted first token matches the paper's own list on **{agree}/{len(tests)}** questions "
                  "(checks that prompt, seed and model match theirs).")
    report.append(f"- Shape of the mean-rank curve vs the paper: Spearman correlation **{rho:.3f}** across the 66 positions.")
    report.append(f"- R4 dips (paper: ranks drop sharply at R4): mean rank at R4 = **{ours_mean[r4_idx].mean():.1f}** "
                  f"(paper {theirs_mean[r4_idx].mean():.1f}); at R1-R3 = **{ours_mean[other_idx].mean():.0f}** "
                  f"(paper {theirs_mean[other_idx].mean():.0f}).")
    report.append(f"- Final state (R4, loop 16): mean rank {ours_mean[-1]:.2f} (paper {theirs_mean[-1]:.2f}); "
                  "should be about 1, since the coda lens at the last R4 is the model's real output.\n")
    report.append("## Fig 3 (coda lens, share of top-5 tokens that are numbers or '-')\n")
    report.append("| Block | Ours (mean over loops) | Paper | Ours at loop 16 | Paper at loop 16 |")
    report.append("| --- | --- | --- | --- | --- |")
    for b in range(4):
        report.append(f"| {BLOCKS[b]} | {ours_share[b].mean():.2f} | {theirs_share[b].mean():.2f} | "
                      f"{ours_share[b, -1]:.2f} | {theirs_share[b, -1]:.2f} |")
    report.append(f"\nLargest gap at any block and loop: {np.abs(ours_share - theirs_share).max():.2f}\n")

    report.append("## Appendix Tables 5-7 (coda lens): top-5 tokens, first 3 questions\n")
    for step in (1, 8, 16):
        report.append(f"**Loop {step}**\n")
        report.append("| Block | Ours | Paper |")
        report.append("| --- | --- | --- |")
        for b in range(4):
            j = 4 * (step - 1) + b
            ours = "<br>".join(str(top5[q][j]) for q in range(min(3, len(top5))))
            theirs = "<br>".join(str(list(their_top5[q][j])) for q in range(min(3, len(top5))))
            report.append(f"| {BLOCKS[b]} | {ours} | {theirs} |")
        report.append("")

    p = plt()
    fig, axes = p.subplots(1, 2, figsize=(12, 4.2))
    x = np.arange(1, 67)
    axes[0].plot(x, theirs_mean, "o-", ms=3, color="#999999", label="paper (their saved results)")
    axes[0].plot(x, ours_mean, "o-", ms=3, color="#1f77b4", label="ours")
    axes[0].set_yscale("log")
    axes[0].set_xlabel("block number (1-2 prelude, then R1..R4 per loop)")
    axes[0].set_ylabel("mean rank of predicted token")
    axes[0].set_title(f"Fig 2 right: coda lens, s1 to s66 (n={len(tests)})")
    axes[0].legend()
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]
    steps = np.arange(1, STEPS + 1)
    for b in range(4):
        axes[1].plot(steps, ours_share[b], color=colors[b], lw=2.2, label=f"{BLOCKS[b]} ours")
        axes[1].plot(steps, theirs_share[b], color=colors[b], lw=1.2, ls="--", label=f"{BLOCKS[b]} paper")
    axes[1].set_ylim(-0.02, 1.02)
    axes[1].set_xlabel("loop")
    axes[1].set_ylabel("share of top-5 that are numeric")
    axes[1].set_title("Fig 3 right: coda lens (solid ours, dashed paper)")
    axes[1].legend(fontsize=7, ncol=2)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig2_fig3_coda.png"), dpi=150)
    print("  saved", os.path.join(OUT, "fig2_fig3_coda.png"))




def run_fig4(runner, args, device, report):
    data = json.load(open(os.path.join(DATA, "filtered_arithmetic_dataset.json")))
    fewshot, tests = data[:4], data[4:][: args.limit]
    their = {k: load_ref(f"coda_arithmetic_{k}_rank_16.pkl")[: len(tests)] for k in ("correct", "inter", "the")}

    ours = {k: np.zeros((len(tests), 64)) for k in ("correct", "inter", "the")}
    n_correct = 0
    t0 = time.time()
    for i, q in enumerate(tests):
        ids = build_input_ids(runner.tok, fewshot, q["context"], device)
        init = runner.make_init_state(ids.shape[1], kind="seed", seed=SEED)
        states, pred = unrolled_states(runner, ids, init)
        n_correct += runner.tok.decode([pred]).strip() == q["completion"].strip()
        inter_ids = runner.digit_ids(q["intermediate"].strip())
        for j, s in enumerate(states[2:]):
            logits = runner.coda_lens(s)
            ours["correct"][i, j] = rank_of_id(logits, pred)
            ours["inter"][i, j] = min(rank_of_id(logits, t) for t in inter_ids)
            ours["the"][i, j] = rank_of_id(logits, runner.the_ids[0])
        if (i + 1) % 10 == 0 or i == len(tests) - 1:
            print(f"  fig4: {i + 1}/{len(tests)} questions, {time.time() - t0:.0f}s")

    os.makedirs(OUT, exist_ok=True)
    for k, v in ours.items():
        np.save(os.path.join(OUT, f"fig4_coda_{k}_ranks.npy"), v)

    def block(arr, b):
        return arr[:, [4 * k + b for k in range(STEPS)]]

    report.append("## Fig 4 (coda lens at R4) and Appendix Fig 6 (R1-R3)\n")
    report.append(f"- Huginn answered **{n_correct}/{len(tests)}** of these correctly in our run "
                  "(the paper picked them because it got all of them right; a few misses are run-to-run noise).")
    report.append("\n| Block | Token | Ours: mean rank loop 1 / 6 / 16 | Paper: loop 1 / 6 / 16 | Curve shape (Spearman) |")
    report.append("| --- | --- | --- | --- | --- |")
    for b in range(4):
        for k in ("correct", "inter", "the"):
            o, t = block(ours[k], b).mean(0), block(their[k], b).mean(0)
            name = {"correct": "final", "inter": "intermediate", "the": "'the'"}[k]
            report.append(f"| {BLOCKS[b]} | {name} | {o[0]:.0f} / {o[5]:.0f} / {o[15]:.0f} | "
                          f"{t[0]:.0f} / {t[5]:.0f} / {t[15]:.0f} | {spearman(o, t):.2f} |")

    o_flip = (block(ours["inter"], 3) < block(ours["correct"], 3)).mean(0)
    t_flip = (block(their["inter"], 3) < block(their["correct"], 3)).mean(0)
    o_final_ahead = (block(ours["correct"], 3).mean(0) < block(ours["inter"], 3).mean(0)).mean()
    report.append(f"\n- Final token ranked better than the intermediate (mean, at R4) on **{o_final_ahead:.0%}** of loops "
                  "(paper: 'the final token consistently maintaining a lower rank').")
    report.append("- Share of questions where the intermediate outranks the final, by loop at R4 (the 'rank reversal'):")
    report.append("  - ours:  " + " ".join(f"{x:.2f}" for x in o_flip))
    report.append("  - paper: " + " ".join(f"{x:.2f}" for x in t_flip))
    def peak(x):
        return f"loop {int(np.argmax(x)) + 1}" if x.max() > 0 else "never"
    report.append(f"  - peak: {peak(o_flip)} for ours, {peak(t_flip)} for the paper\n")

    p = plt()

    def band(ax, arr, color, label, ls="-"):
        m = arr.mean(0)
        sd = arr.std(0, ddof=1) if len(arr) > 1 else np.zeros_like(m)
        rel = np.where(m > 1e-12, sd / np.maximum(m, 1e-12), 0)
        f = 10 ** (0.434 * rel)
        steps = np.arange(1, STEPS + 1)
        ax.plot(steps, m, color=color, ls=ls, lw=2, label=label)
        if ls == "-":
            ax.fill_between(steps, m / f, m * f, color=color, alpha=0.15)

    colors = {"inter": "#1f77b4", "correct": "#ff7f0e", "the": "#2ca02c"}
    names = {"inter": "intermediate", "correct": "final", "the": "'the'"}
    fig, axes = p.subplots(1, 4, figsize=(18, 4.2), sharey=True)
    for b in range(4):
        ax = axes[b]
        for k in ("inter", "correct", "the"):
            band(ax, block(ours[k], b), colors[k], f"{names[k]} (ours)")
            ax.plot(np.arange(1, STEPS + 1), block(their[k], b).mean(0), color=colors[k], ls="--", lw=1.2,
                    label=f"{names[k]} (paper)")
        ax.set_yscale("log")
        ax.set_title(f"Coda lens at {BLOCKS[b]}" + ("  (= Fig 4 right)" if b == 3 else "  (Fig 6)"))
        ax.set_xlabel("loop")
    axes[0].set_ylabel(f"rank (mean over {len(tests)} questions)")
    axes[3].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig4_fig6_coda.png"), dpi=150)
    print("  saved", os.path.join(OUT, "fig4_fig6_coda.png"))




def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--part", choices=["fig23", "fig4", "all"], default="all")
    p.add_argument("--limit", type=int, default=None, help="only the first N questions (quick look)")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--tiny", action="store_true", help="tiny random Huginn, for testing the code only")
    p.add_argument("--out", default="results_coda", help="output folder (relative to lu_replication/)")
    args = p.parse_args()
    global OUT
    OUT = os.path.join(HERE, args.out)

    if args.tiny:
        from tests.tiny_huginn import load_tiny
        model, tok = load_tiny()
        model.to(args.device)
    else:
        dtype = torch.bfloat16 if args.device.startswith("cuda") else torch.float32
        model, tok = load_huginn(device=args.device, dtype=dtype)
    runner = Runner(model, tok)

    report = ["# Coda lens replication of Lu et al. (2025)\n",
              f"Model huginn-0125 @ 2a364bd, {STEPS} loops, seed {SEED}, device {args.device}"
              + (" (TINY TEST MODEL, numbers meaningless)" if args.tiny else "")
              + (f", first {args.limit} questions only" if args.limit else "") + "\n",
              "Exact equality with the paper isn't expected (GPU kernels and library versions differ, and "
              "bf16 ties can reorder very low-ranked tokens). The shapes are what should match.\n"]
    t0 = time.time()
    if args.part in ("fig23", "all"):
        run_fig23(runner, args, args.device, report)
    if args.part in ("fig4", "all"):
        run_fig4(runner, args, args.device, report)
    report.append(f"\nTotal time: {time.time() - t0:.0f}s")

    os.makedirs(OUT, exist_ok=True)
    path = os.path.join(OUT, f"summary_{args.part}.md")
    open(path, "w").write("\n".join(report))
    print("\n" + "\n".join(report))
    print(f"\nsaved {path}")


if __name__ == "__main__":
    main()
