"""Effect of the starting state: several seeds, zeros, and scaled noise.
Reports accuracy per loop, cross-seed agreement, lock-in loop, state distance
between seeds, and per-loop state change.

  python exp3_random_start.py
  python exp3_random_start.py --limit 10 --seeds 4
"""

import numpy as np
import torch

from common import base_parser, get_runner, load_json, plt, progress, results_path, save_json
from huginn_harness import R4, build_input_ids


def main():
    p = base_parser(__doc__)
    p.add_argument("--data", default="two_step_50.json")
    p.add_argument("--steps", type=int, default=32)
    p.add_argument("--seeds", type=int, default=8)
    args = p.parse_args()

    runner = get_runner(args)
    fewshot = load_json("fewshot.json")
    items = load_json(args.data)[: args.limit]

    seed_names = [f"seed{s}" for s in range(args.seeds)]
    extra = [("zero", "zero", 0, 1.0), ("seed0 x0.5", "seed", 0, 0.5), ("seed0 x2", "seed", 0, 2.0)]
    conditions = [(n, "seed", s, 1.0) for n, s in zip(seed_names, range(args.seeds))] + extra
    cond_names = [c[0] for c in conditions]

    R = args.steps
    correct = {n: np.zeros((len(items), R), dtype=bool) for n in cond_names}
    preds = {n: [[None] * R for _ in items] for n in cond_names}
    seed_dist = np.zeros((len(items), R))      # mean pairwise relative distance across seeds
    step_change = np.zeros((len(items), R))    # ||s_k - s_{k-1}|| / ||s_k||, seed0

    for i, item in enumerate(progress(items, "exp3")):
        ids = build_input_ids(runner.tok, fewshot, item["context"], args.device)
        last_states = {}
        for name, kind, seed, scale in conditions:
            init = runner.make_init_state(ids.shape[1], kind=kind, seed=seed, scale=scale)
            out = runner.run(ids, num_steps=R, init_state=init, capture_blocks=(R4,))
            states = out.block_states[R4]
            for k, s in enumerate(states):
                logits = runner.coda_lens(s)
                pred = runner.tok.decode([int(logits.argmax())]).strip()
                preds[name][i][k] = pred
                correct[name][i, k] = pred == item["final"]
            if name in seed_names:
                last_states[name] = torch.stack([s[0, -1].float() for s in states])  # [R, d]

        seed0 = last_states[seed_names[0]]
        norms = seed0.norm(dim=-1)
        step_change[i, 0] = np.nan
        step_change[i, 1:] = ((seed0[1:] - seed0[:-1]).norm(dim=-1) / norms[1:]).cpu().numpy()
        stack = torch.stack([last_states[n] for n in seed_names])  # [S, R, d]
        dists = []
        for a in range(len(seed_names)):
            for b in range(a + 1, len(seed_names)):
                dists.append(((stack[a] - stack[b]).norm(dim=-1) / norms).cpu().numpy())
        seed_dist[i] = np.mean(dists, axis=0) if dists else np.nan

    # agreement across seeds, and the per-problem lock-in loop
    agree = np.zeros((len(items), R), dtype=bool)
    for i in range(len(items)):
        for k in range(R):
            agree[i, k] = len({preds[n][i][k] for n in seed_names}) == 1
    lock_in = []
    for i in range(len(items)):
        k_lock = None
        for k in range(R):
            if agree[i, k:].all():
                k_lock = k + 1
                break
        lock_in.append(k_lock)

    loops = np.arange(1, R + 1)
    step_change_mean = [None] + np.mean(step_change[:, 1:], 0).round(4).tolist()  # loop 1 has no previous loop
    seed_acc = np.stack([correct[n].mean(0) for n in seed_names])  # [S, R]
    summary = {
        "n_items": len(items),
        "loops": R,
        "accuracy_by_loop": {n: correct[n].mean(0).round(3).tolist() for n in cond_names},
        "seed_accuracy_spread_at": {str(r): [float(seed_acc[:, r - 1].min()), float(seed_acc[:, r - 1].max())]
                                    for r in (4, 8, 16, 32) if r <= R},
        "all_seeds_agree_by_loop": agree.mean(0).round(3).tolist(),
        "lock_in_loop_per_item": lock_in,
        "median_lock_in_loop": float(np.median([k for k in lock_in if k is not None])) if any(lock_in) else None,
        "never_locked_in": sum(k is None for k in lock_in),
        "seed_distance_by_loop": np.nanmean(seed_dist, 0).round(4).tolist(),
        "step_change_by_loop": step_change_mean,
        "items": [{"context": it["context"], "final": it["final"], "lock_in": lk,
                   "preds_seed0": preds[seed_names[0]][i]} for i, (it, lk) in enumerate(zip(items, lock_in))],
    }
    save_json(summary, "exp3_random_start.json")

    pyplot = plt()
    fig, axes = pyplot.subplots(1, 3, figsize=(15, 4.2))
    ax = axes[0]
    ax.fill_between(loops, seed_acc.min(0), seed_acc.max(0), color="#1f77b4", alpha=0.2,
                    label=f"{len(seed_names)} random seeds (min to max)")
    ax.plot(loops, seed_acc.mean(0), color="#1f77b4", label="seed mean")
    for name, color in (("zero", "#2ca02c"), ("seed0 x0.5", "#ff7f0e"), ("seed0 x2", "#d62728")):
        ax.plot(loops, correct[name].mean(0), color=color, lw=1.2, label=name)
    ax.set_xlabel("loops")
    ax.set_ylabel("accuracy")
    ax.set_title("accuracy vs loops by starting state")
    ax.legend(fontsize=8)

    ax = axes[1]
    ax.plot(loops, agree.mean(0), color="#9467bd")
    ax.set_ylim(0, 1.02)
    ax.set_xlabel("loop")
    ax.set_ylabel("fraction of problems")
    ax.set_title("all seeds give the same answer")

    ax = axes[2]
    ax.plot(loops, np.nanmean(seed_dist, 0), label="distance between seeds")
    ax.plot(loops[1:], step_change_mean[1:], label="change from previous loop")
    ax.set_yscale("log")
    ax.set_xlabel("loop")
    ax.set_ylabel("relative L2 (answer position)")
    ax.set_title("do different starts converge?")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(results_path("exp3_random_start.png"), dpi=150)

    print(f"\n{len(items)} problems, {R} loops, {len(seed_names)} seeds + zero + scaled")
    for r in (4, 8, 16, 32):
        if r <= R:
            lo, hi = summary["seed_accuracy_spread_at"][str(r)]
            print(f"  {r:>2} loops: seed accuracy {lo:.0%} to {hi:.0%} | zero {correct['zero'][:, r-1].mean():.0%}"
                  f" | x0.5 {correct['seed0 x0.5'][:, r-1].mean():.0%} | x2 {correct['seed0 x2'][:, r-1].mean():.0%}"
                  f" | all seeds agree on {agree[:, r-1].mean():.0%}")
    print(f"  median lock-in loop: {summary['median_lock_in_loop']}  (never locked in: {summary['never_locked_in']})")
    print("  plot -> results/exp3_random_start.png")


if __name__ == "__main__":
    main()
