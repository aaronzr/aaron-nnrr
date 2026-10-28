"""Accuracy on the two-step set at several loop counts.
Saves the items answered correctly at 16 loops to results/correct_r16.json.

  python exp1_accuracy.py
  python exp1_accuracy.py --data two_step_pool.json --out-prefix pool
"""

from common import Timer, base_parser, get_runner, load_json, progress, save_json
from huginn_harness import build_input_ids


def main():
    p = base_parser(__doc__)
    p.add_argument("--data", default="two_step_50.json")
    p.add_argument("--steps", type=int, nargs="+", default=[4, 8, 16, 32])
    p.add_argument("--seed", type=int, default=0, help="seed for Huginn's random starting state")
    p.add_argument("--out-prefix", default="exp1")
    args = p.parse_args()

    runner = get_runner(args)
    fewshot = load_json("fewshot.json")
    items = load_json(args.data)[: args.limit]

    per_item = []
    with Timer() as t:
        for item in progress(items, "exp1"):
            ids = build_input_ids(runner.tok, fewshot, item["context"], args.device)
            init = runner.make_init_state(ids.shape[1], kind="seed", seed=args.seed)
            row = {"context": item["context"], "final": item["final"],
                   "intermediate": item["intermediate"], "pattern": item["pattern"], "by_steps": {}}
            for r in args.steps:
                out = runner.run(ids, num_steps=r, init_state=init, capture_blocks=())
                row["by_steps"][str(r)] = {
                    "pred": out.pred_text,
                    "correct": out.pred_text == item["final"],
                    "final_rank": runner.digit_rank(out.logits, item["final"]),
                    "inter_rank": runner.digit_rank(out.logits, item["intermediate"]),
                }
            per_item.append(row)

    acc = {r: sum(x["by_steps"][str(r)]["correct"] for x in per_item) / len(per_item) for r in args.steps}
    by_pattern = {}
    ref = str(16 if 16 in args.steps else args.steps[-1])
    for x in per_item:
        b = by_pattern.setdefault(x["pattern"], [0, 0])
        b[0] += x["by_steps"][ref]["correct"]
        b[1] += 1

    save_json({"accuracy_by_steps": acc, "seed": args.seed, "n": len(per_item),
               "seconds": round(t.seconds, 1), "items": per_item}, f"{args.out_prefix}_accuracy.json")

    src = {i["context"]: i for i in load_json(args.data)}
    correct = [src[x["context"]] for x in per_item if x["by_steps"][ref]["correct"]]
    correct_name = f"correct_r{ref}.json" if args.out_prefix == "exp1" else f"{args.out_prefix}_correct_r{ref}.json"
    save_json(correct, correct_name)

    print(f"\n{len(per_item)} problems, starting-state seed {args.seed}, {t.seconds:.0f}s")
    for r in args.steps:
        print(f"  {r:>3} loops: accuracy {acc[r]:.0%}")
    print(f"  accuracy by op pattern at {ref} loops:",
          {k: f"{v[0]}/{v[1]}" for k, v in sorted(by_pattern.items())})
    print(f"  {len(correct)} problems correct at {ref} loops -> results/{correct_name}")
    wrong_as_inter = sum(x["by_steps"][ref]["pred"] == x["intermediate"] for x in per_item)
    print(f"  wrong answers that equal the intermediate result: {wrong_as_inter}")


if __name__ == "__main__":
    main()
